#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "assets" / "wildcard_sources.json"
DEFAULT_DESTINATION = ROOT / "template" / "wildcard"
DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "comfy-dpo" / "wildcards"
MAX_ARCHIVE_MEMBERS = 10_000
CHUNK_SIZE = 1024 * 1024


class InstallerError(RuntimeError):
    pass


class TermsNotAccepted(InstallerError):
    pass


class UnsafeArchive(InstallerError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and install the optional third-party wildcard packs.")
    parser.add_argument(
        "--accept-third-party-terms",
        action="store_true",
        help="Confirm that you chose these downloads and accept the source terms",
    )
    parser.add_argument("--force", action="store_true", help="Replace installed wildcard files that differ")
    parser.add_argument("--verify-only", action="store_true", help="Verify installed files without downloading")
    parser.add_argument("--list-sources", action="store_true", help="List pinned sources without downloading")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Source manifest path")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION, help="Wildcard destination root")
    parser.add_argument("--archive-dir", type=Path, help="Use already-downloaded archives from this directory")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Verified archive cache directory")
    parser.add_argument(
        "--civitai-proxy",
        default=os.environ.get("CIVITAI_PROXY", ""),
        help="Proxy used only for Civitai hosts; CIVITAI_PROXY is also supported",
    )
    parser.add_argument(
        "--civitai-ca-bundle",
        type=Path,
        default=os.environ.get("CIVITAI_CA_BUNDLE") or None,
        help="Additional CA certificate used only with the Civitai proxy",
    )
    parser.add_argument(
        "--allow-legacy-proxy-ca",
        action="store_true",
        help="Allow an explicitly trusted proxy CA that lacks strict X.509 extensions",
    )
    parser.add_argument(
        "--civitai-wait-seconds",
        type=float,
        default=os.environ.get("CIVITAI_WAIT_SECONDS", "0"),
        help="Delay between Civitai archive requests",
    )
    parser.add_argument(
        "--civitai-token-query",
        action="store_true",
        help="Send the API token in the first Civitai URL when a proxy strips Authorization",
    )
    parser.add_argument(
        "--civitai-direct-redirects",
        action="store_true",
        help="Bypass the Civitai proxy after the first HTTPS redirect",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Download attempts")
    parser.add_argument("--allow-insecure-test-urls", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.list_sources:
            print_sources(manifest)
            return 0
        if args.verify_only:
            verify_install(manifest, args.destination.resolve())
            print(f"Wildcard installation verified: {args.destination.resolve()}")
            return 0
        install(
            manifest,
            destination=args.destination.resolve(),
            accept_terms=bool(args.accept_third_party_terms),
            force=bool(args.force),
            archive_dir=args.archive_dir.resolve() if args.archive_dir else None,
            cache_dir=args.cache_dir.resolve(),
            civitai_proxy=str(args.civitai_proxy).strip() or None,
            civitai_ca_bundle=args.civitai_ca_bundle.expanduser().resolve() if args.civitai_ca_bundle else None,
            allow_legacy_proxy_ca=bool(args.allow_legacy_proxy_ca),
            civitai_wait_seconds=float(args.civitai_wait_seconds),
            civitai_token_query=bool(args.civitai_token_query),
            civitai_direct_redirects=bool(args.civitai_direct_redirects),
            token=os.environ.get("CIVITAI_API_TOKEN", "").strip() or None,
            timeout=float(args.timeout),
            retries=int(args.retries),
            allow_insecure_test_urls=bool(args.allow_insecure_test_urls),
        )
    except InstallerError as exc:
        print(f"Wildcard installation failed: {exc}", file=sys.stderr)
        return 2
    return 0


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallerError(f"Cannot load manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise InstallerError(f"Unsupported wildcard manifest schema in {path}")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise InstallerError(f"Manifest has no wildcard sources: {path}")
    seen_ids: set[str] = set()
    seen_destinations: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise InstallerError("Each wildcard source must be an object")
        source_id = required_string(source, "id")
        destination = required_string(source, "destination")
        safe_relative_path(destination)
        safe_relative_path(required_string(source, "sentinel"))
        if source_id in seen_ids:
            raise InstallerError(f"Duplicate wildcard source id: {source_id}")
        if destination in seen_destinations:
            raise InstallerError(f"Duplicate wildcard destination: {destination}")
        seen_ids.add(source_id)
        seen_destinations.add(destination)
    return payload


def required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InstallerError(f"Manifest field {key!r} must be a non-empty string")
    return value


def safe_relative_path(value: str) -> Path:
    if "\x00" in value or "\\" in value:
        raise InstallerError(f"Unsafe relative path: {value!r}")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or ":" in pure.parts[0]
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise InstallerError(f"Unsafe relative path: {value!r}")
    return Path(*pure.parts)


def print_sources(manifest: dict[str, Any]) -> None:
    print(manifest.get("terms", {}).get("notice", "Third-party wildcard sources:"))
    for source in manifest["sources"]:
        print(
            f"- {source['archive_filename']} | creator={source['creator']} | "
            f"model={source['model_id']} version={source['version_id']} | "
            f"license={source.get('license_status', 'unknown')} | "
            f"provenance={source.get('provenance_status', 'unknown')} | {source['model_url']}"
        )


def install(
    manifest: dict[str, Any],
    *,
    destination: Path,
    accept_terms: bool,
    force: bool,
    archive_dir: Path | None,
    cache_dir: Path,
    civitai_proxy: str | None,
    token: str | None,
    timeout: float,
    retries: int,
    civitai_ca_bundle: Path | None = None,
    allow_legacy_proxy_ca: bool = False,
    civitai_wait_seconds: float = 0.0,
    civitai_token_query: bool = False,
    civitai_direct_redirects: bool = False,
    allow_insecure_test_urls: bool = False,
) -> None:
    if not accept_terms:
        notice = manifest.get("terms", {}).get("notice", "Third-party terms must be accepted.")
        raise TermsNotAccepted(f"{notice} Re-run with --accept-third-party-terms.")
    if timeout <= 0:
        raise InstallerError("Timeout must be greater than zero")
    if retries < 1:
        raise InstallerError("Retries must be at least one")
    if civitai_wait_seconds < 0:
        raise InstallerError("Civitai wait time cannot be negative")
    validate_proxy(civitai_proxy)
    validate_proxy_tls(civitai_proxy, civitai_ca_bundle, allow_legacy_proxy_ca)
    validate_auth_routing(
        civitai_proxy,
        token,
        token_in_query=civitai_token_query,
        direct_redirects=civitai_direct_redirects,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archives: dict[str, Path] = {}
    for index, source in enumerate(manifest["sources"]):
        if index and civitai_wait_seconds:
            time.sleep(civitai_wait_seconds)
        archives[source["id"]] = acquire_archive(
            source,
            archive_dir=archive_dir,
            cache_dir=cache_dir,
            civitai_proxy=civitai_proxy,
            civitai_ca_bundle=civitai_ca_bundle,
            allow_legacy_proxy_ca=allow_legacy_proxy_ca,
            civitai_token_query=civitai_token_query,
            civitai_direct_redirects=civitai_direct_redirects,
            token=token,
            timeout=timeout,
            retries=retries,
            allow_insecure_test_urls=allow_insecure_test_urls,
        )

    with tempfile.TemporaryDirectory(prefix=".wildcard-install-", dir=destination.parent) as tmpdir:
        temporary_root = Path(tmpdir)
        stage_root = temporary_root / "stage"
        extraction_root = temporary_root / "extract"
        stage_root.mkdir()
        extraction_root.mkdir()

        for source in manifest["sources"]:
            print(f"Staging {source['archive_filename']}")
            source_extract = extraction_root / source["id"]
            safe_extract_zip(
                archives[source["id"]],
                source_extract,
                max_uncompressed_bytes=int(source["max_uncompressed_bytes"]),
            )
            payload_root = locate_payload_root(
                source_extract,
                sentinel=safe_relative_path(source["sentinel"]),
                expected_directory=source["destination"],
            )
            include_files = source.get("include_files")
            copy_payload(
                payload_root,
                stage_root / safe_relative_path(source["destination"]),
                include_files=include_files,
            )

        apply_text_replacements(stage_root, manifest.get("text_replacements", []))
        apply_writes(stage_root, manifest.get("writes", []))
        apply_file_copies(stage_root, manifest.get("copies", []))
        apply_directory_copies(stage_root, manifest.get("directory_copies", []))
        apply_derived_line_sets(stage_root, destination, manifest.get("derived_line_sets", []))
        verify_staged_install(manifest, stage_root)
        commit_staged_install(manifest, stage_root, destination, force=force, backup_root=temporary_root / "backup")

    verify_install(manifest, destination)
    print(f"Wildcard installation complete: {destination}")


def validate_proxy(proxy: str | None) -> None:
    if not proxy:
        return
    parsed = urllib.parse.urlparse(proxy)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise InstallerError("Civitai proxy must be an http:// or https:// URL")


def validate_proxy_tls(proxy: str | None, ca_bundle: Path | None, allow_legacy_proxy_ca: bool) -> None:
    if ca_bundle is not None and not proxy:
        raise InstallerError("A Civitai CA bundle can only be used with a Civitai proxy")
    if allow_legacy_proxy_ca and ca_bundle is None:
        raise InstallerError("--allow-legacy-proxy-ca requires --civitai-ca-bundle")
    if ca_bundle is not None and not ca_bundle.is_file():
        raise InstallerError(f"Civitai CA bundle does not exist: {ca_bundle}")


def validate_auth_routing(
    proxy: str | None,
    token: str | None,
    *,
    token_in_query: bool,
    direct_redirects: bool,
) -> None:
    if token_in_query and not token:
        raise InstallerError("--civitai-token-query requires CIVITAI_API_TOKEN")
    if direct_redirects and not proxy:
        raise InstallerError("--civitai-direct-redirects requires --civitai-proxy")


def build_proxy_ssl_context(ca_bundle: Path, *, allow_legacy_proxy_ca: bool) -> ssl.SSLContext:
    context = ssl.create_default_context()
    try:
        context.load_verify_locations(cafile=str(ca_bundle))
    except (OSError, ssl.SSLError) as exc:
        raise InstallerError(f"Cannot load Civitai proxy CA bundle {ca_bundle}: {exc}") from exc
    if allow_legacy_proxy_ca and hasattr(ssl, "VERIFY_X509_STRICT"):
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


def is_civitai_url(url: str) -> bool:
    hostname = (urllib.parse.urlparse(url).hostname or "").lower().rstrip(".")
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in ("civitai.com", "civitai.red"))


def proxy_for_url(url: str, civitai_proxy: str | None) -> str | None:
    return civitai_proxy if civitai_proxy and is_civitai_url(url) else None


def request_headers(url: str, token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/octet-stream",
        "Accept-Encoding": "identity",
        "User-Agent": "comfy-dpo-wildcard-installer/1",
    }
    if token and is_civitai_url(url):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def add_query_token(url: str, token: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query = [(key, value) for key, value in query if key.lower() != "token"]
    query.append(("token", token))
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )


def remove_query_secret(url: str, secret: str | None) -> str:
    if not secret:
        return url
    parsed = urllib.parse.urlsplit(url)
    query_parts: list[str] = []
    for part in parsed.query.split("&"):
        raw_key, separator, raw_value = part.partition("=")
        key = urllib.parse.unquote_plus(raw_key)
        value = urllib.parse.unquote_plus(raw_value) if separator else ""
        if key.lower() == "token" and value == secret:
            continue
        query_parts.append(part)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "&".join(query_parts), parsed.fragment)
    )


def redacted_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def acquire_archive(
    source: dict[str, Any],
    *,
    archive_dir: Path | None,
    cache_dir: Path,
    civitai_proxy: str | None,
    civitai_ca_bundle: Path | None,
    allow_legacy_proxy_ca: bool,
    civitai_token_query: bool,
    civitai_direct_redirects: bool,
    token: str | None,
    timeout: float,
    retries: int,
    allow_insecure_test_urls: bool,
) -> Path:
    filename = required_string(source, "archive_filename")
    if Path(filename).name != filename:
        raise InstallerError(f"Unsafe archive filename: {filename!r}")
    if archive_dir is not None:
        local_archive = archive_dir / filename
        if local_archive.is_file():
            verify_archive(local_archive, source)
            print(f"Using local archive: {filename}", flush=True)
            return local_archive

    cached_archive = cache_dir / filename
    if cached_archive.is_file():
        try:
            verify_archive(cached_archive, source)
        except InstallerError:
            pass
        else:
            print(f"Using verified cache: {filename}", flush=True)
            return cached_archive

    url = required_string(source, "download_url")
    validate_download_url(url, allow_insecure_test_urls=allow_insecure_test_urls)
    print(f"Downloading {filename} from Civitai", flush=True)
    download_archive(
        url,
        cached_archive,
        source,
        civitai_proxy=civitai_proxy,
        civitai_ca_bundle=civitai_ca_bundle,
        allow_legacy_proxy_ca=allow_legacy_proxy_ca,
        civitai_token_query=civitai_token_query,
        civitai_direct_redirects=civitai_direct_redirects,
        token=token,
        timeout=timeout,
        retries=retries,
    )
    return cached_archive


def validate_download_url(url: str, *, allow_insecure_test_urls: bool) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https" and parsed.hostname:
        return
    if allow_insecure_test_urls and parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return
    raise InstallerError(f"Refusing non-HTTPS download URL: {redacted_url(url)}")


def verify_archive(path: Path, source: dict[str, Any]) -> None:
    expected_size = int(source["archive_size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise InstallerError(
            f"Archive size mismatch for {path.name}: expected {expected_size}, received {actual_size}"
        )
    expected_hash = required_string(source, "archive_sha256").lower()
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise InstallerError(
            f"Archive checksum mismatch for {path.name}: expected {expected_hash}, received {actual_hash}"
        )


def download_archive(
    url: str,
    target: Path,
    source: dict[str, Any],
    *,
    civitai_proxy: str | None,
    civitai_ca_bundle: Path | None,
    allow_legacy_proxy_ca: bool,
    civitai_token_query: bool,
    civitai_direct_redirects: bool,
    token: str | None,
    timeout: float,
    retries: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False) as tmp:
                temporary_path = Path(tmp.name)
                response = open_with_scoped_proxy(
                    url,
                    civitai_proxy=civitai_proxy,
                    civitai_ca_bundle=civitai_ca_bundle,
                    allow_legacy_proxy_ca=allow_legacy_proxy_ca,
                    token_in_query=civitai_token_query,
                    direct_redirects=civitai_direct_redirects,
                    token=token,
                    timeout=timeout,
                )
                with response:
                    expected_size = int(source["archive_size_bytes"])
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > expected_size:
                        raise InstallerError(
                            f"Download is larger than the pinned archive size for {source['archive_filename']}"
                        )
                    digest = hashlib.sha256()
                    size = 0
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > expected_size:
                            raise InstallerError(
                                f"Download exceeded the pinned archive size for {source['archive_filename']}"
                            )
                        digest.update(chunk)
                        tmp.write(chunk)
            if size != int(source["archive_size_bytes"]):
                raise InstallerError(
                    f"Downloaded size mismatch for {source['archive_filename']}: "
                    f"expected {source['archive_size_bytes']}, received {size}"
                )
            if digest.hexdigest() != source["archive_sha256"].lower():
                raise InstallerError(f"Downloaded checksum mismatch for {source['archive_filename']}")
            os.replace(temporary_path, target)
            temporary_path = None
            return
        except (InstallerError, OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))
    raise InstallerError(f"Could not download {source['archive_filename']}: {last_error}") from last_error


def open_with_scoped_proxy(
    url: str,
    *,
    civitai_proxy: str | None,
    civitai_ca_bundle: Path | None = None,
    allow_legacy_proxy_ca: bool = False,
    token_in_query: bool = False,
    direct_redirects: bool = False,
    token: str | None,
    timeout: float,
    max_redirects: int = 8,
) -> Any:
    current_url = add_query_token(url, token) if token and token_in_query else url
    proxy_context = (
        build_proxy_ssl_context(civitai_ca_bundle, allow_legacy_proxy_ca=allow_legacy_proxy_ca)
        if civitai_proxy and civitai_ca_bundle
        else None
    )
    for redirect_count in range(max_redirects + 1):
        proxy_allowed = redirect_count == 0 or not direct_redirects
        proxy = proxy_for_url(current_url, civitai_proxy) if proxy_allowed else None
        proxy_map = {"http": proxy, "https": proxy} if proxy else {}
        handlers: list[Any] = [urllib.request.ProxyHandler(proxy_map), _NoRedirect()]
        if proxy and proxy_context is not None:
            handlers.append(urllib.request.HTTPSHandler(context=proxy_context))
        opener = urllib.request.build_opener(*handlers)
        request_token = token if redirect_count == 0 and not token_in_query else None
        try:
            request = urllib.request.Request(
                current_url,
                headers=request_headers(current_url, request_token),
                method="GET",
            )
            return opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                location = exc.headers.get("Location")
                exc.close()
                if not location:
                    raise InstallerError(f"Redirect without Location from {redacted_url(current_url)}") from None
                current_url = remove_query_secret(urllib.parse.urljoin(current_url, location), token)
                validate_download_url(current_url, allow_insecure_test_urls=False)
                continue
            status = exc.code
            exc.close()
            if status in {401, 403} and is_civitai_url(current_url):
                raise InstallerError(
                    "Civitai denied the download. Check CIVITAI_API_TOKEN and CIVITAI_PROXY; "
                    f"HTTP {status} from {redacted_url(current_url)}"
                ) from None
            raise InstallerError(f"HTTP {status} while downloading {redacted_url(current_url)}") from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            reason = str(exc.reason) if isinstance(exc, urllib.error.URLError) else str(exc)
            if token:
                reason = reason.replace(token, "[redacted]")
            raise InstallerError(
                f"Network error while downloading {redacted_url(current_url)}: {reason}"
            ) from None
    raise InstallerError(f"Too many redirects while downloading {redacted_url(url)}")


def safe_extract_zip(archive_path: Path, destination: Path, *, max_uncompressed_bytes: int) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    destination_resolved = destination.resolve()
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsafeArchive(f"Invalid ZIP archive {archive_path.name}: {exc}") from exc
    with archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise UnsafeArchive(f"Archive has too many members: {archive_path.name}")
        total_size = 0
        normalized_names: set[str] = set()
        for member in members:
            relative = validate_zip_member(member)
            normalized = relative.as_posix().casefold()
            if normalized in normalized_names:
                raise UnsafeArchive(f"Duplicate or case-colliding ZIP member: {member.filename}")
            normalized_names.add(normalized)
            total_size += member.file_size
            if total_size > max_uncompressed_bytes:
                raise UnsafeArchive(f"Archive expands beyond its configured limit: {archive_path.name}")
            target = (destination / relative).resolve()
            try:
                target.relative_to(destination_resolved)
            except ValueError as exc:
                raise UnsafeArchive(f"ZIP member escapes destination: {member.filename}") from exc

        for member in members:
            relative = safe_relative_path(member.filename.rstrip("/")) if member.filename.rstrip("/") else None
            if relative is None:
                continue
            target = destination / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source_handle, target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=CHUNK_SIZE)


def validate_zip_member(member: zipfile.ZipInfo) -> Path:
    name = member.filename
    if not name or "\x00" in name or "\\" in name:
        raise UnsafeArchive(f"Unsafe ZIP member path: {name!r}")
    stripped = name.rstrip("/")
    if not stripped:
        return Path(".")
    try:
        relative = safe_relative_path(stripped)
    except InstallerError as exc:
        raise UnsafeArchive(str(exc)) from exc
    mode = (member.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise UnsafeArchive(f"ZIP symlinks are not allowed: {name}")
    if member.flag_bits & 0x1:
        raise UnsafeArchive(f"Encrypted ZIP members are not allowed: {name}")
    return relative


def locate_payload_root(extraction_root: Path, *, sentinel: Path, expected_directory: str) -> Path:
    candidates: list[Path] = []
    sentinel_parts = sentinel.parts
    for path in extraction_root.rglob(sentinel.name):
        if not path.is_file():
            continue
        relative = path.relative_to(extraction_root)
        if len(relative.parts) < len(sentinel_parts) or relative.parts[-len(sentinel_parts) :] != sentinel_parts:
            continue
        root = path.parents[len(sentinel_parts) - 1]
        candidates.append(root)
    preferred = [path for path in candidates if path.name == expected_directory]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise InstallerError(
        f"Could not identify one payload root for {expected_directory}; found {len(candidates)} candidates"
    )


def copy_payload(payload_root: Path, target: Path, *, include_files: Any) -> None:
    target.mkdir(parents=True, exist_ok=False)
    if include_files is not None:
        if not isinstance(include_files, list) or not include_files:
            raise InstallerError("include_files must be a non-empty list")
        relative_files = [safe_relative_path(str(value)) for value in include_files]
    else:
        relative_files = [path.relative_to(payload_root) for path in payload_root.rglob("*") if path.is_file()]
    for relative in relative_files:
        source = payload_root / relative
        if not source.is_file() or source.is_symlink():
            raise InstallerError(f"Expected wildcard payload file is missing: {relative.as_posix()}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def apply_text_replacements(stage_root: Path, operations: Any) -> None:
    for operation in operations:
        path = stage_root / safe_relative_path(required_string(operation, "path"))
        if operation.get("normalize_newlines", True) is False:
            apply_binary_text_replacements(path, operation)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise InstallerError(f"Cannot edit installed wildcard file {path}: {exc}") from exc
        for replacement in operation.get("replacements", []):
            old = required_string(replacement, "old")
            new = required_string(replacement, "new")
            expected = int(replacement.get("expected_occurrences", 1))
            actual = text.count(old)
            if actual != expected:
                raise InstallerError(
                    f"Replacement precondition failed for {path}: expected {expected} occurrences, found {actual}"
                )
            text = text.replace(old, new)
        path.write_text("\n".join(text.splitlines()) + "\n", encoding="utf-8")


def apply_binary_text_replacements(path: Path, operation: dict[str, Any]) -> None:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise InstallerError(f"Cannot edit installed wildcard file {path}: {exc}") from exc
    for replacement in operation.get("replacements", []):
        old = required_string(replacement, "old").encode("utf-8")
        new = required_string(replacement, "new").encode("utf-8")
        expected = int(replacement.get("expected_occurrences", 1))
        actual = payload.count(old)
        if actual != expected:
            raise InstallerError(
                f"Replacement precondition failed for {path}: expected {expected} occurrences, found {actual}"
            )
        payload = payload.replace(old, new)
    if not payload.endswith(b"\n"):
        payload += b"\n"
    path.write_bytes(payload)


def apply_writes(stage_root: Path, operations: Any) -> None:
    for operation in operations:
        path = stage_root / safe_relative_path(required_string(operation, "path"))
        content = operation.get("content")
        if not isinstance(content, str):
            raise InstallerError(f"Write content must be text: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def apply_file_copies(stage_root: Path, operations: Any) -> None:
    for operation in operations:
        source = stage_root / safe_relative_path(required_string(operation, "source"))
        target = stage_root / safe_relative_path(required_string(operation, "target"))
        if not source.is_file():
            raise InstallerError(f"Copy source is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def apply_directory_copies(stage_root: Path, operations: Any) -> None:
    for operation in operations:
        source = stage_root / safe_relative_path(required_string(operation, "source"))
        target = stage_root / safe_relative_path(required_string(operation, "target"))
        if not source.is_dir():
            raise InstallerError(f"Directory copy source is missing: {source}")
        shutil.copytree(source, target)


def apply_derived_line_sets(stage_root: Path, installed_root: Path, operations: Any) -> None:
    for operation in operations:
        target = stage_root / safe_relative_path(required_string(operation, "target"))
        output_lines: list[str] = []
        for selector in operation.get("selectors", []):
            relative_source = safe_relative_path(required_string(selector, "source"))
            source = stage_root / relative_source
            if not source.is_file():
                source = installed_root / relative_source
            if not source.is_file():
                raise InstallerError(f"Derived wildcard source is missing: {relative_source.as_posix()}")
            lines = source.read_text(encoding="utf-8").splitlines()
            for line_number in selector.get("lines", []):
                index = int(line_number) - 1
                if index < 0 or index >= len(lines):
                    raise InstallerError(f"Line {line_number} is missing from {relative_source.as_posix()}")
                value = lines[index].strip()
                if not value:
                    raise InstallerError(f"Line {line_number} is empty in {relative_source.as_posix()}")
                output_lines.append(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        expected_hash = required_string(operation, "sha256").lower()
        actual_hash = file_sha256(target)
        if actual_hash != expected_hash:
            raise InstallerError(
                f"Derived wildcard checksum mismatch for {target.relative_to(stage_root).as_posix()}"
            )


def verify_staged_install(manifest: dict[str, Any], stage_root: Path) -> None:
    for source in manifest["sources"]:
        verify_tree(
            stage_root / safe_relative_path(source["destination"]),
            expected_count=int(source["expected_file_count"]),
            expected_hash=source["expected_tree_sha256"],
        )
    for operation in manifest.get("directory_copies", []):
        verify_tree(
            stage_root / safe_relative_path(operation["target"]),
            expected_count=int(operation["expected_file_count"]),
            expected_hash=operation["expected_tree_sha256"],
        )


def verify_install(manifest: dict[str, Any], destination: Path) -> None:
    for source in manifest["sources"]:
        verify_tree(
            destination / safe_relative_path(source["destination"]),
            expected_count=int(source["expected_file_count"]),
            expected_hash=source["expected_tree_sha256"],
        )
    for operation in manifest.get("directory_copies", []):
        verify_tree(
            destination / safe_relative_path(operation["target"]),
            expected_count=int(operation["expected_file_count"]),
            expected_hash=operation["expected_tree_sha256"],
        )
    for operation in manifest.get("derived_line_sets", []):
        path = destination / safe_relative_path(operation["target"])
        if not path.is_file() or file_sha256(path) != operation["sha256"].lower():
            raise InstallerError(f"Derived wildcard verification failed: {path}")
    derived_tree = manifest.get("derived_tree")
    if derived_tree:
        verify_tree(
            destination / safe_relative_path(derived_tree["path"]),
            expected_count=int(derived_tree["expected_file_count"]),
            expected_hash=derived_tree["expected_tree_sha256"],
        )


def verify_tree(path: Path, *, expected_count: int, expected_hash: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise InstallerError(f"Installed wildcard directory is missing: {path}")
    count, digest = tree_digest(path)
    if count != expected_count or digest != expected_hash.lower():
        raise InstallerError(
            f"Wildcard tree mismatch for {path}: expected {expected_count}/{expected_hash}, received {count}/{digest}"
        )


def tree_digest(root: Path) -> tuple[int, str]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink():
            raise InstallerError(f"Wildcard tree contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit_staged_install(
    manifest: dict[str, Any],
    stage_root: Path,
    destination: Path,
    *,
    force: bool,
    backup_root: Path,
) -> None:
    directory_targets = [safe_relative_path(source["destination"]) for source in manifest["sources"]]
    directory_targets.extend(
        safe_relative_path(operation["target"]) for operation in manifest.get("directory_copies", [])
    )
    file_targets = [safe_relative_path(operation["target"]) for operation in manifest.get("derived_line_sets", [])]
    targets = directory_targets + file_targets
    replacements: list[tuple[Path, Path | None]] = []
    pending: list[Path] = []
    for relative in targets:
        staged = stage_root / relative
        installed = destination / relative
        if paths_equal(staged, installed):
            continue
        if (installed.exists() or installed.is_symlink()) and not force:
            raise InstallerError(f"Refusing to replace existing wildcard path without --force: {installed}")
        pending.append(relative)

    backup_root.mkdir(parents=True, exist_ok=True)
    try:
        for relative in pending:
            staged = stage_root / relative
            installed = destination / relative
            backup: Path | None = None
            installed.parent.mkdir(parents=True, exist_ok=True)
            if installed.exists() or installed.is_symlink():
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(installed, backup)
            replacements.append((installed, backup))
            os.replace(staged, installed)
        verify_install(manifest, destination)
    except Exception:
        for installed, backup in reversed(replacements):
            remove_path(installed)
            if backup is not None and (backup.exists() or backup.is_symlink()):
                installed.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, installed)
        raise


def paths_equal(first: Path, second: Path) -> bool:
    if first.is_file():
        return second.is_file() and not second.is_symlink() and file_sha256(first) == file_sha256(second)
    if first.is_dir():
        return second.is_dir() and not second.is_symlink() and tree_digest(first) == tree_digest(second)
    raise InstallerError(f"Staged wildcard path is missing: {first}")


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
