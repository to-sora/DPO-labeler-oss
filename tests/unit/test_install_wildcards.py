from __future__ import annotations

import copy
import hashlib
import io
import json
import ssl
import stat
import tempfile
import traceback
import unittest
import urllib.error
import urllib.parse
import zipfile
from pathlib import Path
from unittest import mock

from scripts import install_wildcards


class WildcardInstallerTests(unittest.TestCase):
    def test_terms_are_required_before_archive_access(self) -> None:
        manifest = install_wildcards.load_manifest(install_wildcards.DEFAULT_MANIFEST)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            install_wildcards, "acquire_archive"
        ) as acquire:
            with self.assertRaises(install_wildcards.TermsNotAccepted):
                install_wildcards.install(
                    manifest,
                    destination=Path(tmpdir) / "wildcard",
                    accept_terms=False,
                    force=False,
                    archive_dir=None,
                    cache_dir=Path(tmpdir) / "cache",
                    civitai_proxy=None,
                    token=None,
                    timeout=1.0,
                    retries=1,
                )
        acquire.assert_not_called()

    def test_proxy_and_token_are_scoped_to_civitai_hosts(self) -> None:
        proxy = "http://proxy.example:3128"
        token = "secret-token"

        self.assertEqual(install_wildcards.proxy_for_url("https://civitai.com/api/x", proxy), proxy)
        self.assertEqual(install_wildcards.proxy_for_url("https://api.civitai.com/x", proxy), proxy)
        self.assertEqual(install_wildcards.proxy_for_url("https://civitai.red/x", proxy), proxy)
        self.assertIsNone(install_wildcards.proxy_for_url("https://civitai.com.example/x", proxy))
        self.assertIsNone(install_wildcards.proxy_for_url("https://cdn.example/x", proxy))
        self.assertEqual(
            install_wildcards.request_headers("https://civitai.com/api/x", token)["Authorization"],
            "Bearer secret-token",
        )
        self.assertNotIn("Authorization", install_wildcards.request_headers("https://cdn.example/x", token))

    def test_proxy_ca_requires_proxy_and_legacy_mode_requires_ca(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ca_bundle = Path(tmpdir) / "proxy.crt"
            ca_bundle.write_text("certificate", encoding="utf-8")
            install_wildcards.validate_proxy_tls("http://proxy.example:8080", ca_bundle, False)
            with self.assertRaises(install_wildcards.InstallerError):
                install_wildcards.validate_proxy_tls(None, ca_bundle, False)
            with self.assertRaises(install_wildcards.InstallerError):
                install_wildcards.validate_proxy_tls("http://proxy.example:8080", None, True)

    @unittest.skipUnless(hasattr(ssl, "VERIFY_X509_STRICT"), "strict X.509 verification is unavailable")
    def test_legacy_proxy_ca_relaxes_only_strict_x509_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ca_bundle = Path(tmpdir) / "proxy.crt"
            ca_bundle.write_text("certificate", encoding="utf-8")
            context = mock.Mock()
            context.verify_flags = ssl.VERIFY_X509_STRICT | ssl.VERIFY_X509_TRUSTED_FIRST
            with mock.patch.object(install_wildcards.ssl, "create_default_context", return_value=context):
                result = install_wildcards.build_proxy_ssl_context(
                    ca_bundle,
                    allow_legacy_proxy_ca=True,
                )

            self.assertIs(result, context)
            context.load_verify_locations.assert_called_once_with(cafile=str(ca_bundle))
            self.assertFalse(context.verify_flags & ssl.VERIFY_X509_STRICT)
            self.assertTrue(context.verify_flags & ssl.VERIFY_X509_TRUSTED_FIRST)

    def test_query_token_proxy_and_ca_are_removed_after_first_redirect(self) -> None:
        proxy_context = object()
        token = "secret token"
        signed_query = "signature=A%2BB%2FC&credential=folder%2Fname"
        redirect_url = "https://civitai.com/storage/archive.zip?" + urllib.parse.urlencode(
            {"token": token}
        ) + f"&{signed_query}"
        redirect = urllib.error.HTTPError(
            "https://civitai.com/api/download/models/1",
            307,
            "Temporary Redirect",
            {"Location": redirect_url},
            io.BytesIO(),
        )
        first_opener = mock.Mock()
        first_opener.open.side_effect = redirect
        final_response = mock.Mock()
        second_opener = mock.Mock()
        second_opener.open.return_value = final_response

        with (
            mock.patch.object(install_wildcards, "build_proxy_ssl_context", return_value=proxy_context),
            mock.patch.object(
                install_wildcards.urllib.request,
                "ProxyHandler",
                side_effect=lambda mapping: ("proxy", mapping),
            ) as proxy_handler,
            mock.patch.object(install_wildcards.urllib.request, "HTTPSHandler", return_value="custom-ca"),
            mock.patch.object(
                install_wildcards.urllib.request,
                "build_opener",
                side_effect=[first_opener, second_opener],
            ) as build_opener,
        ):
            response = install_wildcards.open_with_scoped_proxy(
                "https://civitai.com/api/download/models/1",
                civitai_proxy="http://proxy.example:8080",
                civitai_ca_bundle=Path("proxy.crt"),
                allow_legacy_proxy_ca=True,
                token_in_query=True,
                direct_redirects=True,
                token=token,
                timeout=1.0,
            )

        self.assertIs(response, final_response)
        self.assertIn("custom-ca", build_opener.call_args_list[0].args)
        self.assertNotIn("custom-ca", build_opener.call_args_list[1].args)
        proxy_calls = proxy_handler.call_args_list
        self.assertEqual(
            proxy_calls[0].args[0],
            {"http": "http://proxy.example:8080", "https": "http://proxy.example:8080"},
        )
        self.assertEqual(proxy_calls[1].args[0], {})

        first_request = first_opener.open.call_args.args[0]
        second_request = second_opener.open.call_args.args[0]
        self.assertEqual(
            urllib.parse.parse_qs(urllib.parse.urlsplit(first_request.full_url).query)["token"],
            [token],
        )
        self.assertNotIn("Authorization", dict(first_request.header_items()))
        self.assertNotIn(token, second_request.full_url)
        self.assertEqual(second_request.full_url, f"https://civitai.com/storage/archive.zip?{signed_query}")
        self.assertNotIn("Authorization", dict(second_request.header_items()))

    def test_auth_routing_requires_its_dependencies(self) -> None:
        with self.assertRaisesRegex(install_wildcards.InstallerError, "CIVITAI_API_TOKEN"):
            install_wildcards.validate_auth_routing(
                "http://proxy.example:8080",
                None,
                token_in_query=True,
                direct_redirects=False,
            )
        with self.assertRaisesRegex(install_wildcards.InstallerError, "--civitai-proxy"):
            install_wildcards.validate_auth_routing(
                None,
                "secret",
                token_in_query=False,
                direct_redirects=True,
            )

    def test_urls_and_network_errors_do_not_expose_query_token(self) -> None:
        token = "secret-token"
        self.assertEqual(
            install_wildcards.redacted_url(f"https://civitai.com/download?token={token}#fragment"),
            "https://civitai.com/download",
        )
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.URLError(f"failed URL token={token}")
        with mock.patch.object(install_wildcards.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(install_wildcards.InstallerError) as raised:
                install_wildcards.open_with_scoped_proxy(
                    "https://civitai.com/api/download/models/1",
                    civitai_proxy="http://proxy.example:8080",
                    token_in_query=True,
                    token=token,
                    timeout=1.0,
                )

        self.assertNotIn(token, str(raised.exception))
        self.assertIn("[redacted]", str(raised.exception))
        self.assertNotIn(token, "".join(traceback.format_exception(raised.exception)))

    def test_safe_extract_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "bad")

            with self.assertRaises(install_wildcards.UnsafeArchive):
                install_wildcards.safe_extract_zip(archive, root / "extract", max_uncompressed_bytes=1024)

            self.assertFalse((root / "escape.txt").exists())

    def test_safe_extract_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "symlink.zip"
            member = zipfile.ZipInfo("pack/link")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(member, "../../outside")

            with self.assertRaises(install_wildcards.UnsafeArchive):
                install_wildcards.safe_extract_zip(archive, root / "extract", max_uncompressed_bytes=1024)

    def test_safe_relative_path_rejects_windows_drive_paths(self) -> None:
        with self.assertRaises(install_wildcards.InstallerError):
            install_wildcards.safe_relative_path("C:/escape.txt")

    def test_failed_replacement_restores_existing_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage_root = root / "stage"
            destination = root / "destination"
            backup_root = root / "backup"
            (stage_root / "Pack").mkdir(parents=True)
            (destination / "Pack").mkdir(parents=True)
            (stage_root / "Pack" / "value.txt").write_text("new\n", encoding="utf-8")
            (destination / "Pack" / "value.txt").write_text("old\n", encoding="utf-8")
            manifest = {"sources": [{"destination": "Pack"}]}
            real_replace = install_wildcards.os.replace

            def fail_staged_move(source: object, target: object) -> None:
                if Path(source) == stage_root / "Pack" and Path(target) == destination / "Pack":
                    raise OSError("simulated replacement failure")
                real_replace(source, target)

            with mock.patch.object(install_wildcards.os, "replace", side_effect=fail_staged_move):
                with self.assertRaisesRegex(OSError, "simulated replacement failure"):
                    install_wildcards.commit_staged_install(
                        manifest,
                        stage_root,
                        destination,
                        force=True,
                        backup_root=backup_root,
                    )

            self.assertEqual((destination / "Pack" / "value.txt").read_text(), "old\n")
            self.assertEqual((stage_root / "Pack" / "value.txt").read_text(), "new\n")

    def test_installs_transforms_verifies_and_replaces_only_with_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_dir = root / "archives"
            archive_dir.mkdir()
            archive = archive_dir / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("wrapper/Pack/raw.txt", b"alpha\r\nold\r\nlast")
                handle.writestr("wrapper/Pack/list.txt", "first\nsecond\nthird\n")

            expected = root / "expected"
            pack = expected / "Pack"
            pack.mkdir(parents=True)
            (pack / "raw.txt").write_text("alpha\nnew\nlast\n", encoding="utf-8")
            (pack / "list.txt").write_text("first\nsecond\nthird\n", encoding="utf-8")
            (pack / "written.txt").write_text("written\n", encoding="utf-8")
            (pack / "copied.txt").write_text("written\n", encoding="utf-8")
            pack_count, pack_hash = install_wildcards.tree_digest(pack)

            derived_payload = "second\nproject\n"
            derived_hash = hashlib.sha256(derived_payload.encode("utf-8")).hexdigest()
            archive_hash = install_wildcards.file_sha256(archive)
            manifest = {
                "schema_version": 1,
                "terms": {"notice": "test terms"},
                "sources": [
                    {
                        "id": "source",
                        "title": "Source",
                        "creator": "tester",
                        "model_id": 1,
                        "version_id": 2,
                        "model_url": "https://civitai.com/models/1?modelVersionId=2",
                        "download_url": "https://civitai.com/api/download/models/2",
                        "archive_filename": archive.name,
                        "archive_sha256": archive_hash,
                        "archive_size_bytes": archive.stat().st_size,
                        "max_uncompressed_bytes": 4096,
                        "destination": "Pack",
                        "sentinel": "raw.txt",
                        "expected_file_count": pack_count,
                        "expected_tree_sha256": pack_hash,
                    }
                ],
                "text_replacements": [
                    {
                        "path": "Pack/raw.txt",
                        "replacements": [{"old": "old", "new": "new", "expected_occurrences": 1}],
                    }
                ],
                "writes": [{"path": "Pack/written.txt", "content": "written\n"}],
                "copies": [{"source": "Pack/written.txt", "target": "Pack/copied.txt"}],
                "directory_copies": [
                    {
                        "source": "Pack",
                        "target": "PackCopy",
                        "expected_file_count": pack_count,
                        "expected_tree_sha256": pack_hash,
                    }
                ],
                "derived_line_sets": [
                    {
                        "target": "research/imported.txt",
                        "selectors": [
                            {"source": "Pack/list.txt", "lines": [2]},
                            {"source": "research/base.txt", "lines": [1]},
                        ],
                        "sha256": derived_hash,
                    }
                ],
            }
            manifest = install_wildcards.load_manifest(write_manifest(root, manifest))
            destination = root / "destination"
            (destination / "research").mkdir(parents=True)
            (destination / "research" / "base.txt").write_text("project\n", encoding="utf-8")

            install_test_manifest(manifest, destination, archive_dir, root / "cache", force=False)
            install_wildcards.verify_install(manifest, destination)
            self.assertEqual((destination / "research" / "imported.txt").read_text(), derived_payload)
            self.assertEqual(install_wildcards.tree_digest(destination / "Pack"), (pack_count, pack_hash))
            self.assertEqual(install_wildcards.tree_digest(destination / "PackCopy"), (pack_count, pack_hash))

            install_test_manifest(manifest, destination, archive_dir, root / "cache", force=False)
            (destination / "Pack" / "raw.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(install_wildcards.InstallerError):
                install_test_manifest(manifest, destination, archive_dir, root / "cache", force=False)
            self.assertEqual((destination / "Pack" / "raw.txt").read_text(), "changed\n")

            install_test_manifest(manifest, destination, archive_dir, root / "cache", force=True)
            install_wildcards.verify_install(manifest, destination)

    def test_production_manifest_has_unique_pinned_sources(self) -> None:
        manifest = install_wildcards.load_manifest(install_wildcards.DEFAULT_MANIFEST)
        self.assertEqual(len(manifest["sources"]), 7)
        self.assertEqual(len({row["archive_sha256"] for row in manifest["sources"]}), 7)
        self.assertTrue(all(len(row["archive_sha256"]) == 64 for row in manifest["sources"]))
        self.assertTrue(all(row["download_url"].startswith("https://civitai.com/") for row in manifest["sources"]))


def write_manifest(root: Path, manifest: dict[str, object]) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def install_test_manifest(
    manifest: dict[str, object],
    destination: Path,
    archive_dir: Path,
    cache_dir: Path,
    *,
    force: bool,
) -> None:
    install_wildcards.install(
        copy.deepcopy(manifest),
        destination=destination,
        accept_terms=True,
        force=force,
        archive_dir=archive_dir,
        cache_dir=cache_dir,
        civitai_proxy=None,
        token=None,
        timeout=1.0,
        retries=1,
    )


if __name__ == "__main__":
    unittest.main()
