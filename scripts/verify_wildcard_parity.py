#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

if __package__:
    from . import install_wildcards
else:
    import install_wildcards


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify installer output against an existing complete asset tree.")
    parser.add_argument("--legacy-bundle", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = install_wildcards.load_manifest(install_wildcards.DEFAULT_MANIFEST)
    with tempfile.TemporaryDirectory(prefix="wildcard-parity-") as tmpdir:
        temporary_root = Path(tmpdir)
        extracted = temporary_root / "legacy"
        install_wildcards.safe_extract_zip(
            args.legacy_bundle.resolve(),
            extracted,
            max_uncompressed_bytes=600 * 1024 * 1024,
        )
        source_root = extracted / "wildcard"
        archive_dir = temporary_root / "archives"
        archive_dir.mkdir()
        test_manifest = make_test_manifest(manifest, source_root, archive_dir)

        destination = temporary_root / "template" / "wildcard"
        copy_project_wildcards(ROOT / "template" / "wildcard", destination)
        install_wildcards.install(
            test_manifest,
            destination=destination,
            accept_terms=True,
            force=False,
            archive_dir=archive_dir,
            cache_dir=temporary_root / "cache",
            civitai_proxy=None,
            token=None,
            timeout=10.0,
            retries=1,
        )

        reference_template = args.reference_root.resolve() / "template"
        compare_trees(destination, reference_template / "wildcard", excluded_suffixes={".html"})
        for directory in ("prompt_lists", "prompt_templates", "workflows"):
            compare_trees(
                ROOT / "template" / directory,
                reference_template / directory,
                excluded_parts={".ipynb_checkpoints"},
            )
    print("Installer output matches the reference prompt and wildcard assets.")
    return 0


def make_test_manifest(manifest: dict[str, Any], source_root: Path, archive_dir: Path) -> dict[str, Any]:
    payload = copy.deepcopy(manifest)
    for source in payload["sources"]:
        package_root = source_root / source["destination"]
        if not package_root.is_dir():
            raise install_wildcards.InstallerError(f"Legacy source package is missing: {package_root}")
        archive_path = archive_dir / source["archive_filename"]
        include_files = source.get("include_files")
        if include_files:
            files = [package_root / install_wildcards.safe_relative_path(value) for value in include_files]
        else:
            files = [path for path in package_root.rglob("*") if path.is_file()]
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(files):
                relative = Path(source["destination"]) / path.relative_to(package_root)
                archive.write(path, relative.as_posix())
        source["archive_size_bytes"] = archive_path.stat().st_size
        source["archive_sha256"] = install_wildcards.file_sha256(archive_path)
    return payload


def copy_project_wildcards(source_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for path in sorted(source_root.iterdir()):
        if path.name == "custom_character_list.txt":
            shutil.copyfile(path, destination / path.name)
        elif path.is_dir() and path.name.startswith("research_v"):
            shutil.copytree(path, destination / path.name, ignore=shutil.ignore_patterns("*.txt") if path.name == "research_v5" else None)
    source_v5 = source_root / "research_v5"
    destination_v5 = destination / "research_v5"
    for path in source_v5.rglob("*.txt"):
        if path.parent.name == "imported":
            continue
        target = destination_v5 / path.relative_to(source_v5)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def compare_trees(
    actual: Path,
    expected: Path,
    *,
    excluded_suffixes: set[str] | None = None,
    excluded_parts: set[str] | None = None,
) -> None:
    actual_files = file_map(actual, excluded_suffixes=excluded_suffixes, excluded_parts=excluded_parts)
    expected_files = file_map(expected, excluded_suffixes=excluded_suffixes, excluded_parts=excluded_parts)
    if actual_files == expected_files:
        return
    missing = sorted(set(expected_files) - set(actual_files))
    unexpected = sorted(set(actual_files) - set(expected_files))
    changed = sorted(path for path in set(actual_files) & set(expected_files) if actual_files[path] != expected_files[path])
    details = []
    if missing:
        details.append(f"missing={missing}")
    if unexpected:
        details.append(f"unexpected={unexpected}")
    if changed:
        details.append(f"changed={changed}")
    raise install_wildcards.InstallerError(f"Asset parity failed for {actual}: {'; '.join(details)}")


def file_map(
    root: Path,
    *,
    excluded_suffixes: set[str] | None,
    excluded_parts: set[str] | None,
) -> dict[str, str]:
    suffixes = excluded_suffixes or set()
    parts = excluded_parts or set()
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in suffixes or parts.intersection(path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[relative] = digest
    return result


if __name__ == "__main__":
    raise SystemExit(main())
