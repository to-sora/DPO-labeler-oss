from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .config import CacheConfig


@dataclass(frozen=True)
class FileFingerprint:
    image_id: str
    size_bytes: int
    mtime_ns: int


def fingerprint_file(path: str | Path, config: CacheConfig) -> FileFingerprint:
    image_path = Path(path)
    stat = image_path.stat()
    if config.fingerprint == "full_sha256":
        digest = _full_sha256(image_path)
    elif config.fingerprint == "path_mtime":
        material = f"{image_path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8")
        digest = hashlib.sha256(material).hexdigest()
    else:
        digest = _sample_sha256(image_path, stat.st_size, config.sample_bytes)
    return FileFingerprint(image_id=digest, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _full_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sample_sha256(path: Path, size_bytes: int, sample_bytes: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"sample_sha256:v1:{size_bytes}:".encode("ascii"))
    with path.open("rb") as handle:
        if size_bytes <= sample_bytes * 3:
            hasher.update(handle.read())
            return hasher.hexdigest()

        offsets = (0, max((size_bytes - sample_bytes) // 2, 0), max(size_bytes - sample_bytes, 0))
        for offset in offsets:
            handle.seek(offset)
            hasher.update(offset.to_bytes(8, "little", signed=False))
            hasher.update(handle.read(sample_bytes))
    return hasher.hexdigest()
