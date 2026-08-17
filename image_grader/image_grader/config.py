from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODELS_ROOT = "models/image_eval"
DEFAULT_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CacheConfig:
    fingerprint: str = "sample_sha256"
    sample_bytes: int = 1024 * 1024

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CacheConfig":
        payload = payload or {}
        fingerprint = str(payload.get("fingerprint", cls.fingerprint)).strip()
        if fingerprint not in {"sample_sha256", "full_sha256", "path_mtime"}:
            raise ConfigError("cache.fingerprint must be one of: sample_sha256, full_sha256, path_mtime")
        sample_bytes = int(payload.get("sample_bytes", cls.sample_bytes))
        if sample_bytes <= 0:
            raise ConfigError("cache.sample_bytes must be greater than zero")
        return cls(fingerprint=fingerprint, sample_bytes=sample_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {"fingerprint": self.fingerprint, "sample_bytes": self.sample_bytes}


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    kind: str
    path: str | None = None
    batch_size: int = 16
    clip_model: str | None = None
    input_size: int | None = None
    score_label: str = "hq"
    trusted_pickle: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, model_id: str, payload: Mapping[str, Any]) -> "ModelConfig":
        if not isinstance(payload, Mapping):
            raise ConfigError(f"models.{model_id} must be an object")
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            raise ConfigError(f"models.{model_id}.kind is required")
        batch_size = int(payload.get("batch_size", 16))
        if batch_size <= 0:
            raise ConfigError(f"models.{model_id}.batch_size must be greater than zero")
        known = {"kind", "path", "batch_size", "clip_model", "input_size", "score_label", "trusted_pickle"}
        options = {str(key): value for key, value in payload.items() if key not in known}
        return cls(
            model_id=model_id,
            kind=kind,
            path=str(payload["path"]) if payload.get("path") is not None else None,
            batch_size=batch_size,
            clip_model=str(payload["clip_model"]) if payload.get("clip_model") is not None else None,
            input_size=int(payload["input_size"]) if payload.get("input_size") is not None else None,
            score_label=str(payload.get("score_label", "hq")),
            trusted_pickle=_parse_bool(payload.get("trusted_pickle", False)),
            options=options,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "batch_size": self.batch_size,
            "score_label": self.score_label,
            "trusted_pickle": self.trusted_pickle,
        }
        if self.path is not None:
            data["path"] = self.path
        if self.clip_model is not None:
            data["clip_model"] = self.clip_model
        if self.input_size is not None:
            data["input_size"] = self.input_size
        data.update(self.options)
        return data


@dataclass(frozen=True)
class ServerConfig:
    allowed_roots: tuple[str, ...] = ()
    max_items_per_request: int = 256

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ServerConfig":
        payload = payload or {}
        raw_roots = payload.get("allowed_roots", ())
        if raw_roots is None:
            raw_roots = ()
        if not isinstance(raw_roots, list | tuple):
            raise ConfigError("server.allowed_roots must be a list of paths")
        roots = tuple(str(item) for item in raw_roots)
        max_items = int(payload.get("max_items_per_request", 256))
        if max_items <= 0:
            raise ConfigError("server.max_items_per_request must be greater than zero")
        return cls(allowed_roots=roots, max_items_per_request=max_items)

    def to_dict(self) -> dict[str, Any]:
        return {"allowed_roots": list(self.allowed_roots), "max_items_per_request": self.max_items_per_request}


@dataclass(frozen=True)
class GraderConfig:
    device: str = "cuda:0"
    dtype: str = "auto"
    models_root: str = DEFAULT_MODELS_ROOT
    enabled_models: tuple[str, ...] = ()
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS
    cache: CacheConfig = field(default_factory=CacheConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GraderConfig":
        if not isinstance(payload, Mapping):
            raise ConfigError("config must be a JSON object")
        models_payload = payload.get("models")
        if not isinstance(models_payload, Mapping) or not models_payload:
            raise ConfigError("config.models must be a non-empty object")
        models = {
            str(model_id): ModelConfig.from_mapping(str(model_id), model_payload)
            for model_id, model_payload in models_payload.items()
        }
        raw_enabled = payload.get("enabled_models", tuple(models))
        if not isinstance(raw_enabled, list | tuple):
            raise ConfigError("enabled_models must be a list")
        enabled = tuple(str(item) for item in raw_enabled)
        missing = [model_id for model_id in enabled if model_id not in models]
        if missing:
            raise ConfigError(f"enabled_models references unknown model ids: {missing}")
        raw_extensions = payload.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS)
        if not isinstance(raw_extensions, list | tuple):
            raise ConfigError("image_extensions must be a list")
        extensions = tuple(_normalize_extension(str(item)) for item in raw_extensions)
        return cls(
            device=str(payload.get("device", "cuda:0")),
            dtype=str(payload.get("dtype", "auto")),
            models_root=str(payload.get("models_root", DEFAULT_MODELS_ROOT)),
            enabled_models=enabled,
            image_extensions=extensions,
            cache=CacheConfig.from_mapping(payload.get("cache")),
            server=ServerConfig.from_mapping(payload.get("server")),
            models=models,
        )

    def selected_model_ids(self, override: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
        selected = tuple(str(item) for item in override) if override else self.enabled_models
        missing = [model_id for model_id in selected if model_id not in self.models]
        if missing:
            raise ConfigError(f"unknown model ids: {missing}")
        return selected

    def resolve_model_path(self, model: ModelConfig) -> Path:
        if model.path is None:
            raise ConfigError(f"models.{model.model_id}.path is required")
        path = Path(model.path)
        if path.is_absolute():
            return path
        return Path(self.models_root) / path

    def model_config_hash(self, model_id: str, *, preprocess_policy: str = "native") -> str:
        model = self.models[model_id]
        material = {
            "version": 1,
            "device_policy": self.device,
            "dtype": self.dtype,
            "cache": self.cache.to_dict(),
            "preprocess_policy": str(preprocess_policy or "native"),
            "model_id": model_id,
            "model": model.to_dict(),
        }
        return sha256_json(material)

    def public_config(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "models_root": self.models_root,
            "enabled_models": list(self.enabled_models),
            "image_extensions": list(self.image_extensions),
            "cache": self.cache.to_dict(),
            "server": self.server.to_dict(),
            "models": {model_id: model.to_dict() for model_id, model in self.models.items()},
        }


def load_config(path: str | Path) -> GraderConfig:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON config {config_path}: {exc}") from exc
    return GraderConfig.from_mapping(payload)


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_extension(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ConfigError("image_extensions cannot contain empty values")
    if not normalized.startswith("."):
        normalized = "." + normalized
    return normalized


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"expected boolean-like value, got {value!r}")
