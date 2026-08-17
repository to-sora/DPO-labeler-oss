from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


CANONICAL_CHECKPOINT_FAMILIES = frozenset(
    {"illustration", "sdxl_anime_base", "pony", "realistic"}
)
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("checkpoint_aliases.yaml")
DEFAULT_LOCAL_REGISTRY_PATH = Path(__file__).with_name("checkpoint_aliases.local.yaml")
REGISTRY_OVERLAY_ENV = "CHECKPOINT_ALIAS_REGISTRY"
VALID_VISIBILITIES = frozenset({"public", "private", "unknown"})


class CheckpointRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class KeywordRule:
    family: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointModel:
    model_id: str
    aliases: tuple[str, ...]
    family: str | None
    visibility: str
    publish: bool


@dataclass(frozen=True)
class KeywordClassification:
    family: str
    keyword: str


@dataclass(frozen=True)
class CheckpointResolution:
    checkpoint: str
    family: str
    family_source: str
    matched_keyword: str | None
    model_id: str | None
    visibility: str
    publish: bool


def _normalize_alias(value: str) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def _keyword_matches(text: str, keyword: str) -> bool:
    normalized_text = _normalize_alias(text)
    normalized_keyword = _normalize_alias(keyword)
    if normalized_keyword in normalized_text:
        return True
    compact_text = re.sub(r"[^a-z0-9]+", "", normalized_text)
    compact_keyword = re.sub(r"[^a-z0-9]+", "", normalized_keyword)
    return bool(compact_keyword) and compact_keyword in compact_text


class CheckpointRegistry:
    def __init__(
        self,
        *,
        default_family: str,
        keyword_rules: Iterable[KeywordRule],
        models: Iterable[CheckpointModel],
    ) -> None:
        if default_family not in CANONICAL_CHECKPOINT_FAMILIES:
            raise CheckpointRegistryError(
                f"Unsupported default checkpoint family {default_family!r}"
            )
        self.default_family = default_family
        self.keyword_rules = tuple(keyword_rules)
        self.models = tuple(models)

        alias_lookup: dict[str, CheckpointModel] = {}
        basename_candidates: dict[str, list[CheckpointModel]] = {}
        model_ids: set[str] = set()
        for model in self.models:
            if model.model_id in model_ids:
                raise CheckpointRegistryError(f"Duplicate checkpoint model id {model.model_id!r}")
            model_ids.add(model.model_id)
            for alias in model.aliases:
                normalized = _normalize_alias(alias)
                if not normalized:
                    raise CheckpointRegistryError(
                        f"Checkpoint model {model.model_id!r} contains an empty alias"
                    )
                previous = alias_lookup.get(normalized)
                if previous is not None and previous.model_id != model.model_id:
                    raise CheckpointRegistryError(
                        f"Checkpoint alias {alias!r} belongs to both "
                        f"{previous.model_id!r} and {model.model_id!r}"
                    )
                alias_lookup[normalized] = model
                basename = normalized.rsplit("/", 1)[-1]
                basename_candidates.setdefault(basename, []).append(model)

        self._alias_lookup = alias_lookup
        self._basename_lookup = {
            basename: candidates[0]
            for basename, candidates in basename_candidates.items()
            if len({candidate.model_id for candidate in candidates}) == 1
        }

    def find_model(self, checkpoint: str) -> CheckpointModel | None:
        normalized = _normalize_alias(checkpoint)
        exact = self._alias_lookup.get(normalized)
        if exact is not None:
            return exact
        return self._basename_lookup.get(normalized.rsplit("/", 1)[-1])

    def classify_family_by_keywords(
        self,
        checkpoint: str,
        *,
        extra_text: Iterable[str] = (),
    ) -> KeywordClassification | None:
        search_text = " ".join([str(checkpoint), *(str(value) for value in extra_text)])
        for rule in self.keyword_rules:
            for keyword in rule.keywords:
                if _keyword_matches(search_text, keyword):
                    return KeywordClassification(family=rule.family, keyword=keyword)
        return None

    def resolve(self, checkpoint: str) -> CheckpointResolution:
        model = self.find_model(checkpoint)
        if model is not None and model.family is not None:
            return CheckpointResolution(
                checkpoint=str(checkpoint),
                family=model.family,
                family_source="registry",
                matched_keyword=None,
                model_id=model.model_id,
                visibility=model.visibility,
                publish=model.publish,
            )

        extra_text: tuple[str, ...] = ()
        if model is not None:
            extra_text = (model.model_id, *model.aliases)
        classified = self.classify_family_by_keywords(checkpoint, extra_text=extra_text)
        if classified is not None:
            return CheckpointResolution(
                checkpoint=str(checkpoint),
                family=classified.family,
                family_source="keyword",
                matched_keyword=classified.keyword,
                model_id=model.model_id if model is not None else None,
                visibility=model.visibility if model is not None else "unknown",
                publish=model.publish if model is not None else False,
            )

        return CheckpointResolution(
            checkpoint=str(checkpoint),
            family=self.default_family,
            family_source="default",
            matched_keyword=None,
            model_id=model.model_id if model is not None else None,
            visibility=model.visibility if model is not None else "unknown",
            publish=model.publish if model is not None else False,
        )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CheckpointRegistryError(f"Cannot read checkpoint registry {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CheckpointRegistryError(f"Invalid checkpoint registry YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointRegistryError(f"Checkpoint registry {path} must contain a mapping")
    if payload.get("version") != 1:
        raise CheckpointRegistryError(
            f"Checkpoint registry {path} must declare version: 1"
        )
    return payload


def _merge_registry_payloads(payloads: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"version": 1, "models": {}}
    for payload in payloads:
        for key in ("default_family", "keyword_rules", "model_defaults"):
            if key in payload:
                merged[key] = payload[key]
        models = payload.get("models", {})
        if not isinstance(models, Mapping):
            raise CheckpointRegistryError("checkpoint registry models must be a mapping")
        merged["models"].update(models)
    return merged


def _parse_registry(payload: Mapping[str, Any]) -> CheckpointRegistry:
    default_family = str(payload.get("default_family", "sdxl_anime_base"))

    raw_rules = payload.get("keyword_rules", [])
    if not isinstance(raw_rules, list) or not raw_rules:
        raise CheckpointRegistryError("checkpoint registry keyword_rules must be a non-empty list")
    keyword_rules: list[KeywordRule] = []
    for index, raw_rule in enumerate(raw_rules, start=1):
        if not isinstance(raw_rule, Mapping):
            raise CheckpointRegistryError(f"keyword_rules[{index}] must be a mapping")
        family = str(raw_rule.get("family", ""))
        if family not in CANONICAL_CHECKPOINT_FAMILIES:
            raise CheckpointRegistryError(
                f"keyword_rules[{index}] has unsupported family {family!r}"
            )
        raw_keywords = raw_rule.get("keywords", [])
        if not isinstance(raw_keywords, list) or not raw_keywords:
            raise CheckpointRegistryError(
                f"keyword_rules[{index}].keywords must be a non-empty list"
            )
        keywords = tuple(str(keyword).strip() for keyword in raw_keywords)
        if any(not keyword for keyword in keywords):
            raise CheckpointRegistryError(
                f"keyword_rules[{index}].keywords contains an empty keyword"
            )
        keyword_rules.append(KeywordRule(family=family, keywords=keywords))

    defaults = payload.get("model_defaults", {})
    if not isinstance(defaults, Mapping):
        raise CheckpointRegistryError("checkpoint registry model_defaults must be a mapping")
    default_visibility = str(defaults.get("visibility", "unknown"))
    default_publish = defaults.get("publish", False)

    raw_models = payload.get("models", {})
    if not isinstance(raw_models, Mapping) or not raw_models:
        raise CheckpointRegistryError("checkpoint registry models must be a non-empty mapping")
    models: list[CheckpointModel] = []
    for raw_model_id, raw_model in raw_models.items():
        model_id = str(raw_model_id).strip()
        if not model_id or not isinstance(raw_model, Mapping):
            raise CheckpointRegistryError(f"Invalid checkpoint model entry {raw_model_id!r}")
        raw_aliases = raw_model.get("aliases", [])
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise CheckpointRegistryError(
                f"Checkpoint model {model_id!r} must provide a non-empty aliases list"
            )
        family_value = raw_model.get("family")
        family = None if family_value in (None, "") else str(family_value)
        if family is not None and family not in CANONICAL_CHECKPOINT_FAMILIES:
            raise CheckpointRegistryError(
                f"Checkpoint model {model_id!r} has unsupported family {family!r}"
            )
        visibility = str(raw_model.get("visibility", default_visibility))
        if visibility not in VALID_VISIBILITIES:
            raise CheckpointRegistryError(
                f"Checkpoint model {model_id!r} has unsupported visibility {visibility!r}"
            )
        publish = raw_model.get("publish", default_publish)
        if not isinstance(publish, bool):
            raise CheckpointRegistryError(
                f"Checkpoint model {model_id!r} publish must be a boolean"
            )
        models.append(
            CheckpointModel(
                model_id=model_id,
                aliases=tuple(str(alias) for alias in raw_aliases),
                family=family,
                visibility=visibility,
                publish=publish,
            )
        )

    return CheckpointRegistry(
        default_family=default_family,
        keyword_rules=keyword_rules,
        models=models,
    )


def load_checkpoint_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
    *,
    overlay_paths: Iterable[str | Path] = (),
) -> CheckpointRegistry:
    paths = [Path(path), *(Path(value) for value in overlay_paths)]
    payloads = [_load_yaml_mapping(candidate) for candidate in paths]
    return _parse_registry(_merge_registry_payloads(payloads))


@lru_cache(maxsize=1)
def get_checkpoint_registry() -> CheckpointRegistry:
    overlays: list[Path] = []
    if DEFAULT_LOCAL_REGISTRY_PATH.is_file():
        overlays.append(DEFAULT_LOCAL_REGISTRY_PATH)
    env_overlay = os.environ.get(REGISTRY_OVERLAY_ENV, "").strip()
    if env_overlay:
        overlays.append(Path(env_overlay))
    return load_checkpoint_registry(DEFAULT_REGISTRY_PATH, overlay_paths=overlays)


def resolve_checkpoint(checkpoint: str) -> CheckpointResolution:
    return get_checkpoint_registry().resolve(checkpoint)


def resolve_checkpoint_family(checkpoint: str) -> str:
    return resolve_checkpoint(checkpoint).family
