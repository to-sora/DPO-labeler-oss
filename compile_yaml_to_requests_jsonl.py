from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml

from checkpoint_registry import (
    CANONICAL_CHECKPOINT_FAMILIES,
    resolve_checkpoint,
    resolve_checkpoint_family,
)
from layout_paths import PROMPT_LISTS_DIR, PROMPT_TEMPLATES_DIR, WILDCARD_DIR
from prompt_generator import get_prompt_generator

COMPILER_VERSION = "v3"


REQUIRED_IMAGE_FIELDS = [
    "image_name",
    "workflow_name",
    "ckpt",
    "lora_stack_config",
    "prompt_generator",
    "sample",
]
REQUIRED_SAMPLE_FIELDS = ["steps", "cfg", "width", "height", "generation_seed_control"]
WILDCARD_PROMPT_GENERATOR_NAME = "wildcard_template_generator"
PROMPT_LIST_V1_GENERATOR_NAME = "prompt_list_v1"
NON_WILDCARD_V1_GENERATOR_NAME = "non_wildcard_v1"
NON_WILDCARD_V2_GENERATOR_NAME = "non_wildcard_v2"
NON_WILDCARD_FAMILY_ROUTED_GENERATORS = frozenset(
    {NON_WILDCARD_V1_GENERATOR_NAME, NON_WILDCARD_V2_GENERATOR_NAME}
)
FAMILY_ROUTED_GENERATORS = frozenset(
    {WILDCARD_PROMPT_GENERATOR_NAME, PROMPT_LIST_V1_GENERATOR_NAME, *NON_WILDCARD_FAMILY_ROUTED_GENERATORS}
)
VALID_SEED_CONTROL_NAMES = frozenset({"global_seed", "session_seed", "image_index_seed"})
OPTION_SPEC_KEYS = frozenset({"seed_control", "options"})
WILDCARD_DROPOUT_KEYS = frozenset({"dropout_items", "dropout_probs", "seed_control_dropout"})
RANDOM_SEGMENT_DROPOUT_KEYS = frozenset(
    {
        "random_segment_dropout_pair_chance",
        "random_segment_dropout_segment_prob",
        "seed_control_random_segment_dropout",
    }
)
VALID_CKPT_FAMILY_NAMES = frozenset(
    {"anime", "illustration", "sdxl_anime_base", "pony", "realistic"}
)
CANONICAL_CKPT_FAMILY_NAMES = CANONICAL_CHECKPOINT_FAMILIES
LEGACY_FAMILY_FALLBACKS = {
    "illustration": ("anime",),
    "sdxl_anime_base": ("anime",),
}


def load_task_yaml(task_yaml: str | Path) -> Dict[str, Any]:
    path = Path(task_yaml)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Task YAML must be a mapping object at the top level")
    return data


def stable_hash_hex(*parts: Any) -> str:
    joined = "||".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def derive_uint64(*parts: Any) -> int:
    digest = stable_hash_hex(*parts)
    return int(digest[:16], 16)


def derive_unit_float(*parts: Any) -> float:
    return derive_uint64(*parts) / float(2**64)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def is_option_spec(value: Any) -> bool:
    return isinstance(value, Mapping) and "options" in value


def _format_path(*parts: Any) -> str:
    if not parts:
        return "<root>"

    rendered: list[str] = []
    for part in parts:
        if isinstance(part, int):
            rendered.append(f"[{part}]")
            continue
        if rendered:
            rendered.append(".")
        rendered.append(str(part))
    return "".join(rendered)


def validate_seed_control_name(value: Any, *path_parts: Any) -> str:
    seed_control = str(value)
    if seed_control not in VALID_SEED_CONTROL_NAMES:
        raise ValueError(
            f"{_format_path(*path_parts)} must be one of {sorted(VALID_SEED_CONTROL_NAMES)}, got {value!r}"
        )
    return seed_control


def resolve_ckpt_family(ckpt: str) -> str:
    return resolve_checkpoint_family(ckpt)


def _resolve_family_mapping_value(
    value: Any,
    *,
    mapping_key: str,
    ckpt_family: str,
    fallback_value: Any,
) -> Any:
    if value in (None, ""):
        return fallback_value
    if not isinstance(value, Mapping):
        raise ValueError(f"{mapping_key} must be a mapping keyed by checkpoint family")

    family_map = {str(key): item for key, item in value.items()}
    invalid_keys = sorted(key for key in family_map if key not in VALID_CKPT_FAMILY_NAMES and key != "default")
    if invalid_keys:
        raise ValueError(
            f"{mapping_key} contains unsupported family keys: {invalid_keys}. "
            f"Expected one of {sorted(VALID_CKPT_FAMILY_NAMES)} or 'default'."
        )

    candidate_keys = [ckpt_family, *LEGACY_FAMILY_FALLBACKS.get(ckpt_family, ())]
    for candidate_key in candidate_keys:
        if candidate_key in family_map and family_map[candidate_key] not in (None, ""):
            return family_map[candidate_key]
    if "default" in family_map and family_map["default"] not in (None, ""):
        return family_map["default"]
    if fallback_value not in (None, ""):
        return fallback_value
    raise ValueError(
        f"{mapping_key} does not provide a value for checkpoint family {ckpt_family!r} "
        "and no fallback value is available"
    )


def apply_prompt_family_routing(
    prompt_generator_name: str,
    prompt_generator_args: Mapping[str, Any],
    *,
    ckpt_family: str,
) -> Dict[str, Any]:
    routed = dict(prompt_generator_args)
    if prompt_generator_name not in FAMILY_ROUTED_GENERATORS:
        return routed

    if prompt_generator_name == WILDCARD_PROMPT_GENERATOR_NAME:
        routed["template"] = _resolve_family_mapping_value(
            routed.get("template_by_ckpt_family"),
            mapping_key="template_by_ckpt_family",
            ckpt_family=ckpt_family,
            fallback_value=routed.get("template"),
        )
    elif prompt_generator_name == PROMPT_LIST_V1_GENERATOR_NAME:
        routed["prompt_list"] = _resolve_family_mapping_value(
            routed.get("prompt_list_by_ckpt_family"),
            mapping_key="prompt_list_by_ckpt_family",
            ckpt_family=ckpt_family,
            fallback_value=routed.get("prompt_list"),
        )

    routed["negative_prompt"] = _resolve_family_mapping_value(
        routed.get("negative_prompt_by_ckpt_family"),
        mapping_key="negative_prompt_by_ckpt_family",
        ckpt_family=ckpt_family,
        fallback_value=routed.get("negative_prompt"),
    )

    if prompt_generator_name in NON_WILDCARD_FAMILY_ROUTED_GENERATORS:
        routed["positive_prefix"] = _resolve_family_mapping_value(
            routed.get("positive_prefix_by_ckpt_family"),
            mapping_key="positive_prefix_by_ckpt_family",
            ckpt_family=ckpt_family,
            fallback_value=routed.get("positive_prefix"),
        )
        # positive_suffix allows empty-string as a legitimate "append nothing"
        # value, so resolve it inline rather than via _resolve_family_mapping_value
        # (which treats "" as missing and triggers the fallback).
        suffix_map = routed.get("positive_suffix_by_ckpt_family")
        if isinstance(suffix_map, Mapping):
            family_map = {str(k): v for k, v in suffix_map.items()}
            candidate_keys = [ckpt_family, *LEGACY_FAMILY_FALLBACKS.get(ckpt_family, ()), "default"]
            resolved_suffix: Any = None
            for candidate_key in candidate_keys:
                if candidate_key in family_map:
                    resolved_suffix = family_map[candidate_key]
                    break
            if resolved_suffix is None:
                resolved_suffix = routed.get("positive_suffix", "")
            routed["positive_suffix"] = "" if resolved_suffix is None else str(resolved_suffix)
        else:
            routed["positive_suffix"] = str(routed.get("positive_suffix", "") or "")

    routed["_resolved_ckpt_family"] = ckpt_family
    return routed


def concat_seed(global_seed: int, suffix: int) -> int:
    return int(f"{int(global_seed)}{int(suffix)}")


def build_runtime_seed_values(
    *,
    global_seed: int,
    session_index: int,
    image_index: int,
    image_count: int,
) -> Dict[str, int]:
    return {
        "global_seed": int(global_seed),
        "session_seed": concat_seed(global_seed, (int(session_index) + 1) * (15 + int(image_count))),
        "image_index_seed": concat_seed(global_seed, (int(session_index) + 1) * (13 + int(image_index))),
    }


def validate_weighted_option_specs(value: Any, *path_parts: Any) -> None:
    if is_option_spec(value):
        spec = dict(value)
        extra_keys = set(spec) - OPTION_SPEC_KEYS
        missing_keys = OPTION_SPEC_KEYS - set(spec)
        if missing_keys:
            raise ValueError(
                f"{_format_path(*path_parts)} is missing required fields for weighted sampling: {sorted(missing_keys)}"
            )
        if extra_keys:
            raise ValueError(
                f"{_format_path(*path_parts)} contains unsupported keys for weighted sampling: {sorted(extra_keys)}"
            )

        validate_seed_control_name(spec["seed_control"], *path_parts, "seed_control")
        options = spec.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError(f"{_format_path(*path_parts)}.options must be a non-empty list")

        for option_index, option in enumerate(options):
            if not isinstance(option, dict) or "value" not in option or "weight" not in option:
                raise ValueError(
                    f"{_format_path(*path_parts)}.options[{option_index}] must contain 'value' and 'weight'"
                )
            weight = float(option["weight"])
            if weight <= 0:
                raise ValueError(
                    f"{_format_path(*path_parts)}.options[{option_index}].weight must be > 0, got {weight!r}"
                )
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            validate_weighted_option_specs(item, *path_parts, str(key))
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_weighted_option_specs(item, *path_parts, index)


def validate_wildcard_dropout_args(args: Mapping[str, Any], *path_parts: Any) -> None:
    present_keys = [key for key in WILDCARD_DROPOUT_KEYS if args.get(key) not in (None, "")]
    if not present_keys:
        return

    missing = [key for key in WILDCARD_DROPOUT_KEYS if args.get(key) in (None, "")]
    if missing:
        raise ValueError(
            f"{_format_path(*path_parts)} must provide {sorted(WILDCARD_DROPOUT_KEYS)} together when enabling dropout"
        )

    dropout_items = args["dropout_items"]
    dropout_probs = args["dropout_probs"]
    if not isinstance(dropout_items, list) or not isinstance(dropout_probs, list):
        raise ValueError(f"{_format_path(*path_parts)}.dropout_items and dropout_probs must both be lists")
    if len(dropout_items) != len(dropout_probs):
        raise ValueError(f"{_format_path(*path_parts)}.dropout_items and dropout_probs must have the same length")

    for index, probability in enumerate(dropout_probs):
        probability_value = float(probability)
        if probability_value < 0.0 or probability_value > 1.0:
            raise ValueError(
                f"{_format_path(*path_parts)}.dropout_probs[{index}] must be within [0.0, 1.0], got {probability!r}"
            )
    validate_seed_control_name(args["seed_control_dropout"], *path_parts, "seed_control_dropout")


def validate_random_segment_dropout_args(args: Mapping[str, Any], *path_parts: Any) -> None:
    pair_chance = args.get("random_segment_dropout_pair_chance")
    segment_prob = args.get("random_segment_dropout_segment_prob")
    seed_control = args.get("seed_control_random_segment_dropout")

    if pair_chance in (None, "") and segment_prob in (None, "") and seed_control in (None, ""):
        return

    missing = [key for key in RANDOM_SEGMENT_DROPOUT_KEYS if args.get(key) in (None, "")]
    if missing:
        raise ValueError(
            f"{_format_path(*path_parts)} must provide {sorted(RANDOM_SEGMENT_DROPOUT_KEYS)} together when enabling random segment dropout"
        )

    pair_chance_value = float(pair_chance)
    if pair_chance_value < 0.0 or pair_chance_value > 1.0:
        raise ValueError(
            f"{_format_path(*path_parts)}.random_segment_dropout_pair_chance must be within [0.0, 1.0], got {pair_chance!r}"
        )

    segment_prob_value = float(segment_prob)
    if segment_prob_value < 0.0 or segment_prob_value > 1.0:
        raise ValueError(
            f"{_format_path(*path_parts)}.random_segment_dropout_segment_prob must be within [0.0, 1.0], got {segment_prob!r}"
        )

    validate_seed_control_name(
        seed_control,
        *path_parts,
        "seed_control_random_segment_dropout",
    )


def resolve_options_recursive(
    value: Any,
    runtime_seed_values: Mapping[str, int],
    *key_parts: Any,
) -> Any:
    """Resolve any nested {'seed_control': ..., 'options': [...]} structure into a sampled value."""
    if is_option_spec(value):
        return select_weighted_value(value, runtime_seed_values, *key_parts)

    if isinstance(value, Mapping):
        resolved: Dict[str, Any] = {}
        for key, item in value.items():
            resolved[str(key)] = resolve_options_recursive(item, runtime_seed_values, *key_parts, str(key))
        return resolved

    if isinstance(value, list):
        return [resolve_options_recursive(item, runtime_seed_values, *key_parts, index) for index, item in enumerate(value)]

    return value


def select_weighted_value(spec: Mapping[str, Any], runtime_seed_values: Mapping[str, int], *key_parts: Any) -> Any:
    seed_control = validate_seed_control_name(spec.get("seed_control"), *key_parts, "seed_control")
    options = spec.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError(f"{_format_path(*key_parts)}.options must be a non-empty list")

    prepared = []
    total_weight = 0.0
    for option in options:
        if not isinstance(option, dict) or "value" not in option or "weight" not in option:
            raise ValueError(f"Each option must contain 'value' and 'weight'. Got: {option!r}")
        weight = float(option["weight"])
        if weight <= 0:
            raise ValueError(f"Weights must be > 0. Got: {weight!r}")
        prepared.append((option["value"], weight))
        total_weight += weight

    selected_seed_value = int(runtime_seed_values[seed_control])
    ticket = derive_unit_float(selected_seed_value, *key_parts) * total_weight
    cumulative = 0.0
    for value, weight in prepared:
        cumulative += weight
        if ticket < cumulative:
            return value
    return prepared[-1][0]


def validate_task_schema(task: Dict[str, Any]) -> None:
    required_top = ["version", "task_name", "global_seed", "session_count", "images"]
    missing_top = [field for field in required_top if field not in task]
    if missing_top:
        raise ValueError(f"Task YAML is missing required top-level fields: {missing_top}")

    if not isinstance(task["images"], list) or not task["images"]:
        raise ValueError("Task YAML field 'images' must be a non-empty list")

    for index, image in enumerate(task["images"], start=1):
        if not isinstance(image, dict):
            raise ValueError(f"images[{index}] must be a mapping")
        missing = [field for field in REQUIRED_IMAGE_FIELDS if field not in image]
        if missing:
            raise ValueError(f"images[{index}] is missing required fields: {missing}")

        prompt_generator = image["prompt_generator"]
        if not isinstance(prompt_generator, dict) or "name" not in prompt_generator or "args" not in prompt_generator:
            raise ValueError(f"images[{index}].prompt_generator must contain 'name' and 'args'")
        if not isinstance(prompt_generator["args"], Mapping):
            raise ValueError(f"images[{index}].prompt_generator.args must be a mapping")

        ckpt = image["ckpt"]
        if isinstance(ckpt, list) or (isinstance(ckpt, Mapping) and not is_option_spec(ckpt)):
            raise ValueError(f"images[{index}].ckpt must be a string or weighted sampling spec")
        validate_weighted_option_specs(ckpt, f"images[{index}].ckpt")

        sample = image["sample"]
        if not isinstance(sample, Mapping):
            raise ValueError(f"images[{index}].sample must be a mapping")
        missing_sample = [field for field in REQUIRED_SAMPLE_FIELDS if field not in sample]
        if missing_sample:
            raise ValueError(f"images[{index}].sample is missing fields: {missing_sample}")
        validate_seed_control_name(sample["generation_seed_control"], f"images[{index}].sample", "generation_seed_control")

        prompt_generator_name = str(prompt_generator["name"])
        prompt_generator_args = prompt_generator["args"]
        validate_weighted_option_specs(prompt_generator_args, f"images[{index}].prompt_generator.args")
        validate_weighted_option_specs(sample, f"images[{index}].sample")
        validate_weighted_option_specs(image.get("workflow_kwargs", {}), f"images[{index}].workflow_kwargs")
        validate_weighted_option_specs(image["lora_stack_config"], f"images[{index}].lora_stack_config")

        if prompt_generator_name == WILDCARD_PROMPT_GENERATOR_NAME:
            if prompt_generator_args.get("seed_control") in (None, ""):
                raise ValueError(f"images[{index}].prompt_generator.args.seed_control is required for wildcard prompts")
            validate_seed_control_name(
                prompt_generator_args["seed_control"], f"images[{index}].prompt_generator.args", "seed_control"
            )
            validate_wildcard_dropout_args(prompt_generator_args, f"images[{index}].prompt_generator.args")
            validate_random_segment_dropout_args(prompt_generator_args, f"images[{index}].prompt_generator.args")


def normalize_prompt_generator_args(
    prompt_generator_name: str,
    prompt_generator_args: Mapping[str, Any],
    task_yaml_dir: Path,
) -> Dict[str, Any]:
    normalized = dict(prompt_generator_args)

    if prompt_generator_name not in {WILDCARD_PROMPT_GENERATOR_NAME, PROMPT_LIST_V1_GENERATOR_NAME}:
        return normalized

    if prompt_generator_name == WILDCARD_PROMPT_GENERATOR_NAME:
        for candidate_group in (
            ("wildcard_root", "wildcards_root", "wildcards_dir"),
            ("template_root", "templates_root", "templates_dir"),
        ):
            normalized = _normalize_relative_root_arg(normalized, task_yaml_dir, *candidate_group)
        return normalized

    normalized = _normalize_relative_root_arg(
        normalized,
        task_yaml_dir,
        "prompt_list_root",
        "prompt_lists_root",
        "prompt_lists_dir",
    )
    return normalized


def _normalize_relative_root_arg(
    normalized: Dict[str, Any],
    task_yaml_dir: Path,
    *candidate_keys: str,
) -> Dict[str, Any]:
    selected_key = None
    for candidate in candidate_keys:
        value = normalized.get(candidate)
        if value not in (None, ""):
            selected_key = candidate
            root_path = Path(str(value))
            break
    else:
        selected_key = candidate_keys[0]
        if "wildcard" in candidate_keys[0]:
            root_path = _detect_default_root(
                task_yaml_dir,
                primary_name="wildcard",
                alternate_names=("wildcards",),
                repo_layout_fallback=WILDCARD_DIR,
            )
        elif "prompt_list" in candidate_keys[0]:
            root_path = _detect_default_root(
                task_yaml_dir,
                primary_name="prompt_lists",
                alternate_names=("prompt_list",),
                repo_layout_fallback=PROMPT_LISTS_DIR,
            )
        else:
            root_path = _detect_default_root(
                task_yaml_dir,
                primary_name="prompt_templates",
                alternate_names=(),
                repo_layout_fallback=PROMPT_TEMPLATES_DIR,
            )

    if not root_path.is_absolute():
        root_path = (task_yaml_dir / root_path).resolve()

    normalized[selected_key] = str(root_path)
    return normalized


def _detect_default_root(
    task_yaml_dir: Path,
    *,
    primary_name: str,
    alternate_names: tuple[str, ...],
    repo_layout_fallback: Path,
) -> Path:
    candidate_names = (primary_name, *alternate_names)
    repo_root = Path(__file__).resolve().parent
    cwd_root = Path.cwd().resolve()
    fallback_root = repo_layout_fallback if repo_layout_fallback.is_absolute() else (repo_root / repo_layout_fallback).resolve()
    candidate_dirs: list[Path] = []
    search_roots = (task_yaml_dir, task_yaml_dir.parent, task_yaml_dir.parent.parent)
    for search_root in search_roots:
        for name in candidate_names:
            candidate_dirs.append(search_root / name)
    for name in candidate_names:
        candidate_dirs.append((cwd_root / name).resolve())
        candidate_dirs.append((cwd_root / "template" / name).resolve())
        candidate_dirs.append((repo_root / name).resolve())
        candidate_dirs.append((repo_root / "template" / name).resolve())
    candidate_dirs.append(fallback_root)

    seen: set[Path] = set()
    for candidate_dir in candidate_dirs:
        resolved = candidate_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate_dir.exists():
            return resolved
    return fallback_root


def apply_global_seed_override(task: Mapping[str, Any], global_seed_override: int | None) -> Dict[str, Any]:
    effective_task = dict(task)
    if global_seed_override is not None:
        effective_task["global_seed"] = int(global_seed_override)
    return effective_task


def compile_requests(
    task: Dict[str, Any],
    task_yaml_path: str | Path,
    global_seed_override: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    effective_task = apply_global_seed_override(task, global_seed_override)
    validate_task_schema(effective_task)

    source_path = Path(task_yaml_path).resolve()
    task_yaml_text = source_path.read_text(encoding="utf-8")
    task_yaml_sha256 = hashlib.sha256(task_yaml_text.encode("utf-8")).hexdigest()

    task_name = str(effective_task["task_name"])
    source_global_seed = int(task["global_seed"])
    global_seed = int(effective_task["global_seed"])
    session_count = int(effective_task["session_count"])
    task_version = effective_task.get("version")
    image_count = len(effective_task["images"])

    records: list[dict[str, Any]] = []

    for session_index in range(session_count):
        session_id = stable_hash_hex(task_name, global_seed, session_index, "session")[:24]

        for image_index, image in enumerate(effective_task["images"]):
            image_name = str(image["image_name"])
            request_id = stable_hash_hex(task_name, global_seed, session_index, image_name, "request")[:24]
            runtime_seed_values = build_runtime_seed_values(
                global_seed=global_seed,
                session_index=session_index,
                image_index=image_index,
                image_count=image_count,
            )
            resolved_ckpt = resolve_options_recursive(image["ckpt"], runtime_seed_values, "ckpt")
            if isinstance(resolved_ckpt, Mapping) or isinstance(resolved_ckpt, list):
                raise ValueError(
                    f"images[{image_index + 1}].ckpt must resolve to a checkpoint string, got {resolved_ckpt!r}"
                )
            ckpt = str(resolved_ckpt)
            if not ckpt:
                raise ValueError(f"images[{image_index + 1}].ckpt resolved to an empty checkpoint string")
            ckpt_resolution = resolve_checkpoint(ckpt)
            ckpt_family = ckpt_resolution.family

            prompt_generator_name = str(image["prompt_generator"]["name"])
            prompt_generator_args = resolve_options_recursive(
                dict(image["prompt_generator"]["args"]),
                runtime_seed_values,
                "prompt_generator_args",
            )
            prompt_generator_args = normalize_prompt_generator_args(
                prompt_generator_name,
                prompt_generator_args,
                source_path.parent,
            )
            prompt_generator_args = apply_prompt_family_routing(
                prompt_generator_name,
                prompt_generator_args,
                ckpt_family=ckpt_family,
            )

            prompt_seed_control: str | None = None
            if prompt_generator_name == WILDCARD_PROMPT_GENERATOR_NAME:
                prompt_seed_control = validate_seed_control_name(
                    prompt_generator_args["seed_control"], "prompt_generator_args", "seed_control"
                )
                prompt_seed = int(runtime_seed_values[prompt_seed_control])
            else:
                prompt_seed = derive_uint64(task_name, global_seed, session_index, image_name, "prompt_seed")

            prompt_generator = get_prompt_generator(prompt_generator_name)
            prompt_generator_call_args = dict(prompt_generator_args)
            if prompt_generator_name == WILDCARD_PROMPT_GENERATOR_NAME:
                prompt_generator_call_args["_resolved_seed_control_value"] = prompt_seed
                prompt_generator_call_args["_resolved_session_id"] = session_id
                prompt_generator_call_args["_resolved_image_index"] = image_index
                prompt_generator_call_args["_resolved_image_count"] = image_count
                if prompt_generator_call_args.get("seed_control_dropout") not in (None, ""):
                    dropout_seed_control = validate_seed_control_name(
                        prompt_generator_call_args["seed_control_dropout"],
                        "prompt_generator_args",
                        "seed_control_dropout",
                    )
                    prompt_generator_call_args["_resolved_dropout_seed_value"] = int(
                        runtime_seed_values[dropout_seed_control]
                    )
                if prompt_generator_call_args.get("seed_control_random_segment_dropout") not in (None, ""):
                    random_segment_dropout_seed_control = validate_seed_control_name(
                        prompt_generator_call_args["seed_control_random_segment_dropout"],
                        "prompt_generator_args",
                        "seed_control_random_segment_dropout",
                    )
                    prompt_generator_call_args["_resolved_random_segment_dropout_seed_value"] = int(
                        runtime_seed_values[random_segment_dropout_seed_control]
                    )
            elif prompt_generator_name == PROMPT_LIST_V1_GENERATOR_NAME:
                prompt_generator_call_args["_resolved_session_index"] = int(session_index)

            prompt_bundle = prompt_generator.generate(prompt_generator_call_args, prompt_seed)

            sampled = resolve_options_recursive(
                dict(image["sample"]),
                runtime_seed_values,
                "sample",
            )
            generation_seed_control = validate_seed_control_name(
                sampled["generation_seed_control"], "sample", "generation_seed_control"
            )
            sampled_steps = int(sampled["steps"])
            sampled_cfg = float(sampled["cfg"])
            sampled_width = int(sampled["width"])
            sampled_height = int(sampled["height"])
            generation_seed = int(runtime_seed_values[generation_seed_control])

            workflow_kwargs_result = resolve_options_recursive(
                dict(image.get("workflow_kwargs", {})),
                runtime_seed_values,
                "workflow_kwargs",
            )

            lora_stack_config = resolve_options_recursive(
                dict(image["lora_stack_config"]),
                runtime_seed_values,
                "lora_stack_config",
            )

            record: Dict[str, Any] = {
                "request_id": request_id,
                "session_id": session_id,
                "task_name": task_name,
                "task_version": task_version,
                "task_yaml_path": str(source_path),
                "task_yaml_sha256": task_yaml_sha256,
                "compiler_version": COMPILER_VERSION,
                "global_seed": global_seed,
                "session_index": session_index,
                "image_index": image_index,
                "image_name": image_name,
                "runtime_seed_values": runtime_seed_values,
                "workflow_name": str(image["workflow_name"]),
                "ckpt": ckpt,
                "ckpt_family": ckpt_family,
                "ckpt_registry_id": ckpt_resolution.model_id,
                "ckpt_family_source": ckpt_resolution.family_source,
                "ckpt_family_keyword": ckpt_resolution.matched_keyword,
                "ckpt_visibility": ckpt_resolution.visibility,
                "ckpt_publish": ckpt_resolution.publish,
                "lora_stack_config": lora_stack_config,
                "prompt_generator_name": prompt_generator_name,
                "prompt_generator_version": prompt_bundle.prompt_metadata["generator_version"],
                "prompt_generator_args": prompt_generator_args,
                "prompt_seed_control": prompt_seed_control,
                "prompt_seed": prompt_seed,
                "positive_prompt": prompt_bundle.positive_prompt,
                "negative_prompt": prompt_bundle.negative_prompt,
                "generation_seed_control": generation_seed_control,
                "seed": generation_seed,
                "steps": sampled_steps,
                "cfg": sampled_cfg,
                "width": sampled_width,
                "height": sampled_height,
                "workflow_kwargs": workflow_kwargs_result,
            }
            records.append(record)

    compile_manifest = {
        "task_name": task_name,
        "task_version": task_version,
        "global_seed": global_seed,
        "session_count": session_count,
        "image_count_per_session": image_count,
        "request_count": len(records),
        "task_yaml_path": str(source_path),
        "task_yaml_sha256": task_yaml_sha256,
        "compiler_version": COMPILER_VERSION,
        "images": [
            {
                "image_name": image["image_name"],
                "workflow_name": image["workflow_name"],
                "ckpt": image["ckpt"],
                "lora_stack_config": image["lora_stack_config"],
                "prompt_generator": image["prompt_generator"],
                "sample": image["sample"],
            }
            for image in effective_task["images"]
        ],
    }
    if global_seed_override is not None and global_seed != source_global_seed:
        compile_manifest["source_global_seed"] = source_global_seed
        compile_manifest["global_seed_override"] = global_seed
    return records, compile_manifest, task_yaml_text


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_compile_outputs(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    compile_manifest: dict[str, Any],
    task_yaml_text: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "requests.jsonl", records)
    (out_dir / "compile_manifest.json").write_text(
        json.dumps(compile_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "task_snapshot.yaml").write_text(task_yaml_text, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a task YAML into concrete requests.jsonl for batch execution."
    )
    parser.add_argument("--task-yaml", required=True, help="Path to task YAML")
    parser.add_argument("--output-dir", required=True, help="Directory to write requests.jsonl and manifest files")
    parser.add_argument(
        "--global-seed",
        type=int,
        help="Override the YAML global_seed for this compile without editing the source task file",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    task = load_task_yaml(args.task_yaml)
    records, compile_manifest, task_yaml_text = compile_requests(
        task,
        args.task_yaml,
        global_seed_override=args.global_seed,
    )
    write_compile_outputs(args.output_dir, records, compile_manifest, task_yaml_text)
    print(f"Wrote requests to: {Path(args.output_dir) / 'requests.jsonl'}")
    print(f"Wrote compile manifest to: {Path(args.output_dir) / 'compile_manifest.json'}")
    print(f"Wrote task snapshot to: {Path(args.output_dir) / 'task_snapshot.yaml'}")


if __name__ == "__main__":
    main()
