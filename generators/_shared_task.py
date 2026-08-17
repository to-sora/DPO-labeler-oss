"""Shared task-YAML builder for the consolidated V1-V7 / SV6-SV7 / batch_v1_7 generators.

The consolidated rule for every generator:

* Workflow  : ``SdxlEaseLoraWorkflow`` (no LoRA, no latent/model upscale).
* Sessions  : 30 per task.
* Checkpoints: three equal-weight tiers via :func:`make_shared_ckpt_spec`.
* Dropout   : 30% pair chance, 10% per-segment probability, image-index-seeded.
* Each unique prompt template yields exactly two tasks — one where both images
  share the base checkpoint (``same`` / ``session_seed``) and one where they
  differ (``diff`` / ``image_index_seed``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping

import yaml


SHARED_WORKFLOW_NAME = "SdxlEaseLoraWorkflow"
SHARED_SESSION_COUNT = 30
SHARED_GLOBAL_SEED = 20260406
SHARED_CFG_VALUES = [7.0, 7.5]
SHARED_STEPS_VALUES = [30, 35, 40]
SHARED_RESOLUTION_VALUES = [768, 1024, 1280, 1536]
SHARED_RESOLUTION_WEIGHTS = [3.0, 3.0, 3.0, 2.0]
SHARED_PAIR_DROPOUT_CHANCE = 0.3
SHARED_SEGMENT_DROPOUT_PROB = 0.1


def _option_spec(seed_control: str, values: Iterable[Any]) -> Dict[str, Any]:
    return {
        "seed_control": seed_control,
        "options": [{"value": v, "weight": 1.0} for v in values],
    }


def _weighted_option_spec(
    seed_control: str, values: Iterable[Any], weights: Iterable[float]
) -> Dict[str, Any]:
    vs = list(values)
    ws = list(weights)
    return {
        "seed_control": seed_control,
        "options": [{"value": v, "weight": float(w)} for v, w in zip(vs, ws)],
    }


def _make_no_lora_config() -> Dict[str, Any]:
    return {
        "toggle": False,
        "mode": "simple",
        "num_loras": 1,
        "lora_1_name": "None",
        "lora_1_strength": 1.0,
        "lora_1_model_strength": 1.0,
        "lora_1_clip_strength": 1.0,
    }


def inject_dropout(prompt_args: Dict[str, Any]) -> Dict[str, Any]:
    """Add shared random-segment dropout knobs to a prompt_generator args dict."""
    prompt_args = dict(prompt_args)
    prompt_args["random_segment_dropout_pair_chance"] = SHARED_PAIR_DROPOUT_CHANCE
    prompt_args["random_segment_dropout_segment_prob"] = SHARED_SEGMENT_DROPOUT_PROB
    prompt_args["seed_control_random_segment_dropout"] = "image_index_seed"
    return prompt_args


def make_shared_image(
    *,
    image_name: str,
    ckpt_seed_control: str,
    prompt_generator_name: str,
    prompt_args: Mapping[str, Any],
) -> Dict[str, Any]:
    # Lazy import avoids a circular dep between sv_checkpoint_pools and the V1
    # generator (which owns ALL_CHECKPOINTS / TIER1_CHECKPOINTS).
    from generators.sv_checkpoint_pools import make_shared_ckpt_spec

    return {
        "image_name": image_name,
        "workflow_name": SHARED_WORKFLOW_NAME,
        "ckpt": make_shared_ckpt_spec(seed_control=ckpt_seed_control),
        "lora_stack_config": _make_no_lora_config(),
        "prompt_generator": {
            "name": prompt_generator_name,
            "args": inject_dropout(dict(prompt_args)),
        },
        "sample": {
            "generation_seed_control": "image_index_seed",
            "steps": _option_spec("image_index_seed", SHARED_STEPS_VALUES),
            "cfg": _option_spec("image_index_seed", SHARED_CFG_VALUES),
            "width": _weighted_option_spec(
                "image_index_seed", SHARED_RESOLUTION_VALUES, SHARED_RESOLUTION_WEIGHTS
            ),
            "height": _weighted_option_spec(
                "image_index_seed", SHARED_RESOLUTION_VALUES, SHARED_RESOLUTION_WEIGHTS
            ),
        },
    }


def build_shared_pair_task(
    *,
    task_name: str,
    pair_mode: str,  # "same" or "diff"
    prompt_generator_name: str,
    prompt_args_image1: Mapping[str, Any],
    prompt_args_image2: Mapping[str, Any] | None = None,
    session_count: int = SHARED_SESSION_COUNT,
    global_seed: int = SHARED_GLOBAL_SEED,
) -> Dict[str, Any]:
    if pair_mode not in ("same", "diff"):
        raise ValueError(f"pair_mode must be 'same' or 'diff', got {pair_mode!r}")
    ckpt_seed_control = "session_seed" if pair_mode == "same" else "image_index_seed"
    image2_args = prompt_args_image2 if prompt_args_image2 is not None else prompt_args_image1
    return {
        "version": 1,
        "task_name": task_name,
        "global_seed": global_seed,
        "session_count": session_count,
        "images": [
            make_shared_image(
                image_name="image1",
                ckpt_seed_control=ckpt_seed_control,
                prompt_generator_name=prompt_generator_name,
                prompt_args=prompt_args_image1,
            ),
            make_shared_image(
                image_name="image2",
                ckpt_seed_control=ckpt_seed_control,
                prompt_generator_name=prompt_generator_name,
                prompt_args=image2_args,
            ),
        ],
    }


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


@dataclass(frozen=True)
class TemplateEntry:
    """One unique prompt template — will emit one ``_same`` and one ``_diff`` task."""

    short_name: str  # e.g. "gpt", "qwen", "compact", "cinematic", "sfw", "nsfw"
    prompt_generator_name: str
    prompt_args: Dict[str, Any]
    # Optional override for image2 (used by V6/V7 runtime split if needed)
    prompt_args_image2: Dict[str, Any] | None = None


def make_research_template_args(
    *,
    family_templates: Mapping[str, str],
    negatives: Mapping[str, str],
    template_root: str,
    wildcard_root: str,
) -> Dict[str, Any]:
    """Build ``wildcard_template_generator`` args for a V2-V7 research axis.

    ``family_templates`` maps ckpt-family key -> template identifier (e.g.
    ``{"anime": "research_v2/hybrid_compact_anime_adult", ...}``). The first
    family entry becomes the default ``template`` / ``negative_prompt``.
    """
    first_family = next(iter(family_templates))
    return {
        "seed_control": "session_seed",
        "template": family_templates[first_family],
        "template_by_ckpt_family": dict(family_templates),
        "template_root": template_root,
        "wildcard_root": wildcard_root,
        "negative_prompt": negatives[first_family],
        "negative_prompt_by_ckpt_family": dict(negatives),
    }


def build_research_templates(
    *,
    prompt_axes_short: Mapping[str, str],
    identifiers: Mapping[str, Mapping[str, str]],
    negatives: Mapping[str, str],
    template_root: str,
    wildcard_root: str,
) -> List[TemplateEntry]:
    """Build :class:`TemplateEntry` list for a V2-V7 style research generator.

    ``prompt_axes_short`` maps long-form axis name (e.g. ``"hybrid_compact"``)
    to the short token used in the task name (e.g. ``"compact"``).
    """
    entries: List[TemplateEntry] = []
    for axis_long, axis_short in prompt_axes_short.items():
        entries.append(
            TemplateEntry(
                short_name=axis_short,
                prompt_generator_name="wildcard_template_generator",
                prompt_args=make_research_template_args(
                    family_templates=identifiers[axis_long],
                    negatives=negatives,
                    template_root=template_root,
                    wildcard_root=wildcard_root,
                ),
            )
        )
    return entries


def emit_version_matrix(
    *,
    version_tag: str,  # e.g. "v1", "v2", "sv6"
    templates: List[TemplateEntry],
    output_dir: Path,
    global_seed: int = SHARED_GLOBAL_SEED,
) -> List[Dict[str, Any]]:
    """Emit ``{version}_{short}_{same|diff}.yaml`` for every template entry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for entry in templates:
        for pair_mode in ("same", "diff"):
            task_name = f"{version_tag}_{entry.short_name}_{pair_mode}"
            payload = build_shared_pair_task(
                task_name=task_name,
                pair_mode=pair_mode,
                prompt_generator_name=entry.prompt_generator_name,
                prompt_args_image1=entry.prompt_args,
                prompt_args_image2=entry.prompt_args_image2,
                global_seed=global_seed,
            )
            rel = Path(f"{task_name}.yaml")
            write_yaml(output_dir / rel, payload)
            manifest.append(
                {
                    "task_name": task_name,
                    "output_yaml": rel.as_posix(),
                    "pair_mode": pair_mode,
                    "template_short_name": entry.short_name,
                    "prompt_generator": entry.prompt_generator_name,
                    "workflow_name": SHARED_WORKFLOW_NAME,
                    "session_count": SHARED_SESSION_COUNT,
                }
            )
    return manifest
