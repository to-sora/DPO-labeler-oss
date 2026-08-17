from __future__ import annotations

from typing import Any, Dict

from generators.generate_dpo_workflow_matrix import ALL_CHECKPOINTS, TIER1_CHECKPOINTS


REMOVED_CHECKPOINTS = frozenset(
    {
        "sdxl_realismByStableYogi_v5XLFP16.safetensors",
        "sdxl_sdxl10ArienmixxlAsian_v45Pruned.safetensors",
        "sdxl_iniverseMixSFWNSFW_realXLV1.safetensors",
        "sdxl_juggernautXL_ragnarokBy.safetensors",
        "sdxl_cyberrealisticPony_v150__2.safetensors",
        "sdxl_iniverseMixSFWNSFW_ponyRealGuofengV50A__2.safetensors",
        "sdxl_epicrealismXL_pureFix.safetensors",
        "sdxl_waiREALCN_v150.safetensors",
        "sdxl_xxmix9realisticsdxl_v10.safetensors",
        "sdxl_counterfeitxl_v25.safetensors",
        "sdxl_astolfokarmixXL_256cBased.safetensors",
        "sdxl_divingIllustriousReal_v70VAE.safetensors",
        "sdxl_jibMixRealisticXL_v180SkinSupreme.safetensors",
        "sdxl_cyberrealisticPony_v150.safetensors",
        "sdxl_ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
        "sdxl_mistoonAnime_v10Illustrious.safetensors",
        "sdxl_sdxlNijiSeven_sdxlNijiSeven.safetensors",
        "sdxl_realPony_illustriousPony.safetensors",
    }
)

SELF_TRAINED_CHECKPOINTS = [
    "kohyass/keep/Animagine_XL_4.0_base/test3_ver4.ckpt",
    "kohyass/keep/Animagine_XL_4.0_base/test4_ver4-000008.ckpt",
    "kohyass/keep/Animagine_XL_4.0_base/test4_ver4.ckpt",
    "kohyass/keep/illlustion_base/path_2/test1_ver4-000014.ckpt",
    "kohyass/keep/illlustion_base/path_2/test1_ver4.ckpt",
    "kohyass/keep/illlustion_base/path_4/test1_ver4-000014.ckpt",
    "kohyass/keep/illlustion_base/path_4/test1_ver4.ckpt",
]

BASE_TIER1_CHECKPOINTS = [
    checkpoint for checkpoint in TIER1_CHECKPOINTS if checkpoint not in REMOVED_CHECKPOINTS
]
BASE_NON_TIER1_CHECKPOINTS = [
    checkpoint
    for checkpoint in ALL_CHECKPOINTS
    if checkpoint not in REMOVED_CHECKPOINTS and checkpoint not in set(TIER1_CHECKPOINTS)
]

# Shared 3-tier checkpoint groups used by every generator (V1-V7, SV6/SV7, batch_v1_7).
# Tier 1 keeps the original 8 entries from TIER1_CHECKPOINTS without REMOVED filtering,
# per the shared-config requirement of "Tier 1 (8 models)".
SHARED_TIER1_CHECKPOINTS = list(TIER1_CHECKPOINTS)
SHARED_CUSTOM_CHECKPOINTS = list(SELF_TRAINED_CHECKPOINTS)
SHARED_REMAINING_CHECKPOINTS = [
    checkpoint
    for checkpoint in ALL_CHECKPOINTS
    if checkpoint not in set(TIER1_CHECKPOINTS) and checkpoint not in REMOVED_CHECKPOINTS
]


def _group_options(checkpoints: list[str], *, total_weight: float) -> list[dict[str, Any]]:
    if not checkpoints:
        return []
    per_checkpoint_weight = float(total_weight) / float(len(checkpoints))
    return [
        {
            "value": checkpoint,
            "weight": per_checkpoint_weight,
        }
        for checkpoint in checkpoints
    ]


def make_sv_ckpt_spec(pool_name: str, *, seed_control: str = "session_seed") -> Dict[str, Any]:
    if pool_name == "tier1":
        options = _group_options(BASE_TIER1_CHECKPOINTS, total_weight=2.0 / 3.0) + _group_options(
            SELF_TRAINED_CHECKPOINTS,
            total_weight=1.0 / 3.0,
        )
    elif pool_name == "all":
        options = (
            _group_options(BASE_TIER1_CHECKPOINTS, total_weight=1.0 / 3.0)
            + _group_options(BASE_NON_TIER1_CHECKPOINTS, total_weight=1.0 / 3.0)
            + _group_options(SELF_TRAINED_CHECKPOINTS, total_weight=1.0 / 3.0)
        )
    else:
        raise ValueError(f"Unsupported checkpoint pool {pool_name!r}")

    if not options:
        raise ValueError(f"Checkpoint pool {pool_name!r} resolved to an empty option list")
    return {
        "seed_control": seed_control,
        "options": options,
    }


def make_shared_ckpt_spec(*, seed_control: str = "session_seed") -> Dict[str, Any]:
    """Shared 3-tier checkpoint spec with equal 1/3 weight per group.

    Within each group every checkpoint gets the same probability.
    Used uniformly by all workflow generators so checkpoint selection behaviour
    is identical across V1-V7, SV6/SV7, and batch_v1_7.
    """
    options = (
        _group_options(SHARED_TIER1_CHECKPOINTS, total_weight=1.0 / 3.0)
        + _group_options(SHARED_CUSTOM_CHECKPOINTS, total_weight=1.0 / 3.0)
        + _group_options(SHARED_REMAINING_CHECKPOINTS, total_weight=1.0 / 3.0)
    )
    if not options:
        raise ValueError("Shared checkpoint spec resolved to an empty option list")
    return {
        "seed_control": seed_control,
        "options": options,
    }


def describe_shared_checkpoint_pools() -> dict[str, Any]:
    return {
        "tier1_checkpoints": list(SHARED_TIER1_CHECKPOINTS),
        "custom_checkpoints": list(SHARED_CUSTOM_CHECKPOINTS),
        "remaining_checkpoints": list(SHARED_REMAINING_CHECKPOINTS),
        "weight_targets": {
            "tier1_total_weight": 1.0 / 3.0,
            "custom_total_weight": 1.0 / 3.0,
            "remaining_total_weight": 1.0 / 3.0,
        },
    }


def describe_sv_checkpoint_pools() -> dict[str, Any]:
    return {
        "removed_checkpoints": list(REMOVED_CHECKPOINTS),
        "self_trained_checkpoints": list(SELF_TRAINED_CHECKPOINTS),
        "base_tier1_checkpoints": list(BASE_TIER1_CHECKPOINTS),
        "base_non_tier1_checkpoints": list(BASE_NON_TIER1_CHECKPOINTS),
        "weight_targets": {
            "tier1_pool": {
                "base_tier1_total_weight": 2.0 / 3.0,
                "self_trained_total_weight": 1.0 / 3.0,
            },
            "all_pool": {
                "base_tier1_total_weight": 1.0 / 3.0,
                "base_non_tier1_total_weight": 1.0 / 3.0,
                "self_trained_total_weight": 1.0 / 3.0,
            },
        },
    }
