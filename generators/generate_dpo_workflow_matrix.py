"""V1 DPO workflow generator (consolidated).

Emits exactly 4 YAMLs per run:

* ``v1_gpt_same.yaml``  / ``v1_gpt_diff.yaml``
* ``v1_qwen_same.yaml`` / ``v1_qwen_diff.yaml``

All tasks use :class:`SdxlEaseLoraWorkflow`, the shared 3-tier checkpoint
config, no LoRA, no upscale, 30 sessions, and 30 %/10 % random segment
dropout.

Legacy constants (``TIER1_CHECKPOINTS``, ``ALL_CHECKPOINTS``,
``NEGATIVE_PROMPTS_BY_FAMILY``, ``CURATED_TEMPLATE_IDENTIFIERS``,
``make_template_args``) are still exported because downstream modules
(``generate_batch_v1_7``, ``sv_checkpoint_pools``, the V2-V7 siblings) import
them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import PROMPT_TEMPLATES_DIR, WILDCARD_DIR, WORKFLOWS_DIR, relative_path as relative_root_path

from generators._shared_task import TemplateEntry, emit_version_matrix, write_yaml


DEFAULT_OUTPUT_DIR = WORKFLOWS_DIR / "dpo_workflow"

CURATED_TEMPLATE_IDENTIFIERS: Dict[str, Dict[str, str]] = {
    "mix_gpt": {
        "anime": "curated/mix_gpt_anime_curated",
        "pony": "curated/mix_gpt_pony_curated",
        "realistic": "curated/mix_gpt_realistic_curated",
    },
    "mix_qwen": {
        "anime": "curated/mix_qwen_anime_curated",
        "pony": "curated/mix_qwen_pony_curated",
        "realistic": "curated/mix_qwen_realistic_curated",
    },
}

NEGATIVE_PROMPTS_BY_FAMILY = {
    "anime": "bad anatomy, malformed hands, extra fingers, lowres, blurry, jpeg artifacts, worst quality",
    "pony": "score_6, score_5, score_4, source_furry, lowres, blurry, bad anatomy, malformed hands",
    "realistic": "anime, illustration, cartoon, painting, cgi, 3d, lowres, blurry, bad anatomy, malformed hands",
}

TIER1_CHECKPOINTS = [
    "sdxl_animagineXL40_v4Opt.safetensors",
    "sdxl_astolfokarmixXL_256cBased.safetensors",
    "sdxl_hassakuXLIllustrious_v32.safetensors",
    "sdxl_illustrij_v20.safetensors",
    "sdxl_novaAnimeXL_ilV150.safetensors",
    "sdxl_perfectdeliberate_v60.safetensors",
    "sdxl_prefectiousXLNSFW_v10.safetensors",
    "sdxl_ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
]

ALL_CHECKPOINTS = [
    "sdxl_albedobaseXL_v31Large.safetensors",
    "sdxl_animagineXL40_v4Opt.safetensors",
    "sdxl_animagineXLV31_v31.safetensors",
    "sdxl_AnythingXL_xl.safetensors",
    "sdxl_astolfokarmixXL_256cBased.safetensors",
    "sdxl_bismuthIllustrious_v60.safetensors",
    "sdxl_bluePencilXL_v700.safetensors",
    "sdxl_boleromixIllustrious_v601.safetensors",
    "sdxl_counterfeitxl_v25.safetensors",
    "sdxl_cyberrealisticPony_v150__2.safetensors",
    "sdxl_cyberrealisticPony_v150.safetensors",
    "sdxl_divingIllustriousReal_v70VAE.safetensors",
    "sdxl_epicrealismXL_pureFix.safetensors",
    "sdxl_hassakuXLIllustrious_v32.safetensors",
    "sdxl_hikariNoobVPred_124.safetensors",
    "sdxl_hosekiLustrousmix_illustNoobaiEPS11V2.safetensors",
    "sdxl_illustrij_v20.safetensors",
    "sdxl_iniverseMixSFWNSFW_ponyRealGuofengV50A__2.safetensors",
    "sdxl_iniverseMixSFWNSFW_ponyRealGuofengV50A.safetensors",
    "sdxl_iniverseMixSFWNSFW_realXLV1.safetensors",
    "sdxl_JANKUTrainedNoobaiRouwei_v60.safetensors",
    "sdxl_jibMixRealisticXL_v180SkinSupreme.safetensors",
    "sdxl_juggernautXL_ragnarokBy.safetensors",
    "sdxl_meichidarkmixReload_meichidarkanimv2Lust.safetensors",
    "sdxl_mistoonAnime_v10Illustrious.safetensors",
    "sdxl_mritualIllustrious_v201.safetensors",
    "sdxl_novaAnimeXL_ilV140.safetensors",
    "sdxl_novaAnimeXL_ilV150.safetensors",
    "sdxl_novaUnrealXL_v100.safetensors",
    "sdxl_ntrMIXIllustriousXL_xiii.safetensors",
    "sdxl_obsessionIllustrious_vPredV11.safetensors",
    "sdxl_perfectdeliberate_v60.safetensors",
    "sdxl_pieModels_applePieV2.safetensors",
    "sdxl_ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
    "sdxl_pornmaster_proSDXLV8.safetensors",
    "sdxl_prefectIllustriousXL_v60.safetensors",
    "sdxl_prefectiousXLNSFW_v10.safetensors",
    "sdxl_prefectPonyXL_v6.safetensors",
    "sdxl_realismByStableYogi_v5XLFP16.safetensors",
    "sdxl_realPony_illustriousPony.safetensors",
    "sdxl_sdxl10ArienmixxlAsian_v45Pruned.safetensors",
    "sdxl_sdxlNijiSeven_sdxlNijiSeven.safetensors",
    "sdxl_steincustom_V13__2.safetensors",
    "sdxl_steincustom_V13.safetensors",
    "sdxl_uncannyValley_VPredV1.safetensors",
    "sdxl_waiIllustriousSDXL_v160.safetensors",
    "sdxl_waiREALCN_v150.safetensors",
    "sdxl_waiREALMIX_v11.safetensors",
    "sdxl_xxmix9realisticsdxl_v10.safetensors",
]

# Short-name map for task naming: template axis -> short token.
SHORT_TEMPLATE_NAMES = {"mix_gpt": "gpt", "mix_qwen": "qwen"}


def make_template_args(
    template_name: str,
    *,
    template_root: str,
    wildcard_root: str,
) -> Dict[str, Any]:
    """Build ``wildcard_template_generator`` args for a V1 template axis.

    Kept for ``generate_batch_v1_7`` compatibility.
    """
    if template_name not in CURATED_TEMPLATE_IDENTIFIERS:
        raise ValueError(f"Unsupported V1 template axis {template_name!r}")
    family_templates = CURATED_TEMPLATE_IDENTIFIERS[template_name]
    return {
        "seed_control": "session_seed",
        "template": family_templates["anime"],
        "template_by_ckpt_family": dict(family_templates),
        "template_root": template_root,
        "wildcard_root": wildcard_root,
        "negative_prompt": NEGATIVE_PROMPTS_BY_FAMILY["anime"],
        "negative_prompt_by_ckpt_family": dict(NEGATIVE_PROMPTS_BY_FAMILY),
    }


def build_v1_templates(*, template_root: str, wildcard_root: str) -> List[TemplateEntry]:
    return [
        TemplateEntry(
            short_name=SHORT_TEMPLATE_NAMES[axis],
            prompt_generator_name="wildcard_template_generator",
            prompt_args=make_template_args(
                axis, template_root=template_root, wildcard_root=wildcard_root
            ),
        )
        for axis in CURATED_TEMPLATE_IDENTIFIERS
    ]


def generate_matrix(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir)
    template_root = relative_root_path(out_dir, PROMPT_TEMPLATES_DIR)
    wildcard_root = relative_root_path(out_dir, WILDCARD_DIR)
    templates = build_v1_templates(template_root=template_root, wildcard_root=wildcard_root)
    manifest = emit_version_matrix(
        version_tag="v1", templates=templates, output_dir=out_dir
    )
    write_yaml(out_dir / "manifest.yaml", {"workflow_count": len(manifest), "workflows": manifest})
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the V1 DPO workflow YAMLs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = generate_matrix(args.output_dir)
    print(f"Wrote {len(rows)} workflow YAML files to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
