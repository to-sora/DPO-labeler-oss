"""V2 DPO workflow generator (consolidated).

Emits 4 YAMLs: ``v2_compact_same/diff``, ``v2_cinematic_same/diff``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from layout_paths import PROMPT_TEMPLATES_DIR, WILDCARD_DIR, WORKFLOWS_DIR, relative_path as relative_root_path

from generators._shared_task import (
    TemplateEntry,
    build_research_templates,
    emit_version_matrix,
    make_research_template_args,
    write_yaml,
)


DEFAULT_OUTPUT_DIR = WORKFLOWS_DIR / "dpo_workflow_v2"

PROMPT_AXES_SHORT = {"hybrid_compact": "compact", "hybrid_cinematic": "cinematic"}

RESEARCH_TEMPLATE_IDENTIFIERS: Dict[str, Dict[str, str]] = {
    "hybrid_compact": {
        "anime": "research_v2/hybrid_compact_anime_adult",
        "pony": "research_v2/hybrid_compact_pony_adult",
        "realistic": "research_v2/hybrid_compact_realistic_adult",
    },
    "hybrid_cinematic": {
        "anime": "research_v2/hybrid_cinematic_anime_adult",
        "pony": "research_v2/hybrid_cinematic_pony_adult",
        "realistic": "research_v2/hybrid_cinematic_realistic_adult",
    },
}

NEGATIVE_PROMPTS_BY_FAMILY = {
    "anime": (
        "lowres, blurry, bad anatomy, bad hands, extra fingers, extra limbs, deformed face, "
        "asymmetrical eyes, cross-eyed, extra eyes, malformed mouth, bad teeth, distorted pupils, "
        "jpeg artifacts, watermark, text, multiple views, worst quality"
    ),
    "pony": (
        "score_6, score_5, score_4, source_furry, lowres, blurry, bad anatomy, bad hands, "
        "extra digits, deformed face, asymmetrical eyes, cross-eyed, extra eyes, malformed mouth, "
        "bad teeth, distorted pupils, watermark, text, multiple views"
    ),
    "realistic": (
        "anime, illustration, cartoon, painting, cgi, 3d, lowres, blurry, bad anatomy, "
        "bad hands, extra fingers, deformed face, asymmetrical eyes, cross-eyed, extra eyes, "
        "malformed mouth, bad teeth, distorted pupils, waxy skin, watermark, text"
    ),
}


def make_template_args(
    prompt_axis: str,
    *,
    template_root: str,
    wildcard_root: str,
) -> Dict[str, Any]:
    """Kept for ``generate_batch_v1_7`` compatibility."""
    if prompt_axis not in RESEARCH_TEMPLATE_IDENTIFIERS:
        raise ValueError(f"Unsupported V2 prompt axis {prompt_axis!r}")
    return make_research_template_args(
        family_templates=RESEARCH_TEMPLATE_IDENTIFIERS[prompt_axis],
        negatives=NEGATIVE_PROMPTS_BY_FAMILY,
        template_root=template_root,
        wildcard_root=wildcard_root,
    )


def generate_matrix(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> List[Dict[str, Any]]:
    out_dir = Path(output_dir)
    template_root = relative_root_path(out_dir, PROMPT_TEMPLATES_DIR)
    wildcard_root = relative_root_path(out_dir, WILDCARD_DIR)
    templates = build_research_templates(
        prompt_axes_short=PROMPT_AXES_SHORT,
        identifiers=RESEARCH_TEMPLATE_IDENTIFIERS,
        negatives=NEGATIVE_PROMPTS_BY_FAMILY,
        template_root=template_root,
        wildcard_root=wildcard_root,
    )
    manifest = emit_version_matrix(version_tag="v2", templates=templates, output_dir=out_dir)
    write_yaml(out_dir / "manifest.yaml", {"workflow_count": len(manifest), "workflows": manifest})
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the V2 DPO workflow YAMLs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = generate_matrix(args.output_dir)
    print(f"Wrote {len(rows)} workflow YAML files to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
