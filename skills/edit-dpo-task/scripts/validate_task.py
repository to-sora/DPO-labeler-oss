#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def find_repo_root(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([Path.cwd(), Path(__file__).resolve().parents[3]])
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "compile_yaml_to_requests_jsonl.py").is_file() and (
            resolved / "checkpoint_aliases.yaml"
        ).is_file():
            return resolved
    raise RuntimeError("Cannot locate repository root; pass --repo-root")


def checkpoint_values(spec: Any) -> list[str]:
    if isinstance(spec, str):
        return [spec]
    if not isinstance(spec, Mapping) or "options" not in spec:
        raise ValueError(f"ckpt must be a string or option specification, got {spec!r}")
    options = spec["options"]
    if not isinstance(options, list):
        raise ValueError("ckpt.options must be a list")
    values: list[str] = []
    for index, option in enumerate(options, start=1):
        if not isinstance(option, Mapping) or "value" not in option:
            raise ValueError(f"ckpt.options[{index}] must contain value")
        values.extend(checkpoint_values(option["value"]))
    return values


def unresolved_prompt_rows(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        str(record["request_id"])
        for record in records
        if "__" in str(record.get("positive_prompt", ""))
        or "{{" in str(record.get("positive_prompt", ""))
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile and validate one Comfy DPO task YAML.")
    parser.add_argument("task_yaml")
    parser.add_argument("--repo-root")
    parser.add_argument("--expected-per-image", type=int)
    parser.add_argument("--publication", action="store_true")
    parser.add_argument("--allow-default-family", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        repo_root = find_repo_root(args.repo_root)
        sys.path.insert(0, str(repo_root))

        from checkpoint_registry import get_checkpoint_registry
        from compile_yaml_to_requests_jsonl import compile_requests, load_task_yaml

        task_path = Path(args.task_yaml)
        if not task_path.is_absolute():
            task_path = (repo_root / task_path).resolve()
        task = load_task_yaml(task_path)
        records, manifest, _ = compile_requests(task, task_path)

        image_counts = Counter(str(record["image_name"]) for record in records)
        errors: list[str] = []
        if args.expected_per_image is not None:
            wrong_counts = {
                image_name: count
                for image_name, count in image_counts.items()
                if count != args.expected_per_image
            }
            if wrong_counts:
                errors.append(
                    f"expected {args.expected_per_image} requests per image, got {wrong_counts}"
                )

        unresolved = unresolved_prompt_rows(records)
        if unresolved:
            errors.append(f"unresolved wildcard tokens in {len(unresolved)} request(s)")

        registry = get_checkpoint_registry()
        checkpoint_rows: list[dict[str, Any]] = []
        seen_checkpoints: set[str] = set()
        for image in task["images"]:
            for checkpoint in checkpoint_values(image["ckpt"]):
                if checkpoint in seen_checkpoints:
                    continue
                seen_checkpoints.add(checkpoint)
                resolved = registry.resolve(checkpoint)
                checkpoint_rows.append(
                    {
                        "checkpoint": checkpoint,
                        "model_id": resolved.model_id,
                        "family": resolved.family,
                        "family_source": resolved.family_source,
                        "matched_keyword": resolved.matched_keyword,
                        "visibility": resolved.visibility,
                        "publish": resolved.publish,
                    }
                )
                if resolved.family_source == "default" and not args.allow_default_family:
                    errors.append(f"checkpoint {checkpoint!r} uses the default family fallback")
                if args.publication and (
                    resolved.visibility != "public" or not resolved.publish
                ):
                    errors.append(
                        f"checkpoint {checkpoint!r} is not registered as publishable public"
                    )

        summary = {
            "task": str(task_path),
            "task_name": manifest["task_name"],
            "sessions": manifest["session_count"],
            "images_per_session": manifest["image_count_per_session"],
            "requests": manifest["request_count"],
            "image_counts": dict(sorted(image_counts.items())),
            "families": sorted({str(record["ckpt_family"]) for record in records}),
            "checkpoints": checkpoint_rows,
            "unresolved_prompts": len(unresolved),
            "errors": errors,
        }
        if args.as_json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"Task: {summary['task_name']}")
            print(f"Sessions: {summary['sessions']}")
            print(f"Requests: {summary['requests']}")
            print(f"Image counts: {summary['image_counts']}")
            print(f"Families: {summary['families']}")
            for checkpoint in checkpoint_rows:
                print(
                    "Checkpoint: "
                    f"{checkpoint['checkpoint']} -> {checkpoint['family']} "
                    f"({checkpoint['family_source']}, {checkpoint['visibility']}, "
                    f"publish={checkpoint['publish']})"
                )
            if errors:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 2
            print("Validation passed.")
            return 0
        return 2 if errors else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
