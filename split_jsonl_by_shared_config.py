from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def has_unresolved_option_spec_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped.startswith("{") or "seed_control" not in stripped or "options" not in stripped:
        return False
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return False
    return isinstance(parsed, dict) and ("seed_control" in parsed or "options" in parsed)


def load_jsonl(input_path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Each JSONL line must be a JSON object. Bad line: {line_number}")
            if "ckpt" not in payload:
                raise ValueError(f"Line {line_number} is missing required field 'ckpt'")
            if "lora_stack_config" not in payload:
                raise ValueError(f"Line {line_number} is missing required field 'lora_stack_config'")
            if not isinstance(payload["ckpt"], str):
                raise ValueError(f"Line {line_number} field 'ckpt' must be a string checkpoint name")
            if has_unresolved_option_spec_string(payload["ckpt"]):
                raise ValueError(
                    f"Line {line_number} field 'ckpt' contains an unresolved weighted option spec; "
                    "recompile requests before splitting"
                )
            records.append(payload)

    if not records:
        raise ValueError(f"No JSON objects found in {path}")
    return records


def make_group_key(record: Dict[str, Any]) -> Tuple[str, str]:
    ckpt = str(record["ckpt"])
    lora_key = canonical_json(record["lora_stack_config"])
    return ckpt, lora_key


def make_group_id(ckpt: str, lora_key: str) -> str:
    digest = hashlib.sha256(f"{ckpt}\n{lora_key}".encode("utf-8")).hexdigest()[:16]
    return digest


def sanitize_filename(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    name = "".join(safe).strip("._")
    return name or "unknown_ckpt"


def split_jsonl_by_shared_config(
    input_jsonl: str | Path,
    output_dir: str | Path,
    manifest_name: str = "manifest.jsonl",
) -> Path:
    records = load_jsonl(input_jsonl)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = make_group_key(record)
        grouped.setdefault(key, []).append(record)

    manifest_path = out_dir / manifest_name
    if manifest_path.exists():
        manifest_path.unlink()

    for index, ((ckpt, lora_key), items) in enumerate(grouped.items(), start=1):
        group_id = make_group_id(ckpt, lora_key)
        ckpt_label = sanitize_filename(Path(ckpt).stem)
        output_name = f"{index:04d}__{ckpt_label}__{group_id}.jsonl"
        output_path = out_dir / output_name

        with output_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        manifest_entry = {
            "group_index": index,
            "group_id": group_id,
            "ckpt": ckpt,
            "lora_stack_config": json.loads(lora_key),
            "record_count": len(items),
            "output_jsonl": output_name,
        }
        with manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split an arbitrary input JSONL into multiple JSONL files grouped by "
            "the unique combination of ckpt + lora_stack_config."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory for grouped JSONL files")
    parser.add_argument(
        "--manifest-name",
        default="manifest.jsonl",
        help="Filename for the manifest JSONL written inside output-dir",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    manifest_path = split_jsonl_by_shared_config(
        input_jsonl=args.input,
        output_dir=args.output_dir,
        manifest_name=args.manifest_name,
    )
    print(f"Wrote grouped JSONL files to: {Path(args.output_dir)}")
    print(f"Wrote manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
