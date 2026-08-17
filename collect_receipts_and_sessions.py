from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Each JSONL line must be a JSON object. Bad line {line_number} in {path}")
            records.append(payload)
    return records


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def merge_request_and_result(request: Dict[str, Any], result: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(request)
    if result is None:
        merged["status"] = "missing_result"
        merged["error"] = "No runner result found for request_id"
        return merged

    merged["status"] = result.get("status", "unknown")
    merged["runner_result"] = result

    if result.get("status") == "success" and isinstance(result.get("receipt"), dict):
        merged.update(result["receipt"])
    else:
        merged["error"] = result.get("error", "Unknown runner error")

    return merged


def collect_receipts_and_sessions(
    requests_jsonl: str | Path,
    run_results_jsonl: str | Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    requests = load_jsonl(requests_jsonl)
    results = load_jsonl(run_results_jsonl)
    result_by_request_id = {row["request_id"]: row for row in results if "request_id" in row}

    receipts: List[Dict[str, Any]] = []
    grouped_sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for request in requests:
        request_id = request["request_id"]
        receipt = merge_request_and_result(request, result_by_request_id.get(request_id))
        receipts.append(receipt)
        grouped_sessions[request["session_id"]].append(receipt)

    sessions: List[Dict[str, Any]] = []
    for session_id, rows in grouped_sessions.items():
        ordered_rows = sorted(rows, key=lambda row: (row.get("image_index", 0), row.get("image_name", "")))
        first = ordered_rows[0]
        session_record = {
            "session_id": session_id,
            "task_name": first.get("task_name"),
            "task_version": first.get("task_version"),
            "task_yaml_path": first.get("task_yaml_path"),
            "task_yaml_sha256": first.get("task_yaml_sha256"),
            "compiler_version": first.get("compiler_version"),
            "global_seed": first.get("global_seed"),
            "session_index": first.get("session_index"),
            "images": ordered_rows,
        }
        sessions.append(session_record)

    sessions.sort(key=lambda row: row.get("session_index", 0))
    return receipts, sessions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join compiled requests.jsonl with runner output JSONL, and produce receipts.jsonl "
            "plus sessions.jsonl. Each receipt preserves all upstream request metadata."
        )
    )
    parser.add_argument("--requests", required=True, help="Path to compiled requests.jsonl")
    parser.add_argument("--run-results", required=True, help="Path to runner output JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory to write receipts.jsonl and sessions.jsonl")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    receipts, sessions = collect_receipts_and_sessions(args.requests, args.run_results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "receipts.jsonl", receipts)
    write_jsonl(out_dir / "sessions.jsonl", sessions)
    print(f"Wrote receipts to: {out_dir / 'receipts.jsonl'}")
    print(f"Wrote sessions to: {out_dir / 'sessions.jsonl'}")


if __name__ == "__main__":
    main()
