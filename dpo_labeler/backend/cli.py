from __future__ import annotations

import argparse
import json
from pathlib import Path

from .app import DpoLabelerApp


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO labeler utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_tasks = subparsers.add_parser("list-tasks", help="List discovered tasks")
    _add_shared_args(list_tasks)

    export = subparsers.add_parser("export", help="Export filtered JSONL")
    _add_shared_args(export)
    export.add_argument("--type", required=True, choices=["label-events", "labels-latest", "preference-pairs", "dpo-pairs"])
    export.add_argument("--filter-json", default="", help="Filter AST as JSON string")
    export.add_argument("--output", required=True, help="Path to write the JSONL export")

    return parser


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True, help="Root directory to recursively scan for sessions.jsonl files")
    parser.add_argument("--state-dir", required=True, help="Directory for labeler state")
    parser.add_argument("--review-round-seed", default="default-round-v1", help="Stable seed for deterministic random review order")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        dest="exclude_dirs",
        metavar="PATTERN",
        help="Exclude any dataset path segment matching this rsync-style glob (repeatable)",
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    app = DpoLabelerApp(
        dataset_root=args.dataset_root,
        state_dir=args.state_dir,
        invite_token="cli-disabled",
        session_secret="cli-disabled",
        rescan_seconds=3600,
        review_round_seed=args.review_round_seed,
        exclude_dirs=args.exclude_dirs,
    )

    if args.command == "list-tasks":
        payload = app.get_catalog()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "export":
        filter_ast = json.loads(args.filter_json) if args.filter_json else None
        output_path = Path(args.output)
        text = app.export_text(args.type, filter_ast)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(str(output_path))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
