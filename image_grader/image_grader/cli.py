from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .io import InputError, iter_image_dir, iter_image_json
from .runtime import RuntimeValidationError, validate_runtime
from .runner import BatchRunner
from .server import serve


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch anime/aesthetic image scoring tool.")
    parser.add_argument("--config", required=True, help="Path to image grader JSON config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_dir = subparsers.add_parser("run-dir", help="Recursively score images from a directory")
    _add_common_run_args(run_dir)
    run_dir.add_argument("--input-dir", required=True, help="Input image directory")
    run_dir.add_argument("--no-recursive", action="store_true", help="Only scan direct children of input-dir")

    run_json = subparsers.add_parser("run-json", help="Score image paths from JSON or JSONL")
    _add_common_run_args(run_json)
    run_json.add_argument("--input", required=True, help="Input JSON or JSONL file")

    server = subparsers.add_parser("serve", help="Start local HTTP scoring server")
    server.add_argument("--state-dir", required=True, help="SQLite cache/state directory")
    server.add_argument("--host", default="127.0.0.1", help="Host to bind")
    server.add_argument("--port", type=int, default=8790, help="Port to bind")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "run-dir":
            jobs = iter_image_dir(args.input_dir, config.image_extensions, recursive=not args.no_recursive)
            _run(config, args, jobs)
            return
        if args.command == "run-json":
            jobs = iter_image_json(args.input)
            _run(config, args, jobs)
            return
        if args.command == "serve":
            serve(config, state_dir=args.state_dir, host=args.host, port=args.port)
            return
        parser.error(f"unknown command: {args.command}")
    except (ConfigError, InputError, RuntimeValidationError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--state-dir", required=True, help="SQLite cache/state directory")
    parser.add_argument("--model", action="append", dest="models", help="Model id to run. Repeat to select multiple.")
    parser.add_argument("--chunk-size", type=int, help="Image chunk size before per-model batching")
    parser.add_argument("--emit-cached", action="store_true", help="Also write rows whose scores were already cached")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output")


def _run(config: object, args: argparse.Namespace, jobs: object) -> None:
    validate_runtime(config)
    runner = BatchRunner(config, state_dir=args.state_dir)
    try:
        stats = runner.score_jobs(
            jobs,
            model_ids=tuple(args.models) if args.models else None,
            output_path=Path(args.output),
            emit_cached=bool(args.emit_cached),
            chunk_size=args.chunk_size,
            show_progress=not args.no_progress and sys.stderr.isatty(),
        )
    finally:
        runner.close()
    print(
        "Finished: "
        f"seen={stats.seen} emitted={stats.emitted} "
        f"computed_scores={stats.computed_scores} cached_scores={stats.cached_scores} "
        f"failed_images={stats.failed_images}"
    )


if __name__ == "__main__":
    main()
