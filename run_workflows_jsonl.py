from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Type

from sdxl_ease_lora_workflow import SdxlEaseLoraWorkflow
from sdxl_ease_lora_latent_upscale_workflow import SdxlEaseLoraLatentUpscaleWorkflow
from sdxl_ease_lora_model_upscale_workflow import SdxlEaseLoraModelUpscaleWorkflow
from workflow_base import WorkflowBase

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

_MISSING_TQDM_WARNING_SHOWN = False


WORKFLOW_REGISTRY: Dict[str, Type[WorkflowBase]] = {
    "SdxlEaseLoraWorkflow": SdxlEaseLoraWorkflow,
    "SdxlEaseLoraLatentUpscaleWorkflow": SdxlEaseLoraLatentUpscaleWorkflow,
    "SdxlEaseLoraModelUpscaleWorkflow": SdxlEaseLoraModelUpscaleWorkflow,
}

DEFAULT_ENV_PATH = ".env"
DEFAULT_OUTPUT_JSONL = "output/run_results.jsonl"
DEFAULT_OUTPUT_DIR = "output/images"


class _NoOpProgress:
    def __init__(self, total: int, initial: int = 0, desc: str | None = None) -> None:
        self.total = int(total)
        self.n = int(initial)
        self.desc = desc

    def update(self, n: int = 1) -> None:
        self.n += int(n)

    def close(self) -> None:
        return None

    def __enter__(self) -> "_NoOpProgress":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def create_progress(total: int, *, initial: int = 0, desc: str = "Running workflows") -> Any:
    global _MISSING_TQDM_WARNING_SHOWN
    if _tqdm is None:
        if not _MISSING_TQDM_WARNING_SHOWN:
            print("tqdm is not installed; continuing without a visual progress bar.")
            _MISSING_TQDM_WARNING_SHOWN = True
        return _NoOpProgress(total=total, initial=initial, desc=desc)
    return _tqdm(total=total, initial=initial, desc=desc, unit="req")


def load_dotenv_defaults(path: str | Path = DEFAULT_ENV_PATH) -> Dict[str, str]:
    env_path = Path(path)
    if not env_path.is_file():
        return {}

    resolved: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        resolved[key] = value
    return resolved


def _env_or_os(env_defaults: Mapping[str, str], key: str) -> str | None:
    if key in os.environ:
        return os.environ[key]
    return env_defaults.get(key)


def _parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}")


def resolve_runner_config(args: argparse.Namespace, env_defaults: Mapping[str, str]) -> Dict[str, Any]:
    input_jsonl = args.input or args.input_path or _env_or_os(env_defaults, "RUN_WORKFLOWS_INPUT")
    if not input_jsonl:
        raise ValueError("Missing input JSONL. Pass --input, a positional input path, or RUN_WORKFLOWS_INPUT in .env")

    output_jsonl = args.output or _env_or_os(env_defaults, "RUN_WORKFLOWS_OUTPUT") or DEFAULT_OUTPUT_JSONL
    output_dir = args.output_dir or _env_or_os(env_defaults, "RUN_WORKFLOWS_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR
    url = args.url or _env_or_os(env_defaults, "RUN_WORKFLOWS_URL") or "http://127.0.0.1"

    port_value = args.port
    if port_value is None:
        port_value = _env_or_os(env_defaults, "RUN_WORKFLOWS_PORT") or 8188

    timeout_value = args.timeout_seconds
    if timeout_value is None:
        timeout_value = _env_or_os(env_defaults, "RUN_WORKFLOWS_TIMEOUT_SECONDS") or 300.0

    if args.allow_insecure is not None:
        allow_insecure = bool(args.allow_insecure)
    else:
        env_allow_insecure = _env_or_os(env_defaults, "RUN_WORKFLOWS_ALLOW_INSECURE")
        allow_insecure = _parse_bool(env_allow_insecure) if env_allow_insecure is not None else False

    return {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_dir": str(output_dir),
        "url": str(url),
        "port": int(port_value),
        "allow_insecure": bool(allow_insecure),
        "timeout_seconds": float(timeout_value),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


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
            records.append(payload)

    if not records:
        raise ValueError(f"No JSON objects found in {path}")
    return records


def load_existing_results(output_path: str | Path) -> List[Dict[str, Any]]:
    path = Path(output_path)
    if not path.is_file():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in existing output {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Each JSONL line must be a JSON object. Bad line: {line_number} in {path}")
            records.append(payload)
    return records


def _result_is_success(result: Mapping[str, Any]) -> bool:
    return result.get("status") == "success" and isinstance(result.get("receipt"), dict)


def select_resume_results(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest_by_request_id: Dict[str, Dict[str, Any]] = {}
    latest_success_by_request_id: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        request_id = row.get("request_id")
        if request_id in (None, ""):
            continue
        request_id_str = str(request_id)
        materialized = dict(row)
        latest_by_request_id[request_id_str] = materialized
        if _result_is_success(materialized):
            latest_success_by_request_id[request_id_str] = materialized

    selected: Dict[str, Dict[str, Any]] = {}
    request_ids = set(latest_by_request_id) | set(latest_success_by_request_id)
    for request_id in request_ids:
        if request_id in latest_success_by_request_id:
            selected[request_id] = latest_success_by_request_id[request_id]
        else:
            selected[request_id] = latest_by_request_id[request_id]
    return selected


def validate_shared_config(records: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    first = records[0]
    if "ckpt" not in first:
        raise ValueError("Input JSONL line 1 must contain 'ckpt'")
    if "lora_stack_config" not in first:
        raise ValueError("Input JSONL line 1 must contain 'lora_stack_config'")

    shared_ckpt = first["ckpt"]
    shared_lora = first["lora_stack_config"]

    shared_lora_key = _canonical_json(shared_lora)

    for index, record in enumerate(records, start=1):
        if record.get("ckpt") != shared_ckpt:
            raise ValueError(
                f"Shared config mismatch at line {index}: expected ckpt={shared_ckpt!r}, got {record.get('ckpt')!r}"
            )
        if _canonical_json(record.get("lora_stack_config")) != shared_lora_key:
            raise ValueError(f"Shared config mismatch at line {index}: lora_stack_config differs from line 1")

    return str(shared_ckpt), dict(shared_lora)


def validate_record_schema(record: Dict[str, Any], line_number: int) -> None:
    required_fields = [
        "request_id",
        "workflow_name",
        "positive_prompt",
        "negative_prompt",
        "seed",
        "steps",
        "cfg",
        "width",
        "height",
    ]
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f"Line {line_number} is missing required fields: {missing}")

    workflow_name = record["workflow_name"]
    if workflow_name not in WORKFLOW_REGISTRY:
        raise ValueError(
            f"Line {line_number} has unsupported workflow_name={workflow_name!r}. "
            f"Supported values: {sorted(WORKFLOW_REGISTRY)}"
        )


def workflow_kwargs_for_record(record: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    nested = record.get("workflow_kwargs")
    if isinstance(nested, dict):
        kwargs.update(nested)

    # Legacy compatibility with earlier flat JSONL fields.
    if "upscale_by" in record and "upscale_by" not in kwargs:
        kwargs["upscale_by"] = record["upscale_by"]
    if "upscale_model_name" in record and "upscale_model_name" not in kwargs:
        kwargs["upscale_model_name"] = record["upscale_model_name"]
    return kwargs


def append_jsonl(output_path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(output_path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def run_batch(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    output_dir: str | Path,
    url: str,
    port: int,
    allow_insecure: bool = True,
    timeout_seconds: float = 300.0,
) -> int:
    records = load_jsonl(input_jsonl)

    for line_number, record in enumerate(records, start=1):
        validate_record_schema(record, line_number)

    shared_ckpt, shared_lora = validate_shared_config(records)

    existing_rows = load_existing_results(output_jsonl)
    resume_results = select_resume_results(existing_rows)
    preserved_success_rows = [
        result
        for result in resume_results.values()
        if _result_is_success(result)
    ]
    preserved_success_rows.sort(key=lambda row: str(row.get("request_id", "")))
    write_jsonl(output_jsonl, preserved_success_rows)

    completed_request_ids = {str(row["request_id"]) for row in preserved_success_rows if "request_id" in row}
    retry_count = len(records) - len(completed_request_ids)
    removed_error_count = max(len(existing_rows) - len(preserved_success_rows), 0)
    print(
        f"Loaded {len(records)} requests. "
        f"Skipping {len(completed_request_ids)} existing successes, retrying {retry_count} requests."
    )
    if existing_rows:
        print(f"Removed {removed_error_count} stale non-success result rows from {Path(output_jsonl)} before retry.")

    failure_count = 0
    with create_progress(len(records), initial=len(completed_request_ids)) as progress:
        for record in records:
            request_id = str(record["request_id"])
            if request_id in completed_request_ids:
                continue

            workflow_cls = WORKFLOW_REGISTRY[record["workflow_name"]]
            workflow = workflow_cls(
                ckpt=shared_ckpt,
                lora_stack_config=shared_lora,
                output_dir=output_dir,
                url=url,
                port=port,
                allow_insecure=allow_insecure,
            )

            try:
                receipt = workflow.generate(
                    positive_prompt=record["positive_prompt"],
                    negative_prompt=record["negative_prompt"],
                    seed=int(record["seed"]),
                    steps=int(record["steps"]),
                    cfg=float(record["cfg"]),
                    width=int(record["width"]),
                    height=int(record["height"]),
                    timeout_seconds=float(timeout_seconds),
                    **workflow_kwargs_for_record(record),
                )
                append_jsonl(
                    output_jsonl,
                    {
                        "request_id": request_id,
                        "status": "success",
                        "workflow_name": record["workflow_name"],
                        "receipt": receipt.to_dict(),
                    },
                )
            except Exception as exc:
                failure_count += 1
                append_jsonl(
                    output_jsonl,
                    {
                        "request_id": request_id,
                        "status": "error",
                        "workflow_name": record["workflow_name"],
                        "error": str(exc),
                    },
                )
            progress.update(1)
    return failure_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ComfyUI workflows from an input JSONL. "
            "The script validates that every line shares the same ckpt + lora_stack_config "
            "before making any API call."
        )
    )
    parser.add_argument("input_path", nargs="?", help="Optional positional path to input JSONL")
    parser.add_argument("--input", help="Path to input JSONL")
    parser.add_argument("--output", help=f"Path to output JSONL. Defaults to {DEFAULT_OUTPUT_JSONL} or .env")
    parser.add_argument(
        "--output-dir",
        help=f"Directory to save generated images. Defaults to {DEFAULT_OUTPUT_DIR} or .env",
    )
    parser.add_argument("--url", help="ComfyUI host or full base URL")
    parser.add_argument("--port", type=int, help="ComfyUI port")
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--allow-insecure",
        action="store_true",
        dest="allow_insecure",
        default=None,
        help="Use insecure HTTP / disable SSL verification",
    )
    transport_group.add_argument(
        "--secure",
        action="store_false",
        dest="allow_insecure",
        help="Force HTTPS / SSL verification even if .env enables insecure mode",
    )
    parser.add_argument("--timeout-seconds", type=float, help="Per-request generation timeout")
    return parser


def main() -> None:
    env_defaults = load_dotenv_defaults()
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        resolved_config = resolve_runner_config(args, env_defaults)
    except ValueError as exc:
        parser.error(str(exc))
    failure_count = run_batch(**resolved_config)
    if failure_count:
        print(f"{failure_count} workflow request(s) failed. See {resolved_config['output_jsonl']}.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
