from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_workflows_jsonl import (
    WORKFLOW_REGISTRY,
    build_arg_parser,
    load_dotenv_defaults,
    load_existing_results,
    main,
    resolve_runner_config,
    run_batch,
    select_resume_results,
)


class RunWorkflowsJsonlCliTests(unittest.TestCase):
    def test_load_dotenv_defaults_reads_simple_key_value_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "RUN_WORKFLOWS_URL=http://comfyui.example:8188/",
                        "RUN_WORKFLOWS_ALLOW_INSECURE=true",
                    ]
                ),
                encoding="utf-8",
            )

            resolved = load_dotenv_defaults(env_path)

            self.assertEqual(resolved["RUN_WORKFLOWS_URL"], "http://comfyui.example:8188/")
            self.assertEqual(resolved["RUN_WORKFLOWS_ALLOW_INSECURE"], "true")

    def test_resolve_runner_config_uses_positional_input_and_env_defaults(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["compiled/requests.jsonl"])

        resolved = resolve_runner_config(
            args,
            {
                "RUN_WORKFLOWS_URL": "http://comfyui.example:8188/",
                "RUN_WORKFLOWS_ALLOW_INSECURE": "true",
                "RUN_WORKFLOWS_OUTPUT": "output/run_results.jsonl",
                "RUN_WORKFLOWS_OUTPUT_DIR": "output/images",
            },
        )

        self.assertEqual(resolved["input_jsonl"], "compiled/requests.jsonl")
        self.assertEqual(resolved["output_jsonl"], "output/run_results.jsonl")
        self.assertEqual(resolved["output_dir"], "output/images")
        self.assertEqual(resolved["url"], "http://comfyui.example:8188/")
        self.assertTrue(resolved["allow_insecure"])

    def test_cli_flags_override_env_defaults(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "compiled/requests.jsonl",
                "--output",
                "custom/results.jsonl",
                "--output-dir",
                "custom/images",
                "--url",
                "https://override.example",
                "--secure",
            ]
        )

        resolved = resolve_runner_config(
            args,
            {
                "RUN_WORKFLOWS_URL": "http://comfyui.example:8188/",
                "RUN_WORKFLOWS_ALLOW_INSECURE": "true",
                "RUN_WORKFLOWS_OUTPUT": "output/run_results.jsonl",
                "RUN_WORKFLOWS_OUTPUT_DIR": "output/images",
            },
        )

        self.assertEqual(resolved["output_jsonl"], "custom/results.jsonl")
        self.assertEqual(resolved["output_dir"], "custom/images")
        self.assertEqual(resolved["url"], "https://override.example")
        self.assertFalse(resolved["allow_insecure"])

    def test_resolve_runner_config_requires_input_from_cli_or_env(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args([])

        with self.assertRaisesRegex(ValueError, "Missing input JSONL"):
            resolve_runner_config(args, {})

    def test_resolve_runner_config_defaults_to_local_http(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["compiled/requests.jsonl"])

        resolved = resolve_runner_config(args, {})

        self.assertEqual(resolved["url"], "http://127.0.0.1")
        self.assertEqual(resolved["port"], 8188)
        self.assertFalse(resolved["allow_insecure"])

    def test_select_resume_results_prefers_success_over_later_error(self) -> None:
        selected = select_resume_results(
            [
                {"request_id": "req-1", "status": "success", "receipt": {"saved_path": "/tmp/a.png"}},
                {"request_id": "req-1", "status": "error", "error": "404"},
                {"request_id": "req-2", "status": "error", "error": "timeout"},
            ]
        )

        self.assertEqual(selected["req-1"]["status"], "success")
        self.assertEqual(selected["req-2"]["status"], "error")

    def test_run_batch_retries_errors_and_skips_existing_success(self) -> None:
        class FakeReceipt:
            def __init__(self, request_id: str) -> None:
                self.request_id = request_id

            def to_dict(self) -> dict[str, str]:
                return {"saved_path": f"/tmp/{self.request_id}.png"}

        calls: list[int] = []

        class FakeWorkflow:
            def __init__(
                self,
                ckpt: str,
                lora_stack_config: dict[str, object],
                output_dir: str,
                url: str,
                port: int,
                allow_insecure: bool,
            ) -> None:
                self.ckpt = ckpt

            def generate(
                self,
                positive_prompt: str,
                negative_prompt: str,
                seed: int,
                steps: int,
                cfg: float,
                width: int,
                height: int,
                timeout_seconds: float,
                **workflow_kwargs: object,
            ) -> FakeReceipt:
                calls.append(seed)
                return FakeReceipt(f"req-{len(calls) + 1}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "requests.jsonl"
            output_path = tmp / "run_results.jsonl"
            request_rows = [
                {
                    "request_id": "req-1",
                    "workflow_name": "SdxlEaseLoraWorkflow",
                    "positive_prompt": "a",
                    "negative_prompt": "b",
                    "seed": 11,
                    "steps": 20,
                    "cfg": 7.0,
                    "width": 512,
                    "height": 512,
                    "ckpt": "model.safetensors",
                    "lora_stack_config": {},
                },
                {
                    "request_id": "req-2",
                    "workflow_name": "SdxlEaseLoraWorkflow",
                    "positive_prompt": "c",
                    "negative_prompt": "d",
                    "seed": 22,
                    "steps": 20,
                    "cfg": 7.0,
                    "width": 512,
                    "height": 512,
                    "ckpt": "model.safetensors",
                    "lora_stack_config": {},
                },
            ]
            with input_path.open("w", encoding="utf-8") as handle:
                for row in request_rows:
                    handle.write(json.dumps(row) + "\n")

            existing_rows = [
                {"request_id": "req-1", "status": "success", "workflow_name": "SdxlEaseLoraWorkflow", "receipt": {}},
                {"request_id": "req-2", "status": "error", "workflow_name": "SdxlEaseLoraWorkflow", "error": "404"},
            ]
            with output_path.open("w", encoding="utf-8") as handle:
                for row in existing_rows:
                    handle.write(json.dumps(row) + "\n")

            with patch.dict(WORKFLOW_REGISTRY, {"SdxlEaseLoraWorkflow": FakeWorkflow}):
                failure_count = run_batch(
                    input_jsonl=input_path,
                    output_jsonl=output_path,
                    output_dir=tmp / "images",
                    url="http://example.test",
                    port=8188,
                    allow_insecure=True,
                )

            rows = load_existing_results(output_path)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], 22)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["request_id"], "req-1")
            self.assertEqual(rows[0]["status"], "success")
            self.assertEqual(rows[1]["request_id"], "req-2")
            self.assertEqual(rows[1]["status"], "success")
            self.assertNotIn("error", rows[1])
            self.assertEqual(failure_count, 0)

    def test_run_batch_returns_failure_count_after_writing_error_rows(self) -> None:
        class FailingWorkflow:
            def __init__(self, **kwargs: object) -> None:
                pass

            def generate(self, **kwargs: object) -> object:
                raise RuntimeError("generation failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "requests.jsonl"
            output_path = tmp / "run_results.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "request_id": "req-fail",
                        "workflow_name": "SdxlEaseLoraWorkflow",
                        "positive_prompt": "a",
                        "negative_prompt": "b",
                        "seed": 11,
                        "steps": 20,
                        "cfg": 7.0,
                        "width": 512,
                        "height": 512,
                        "ckpt": "model.safetensors",
                        "lora_stack_config": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.dict(WORKFLOW_REGISTRY, {"SdxlEaseLoraWorkflow": FailingWorkflow}):
                failure_count = run_batch(
                    input_jsonl=input_path,
                    output_jsonl=output_path,
                    output_dir=tmp / "images",
                    url="http://example.test",
                    port=8188,
                )

            rows = load_existing_results(output_path)
            self.assertEqual(failure_count, 1)
            self.assertEqual(rows[0]["status"], "error")
            self.assertIn("generation failed", rows[0]["error"])

    def test_main_exits_nonzero_when_batch_contains_failures(self) -> None:
        resolved = {
            "input_jsonl": "requests.jsonl",
            "output_jsonl": "run_results.jsonl",
            "output_dir": "images",
            "url": "http://127.0.0.1",
            "port": 8188,
            "allow_insecure": False,
            "timeout_seconds": 300.0,
        }
        with (
            patch("run_workflows_jsonl.load_dotenv_defaults", return_value={}),
            patch("run_workflows_jsonl.resolve_runner_config", return_value=resolved),
            patch("run_workflows_jsonl.run_batch", return_value=2),
            patch("sys.argv", ["run_workflows_jsonl.py", "requests.jsonl"]),
        ):
            with self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
