from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compile_yaml_to_requests_jsonl import build_arg_parser, compile_requests, load_task_yaml, write_compile_outputs
from collect_receipts_and_sessions import collect_receipts_and_sessions, write_jsonl
from tests._paths import REPO_ROOT


class PipelineSmokeTests(unittest.TestCase):
    def _write_failing_python_wrapper(self, path: Path) -> None:
        real_python = shlex.quote(sys.executable)
        path.write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" != "run_workflows_jsonl.py" ]]; then
  exec {real_python} "$@"
fi

input_jsonl="$2"
shift 2
output_jsonl=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      output_jsonl="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

{real_python} - "$input_jsonl" "$output_jsonl" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
output_path.parent.mkdir(parents=True, exist_ok=True)
with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
    for line in source:
        if line.strip():
            request = json.loads(line)
            result = {{
                "request_id": request["request_id"],
                "status": "error",
                "error": "simulated generation failure",
            }}
            target.write(json.dumps(result) + "\\n")
PY
exit 1
""",
            encoding="utf-8",
        )
        path.chmod(0o755)

    def test_example_requests_file_is_valid_jsonl(self) -> None:
        rows = [json.loads(line) for line in (REPO_ROOT / "example_requests.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["request_id"] for row in rows], ["req-0001", "req-0002", "req-0003"])

    def test_compile_requests_accepts_global_seed_override(self) -> None:
        base_dir = REPO_ROOT
        task_yaml_path = base_dir / "template" / "tasks" / "example_task.yaml"
        task = load_task_yaml(task_yaml_path)

        default_requests, default_manifest, _ = compile_requests(task, task_yaml_path)
        overridden_requests, overridden_manifest, _ = compile_requests(task, task_yaml_path, global_seed_override=9999)

        self.assertEqual(default_manifest["global_seed"], int(task["global_seed"]))
        self.assertEqual(overridden_manifest["global_seed"], 9999)
        self.assertEqual(overridden_manifest["source_global_seed"], int(task["global_seed"]))
        self.assertEqual(overridden_manifest["global_seed_override"], 9999)
        self.assertNotEqual(default_requests[0]["request_id"], overridden_requests[0]["request_id"])
        self.assertEqual(overridden_requests[0]["global_seed"], 9999)
        self.assertEqual(overridden_requests[0]["runtime_seed_values"]["global_seed"], 9999)

    def test_build_arg_parser_accepts_global_seed_override(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            ["--task-yaml", "template/tasks/example_task.yaml", "--output-dir", "compiled_run", "--global-seed", "1234"]
        )
        self.assertEqual(args.global_seed, 1234)

    def test_compile_and_collect_round_trip(self) -> None:
        base_dir = REPO_ROOT
        task_yaml_path = base_dir / "template" / "tasks" / "example_task.yaml"
        task = load_task_yaml(task_yaml_path)
        requests, manifest, task_yaml_text = compile_requests(task, task_yaml_path)
        self.assertTrue(requests)
        self.assertEqual(manifest["request_count"], len(requests))
        self.assertIn("task_name", manifest)
        self.assertNotIn("sample", requests[0])
        self.assertIn("workflow_kwargs", requests[1])

        fake_results = []
        for request in requests:
            fake_results.append(
                {
                    "request_id": request["request_id"],
                    "status": "success",
                    "workflow_name": request["workflow_name"],
                    "receipt": {
                        "workflow_name": request["workflow_name"],
                        "prompt_id": f"prompt-{request['request_id']}",
                        "positive_prompt": request["positive_prompt"],
                        "negative_prompt": request["negative_prompt"],
                        "seed": request["seed"],
                        "steps": request["steps"],
                        "cfg": request["cfg"],
                        "width": request["width"],
                        "height": request["height"],
                        "ckpt": request["ckpt"],
                        "lora_stack_config": request["lora_stack_config"],
                        "original_filename": f"{request['request_id']}.png",
                        "saved_filename": f"{request['request_id']}.png",
                        "saved_path": f"/tmp/{request['request_id']}.png",
                        "image_sha256": request["request_id"],
                        "image_size_bytes": 123,
                        "image_width": request["width"],
                        "image_height": request["height"],
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            write_compile_outputs(tmp / "compiled", requests, manifest, task_yaml_text)
            write_jsonl(tmp / "run_results.jsonl", fake_results)
            receipts, sessions = collect_receipts_and_sessions(
                tmp / "compiled" / "requests.jsonl",
                tmp / "run_results.jsonl",
            )
            self.assertEqual(len(receipts), len(requests))
            self.assertEqual(len(sessions), task["session_count"])
            self.assertEqual(len(sessions[0]["images"]), len(task["images"]))
            self.assertEqual(receipts[0]["status"], "success")
            self.assertIn("task_yaml_sha256", receipts[0])
            self.assertIn("saved_path", receipts[0])

    def test_cycle_scripts_collect_errors_and_exit_nonzero(self) -> None:
        task_path = REPO_ROOT / "template" / "tasks" / "example_task.yaml"
        scripts = (
            ("run_full_cycle.sh", "Cycle completed with generation errors."),
            ("run_multi_task_scheduled_cycle.sh", "Scheduled cycle completed with generation errors."),
        )

        for script_name, expected_error in scripts:
            with self.subTest(script=script_name), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                wrapper = root / "failing_python"
                self._write_failing_python_wrapper(wrapper)
                output_root = root / "output"
                result = subprocess.run(
                    [
                        "bash",
                        str(REPO_ROOT / script_name),
                        "--task-yaml",
                        str(task_path),
                        "--output-root",
                        str(output_root),
                        "--python",
                        str(wrapper),
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
                self.assertIn(expected_error, result.stderr)
                self.assertNotIn("Cycle complete.\n", result.stdout)
                self.assertNotIn("Scheduled cycle complete.\n", result.stdout)

                receipt_paths = list(output_root.rglob("collected/receipts.jsonl"))
                self.assertEqual(len(receipt_paths), 1)
                receipts = [
                    json.loads(line)
                    for line in receipt_paths[0].read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertTrue(receipts)
                self.assertTrue(all(receipt["status"] == "error" for receipt in receipts))

    def test_compile_requests_routes_legacy_and_family_specific_prompt_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            template_root = tmp / "prompt_templates"
            wildcard_root = tmp / "wildcard"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "anime_prompt.txt").write_text("anime prompt\n", encoding="utf-8")
            (template_root / "illustration_prompt.txt").write_text("illustration prompt\n", encoding="utf-8")
            (template_root / "sdxl_base_prompt.txt").write_text("sdxl base prompt\n", encoding="utf-8")
            (template_root / "pony_prompt.txt").write_text("pony prompt\n", encoding="utf-8")
            (template_root / "realistic_prompt.txt").write_text("realistic prompt\n", encoding="utf-8")

            task_yaml_path = tmp / "task.yaml"
            task_yaml_path.write_text("version: 1\n", encoding="utf-8")
            task = {
                "version": 1,
                "task_name": "family_routing_smoke",
                "global_seed": 2025,
                "session_count": 1,
                "images": [
                    {
                        "image_name": "illustration",
                        "workflow_name": "SdxlEaseLoraLatentUpscaleWorkflow",
                        "ckpt": "sdxl_illustrij_v20.safetensors",
                        "lora_stack_config": {"toggle": False, "mode": "simple", "num_loras": 1, "lora_1_name": "None"},
                        "prompt_generator": {
                            "name": "wildcard_template_generator",
                            "args": {
                                "seed_control": "session_seed",
                                "template_by_ckpt_family": {
                                    "anime": "anime_prompt",
                                    "pony": "pony_prompt",
                                    "realistic": "realistic_prompt",
                                },
                                "negative_prompt_by_ckpt_family": {
                                    "anime": "anime negative",
                                    "pony": "pony negative",
                                    "realistic": "realistic negative",
                                },
                                "template_root": str(template_root),
                                "wildcard_root": str(wildcard_root),
                            },
                        },
                        "sample": {
                            "generation_seed_control": "image_index_seed",
                            "steps": 30,
                            "cfg": 7.0,
                            "width": 1024,
                            "height": 1024,
                        },
                    },
                    {
                        "image_name": "sdxl_base",
                        "workflow_name": "SdxlEaseLoraLatentUpscaleWorkflow",
                        "ckpt": "sdxl_animagineXL40_v4Opt.safetensors",
                        "lora_stack_config": {"toggle": False, "mode": "simple", "num_loras": 1, "lora_1_name": "None"},
                        "prompt_generator": {
                            "name": "wildcard_template_generator",
                            "args": {
                                "seed_control": "session_seed",
                                "template_by_ckpt_family": {
                                    "anime": "anime_prompt",
                                    "pony": "pony_prompt",
                                    "realistic": "realistic_prompt",
                                },
                                "negative_prompt_by_ckpt_family": {
                                    "anime": "anime negative",
                                    "pony": "pony negative",
                                    "realistic": "realistic negative",
                                },
                                "template_root": str(template_root),
                                "wildcard_root": str(wildcard_root),
                            },
                        },
                        "sample": {
                            "generation_seed_control": "image_index_seed",
                            "steps": 30,
                            "cfg": 7.0,
                            "width": 1024,
                            "height": 1024,
                        },
                    },
                    {
                        "image_name": "pony",
                        "workflow_name": "SdxlEaseLoraLatentUpscaleWorkflow",
                        "ckpt": "sdxl_ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
                        "lora_stack_config": {"toggle": False, "mode": "simple", "num_loras": 1, "lora_1_name": "None"},
                        "prompt_generator": {
                            "name": "wildcard_template_generator",
                            "args": {
                                "seed_control": "session_seed",
                                "template_by_ckpt_family": {
                                    "anime": "anime_prompt",
                                    "pony": "pony_prompt",
                                    "realistic": "realistic_prompt",
                                },
                                "negative_prompt_by_ckpt_family": {
                                    "anime": "anime negative",
                                    "pony": "pony negative",
                                    "realistic": "realistic negative",
                                },
                                "template_root": str(template_root),
                                "wildcard_root": str(wildcard_root),
                            },
                        },
                        "sample": {
                            "generation_seed_control": "image_index_seed",
                            "steps": 30,
                            "cfg": 7.0,
                            "width": 1024,
                            "height": 1024,
                        },
                    },
                    {
                        "image_name": "realistic",
                        "workflow_name": "SdxlEaseLoraLatentUpscaleWorkflow",
                        "ckpt": "sdxl_perfectdeliberate_v60.safetensors",
                        "lora_stack_config": {"toggle": False, "mode": "simple", "num_loras": 1, "lora_1_name": "None"},
                        "prompt_generator": {
                            "name": "wildcard_template_generator",
                            "args": {
                                "seed_control": "session_seed",
                                "template_by_ckpt_family": {
                                    "anime": "anime_prompt",
                                    "pony": "pony_prompt",
                                    "realistic": "realistic_prompt",
                                },
                                "negative_prompt_by_ckpt_family": {
                                    "anime": "anime negative",
                                    "pony": "pony negative",
                                    "realistic": "realistic negative",
                                },
                                "template_root": str(template_root),
                                "wildcard_root": str(wildcard_root),
                            },
                        },
                        "sample": {
                            "generation_seed_control": "image_index_seed",
                            "steps": 30,
                            "cfg": 7.0,
                            "width": 1024,
                            "height": 1024,
                        },
                    },
                ],
            }

            requests, _, _ = compile_requests(task, task_yaml_path)
            by_name = {request["image_name"]: request for request in requests}

            self.assertEqual(by_name["illustration"]["ckpt_family"], "illustration")
            self.assertEqual(by_name["sdxl_base"]["ckpt_family"], "sdxl_anime_base")
            self.assertEqual(by_name["pony"]["ckpt_family"], "pony")
            self.assertEqual(by_name["realistic"]["ckpt_family"], "realistic")
            self.assertEqual(by_name["illustration"]["positive_prompt"], "anime prompt")
            self.assertEqual(by_name["sdxl_base"]["positive_prompt"], "anime prompt")
            self.assertEqual(by_name["pony"]["positive_prompt"], "pony prompt")
            self.assertEqual(by_name["realistic"]["positive_prompt"], "realistic prompt")
            self.assertEqual(by_name["illustration"]["negative_prompt"], "anime negative")
            self.assertEqual(by_name["sdxl_base"]["negative_prompt"], "anime negative")
            self.assertEqual(by_name["pony"]["negative_prompt"], "pony negative")
            self.assertEqual(by_name["realistic"]["negative_prompt"], "realistic negative")


if __name__ == "__main__":
    unittest.main()
