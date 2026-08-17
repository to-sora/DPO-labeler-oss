from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from image_grader.config import GraderConfig
from image_grader.io import ImageJob
from image_grader_adapter_ui.app import (
    AdapterApp,
    ScoreRecord,
    aspect_ratio_key,
    checkpoint_display_name,
    prompt_template_key,
    rank_expression_score,
)
from image_grader_adapter_ui.server import resolve_bind_host


class FakeRunner:
    def __init__(self, config: GraderConfig, state_dir: Path) -> None:
        self.config = config
        self.state_dir = state_dir

    def score_job_chunk(
        self,
        jobs: list[ImageJob],
        *,
        selected_models: tuple[str, ...],
        preprocess_policy: str = "native",
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for job in jobs:
            score = 8.0 if "good" in job.image_path.name else 3.0
            rows.append(
                {
                    "request_id": job.request_id,
                    "ok": True,
                    "image": {"path": str(job.image_path), "image_id": job.image_path.stem, "width": 8, "height": 8},
                    "metadata": {},
                    "preprocess_policy": preprocess_policy,
                    "scores": {
                        model_id: {
                            "ok": True,
                            "score": score,
                            "scale": "0_10",
                            "raw": {"fake": score},
                            "error": None,
                            "cached": False,
                        }
                        for model_id in selected_models
                    },
                    "error": None,
                    "elapsed_ms": 1.0,
                }
            )
        return rows

    def close(self) -> None:
        return None


def _config(tmpdir: str) -> GraderConfig:
    return GraderConfig.from_mapping(
        {
            "models_root": tmpdir,
            "enabled_models": ["fake_eval"],
            "models": {"fake_eval": {"kind": "noop", "path": "fake.bin"}},
        }
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _make_dataset(root: Path) -> None:
    dataset_dir = root / "collected"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    for name in ("good_a.png", "bad_b.png", "bad_a.png", "bad_b2.png"):
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images_dir / name)
    sessions = [
        {
            "session_id": "session-good",
            "session_index": 1,
            "task_name": "portrait_task",
            "task_yaml_path": "template/tasks/portrait.yaml",
            "task_yaml_sha256": "sha-portrait",
            "images": [
                _image_row(0, "images/good_a.png", "model_good.safetensors"),
                _image_row(1, "images/bad_b.png", "model_good.safetensors"),
            ],
        },
        {
            "session_id": "session-bad",
            "session_index": 2,
            "task_name": "portrait_task",
            "task_yaml_path": "template/tasks/portrait.yaml",
            "task_yaml_sha256": "sha-portrait",
            "images": [
                _image_row(0, "images/bad_a.png", "model_bad.safetensors"),
                _image_row(1, "images/bad_b2.png", "model_bad.safetensors"),
            ],
        },
    ]
    _write_jsonl(dataset_dir / "sessions.jsonl", sessions)


def _image_row(index: int, path: str, ckpt: str) -> dict[str, object]:
    return {
        "image_index": index,
        "image_name": f"image_{index}",
        "saved_path": path,
        "workflow_name": "SdxlEaseLoraWorkflow",
        "ckpt": ckpt,
        "ckpt_family": "illustration",
        "width": 768,
        "height": 768,
        "positive_prompt": "portrait",
        "negative_prompt": "low quality",
        "prompt_generator_name": "wildcard_template_generator",
        "prompt_generator_args": {"template": "portrait"},
    }


def _score_record(score: float, *, model: str = "fake_eval", policy: str = "native") -> ScoreRecord:
    return ScoreRecord(
        run_id="run",
        session_key="session",
        dataset_id="dataset",
        session_id="session",
        image_index=0,
        image_path="image.png",
        task_name="task",
        task_yaml_name="task.yaml",
        workflow_name="workflow",
        ckpt="ckpt",
        ckpt_family="family",
        prompt_template_key="template",
        aspect_ratio="1:1",
        orientation="square",
        eval_model=model,
        preprocess_policy=policy,
        ok=True,
        score=score,
        error=None,
    )


class AdapterUiTests(unittest.TestCase):
    def test_discovers_sessions_and_extracts_prompt_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "output"
            _make_dataset(root)
            app = AdapterApp(
                work_dir=Path(tmpdir) / "work",
                dataset_root=root,
                grader_config=_config(tmpdir),
                runner_factory=FakeRunner,
            )

            sessions = app.list_sessions()["items"]

        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0]["prompt_template_keys"], ["portrait"])
        self.assertEqual(sessions[0]["aspect_ratios"], ["1:1"])

    def test_checkpoint_aliases_preserve_original_filter_values(self) -> None:
        checkpoint = "sdxl_bluePencilXL_v700.safetensors"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "output"
            _make_dataset(root)
            sessions_path = root / "collected" / "sessions.jsonl"
            rows = [json.loads(line) for line in sessions_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["images"][0]["ckpt"] = checkpoint
            _write_jsonl(sessions_path, rows)
            app = AdapterApp(
                work_dir=Path(tmpdir) / "work",
                dataset_root=root,
                grader_config=_config(tmpdir),
                runner_factory=FakeRunner,
            )

            facets = app.get_facets()
            sessions = app.list_sessions()["items"]

        self.assertIn(checkpoint, facets["facets"]["ckpts"])
        self.assertEqual(facets["checkpoint_aliases"][checkpoint], "blue-pencil-xl-v700")
        matching = next(item for item in sessions if checkpoint in item["ckpts"])
        self.assertIn(
            {"name": checkpoint, "alias": "blue-pencil-xl-v700"},
            matching["checkpoints"],
        )

    def test_unknown_checkpoint_alias_falls_back_to_basename(self) -> None:
        self.assertEqual(checkpoint_display_name("private/path/custom.ckpt"), "custom.ckpt")

    def test_discovers_sessions_with_repo_relative_output_image_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            root = repo_root / "output"
            dataset_dir = root / "collected"
            images_dir = root / "images"
            images_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images_dir / "a.png")
            Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images_dir / "b.png")
            _write_jsonl(
                dataset_dir / "sessions.jsonl",
                [
                    {
                        "session_id": "repo-relative",
                        "session_index": 1,
                        "task_name": "task",
                        "task_yaml_path": "template/tasks/task.yaml",
                        "images": [
                            _image_row(0, "output/images/a.png", "model.safetensors"),
                            _image_row(1, "output/images/b.png", "model.safetensors"),
                        ],
                    }
                ],
            )
            app = AdapterApp(
                work_dir=root / "ai_grader_admin",
                dataset_root=root,
                repo_root=repo_root,
                grader_config=_config(tmpdir),
                runner_factory=FakeRunner,
            )

            sessions = app.list_sessions()["items"]

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "repo-relative")

    def test_report_flags_bad_generation_model_for_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "output"
            _make_dataset(root)
            app = AdapterApp(
                work_dir=Path(tmpdir) / "work",
                dataset_root=root,
                grader_config=_config(tmpdir),
                runner_factory=FakeRunner,
            )
            template = app.save_template({"name": "ai_eval_v1", "models": ["fake_eval"]})["template"]

            result = app.run_report({"template": template})

        flags = result["report"]["bad_fit_flags"]
        self.assertTrue(any(flag["ckpt"] == "model_bad.safetensors" for flag in flags))
        table = result["report"]["tables"]["prompt_template_model_aspect"]
        self.assertTrue(any(row["prompt_template_key"] == "portrait" and row["aspect_ratio"] == "1:1" for row in table))

    def test_pagination_and_playground_exact_session_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "output"
            _make_dataset(root)
            app = AdapterApp(
                work_dir=Path(tmpdir) / "work",
                dataset_root=root,
                grader_config=_config(tmpdir),
                runner_factory=FakeRunner,
            )
            page = app.list_sessions(limit=1)
            selected_key = page["items"][0]["session_key"]
            template = app.save_template({"name": "playground_eval", "models": ["fake_eval"]})["template"]

            result = app.run_playground(
                {
                    "template": template,
                    "session_keys": [selected_key],
                    "filters": {"ckpts": ["model_bad.safetensors"]},
                    "limit": 10,
                }
            )

        self.assertEqual(page["total"], 2)
        self.assertEqual(len(page["items"]), 1)
        self.assertEqual(page["cursor"], 0)
        self.assertEqual(page["next_cursor"], 1)
        self.assertEqual(result["report"]["summary"]["session_count"], 1)
        self.assertEqual(result["score_count"], 2)
        self.assertEqual(len(result["scores"]), 2)
        self.assertEqual({row["session_key"] for row in result["scores"]}, {selected_key})
        self.assertEqual(result["playground_ranking"]["expression"], '(score["native"]["fake_eval"]) / 1')

    def test_playground_ranks_best_and_worst_percent_with_expression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "output"
            _make_dataset(root)
            app = AdapterApp(
                work_dir=Path(tmpdir) / "work",
                dataset_root=root,
                grader_config=_config(tmpdir),
                runner_factory=FakeRunner,
            )

            result = app.run_playground(
                {
                    "template": {
                        "name": "rank_eval",
                        "models": ["fake_eval"],
                        "preprocess_policies": ["native", "fit_pad_square"],
                    },
                    "limit": 10,
                    "rank_expression": '(score["native"]["fake_eval"] + score["fit_pad_square"]["fake_eval"]) / 2',
                    "rank_percent": 25,
                }
            )

        ranking = result["playground_ranking"]
        self.assertEqual(ranking["ranked_count"], 4)
        self.assertEqual(ranking["bucket_size"], 1)
        self.assertEqual(ranking["top"][0]["score"], 8.0)
        self.assertEqual(ranking["bottom"][0]["score"], 3.0)
        self.assertEqual(ranking["top"][0]["image_index"], 0)

    def test_rank_expression_supports_percentiles_over_image_scores(self) -> None:
        records = [
            _score_record(1.0, model="m1", policy="native"),
            _score_record(2.0, model="m2", policy="native"),
            _score_record(3.0, model="m1", policy="fit_pad_square"),
            _score_record(100.0, model="m2", policy="fit_pad_square"),
        ]

        lower, lower_errors = rank_expression_score("percentile(scores(), 25)", records)
        upper, upper_errors = rank_expression_score("p75(scores())", records)
        direct, direct_errors = rank_expression_score(
            'max(min(score["native"]["m1"], score["native"]["m2"], score["fit_pad_square"]["m1"]), '
            'percentile(score["native"]["m1"], score["native"]["m2"], score["fit_pad_square"]["m1"]; 75))',
            records,
        )
        alias, alias_errors = rank_expression_score(
            'p75(score["native"]["m1"], score["native"]["m2"], score["fit_pad_square"]["m1"])',
            records,
        )

        self.assertEqual(lower_errors, [])
        self.assertEqual(upper_errors, [])
        self.assertEqual(direct_errors, [])
        self.assertEqual(alias_errors, [])
        self.assertEqual(lower, 1.75)
        self.assertEqual(upper, 27.25)
        self.assertEqual(direct, 2.5)
        self.assertEqual(alias, 2.5)

    def test_ai_label_events_are_schema_aligned_and_skip_human_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "output"
            _make_dataset(root)
            human_event = {
                "event_id": "human-1",
                "created_at": "2026-01-01T00:00:00.000Z",
                "client_ts": "2026-01-01T00:00:00.000Z",
                "app_version": "test",
                "dataset_id": "collected",
                "session_id": "session-bad",
                "task_key": "collected::sha-portrait",
                "task_name": "portrait_task",
                "task_yaml_name": "portrait.yaml",
                "workflow_name": "SdxlEaseLoraWorkflow",
                "primary_ckpt": "model_bad.safetensors",
                "reviewer_username": "human",
                "client_instance_id": "client",
                "review_id": "review",
                "decision": "skip",
                "defects_a": [],
                "defects_b": [],
                "display_order": [0, 1],
                "chosen_image_indices": [],
                "defects_by_image_index": {"0": [], "1": []},
                "note": "",
            }
            _write_jsonl(root / "labeler_state" / "label_events.jsonl", [human_event])
            app = AdapterApp(
                work_dir=Path(tmpdir) / "work",
                dataset_root=root,
                grader_config=_config(tmpdir),
                runner_factory=FakeRunner,
            )
            template = app.save_template({"name": "quality_gate", "models": ["fake_eval"]})["template"]

            result = app.write_ai_labels({"template": template})
            rows = [
                json.loads(line)
                for line in Path(result["ai_label_events_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(result["written_events"], 1)
        self.assertEqual(rows[0]["reviewer_username"], "AI_quality_gate")
        self.assertEqual(rows[0]["decision"], "a_good")
        self.assertEqual(rows[0]["chosen_image_indices"], [0])
        self.assertEqual(result["labels"]["skipped_count"], 1)

    def test_helpers(self) -> None:
        self.assertEqual(aspect_ratio_key(768, 1024), "3:4")
        self.assertEqual(prompt_template_key({"prompt_generator_args": {"template_name": "x"}}), "x")

    def test_tailscale_auto_falls_back_to_localhost(self) -> None:
        with patch("image_grader_adapter_ui.server.detect_tailscale_host", return_value="127.0.0.1"):
            self.assertEqual(resolve_bind_host("tailscale-auto"), "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
