from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from image_grader.config import GraderConfig
from image_grader.io import ImageJob
from image_grader.runner import BatchRunner, PreparedImage, preprocess_pil_image
from image_grader.server import GraderServerError, ImageGraderServerApp


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    def score_batch(self, images: list[PreparedImage]) -> list[dict[str, object]]:
        self.calls += 1
        return [
            {"ok": True, "score": float(index + 1), "scale": "0_10", "raw": {"fake": True}, "error": None}
            for index, _image in enumerate(images)
        ]


class FakeRegistry:
    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend

    def get(self, model_id: str) -> FakeBackend:
        return self.backend

    def close(self) -> None:
        return None


def _config(tmpdir: str) -> GraderConfig:
    return GraderConfig.from_mapping(
        {
            "models_root": tmpdir,
            "enabled_models": ["fake"],
            "models": {"fake": {"kind": "noop", "path": "fake.bin", "batch_size": 2}},
        }
    )


def _prepared(job: ImageJob, config: GraderConfig, preprocess_policy: str = "native") -> PreparedImage:
    image_id = job.image_path.stem
    return PreparedImage(
        request_id=job.request_id,
        image_path=job.image_path,
        image_id=image_id,
        size_bytes=1,
        mtime_ns=2,
        width=3,
        height=4,
        preprocess_policy=preprocess_policy,
        metadata=dict(job.metadata),
        image=None,
    )


class RunnerTests(unittest.TestCase):
    def test_runner_caches_scores_and_does_not_reemit_cached_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _config(tmpdir)
            backend = FakeBackend()
            runner = BatchRunner(config, state_dir=Path(tmpdir) / "state", registry=FakeRegistry(backend))
            output = Path(tmpdir) / "results.jsonl"
            jobs = [ImageJob(image_path=Path(tmpdir) / "one.png", request_id="req-1")]
            try:
                with patch("image_grader.runner.prepare_image", side_effect=_prepared):
                    first = runner.score_jobs(jobs, output_path=output, show_progress=False)
                    second = runner.score_jobs(jobs, output_path=output, show_progress=False)
            finally:
                runner.close()

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(first.computed_scores, 1)
        self.assertEqual(first.emitted, 1)
        self.assertEqual(second.cached_scores, 1)
        self.assertEqual(second.emitted, 0)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["scores"]["fake"]["cached"])

    def test_preprocess_policy_uses_separate_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _config(tmpdir)
            backend = FakeBackend()
            runner = BatchRunner(config, state_dir=Path(tmpdir) / "state", registry=FakeRegistry(backend))
            jobs = [ImageJob(image_path=Path(tmpdir) / "one.png", request_id="req-1")]
            try:
                with patch("image_grader.runner.prepare_image", side_effect=_prepared):
                    runner.score_jobs(jobs, show_progress=False, preprocess_policy="native")
                    runner.score_jobs(jobs, show_progress=False, preprocess_policy="fit_pad_square")
                    runner.score_jobs(jobs, show_progress=False, preprocess_policy="native")
            finally:
                runner.close()

        self.assertEqual(backend.calls, 2)

    def test_duplicate_image_ids_keep_scores_on_each_request_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _config(tmpdir)
            backend = FakeBackend()
            runner = BatchRunner(config, state_dir=Path(tmpdir) / "state", registry=FakeRegistry(backend))
            jobs = [
                ImageJob(image_path=Path(tmpdir) / "a" / "same.png", request_id="req-a"),
                ImageJob(image_path=Path(tmpdir) / "b" / "same.png", request_id="req-b"),
            ]
            try:
                with patch("image_grader.runner.prepare_image", side_effect=_prepared):
                    rows = runner.score_job_chunk(jobs, selected_models=("fake",))
            finally:
                runner.close()

        self.assertEqual([row["request_id"] for row in rows], ["req-a", "req-b"])
        self.assertEqual([row["scores"]["fake"]["score"] for row in rows], [1.0, 2.0])
        self.assertTrue(all(row["ok"] for row in rows))

    def test_square_preprocess_policies_transform_image_shape(self) -> None:
        from PIL import Image

        image = Image.new("RGB", (4, 2), color=(20, 30, 40))
        padded = preprocess_pil_image(image, "fit_pad_square")
        cropped = preprocess_pil_image(image, "center_crop_square")

        self.assertEqual(padded.size, (4, 4))
        self.assertEqual(cropped.size, (2, 2))


class ServerAppTests(unittest.TestCase):
    def test_rejects_paths_outside_allowed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GraderConfig.from_mapping(
                {
                    "server": {"allowed_roots": [tmpdir]},
                    "models": {"fake": {"kind": "noop", "path": "fake.bin"}},
                }
            )
            backend = FakeBackend()
            runner = BatchRunner(config, state_dir=Path(tmpdir) / "state", registry=FakeRegistry(backend))
            app = ImageGraderServerApp(config, runner)
            try:
                with self.assertRaises(GraderServerError):
                    app.score({"items": [{"image_path": "/outside.png"}]})
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
