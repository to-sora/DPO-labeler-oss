from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from image_grader.config import ConfigError, GraderConfig
from image_grader.io import iter_image_json
from image_grader.state import ScoreState


class ConfigTests(unittest.TestCase):
    def test_loads_minimal_config(self) -> None:
        config = GraderConfig.from_mapping(
            {
                "models": {
                    "fake": {
                        "kind": "noop",
                        "path": "fake.bin",
                        "batch_size": 3,
                    }
                }
            }
        )

        self.assertEqual(config.enabled_models, ("fake",))
        self.assertEqual(config.models["fake"].batch_size, 3)
        self.assertEqual(config.cache.fingerprint, "sample_sha256")

    def test_rejects_unknown_enabled_model(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown model"):
            GraderConfig.from_mapping(
                {
                    "enabled_models": ["missing"],
                    "models": {"fake": {"kind": "noop"}},
                }
            )


class JsonInputTests(unittest.TestCase):
    def test_json_array_accepts_objects_and_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.json"
            path.write_text(
                json.dumps(
                    [
                        {"request_id": "a", "image_path": "one.png", "metadata": {"rank": 1}},
                        "two.jpg",
                    ]
                ),
                encoding="utf-8",
            )

            jobs = list(iter_image_json(path))

        self.assertEqual(jobs[0].request_id, "a")
        self.assertEqual(jobs[0].metadata, {"rank": 1})
        self.assertEqual(jobs[0].image_path.name, "one.png")
        self.assertEqual(jobs[1].image_path.name, "two.jpg")

    def test_jsonl_reads_each_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"request_id": "a", "image_path": "one.png"}),
                        json.dumps({"request_id": "b", "path": "two.png"}),
                    ]
                ),
                encoding="utf-8",
            )

            jobs = list(iter_image_json(path))

        self.assertEqual([job.request_id for job in jobs], ["a", "b"])
        self.assertEqual([job.image_path.name for job in jobs], ["one.png", "two.png"])


class ScoreStateTests(unittest.TestCase):
    def test_put_and_get_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ScoreState(tmpdir)
            try:
                state.put_score(
                    image_id="img",
                    model_id="model",
                    config_hash="hash",
                    image_path="/tmp/a.png",
                    size_bytes=10,
                    width=1,
                    height=2,
                    score={"ok": True, "score": 8.5, "scale": "0_10", "raw": {"x": 1}, "error": None},
                )
                cached = state.get_score("img", "model", "hash")
            finally:
                state.close()

        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertTrue(cached.ok)
        self.assertEqual(cached.score, 8.5)
        self.assertEqual(cached.raw, {"x": 1})


if __name__ == "__main__":
    unittest.main()
