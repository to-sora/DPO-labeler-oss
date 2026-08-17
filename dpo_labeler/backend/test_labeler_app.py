from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

from PIL import Image

from dpo_labeler.backend.app import DpoLabelerApp, LabelEventValidationError
from dpo_labeler.backend.filters import FilterValidationError


def _write_png(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def _session_row(
    session_id: str,
    image_paths: tuple[Path, Path],
    *,
    session_index: int,
    task_name: str = "Portrait Pair",
    task_yaml_path: str = "tasks/portrait_pair.yaml",
    task_yaml_sha256: str = "task-sha-001",
    positive_prompt: str = "portrait of a traveler",
    negative_prompt: str = "low quality",
    ckpt: str = "sdxl/base.safetensors",
) -> dict[str, object]:
    images = []
    for image_index, image_path in enumerate(image_paths):
        images.append(
            {
                "image_index": image_index,
                "image_name": f"image_{image_index}",
                "saved_path": str(image_path),
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "ckpt": ckpt,
                "seed": 1000 + image_index,
                "status": "success",
                "workflow_name": "sdxl_ease_lora",
                "task_yaml_path": task_yaml_path,
                "prompt_seed": 2000 + image_index,
                "prompt_seed_control": "fixed",
                "generation_seed_control": "fixed",
                "width": 1024,
                "height": 1024,
                "cfg": 7.0,
                "steps": 30,
                "runtime_seed_values": {"generation_seed": 3000 + image_index},
                "lora_stack_config": [],
                "runner_result": {"status": "success"},
                "saved_filename": image_path.name,
                "original_filename": f"comfy_{image_index}.png",
            }
        )
    return {
        "session_id": session_id,
        "session_index": session_index,
        "task_name": task_name,
        "task_yaml_path": task_yaml_path,
        "task_yaml_sha256": task_yaml_sha256,
        "compiler_version": "test-compiler",
        "global_seed": 4242,
        "images": images,
    }


class DpoLabelerAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_root = self.root / "datasets"
        self.state_dir = self.root / "state"
        self.prompt_bold_path = self.root / "bold.txt"
        self.sessions_path = self.dataset_root / "alpha" / "collected" / "sessions.jsonl"
        self.second_sessions_path = self.dataset_root / "beta" / "collected" / "sessions.jsonl"
        self.prompt_bold_path.write_text(
            "\n".join(
                [
                    r"(?:turquoise|lavender|charcoal|burgundy|magenta|violet|purple|silver|yellow|orange|indigo|scarlet|crimson|coral|bronze|salmon|azure|green|black|white|brown|olive|cream|peach|amber|teal|blue|pink|gold|gray|grey|beige|maroon|navy|cyan|mint|lime|plum|aqua|ivory|red|tan)",
                    r"(?:non[-_\s]?binary|genderfluid|genderqueer|transgender|androgyn(?:ous|e)?|demiboy|demboy|demigirl|femboy|tomboy|agender|bigender|female|femla|woman|girl|male|man|boy|trans|enby)",
                    r"\d+(?:\.\d+)?",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_catalog_quarantines_invalid_sessions(self) -> None:
        valid_a = self.root / "images" / "valid_a.png"
        valid_b = self.root / "images" / "valid_b.png"
        _write_png(valid_a, "red")
        _write_png(valid_b, "blue")

        valid_row = _session_row("session-valid", (valid_a, valid_b), session_index=0)
        invalid_row = _session_row(
            "session-invalid",
            (valid_a, valid_b),
            session_index=1,
        )
        invalid_row["images"][1]["saved_path"] = "missing/not-found.png"  # type: ignore[index]

        self._write_sessions(self.sessions_path, [valid_row, invalid_row])
        app = self._create_app()

        catalog = app.get_catalog()
        self.assertEqual(catalog["catalog_version"], app.catalog_service.get_snapshot().catalog_version)
        self.assertEqual(len(catalog["warnings"]), 1)
        self.assertEqual(catalog["warnings"][0]["reason"], "missing_image_path")
        task = catalog["datasets"][0]["tasks"][0]
        self.assertEqual(task["total_pairs"], 1)
        self.assertEqual(task["invalid_pair_count"], 1)
        self.assertEqual(task["unlabeled_pairs"], 1)

    def test_session_review_and_dpo_export(self) -> None:
        image_a = self.root / "images" / "pair_a.png"
        image_b = self.root / "images" / "pair_b.png"
        _write_png(image_a, "green")
        _write_png(image_b, "yellow")

        self._write_sessions(self.sessions_path, [_session_row("session-001", (image_a, image_b), session_index=0)])
        app = self._create_app()
        session_payload, cookie_header = app.start_session("invite-123", "reviewer.one", "client-000001")
        self.assertEqual(session_payload["session"]["reviewer_username"], "reviewer.one")

        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        task_key = catalog["datasets"][0]["tasks"][0]["task_key"]
        review_payload = app.create_review_session(
            cookie_header,
            {"task_keys": [task_key, task_key], "mode": "sequence"},
        )
        self.assertEqual(review_payload["review"]["task_keys"], [task_key])
        self.assertEqual(review_payload["queue_total"], 1)

        pair_payload = app.get_review_pair(
            cookie_header,
            review_payload["review"]["review_id"],
            dataset_id,
            "session-001",
        )
        self.assertEqual(pair_payload["pair_key"], f"{dataset_id}::session-001")
        self.assertIsNone(pair_payload["latest_label"])
        self.assertEqual(pair_payload["images"][0]["positive_prompt_segments"], [{"text": "portrait of a traveler", "bold": False}])

        event = app.submit_label_event(
            cookie_header,
            {
                "event_id": "event-001",
                "review_id": review_payload["review"]["review_id"],
                "dataset_id": dataset_id,
                "session_id": "session-001",
                "decision": "a_good",
                "display_order": [1, 0],
                "chosen_image_indices": [1],
                "defects_a": ["eyes_off"],
                "defects_b": [],
                "defects_by_image_index": {
                    "1": ["eyes_off"],
                    "0": [],
                },
                "note": "better anatomy",
                "reviewer_username": "spoofed.user",
            },
        )
        self.assertEqual(event["reviewer_username"], "reviewer.one")
        self.assertEqual(event["client_instance_id"], "client-000001")
        self.assertEqual(event["chosen_image_indices"], [1])

        with self.assertRaises(KeyError):
            app.get_review_pair(
                cookie_header,
                review_payload["review"]["review_id"],
                dataset_id,
                "session-001",
            )

        preview = app.preview_export(
            cookie_header,
            {
                "export_type": "dpo-pairs",
                "filter": {
                    "type": "group",
                    "operator": "and",
                    "conditions": [
                        {
                            "type": "rule",
                            "field": "reviewer_username",
                            "operator": "eq",
                            "value": "reviewer.one",
                        }
                    ],
                },
            },
        )
        self.assertEqual(preview["count"], 1)

        export_text = app.export_text(
            "dpo-pairs",
            {
                "type": "group",
                "operator": "and",
                "conditions": [
                    {
                        "type": "rule",
                        "field": "decision",
                        "operator": "eq",
                        "value": "a_good",
                    }
                ],
            },
        )
        rows = [json.loads(line) for line in export_text.splitlines() if line.strip()]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["strict_dpo"])
        self.assertEqual(rows[0]["reviewer_username"], "reviewer.one")
        self.assertEqual(rows[0]["chosen_index"], 1)
        self.assertEqual(rows[0]["chosen_image"]["saved_path"], str(image_b))
        self.assertEqual(rows[0]["rejected_image"]["saved_path"], str(image_a))

        with self.assertRaises(LabelEventValidationError):
            app.export_text("unsupported-export", None)

    def test_both_good_is_stored_but_excluded_from_dpo_export(self) -> None:
        image_a = self.root / "images" / "both_a.png"
        image_b = self.root / "images" / "both_b.png"
        _write_png(image_a, "navy")
        _write_png(image_b, "silver")

        self._write_sessions(self.sessions_path, [_session_row("session-both", (image_a, image_b), session_index=0)])
        app = self._create_app()
        _, cookie_header = app.start_session("invite-123", "reviewer.two", "client-000002")

        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        task_key = catalog["datasets"][0]["tasks"][0]["task_key"]
        review_payload = app.create_review_session(
            cookie_header,
            {"task_keys": [task_key], "mode": "sequence"},
        )

        event = app.submit_label_event(
            cookie_header,
            {
                "event_id": "event-both-001",
                "review_id": review_payload["review"]["review_id"],
                "dataset_id": dataset_id,
                "session_id": "session-both",
                "decision": "both_good",
                "display_order": [1, 0],
                "chosen_image_indices": [0, 1],
                "defects_a": [],
                "defects_b": ["bad_composition"],
                "defects_by_image_index": {
                    "1": [],
                    "0": ["bad_composition"],
                },
                "note": "both usable",
            },
        )
        self.assertEqual(event["decision"], "both_good")
        self.assertEqual(event["chosen_image_indices"], [0, 1])

        latest_rows = [
            json.loads(line)
            for line in app.export_text("labels-latest", {"type": "group", "operator": "and", "conditions": []}).splitlines()
            if line.strip()
        ]
        self.assertEqual(len(latest_rows), 1)
        self.assertEqual(latest_rows[0]["label"]["decision"], "both_good")
        self.assertEqual(latest_rows[0]["label"]["chosen_image_indices"], [0, 1])

        dpo_rows = [
            json.loads(line)
            for line in app.export_text("dpo-pairs", {"type": "group", "operator": "and", "conditions": []}).splitlines()
            if line.strip()
        ]
        self.assertEqual(dpo_rows, [])

    def test_create_twitter_share_returns_public_share_page_and_preview(self) -> None:
        image_a = self.root / "images" / "share_a.png"
        image_b = self.root / "images" / "share_b.png"
        _write_png(image_a, "orange")
        _write_png(image_b, "purple")

        self._write_sessions(
            self.sessions_path,
            [_session_row("session-share", (image_a, image_b), session_index=0, positive_prompt="blue jacket portrait")],
        )
        app = self._create_app()
        _, cookie_header = app.start_session("invite-123", "reviewer.share", "client-000999")

        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        payload = app.create_twitter_share(
            cookie_header,
            {
                "dataset_id": dataset_id,
                "session_id": "session-share",
                "image_index": 0,
            },
            "https://labeler.example",
        )
        self.assertEqual(payload["text"], "#AIGenerated\n\nblue jacket portrait")
        self.assertTrue(payload["share_url"].startswith("https://labeler.example/public-share/"))

        parsed = urlparse(payload["twitter_intent_url"])
        self.assertEqual(parsed.netloc, "twitter.com")
        query = parse_qs(parsed.query)
        self.assertEqual(query["text"][0], "#AIGenerated\n\nblue jacket portrait")
        self.assertEqual(query["url"][0], payload["share_url"])

        share_token = payload["share_url"].rsplit("/", 1)[-1]
        html = app.render_public_share_page(share_token, "https://labeler.example")
        self.assertIn("summary_large_image", html)
        self.assertIn("https://labeler.example/public-share/media/", html)
        self.assertIn("blue jacket portrait", html)

        preview_path = app.get_public_share_preview_path(share_token)
        self.assertTrue(preview_path.exists())
        self.assertEqual(preview_path.suffix.lower(), ".jpg")

    def test_post_twitter_image_uses_configured_x_poster(self) -> None:
        image_a = self.root / "images" / "post_a.png"
        image_b = self.root / "images" / "post_b.png"
        _write_png(image_a, "teal")
        _write_png(image_b, "maroon")

        self._write_sessions(
            self.sessions_path,
            [_session_row("session-post", (image_a, image_b), session_index=0, positive_prompt="green dress portrait")],
        )
        app = self._create_app()
        app.x_poster = mock.Mock()
        app.x_poster.is_configured = True
        app.x_poster.post_image_tweet.return_value = {
            "tweet_id": "12345",
            "tweet_url": "https://x.com/test_account/status/12345",
            "screen_name": "test_account",
        }
        _, cookie_header = app.start_session("invite-123", "reviewer.poster", "client-001111")

        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        posted = app.post_twitter_image(
            cookie_header,
            {
                "dataset_id": dataset_id,
                "session_id": "session-post",
                "image_index": 1,
            },
        )

        self.assertEqual(posted["tweet_id"], "12345")
        self.assertEqual(posted["tweet_url"], "https://x.com/test_account/status/12345")
        self.assertEqual(posted["screen_name"], "test_account")
        self.assertEqual(posted["text"], "#AIGenerated\n\ngreen dress portrait")
        app.x_poster.post_image_tweet.assert_called_once()

    def test_review_queue_excludes_persisted_labels_and_skips_after_restart(self) -> None:
        first_a = self.root / "images" / "first_a.png"
        first_b = self.root / "images" / "first_b.png"
        second_a = self.root / "images" / "second_a.png"
        second_b = self.root / "images" / "second_b.png"
        third_a = self.root / "images" / "third_a.png"
        third_b = self.root / "images" / "third_b.png"
        _write_png(first_a, "red")
        _write_png(first_b, "blue")
        _write_png(second_a, "green")
        _write_png(second_b, "yellow")
        _write_png(third_a, "white")
        _write_png(third_b, "black")

        self._write_sessions(
            self.sessions_path,
            [
                _session_row("session-001", (first_a, first_b), session_index=0),
                _session_row("session-002", (second_a, second_b), session_index=1),
                _session_row("session-003", (third_a, third_b), session_index=2),
            ],
        )
        app = self._create_app()
        _, cookie_header = app.start_session("invite-123", "reviewer.three", "client-000003")
        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        task = catalog["datasets"][0]["tasks"][0]
        review_payload = app.create_review_session(cookie_header, {"task_keys": [task["task_key"]], "mode": "sequence"})

        app.submit_label_event(
            cookie_header,
            {
                "event_id": "event-keep-out-001",
                "review_id": review_payload["review"]["review_id"],
                "dataset_id": dataset_id,
                "session_id": "session-001",
                "decision": "a_good",
                "display_order": [0, 1],
                "chosen_image_indices": [0],
                "defects_a": [],
                "defects_b": [],
                "defects_by_image_index": {"0": [], "1": []},
                "note": "",
            },
        )
        app.submit_label_event(
            cookie_header,
            {
                "event_id": "event-keep-out-002",
                "review_id": review_payload["review"]["review_id"],
                "dataset_id": dataset_id,
                "session_id": "session-002",
                "decision": "skip",
                "display_order": [0, 1],
                "chosen_image_indices": [],
                "defects_a": [],
                "defects_b": [],
                "defects_by_image_index": {"0": [], "1": []},
                "note": "",
            },
        )

        restarted = self._create_app()
        restarted_catalog = restarted.get_catalog()
        restarted_task = restarted_catalog["datasets"][0]["tasks"][0]
        self.assertEqual(restarted_task["reviewed_pairs"], 2)
        self.assertEqual(restarted_task["remaining_pairs"], 1)
        self.assertEqual(restarted_task["reviewed_percent"], 67)

        review_after_restart = restarted.create_review_session(
            cookie_header,
            {"task_keys": [restarted_task["task_key"]], "mode": "sequence"},
        )
        self.assertEqual(review_after_restart["queue_total"], 1)
        queue = restarted.get_review_queue(cookie_header, review_after_restart["review"]["review_id"], 0, 10)
        self.assertEqual([item["session_id"] for item in queue["items"]], ["session-003"])

    def test_label_store_ignores_truncated_final_line_after_restart(self) -> None:
        image_a = self.root / "images" / "truncated_a.png"
        image_b = self.root / "images" / "truncated_b.png"
        _write_png(image_a, "red")
        _write_png(image_b, "blue")
        self._write_sessions(self.sessions_path, [_session_row("session-001", (image_a, image_b), session_index=0)])

        app = self._create_app()
        _, cookie_header = app.start_session("invite-123", "reviewer.four", "client-000004")
        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        task_key = catalog["datasets"][0]["tasks"][0]["task_key"]
        review_payload = app.create_review_session(cookie_header, {"task_keys": [task_key], "mode": "sequence"})
        app.submit_label_event(
            cookie_header,
            {
                "event_id": "event-truncated-001",
                "review_id": review_payload["review"]["review_id"],
                "dataset_id": dataset_id,
                "session_id": "session-001",
                "decision": "a_good",
                "display_order": [0, 1],
                "chosen_image_indices": [0],
                "defects_a": [],
                "defects_b": [],
                "defects_by_image_index": {"0": [], "1": []},
                "note": "",
            },
        )

        with (self.state_dir / "label_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"event_id": "partial"')

        restarted = self._create_app()
        restarted_catalog = restarted.get_catalog()
        task = restarted_catalog["datasets"][0]["tasks"][0]
        self.assertEqual(task["reviewed_pairs"], 1)
        self.assertEqual(task["remaining_pairs"], 0)

    def test_invalid_nonfinal_label_line_still_fails_startup(self) -> None:
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "label_events.jsonl").write_text(
            "\n".join(
                [
                    '{"event_id": "broken"',
                    '{"event_id": "event-002", "dataset_id": "alpha-collected", "session_id": "session-001", "decision": "skip"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            self._create_app()

    def test_prompt_segments_follow_bold_rules(self) -> None:
        image_a = self.root / "images" / "segment_a.png"
        image_b = self.root / "images" / "segment_b.png"
        _write_png(image_a, "red")
        _write_png(image_b, "blue")
        self._write_sessions(
            self.sessions_path,
            [
                _session_row(
                    "session-segments",
                    (image_a, image_b),
                    session_index=0,
                    positive_prompt="female traveler in blue coat 123",
                )
            ],
        )

        app = self._create_app()
        _, cookie_header = app.start_session("invite-123", "reviewer.five", "client-000005")
        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        task_key = catalog["datasets"][0]["tasks"][0]["task_key"]
        review_payload = app.create_review_session(cookie_header, {"task_keys": [task_key], "mode": "sequence"})
        pair_payload = app.get_review_pair(cookie_header, review_payload["review"]["review_id"], dataset_id, "session-segments")
        segments = pair_payload["images"][0]["positive_prompt_segments"]
        self.assertEqual("".join(segment["text"] for segment in segments), "female traveler in blue coat 123")
        self.assertEqual(
            [segment["text"] for segment in segments if segment["bold"]],
            ["female", "blue", "123"],
        )

    def test_prompt_segments_match_case_insensitive_substrings(self) -> None:
        image_a = self.root / "images" / "substring_a.png"
        image_b = self.root / "images" / "substring_b.png"
        _write_png(image_a, "red")
        _write_png(image_b, "blue")
        prompt = "1girl with BlueHair, silverdress, DEMBOY, non-binary look, CFG7.5 and 35mm"
        self._write_sessions(
            self.sessions_path,
            [
                _session_row(
                    "session-substrings",
                    (image_a, image_b),
                    session_index=0,
                    positive_prompt=prompt,
                )
            ],
        )

        app = self._create_app()
        _, cookie_header = app.start_session("invite-123", "reviewer.six", "client-000006")
        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        task_key = catalog["datasets"][0]["tasks"][0]["task_key"]
        review_payload = app.create_review_session(cookie_header, {"task_keys": [task_key], "mode": "sequence"})
        pair_payload = app.get_review_pair(cookie_header, review_payload["review"]["review_id"], dataset_id, "session-substrings")
        segments = pair_payload["images"][0]["positive_prompt_segments"]

        self.assertEqual("".join(segment["text"] for segment in segments), prompt)
        self.assertEqual(
            [segment["text"] for segment in segments if segment["bold"]],
            ["1girl", "Blue", "silver", "DEMBOY", "non-binary", "7.5", "35"],
        )

    def test_default_prompt_bold_path_uses_package_file_when_cwd_differs(self) -> None:
        image_a = self.root / "images" / "default_path_a.png"
        image_b = self.root / "images" / "default_path_b.png"
        _write_png(image_a, "red")
        _write_png(image_b, "blue")
        prompt = (
            "masterpiece, best quality, anime illustration, 1girl, adult woman, upper body focus, "
            "composed expression, office styling, Preppy Perfection: A light blue collared shirt layered "
            "under a cable-knit sweater, paired with a plaid flared skirt. Complete the preppy look with "
            "knee-high socks and loafers., window light at night, from below, direct flash photography, Refined"
        )
        self._write_sessions(
            self.sessions_path,
            [
                _session_row(
                    "session-default-bold-path",
                    (image_a, image_b),
                    session_index=0,
                    positive_prompt=prompt,
                )
            ],
        )

        launch_dir = self.root / "launch-dir"
        launch_dir.mkdir(parents=True, exist_ok=True)
        stdout = io.StringIO()
        with mock.patch("dpo_labeler.backend.app.Path.cwd", return_value=launch_dir):
            with contextlib.redirect_stdout(stdout):
                app = DpoLabelerApp(
                    dataset_root=self.dataset_root,
                    state_dir=self.state_dir,
                    invite_token="invite-123",
                    session_secret="secret-123",
                    rescan_seconds=3600,
                )
        self.addCleanup(app.close)

        self.assertIn("Loaded prompt bold config from", stdout.getvalue())
        _, cookie_header = app.start_session("invite-123", "reviewer.seven", "client-000007")
        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        task_key = catalog["datasets"][0]["tasks"][0]["task_key"]
        review_payload = app.create_review_session(cookie_header, {"task_keys": [task_key], "mode": "sequence"})
        pair_payload = app.get_review_pair(
            cookie_header,
            review_payload["review"]["review_id"],
            dataset_id,
            "session-default-bold-path",
        )
        segments = pair_payload["images"][0]["positive_prompt_segments"]
        bold_text = [segment["text"] for segment in segments if segment["bold"]]

        self.assertEqual("".join(segment["text"] for segment in segments), prompt)
        self.assertIn("1girl", bold_text)
        self.assertIn("woman", bold_text)
        self.assertIn("blue", bold_text)

    def test_missing_prompt_bold_path_warns_and_disables_bolding(self) -> None:
        image_a = self.root / "images" / "missing_bold_a.png"
        image_b = self.root / "images" / "missing_bold_b.png"
        _write_png(image_a, "red")
        _write_png(image_b, "blue")
        prompt = "1girl in blue with woman styling"
        self._write_sessions(
            self.sessions_path,
            [
                _session_row(
                    "session-missing-bold",
                    (image_a, image_b),
                    session_index=0,
                    positive_prompt=prompt,
                )
            ],
        )

        missing_bold_path = self.root / "missing-bold.txt"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            app = self._create_app(prompt_bold_path=missing_bold_path)

        self.assertIn("Warning: prompt bold config not found", stdout.getvalue())
        _, cookie_header = app.start_session("invite-123", "reviewer.eight", "client-000008")
        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        task_key = catalog["datasets"][0]["tasks"][0]["task_key"]
        review_payload = app.create_review_session(cookie_header, {"task_keys": [task_key], "mode": "sequence"})
        pair_payload = app.get_review_pair(cookie_header, review_payload["review"]["review_id"], dataset_id, "session-missing-bold")
        self.assertEqual(
            pair_payload["images"][0]["positive_prompt_segments"],
            [{"text": prompt, "bold": False}],
        )

    def test_invalid_bold_regex_fails_startup(self) -> None:
        invalid_bold_path = self.root / "invalid-bold.txt"
        invalid_bold_path.write_text("(\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self._create_app(prompt_bold_path=invalid_bold_path)

    def test_force_rescan_picks_up_new_sessions(self) -> None:
        first_a = self.root / "images" / "first_a.png"
        first_b = self.root / "images" / "first_b.png"
        second_a = self.root / "images" / "second_a.png"
        second_b = self.root / "images" / "second_b.png"
        _write_png(first_a, "purple")
        _write_png(first_b, "orange")
        _write_png(second_a, "white")
        _write_png(second_b, "black")

        first_row = _session_row("session-001", (first_a, first_b), session_index=0)
        self._write_sessions(self.sessions_path, [first_row])
        app = self._create_app()
        initial_catalog = app.get_catalog()
        self.assertEqual(initial_catalog["datasets"][0]["tasks"][0]["total_pairs"], 1)

        second_row = _session_row("session-002", (second_a, second_b), session_index=1)
        self._write_sessions(self.sessions_path, [first_row, second_row])
        app.catalog_service.force_rescan()
        updated_catalog = app.get_catalog()
        self.assertEqual(updated_catalog["datasets"][0]["tasks"][0]["total_pairs"], 2)
        self.assertNotEqual(initial_catalog["catalog_version"], updated_catalog["catalog_version"])

    def test_catalog_payload_cache_reuses_result_until_label_changes(self) -> None:
        image_a = self.root / "images" / "cache_catalog_a.png"
        image_b = self.root / "images" / "cache_catalog_b.png"
        _write_png(image_a, "red")
        _write_png(image_b, "blue")

        self._write_sessions(self.sessions_path, [_session_row("session-cache", (image_a, image_b), session_index=0)])
        app = self._create_app()
        with mock.patch.object(app, "_build_catalog_payload", wraps=app._build_catalog_payload) as build_catalog:
            first_catalog = app.get_catalog()
            second_catalog = app.get_catalog()

            self.assertEqual(build_catalog.call_count, 1)
            self.assertEqual(first_catalog["catalog_version"], second_catalog["catalog_version"])

            _, cookie_header = app.start_session("invite-123", "reviewer.cache", "client-cache")
            dataset_id = first_catalog["datasets"][0]["dataset_id"]
            task_key = first_catalog["datasets"][0]["tasks"][0]["task_key"]
            review_payload = app.create_review_session(cookie_header, {"task_keys": [task_key], "mode": "sequence"})
            app.submit_label_event(
                cookie_header,
                {
                    "event_id": "event-cache-001",
                    "review_id": review_payload["review"]["review_id"],
                    "dataset_id": dataset_id,
                    "session_id": "session-cache",
                    "decision": "a_good",
                    "display_order": [0, 1],
                    "chosen_image_indices": [0],
                    "defects_a": [],
                    "defects_b": [],
                    "defects_by_image_index": {"0": [], "1": []},
                    "note": "",
                },
            )

            updated_catalog = app.get_catalog()
            self.assertEqual(build_catalog.call_count, 2)
            self.assertEqual(updated_catalog["datasets"][0]["tasks"][0]["reviewed_pairs"], 1)

    def test_review_queue_cache_reuses_result_until_label_changes(self) -> None:
        first_a = self.root / "images" / "queue_cache_a1.png"
        first_b = self.root / "images" / "queue_cache_b1.png"
        second_a = self.root / "images" / "queue_cache_a2.png"
        second_b = self.root / "images" / "queue_cache_b2.png"
        _write_png(first_a, "green")
        _write_png(first_b, "yellow")
        _write_png(second_a, "white")
        _write_png(second_b, "black")

        self._write_sessions(
            self.sessions_path,
            [
                _session_row("session-queue-001", (first_a, first_b), session_index=0),
                _session_row("session-queue-002", (second_a, second_b), session_index=1),
            ],
        )
        app = self._create_app()
        _, cookie_header = app.start_session("invite-123", "reviewer.queue", "client-queue")
        catalog = app.get_catalog()
        dataset_id = catalog["datasets"][0]["dataset_id"]
        task_key = catalog["datasets"][0]["tasks"][0]["task_key"]

        with mock.patch.object(app.review_service, "build_queue", wraps=app.review_service.build_queue) as build_queue:
            review_payload = app.create_review_session(cookie_header, {"task_keys": [task_key], "mode": "sequence"})
            self.assertEqual(build_queue.call_count, 1)

            first_page = app.get_review_queue(cookie_header, review_payload["review"]["review_id"], 0, 10)
            second_page = app.get_review_queue(cookie_header, review_payload["review"]["review_id"], 0, 10)
            self.assertEqual(build_queue.call_count, 1)
            self.assertEqual(len(first_page["items"]), 2)
            self.assertEqual(len(second_page["items"]), 2)

            app.submit_label_event(
                cookie_header,
                {
                    "event_id": "event-queue-cache-001",
                    "review_id": review_payload["review"]["review_id"],
                    "dataset_id": dataset_id,
                    "session_id": "session-queue-001",
                    "decision": "skip",
                    "display_order": [0, 1],
                    "chosen_image_indices": [],
                    "defects_a": [],
                    "defects_b": [],
                    "defects_by_image_index": {"0": [], "1": []},
                    "note": "",
                },
            )

            updated_page = app.get_review_queue(cookie_header, review_payload["review"]["review_id"], 0, 10)
            self.assertEqual(build_queue.call_count, 2)
            self.assertEqual([item["session_id"] for item in updated_page["items"]], ["session-queue-002"])

    def test_get_snapshot_does_not_rescan_inline(self) -> None:
        image_a = self.root / "images" / "snapshot_inline_a.png"
        image_b = self.root / "images" / "snapshot_inline_b.png"
        _write_png(image_a, "purple")
        _write_png(image_b, "orange")
        self._write_sessions(self.sessions_path, [_session_row("session-inline", (image_a, image_b), session_index=0)])

        app = self._create_app()
        initial_version = app.catalog_service.get_snapshot().catalog_version
        with mock.patch.object(app.catalog_service, "_scan_root", side_effect=AssertionError("inline rescan")):
            self.assertEqual(app.catalog_service.get_snapshot().catalog_version, initial_version)
            self.assertEqual(app.catalog_service.get_snapshot().catalog_version, initial_version)

    def test_recursive_scan_discovers_multiple_sessions_files(self) -> None:
        alpha_a = self.root / "images" / "alpha_a.png"
        alpha_b = self.root / "images" / "alpha_b.png"
        beta_a = self.root / "images" / "beta_a.png"
        beta_b = self.root / "images" / "beta_b.png"
        _write_png(alpha_a, "red")
        _write_png(alpha_b, "blue")
        _write_png(beta_a, "green")
        _write_png(beta_b, "yellow")

        self._write_sessions(self.sessions_path, [_session_row("session-alpha", (alpha_a, alpha_b), session_index=0)])
        self._write_sessions(self.second_sessions_path, [_session_row("session-beta", (beta_a, beta_b), session_index=0)])

        app = self._create_app()
        catalog = app.get_catalog()

        self.assertEqual(len(catalog["datasets"]), 2)
        display_names = [dataset["display_name"] for dataset in catalog["datasets"]]
        self.assertIn("alpha/collected", display_names)
        self.assertIn("beta/collected", display_names)

    def test_filter_rejects_inverted_datetime_range(self) -> None:
        app = self._create_app()
        with self.assertRaises(FilterValidationError):
            app.filter_engine.validate(
                {
                    "type": "group",
                    "operator": "and",
                    "conditions": [
                        {
                            "type": "rule",
                            "field": "label_created_at",
                            "operator": "between",
                            "value": {
                                "start": "2026-03-29T00:00:00Z",
                                "end": "2026-03-28T00:00:00Z",
                            },
                        }
                    ],
                }
            )

    def _create_app(self, prompt_bold_path: Path | None = None) -> DpoLabelerApp:
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        app = DpoLabelerApp(
            dataset_root=self.dataset_root,
            state_dir=self.state_dir,
            invite_token="invite-123",
            session_secret="secret-123",
            rescan_seconds=30,
            review_round_seed="round-seed-1",
            prompt_bold_path=prompt_bold_path or self.prompt_bold_path,
        )
        self.addCleanup(app.close)
        return app

    def _write_sessions(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
