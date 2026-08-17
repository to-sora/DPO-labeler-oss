from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generators.generate_prompt_lists_from_local_llm import (
    PONY_PREFIX,
    build_chat_messages,
    generate_prompt_lists_from_local_llm,
    load_family_prompt_lines,
    parse_family_response,
)


class _FakeResponse:
    def __init__(self, content: str | dict[str, object]) -> None:
        self._message = content if isinstance(content, dict) else {"content": content}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"choices": [{"message": self._message}]}


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def post(self, url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float) -> _FakeResponse:
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError("No more fake responses queued")
        return self._responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _write_family_examples(base_dir: Path) -> Path:
    example_dir = base_dir / "prompt_list_demo_v1"
    example_dir.mkdir(parents=True, exist_ok=True)
    (example_dir / "illustration.txt").write_text("scene one\nscene two\n", encoding="utf-8")
    (example_dir / "anime.txt").write_text("anime one\nanime two\n", encoding="utf-8")
    (example_dir / "pony.txt").write_text(
        f"{PONY_PREFIX}, scene one\n{PONY_PREFIX}, scene two\n",
        encoding="utf-8",
    )
    return example_dir


class GeneratePromptListsFromLocalLlmTests(unittest.TestCase):
    def test_load_family_prompt_lines_rejects_mismatched_line_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "illustration.txt").write_text("one\ntwo\n", encoding="utf-8")
            (base / "anime.txt").write_text("one\n", encoding="utf-8")
            (base / "pony.txt").write_text(f"{PONY_PREFIX}, one\n{PONY_PREFIX}, two\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must have the same usable line count"):
                load_family_prompt_lines(base)

    def test_build_chat_messages_embeds_examples_and_scene(self) -> None:
        prompt_lines = {
            "illustration": ["illustration alpha"],
            "anime": ["anime alpha"],
            "pony": [f"{PONY_PREFIX}, pony alpha"],
        }

        messages = build_chat_messages(scene="rainy city street with chara", prompt_lines=prompt_lines)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Example 1:", messages[1]["content"])
        self.assertIn("illustration: illustration alpha", messages[1]["content"])
        self.assertIn("anime: anime alpha", messages[1]["content"])
        self.assertIn("Rough scene: rainy city street with chara", messages[1]["content"])

    def test_parse_family_response_validates_required_shape(self) -> None:
        valid = parse_family_response(
            '{"illustration":"coherent illustration prompt","anime":"1girl, city night","pony":"score_9, score_8_up, score_7_up, score_6_up, 1girl, city night"}'
        )
        self.assertEqual(valid["illustration"], "coherent illustration prompt")

        with self.assertRaisesRegex(ValueError, "missing required key: anime"):
            parse_family_response(
                '{"illustration":"a","pony":"score_9, score_8_up, score_7_up, score_6_up, prompt"}'
            )

        with self.assertRaisesRegex(ValueError, "must be a single line"):
            parse_family_response(
                '{"illustration":"line1\\nline2","anime":"1girl","pony":"score_9, score_8_up, score_7_up, score_6_up, 1girl"}'
            )

        with self.assertRaisesRegex(ValueError, "Pony output must start"):
            parse_family_response(
                '{"illustration":"a","anime":"1girl","pony":"1girl, city night"}'
            )

    def test_parse_family_response_extracts_strict_json_from_reasoning_text(self) -> None:
        valid = parse_family_response(
            'Thinking Process: first draft {"illustration":"...","anime":"...","pony":"..."} '
            'Final answer: {"illustration":"rainy window illustration","anime":"2people, rainy window",'
            '"pony":"score_9, score_8_up, score_7_up, score_6_up, 2people, rainy window"}'
        )

        self.assertEqual(valid["illustration"], "rainy window illustration")
        self.assertEqual(valid["anime"], "2people, rainy window")

    def test_generate_prompt_lists_writes_outputs_and_regenerates_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            prompt_list_root = base / "prompt_lists"
            example_dir = _write_family_examples(prompt_list_root)
            output_dir = prompt_list_root / "prompt_list_demo_v1"
            workflow_output_dir = base / "workflows" / "prompt_list_demo"
            fake_session = _FakeSession(
                [
                    _FakeResponse(
                        '{"illustration":"illustration row one with chara","anime":"1girl, rainy city street","pony":"score_9, score_8_up, score_7_up, score_6_up, 1girl, rainy city street"}'
                    ),
                    _FakeResponse(
                        '{"illustration":"illustration row two with chara and charb","anime":"1boy and 1girl, cafe table","pony":"score_9, score_8_up, score_7_up, score_6_up, 1boy and 1girl, cafe table"}'
                    ),
                ]
            )
            regen_calls: list[dict[str, Path]] = []

            outputs = generate_prompt_lists_from_local_llm(
                model="local-model",
                scenes=["rough rainy scene", "rough cafe scene"],
                base_url="http://127.0.0.1:9999/v1",
                example_dir=example_dir,
                output_dir=output_dir,
                prompt_list_root=prompt_list_root,
                workflow_output_dir=workflow_output_dir,
                regenerate_workflow=True,
                api_session=fake_session,
                workflow_regenerator=lambda **kwargs: regen_calls.append(kwargs),
            )

            self.assertEqual(len(fake_session.calls), 2)
            self.assertEqual(
                fake_session.calls[0]["url"],
                "http://127.0.0.1:9999/v1/chat/completions",
            )
            request_json = fake_session.calls[0]["json"]
            self.assertEqual(request_json["model"], "local-model")
            self.assertEqual(request_json["response_format"], {"type": "json_object"})
            self.assertEqual(outputs["illustration"][0], "illustration row one with chara")
            self.assertEqual(
                (output_dir / "illustration.txt").read_text(encoding="utf-8"),
                "illustration row one with chara\nillustration row two with chara and charb\n",
            )
            self.assertEqual(
                (output_dir / "anime.txt").read_text(encoding="utf-8"),
                "1girl, rainy city street\n1boy and 1girl, cafe table\n",
            )
            self.assertEqual(
                (output_dir / "pony.txt").read_text(encoding="utf-8"),
                f"{PONY_PREFIX}, 1girl, rainy city street\n{PONY_PREFIX}, 1boy and 1girl, cafe table\n",
            )
            self.assertEqual(len(regen_calls), 1)
            self.assertEqual(regen_calls[0]["prompt_list_root"], prompt_list_root)
            self.assertEqual(regen_calls[0]["prompt_list_source_dir"], output_dir)
            self.assertEqual(regen_calls[0]["workflow_output_dir"], workflow_output_dir)

    def test_generate_prompt_lists_retries_after_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            prompt_list_root = base / "prompt_lists"
            example_dir = _write_family_examples(prompt_list_root)
            output_dir = prompt_list_root / "custom_pack"
            fake_session = _FakeSession(
                [
                    _FakeResponse("not json"),
                    _FakeResponse(
                        '{"illustration":"final illustration","anime":"1girl, rooftop at night","pony":"score_9, score_8_up, score_7_up, score_6_up, 1girl, rooftop at night"}'
                    ),
                ]
            )

            outputs = generate_prompt_lists_from_local_llm(
                model="retry-model",
                scenes=["neon rooftop"],
                example_dir=example_dir,
                output_dir=output_dir,
                prompt_list_root=prompt_list_root,
                regenerate_workflow=False,
                retries=2,
                api_session=fake_session,
            )

            self.assertEqual(len(fake_session.calls), 2)
            self.assertEqual(outputs["anime"], ["1girl, rooftop at night"])
            self.assertEqual((output_dir / "illustration.txt").read_text(encoding="utf-8"), "final illustration\n")

    def test_generate_prompt_lists_reads_reasoning_content_on_first_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            prompt_list_root = base / "prompt_lists"
            example_dir = _write_family_examples(prompt_list_root)
            output_dir = prompt_list_root / "custom_pack"
            fake_session = _FakeSession(
                [
                    _FakeResponse(
                        {
                            "content": "",
                            "reasoning_content": (
                                "Thinking Process: draft text. "
                                '{"illustration":"velvet bench by rainy glass",'
                                '"anime":"2people, velvet bench, rainy glass",'
                                '"pony":"score_9, score_8_up, score_7_up, score_6_up, 2people, velvet bench, rainy glass"}'
                            ),
                        }
                    ),
                ]
            )

            outputs = generate_prompt_lists_from_local_llm(
                model="reasoning-model",
                scenes=["rainy window bench"],
                example_dir=example_dir,
                output_dir=output_dir,
                prompt_list_root=prompt_list_root,
                regenerate_workflow=False,
                retries=1,
                api_session=fake_session,
            )

            self.assertEqual(len(fake_session.calls), 1)
            self.assertEqual(outputs["illustration"], ["velvet bench by rainy glass"])
            self.assertEqual(
                (output_dir / "pony.txt").read_text(encoding="utf-8"),
                f"{PONY_PREFIX}, 2people, velvet bench, rainy glass\n",
            )


if __name__ == "__main__":
    unittest.main()
