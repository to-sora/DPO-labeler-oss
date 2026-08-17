from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from compile_yaml_to_requests_jsonl import compile_requests
from prompt_generator import WildcardTemplatePromptGenerator, get_prompt_generator, main, preview_wildcard_template_samples
from tests._paths import REPO_ROOT


class WildcardTemplatePromptGeneratorTests(unittest.TestCase):
    def _selection_tuples(self, bundle: object) -> list[tuple[str, int, str]]:
        selection_usage = bundle.prompt_metadata["wildcard_selection_usage"]
        return [
            (str(row["resolved_token"]), int(row["occurrence_index"]), str(row["selected_value"]))
            for row in selection_usage
        ]

    def test_registry_returns_wildcard_generator(self) -> None:
        generator = get_prompt_generator("wildcard_template_generator")
        self.assertIsInstance(generator, WildcardTemplatePromptGenerator)

    def test_expands_recursive_wildcards_from_template_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            (wildcard_root / "class").mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("masterpiece, __character__, __style__\n", encoding="utf-8")
            (wildcard_root / "character.txt").write_text("# comment\n__class/knight__\n", encoding="utf-8")
            (wildcard_root / "class" / "knight.txt").write_text("Altria Saber\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("\nclean anime\n", encoding="utf-8")
            (wildcard_root / "negative").mkdir(parents=True, exist_ok=True)
            (wildcard_root / "negative" / "common.txt").write_text("bad hands\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "portrait",
                    "template_root": str(template_root),
                    "negative_template": "__negative/common__",
                    "wildcard_root": str(wildcard_root),
                },
                seed=123,
            )

            self.assertEqual(bundle.positive_prompt, "masterpiece, Altria Saber, clean anime")
            self.assertEqual(bundle.negative_prompt, "bad hands")
            self.assertEqual(bundle.prompt_metadata["generator_version"], "v2")
            self.assertEqual(bundle.prompt_metadata["template_identifier"], "portrait")
            self.assertEqual(bundle.prompt_metadata["template_path"], str(template_root / "portrait.txt"))
            self.assertEqual(
                bundle.prompt_metadata["wildcard_usage"],
                ["character", "style", "class/knight", "negative/common"],
            )

    def test_repeated_same_wildcard_uses_stable_per_occurrence_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "repeat.txt").write_text("__tone__, __tone__, __tone__\n", encoding="utf-8")
            (wildcard_root / "tone.txt").write_text("red\nblue\ngreen\nyellow\n", encoding="utf-8")

            bundle_a = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "repeat",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=42,
            )
            bundle_b = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "repeat",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=42,
            )

            self.assertEqual(bundle_a.positive_prompt, bundle_b.positive_prompt)
            self.assertEqual(
                self._selection_tuples(bundle_a),
                [
                    ("tone", 0, self._selection_tuples(bundle_a)[0][2]),
                    ("tone", 1, self._selection_tuples(bundle_a)[1][2]),
                    ("tone", 2, self._selection_tuples(bundle_a)[2][2]),
                ],
            )
            self.assertEqual(self._selection_tuples(bundle_a), self._selection_tuples(bundle_b))

    def test_same_shared_wildcard_occurrences_align_across_family_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "illustration.txt").write_text(
                "masterpiece, __rating__, __mood__, __mood__, BREAK, __scene__\n",
                encoding="utf-8",
            )
            (template_root / "pony.txt").write_text(
                "score_9, __rating__, __mood__, __mood__, __scene__\n",
                encoding="utf-8",
            )
            (wildcard_root / "rating.txt").write_text("safe\nquestionable\nexplicit\n", encoding="utf-8")
            (wildcard_root / "mood.txt").write_text("calm\nconfident\nshy\nteasing\n", encoding="utf-8")
            (wildcard_root / "scene.txt").write_text("bedroom\nclassroom\nrooftop\nstudio\n", encoding="utf-8")

            illustration_bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "session_seed",
                    "template": "illustration",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=2025,
            )
            pony_bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "session_seed",
                    "template": "pony",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=2025,
            )

            self.assertEqual(
                self._selection_tuples(illustration_bundle),
                self._selection_tuples(pony_bundle),
            )

    def test_unrelated_extra_wildcards_do_not_shift_shared_token_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "base.txt").write_text("__mood__, __mood__, __scene__\n", encoding="utf-8")
            (template_root / "extra.txt").write_text("__extra__, __mood__, __mood__, __scene__\n", encoding="utf-8")
            (wildcard_root / "extra.txt").write_text("sparkles\nflowers\nmoonlight\n", encoding="utf-8")
            (wildcard_root / "mood.txt").write_text("calm\nconfident\nshy\nteasing\n", encoding="utf-8")
            (wildcard_root / "scene.txt").write_text("bedroom\nclassroom\nrooftop\nstudio\n", encoding="utf-8")

            base_bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "session_seed",
                    "template": "base",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=2025,
            )
            extra_bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "session_seed",
                    "template": "extra",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=2025,
            )

            def to_map(bundle: object) -> dict[tuple[str, int], str]:
                return {
                    (resolved_token, occurrence_index): selected_value
                    for resolved_token, occurrence_index, selected_value in self._selection_tuples(bundle)
                    if resolved_token in {"mood", "scene"}
                }

            self.assertEqual(to_map(base_bundle), to_map(extra_bundle))

    def test_random_segment_dropout_targets_only_one_image_when_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "segments.txt").write_text("alpha, beta, gamma, delta\n", encoding="utf-8")

            common_args = {
                "seed_control": "session_seed",
                "template": "segments",
                "template_root": str(template_root),
                "wildcard_root": str(wildcard_root),
                "_resolved_session_id": "0000000000000000000000aa",
                "_resolved_image_count": 2,
                "_resolved_random_segment_dropout_seed_value": 2025,
                "random_segment_dropout_pair_chance": 1.0,
                "random_segment_dropout_segment_prob": 1.0,
                "seed_control_random_segment_dropout": "session_seed",
            }

            bundle_image0 = WildcardTemplatePromptGenerator().generate(
                {
                    **common_args,
                    "_resolved_image_index": 0,
                },
                seed=2025,
            )
            bundle_image1 = WildcardTemplatePromptGenerator().generate(
                {
                    **common_args,
                    "_resolved_image_index": 1,
                },
                seed=2025,
            )

            prompts = [bundle_image0.positive_prompt, bundle_image1.positive_prompt]
            self.assertEqual(sum(prompt == "alpha" for prompt in prompts), 1)
            self.assertEqual(sum(prompt == "alpha, beta, gamma, delta" for prompt in prompts), 1)
            self.assertEqual(
                sum(
                    bool(bundle.prompt_metadata["random_segment_dropout"]["applied"])
                    for bundle in (bundle_image0, bundle_image1)
                ),
                1,
            )

    def test_random_segment_dropout_pair_chance_zero_disables_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "segments.txt").write_text("alpha, beta, gamma, delta\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "session_seed",
                    "template": "segments",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                    "_resolved_session_id": "0000000000000000000000aa",
                    "_resolved_image_index": 0,
                    "_resolved_image_count": 2,
                    "_resolved_random_segment_dropout_seed_value": 2025,
                    "random_segment_dropout_pair_chance": 0.0,
                    "random_segment_dropout_segment_prob": 1.0,
                    "seed_control_random_segment_dropout": "session_seed",
                },
                seed=2025,
            )

            self.assertEqual(bundle.positive_prompt, "alpha, beta, gamma, delta")
            self.assertFalse(bundle.prompt_metadata["random_segment_dropout"]["active"])
            self.assertFalse(bundle.prompt_metadata["random_segment_dropout"]["applied"])

    def test_random_segment_dropout_keeps_prompt_non_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "segments.txt").write_text("alpha, beta, gamma\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "session_seed",
                    "template": "segments",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                    "_resolved_session_id": "0000000000000000000000aa",
                    "_resolved_image_index": 0,
                    "_resolved_image_count": 1,
                    "_resolved_random_segment_dropout_seed_value": 2025,
                    "random_segment_dropout_pair_chance": 1.0,
                    "random_segment_dropout_segment_prob": 1.0,
                    "seed_control_random_segment_dropout": "session_seed",
                },
                seed=2025,
            )

            self.assertEqual(bundle.positive_prompt, "alpha")
            self.assertEqual(bundle.prompt_metadata["random_segment_dropout"]["dropped_segments"], ["beta", "gamma"])
            self.assertTrue(bundle.prompt_metadata["random_segment_dropout"]["applied"])

    def test_multiline_template_file_selects_by_session_id_mod_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "mix.txt").write_text(
                "line one, __style__\n\nline two, __style__\nline three, __style__\n",
                encoding="utf-8",
            )
            (wildcard_root / "style.txt").write_text("clean\n", encoding="utf-8")

            bundle_a = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "mix",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                    "_resolved_session_id": "000000000000000000000005",
                },
                seed=42,
            )
            bundle_b = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "mix",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                    "_resolved_session_id": "000000000000000000000005",
                },
                seed=42,
            )
            bundle_c = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "mix",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                    "_resolved_session_id": "000000000000000000000006",
                },
                seed=42,
            )

            self.assertEqual(bundle_a.positive_prompt, bundle_b.positive_prompt)
            self.assertEqual(bundle_a.prompt_metadata["template_line_count"], 3)
            self.assertEqual(bundle_a.prompt_metadata["template_line_index"], bundle_b.prompt_metadata["template_line_index"])
            self.assertEqual(bundle_a.prompt_metadata["template_line_index"], int("5", 16) % 3)
            self.assertEqual(bundle_c.prompt_metadata["template_line_index"], int("6", 16) % 3)
            self.assertNotEqual(bundle_a.prompt_metadata["template_line_index"], bundle_c.prompt_metadata["template_line_index"])

    def test_resolves_nested_wildcards_relative_to_pack_root_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcard"
            template_root.mkdir(parents=True, exist_ok=True)
            (wildcard_root / "pack" / "clothing" / "tops").mkdir(parents=True, exist_ok=True)
            (wildcard_root / "pack" / "Colors").mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("__pack/clothing/tops/shirt__\n", encoding="utf-8")
            (wildcard_root / "pack" / "clothing" / "tops" / "shirt.txt").write_text(
                "__Colors/Color__ shirt\n",
                encoding="utf-8",
            )
            (wildcard_root / "pack" / "Colors" / "Color.txt").write_text("red\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "portrait",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=11,
            )

            self.assertEqual(bundle.positive_prompt, "red shirt")
            self.assertEqual(bundle.prompt_metadata["wildcard_usage"], ["pack/clothing/tops/shirt", "Colors/Color"])

    def test_resolves_nested_wildcards_relative_to_current_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcard"
            template_root.mkdir(parents=True, exist_ok=True)
            (wildcard_root / "pack" / "wildcards").mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("__pack/wildcards/choker__\n", encoding="utf-8")
            (wildcard_root / "pack" / "wildcards" / "choker.txt").write_text(
                "__color__ choker\n",
                encoding="utf-8",
            )
            (wildcard_root / "pack" / "wildcards" / "color.txt").write_text("red\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "portrait",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=17,
            )

            self.assertEqual(bundle.positive_prompt, "red choker")
            self.assertEqual(bundle.prompt_metadata["wildcard_usage"], ["pack/wildcards/choker", "color"])

    def test_missing_template_file_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)

            with self.assertRaisesRegex(ValueError, "Template file not found"):
                WildcardTemplatePromptGenerator().generate(
                    {
                        "seed_control": "global_seed",
                        "template": "missing_template",
                        "template_root": str(template_root),
                        "wildcard_root": str(wildcard_root),
                    },
                    seed=1,
                )

    def test_expands_brace_choice_syntax_from_wildcard_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("portrait, __style__\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("{clean|gritty}\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "portrait",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=123,
            )

            self.assertIn(bundle.positive_prompt, {"portrait, clean", "portrait, gritty"})
            self.assertNotIn("{", bundle.positive_prompt)
            self.assertNotIn("}", bundle.positive_prompt)

    def test_resolves_wildcards_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            (wildcard_root / "Colors").mkdir(parents=True, exist_ok=True)
            template_root.mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("tone, __colors/blue__\n", encoding="utf-8")
            (wildcard_root / "Colors" / "Blue.txt").write_text("sapphire\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "portrait",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=7,
            )

            self.assertEqual(bundle.positive_prompt, "tone, sapphire")

    def test_allows_wildcard_tokens_with_spaces_in_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            (wildcard_root / "outfits").mkdir(parents=True, exist_ok=True)
            template_root.mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("wearing __outfits/evening dress__\n", encoding="utf-8")
            (wildcard_root / "outfits" / "evening dress.txt").write_text("silk gown\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "portrait",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=99,
            )

            self.assertEqual(bundle.positive_prompt, "wearing silk gown")

    def test_preserves_braces_without_choice_separator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("portrait, __style__\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("{prompt}\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": "portrait",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                },
                seed=5,
            )

            self.assertEqual(bundle.positive_prompt, "portrait, {prompt}")

    def test_curated_prompt_templates_are_well_formed_and_expand(self) -> None:
        template_root = REPO_ROOT / "template" / "prompt_templates"
        wildcard_root = REPO_ROOT / "template" / "wildcard"
        curated_paths = sorted((template_root / "curated").glob("*_curated.txt"))

        self.assertTrue(curated_paths)
        for template_path in curated_paths:
            raw_text = template_path.read_text(encoding="utf-8")
            self.assertNotIn("detail____", raw_text, msg=str(template_path))
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            self.assertTrue(lines, msg=str(template_path))

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": f"curated/{template_path.stem}",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                    "_resolved_session_id": "000000000000000000000001",
                },
                seed=2025,
            )

            self.assertTrue(bundle.positive_prompt)
            self.assertTrue(bundle.negative_prompt)

    def test_research_v2_prompt_templates_are_well_formed_and_expand(self) -> None:
        template_root = REPO_ROOT / "template" / "prompt_templates"
        wildcard_root = REPO_ROOT / "template" / "wildcard"
        research_paths = sorted((template_root / "research_v2").glob("*.txt"))

        self.assertTrue(research_paths)
        for template_path in research_paths:
            raw_text = template_path.read_text(encoding="utf-8")
            self.assertNotIn("detail____", raw_text, msg=str(template_path))
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            self.assertTrue(lines, msg=str(template_path))

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "global_seed",
                    "template": f"research_v2/{template_path.stem}",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                    "_resolved_session_id": "000000000000000000000001",
                },
                seed=2025,
            )

            self.assertTrue(bundle.positive_prompt)
            self.assertTrue(bundle.negative_prompt)

    def test_preview_wildcard_template_samples_returns_multiple_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("portrait, __style__\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("clean\ngritty\nsoft\n", encoding="utf-8")

            bundles = preview_wildcard_template_samples(
                template="portrait",
                template_root=template_root,
                wildcard_root=wildcard_root,
                seed=9,
                count=3,
            )

            self.assertEqual(len(bundles), 3)
            self.assertTrue(all(bundle.positive_prompt.startswith("portrait, ") for bundle in bundles))

    def test_defaults_to_repo_wildcard_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = Path.cwd()
            base_dir = Path(tmpdir)
            template_root = base_dir / "template" / "prompt_templates"
            wildcard_root = base_dir / "template" / "wildcard"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("portrait, __style__\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("clean\n", encoding="utf-8")

            try:
                import os

                os.chdir(base_dir)
                bundle = WildcardTemplatePromptGenerator().generate(
                    {"seed_control": "global_seed", "template": "portrait"},
                    seed=3,
                )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(bundle.positive_prompt, "portrait, clean")

    def test_main_preview_command_prints_sample_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = Path.cwd()
            base_dir = Path(tmpdir)
            template_root = base_dir / "template" / "prompt_templates"
            wildcard_root = base_dir / "template" / "wildcard"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("portrait, __style__\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("clean\n", encoding="utf-8")

            output = io.StringIO()
            try:
                import os
                import sys

                os.chdir(base_dir)
                original_argv = sys.argv
                sys.argv = [
                    "prompt_generator.py",
                    "preview-wildcard-template",
                    "--template",
                    "portrait",
                    "--count",
                    "1",
                ]
                with redirect_stdout(output):
                    main()
            finally:
                os.chdir(previous_cwd)
                sys.argv = original_argv

            rendered = output.getvalue()
            self.assertIn("Sample 1", rendered)
            self.assertIn("Positive: portrait, clean", rendered)
            self.assertIn("Negative: bad anatomy, worst quality", rendered)

    def test_requires_seed_control_for_wildcard_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("portrait, __style__\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("clean\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "seed_control"):
                WildcardTemplatePromptGenerator().generate(
                    {
                        "template": "portrait",
                        "template_root": str(template_root),
                        "wildcard_root": str(wildcard_root),
                    },
                    seed=1,
                )

    def test_applies_dropout_before_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = Path(tmpdir) / "prompt_templates"
            wildcard_root = Path(tmpdir) / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "portrait.txt").write_text("portrait, __style__, keepme\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("clean\n", encoding="utf-8")

            bundle = WildcardTemplatePromptGenerator().generate(
                {
                    "seed_control": "session_seed",
                    "template": "portrait",
                    "template_root": str(template_root),
                    "wildcard_root": str(wildcard_root),
                    "dropout_items": ["__style__"],
                    "dropout_probs": [1.0],
                    "seed_control_dropout": "image_index_seed",
                    "_resolved_dropout_seed_value": 12345,
                },
                seed=99,
            )

            self.assertEqual(bundle.positive_prompt, "portrait, keepme")
            self.assertEqual(bundle.prompt_metadata["dropped_items"], ["__style__"])


class MixedPromptGeneratorCompileTests(unittest.TestCase):
    def test_compile_requests_uses_repo_defaults_for_external_task_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            task = {
                "version": 1,
                "task_name": "external_defaults",
                "global_seed": 42,
                "session_count": 1,
                "images": [
                    {
                        "image_name": "image1",
                        "workflow_name": "SdxlEaseLoraWorkflow",
                        "ckpt": "model.safetensors",
                        "lora_stack_config": {},
                        "prompt_generator": {
                            "name": "wildcard_template_generator",
                            "args": {
                                "seed_control": "session_seed",
                                "template": "mix_gpt",
                                "negative_prompt": "bad anatomy",
                            },
                        },
                        "sample": {
                            "generation_seed_control": "image_index_seed",
                            "steps": 24,
                            "cfg": 6.5,
                            "width": 512,
                            "height": 512,
                        },
                    }
                ],
            }
            task_yaml_path = base_dir / "task.yaml"
            task_yaml_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

            records, manifest, _ = compile_requests(task, task_yaml_path)

            self.assertEqual(len(records), 1)
            self.assertEqual(manifest["request_count"], 1)
            self.assertTrue(records[0]["positive_prompt"].strip())
            self.assertEqual(
                records[0]["prompt_generator_args"]["template_root"],
                str((REPO_ROOT / "template" / "prompt_templates").resolve()),
            )
            self.assertEqual(
                records[0]["prompt_generator_args"]["wildcard_root"],
                str((REPO_ROOT / "template" / "wildcard").resolve()),
            )

    def test_compile_requests_defaults_to_repo_wildcard_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            template_root = base_dir / "prompt_templates"
            wildcard_root = base_dir / "wildcard"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "image1.txt").write_text("masterpiece, __character__, __style__\n", encoding="utf-8")
            (wildcard_root / "character.txt").write_text("saber\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("portrait\n", encoding="utf-8")

            task = {
                "version": 1,
                "task_name": "wildcard_defaults",
                "global_seed": 42,
                "session_count": 1,
                "images": [
                    {
                        "image_name": "image1",
                        "workflow_name": "SdxlEaseLoraWorkflow",
                        "ckpt": "model.safetensors",
                        "lora_stack_config": {},
                        "prompt_generator": {
                            "name": "wildcard_template_generator",
                            "args": {
                                "seed_control": "session_seed",
                                "template": "image1",
                                "negative_prompt": "bad anatomy",
                            },
                        },
                        "sample": {
                            "generation_seed_control": "image_index_seed",
                            "steps": 24,
                            "cfg": 6.5,
                            "width": 512,
                            "height": 512,
                        },
                    }
                ],
            }
            task_yaml_path = base_dir / "task.yaml"
            task_yaml_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

            records, manifest, _ = compile_requests(task, task_yaml_path)

            self.assertEqual(len(records), 1)
            self.assertEqual(manifest["request_count"], 1)
            self.assertEqual(records[0]["positive_prompt"], "masterpiece, saber, portrait")
            self.assertEqual(records[0]["negative_prompt"], "bad anatomy")
            self.assertEqual(
                records[0]["prompt_generator_args"]["template_root"],
                str((base_dir / "prompt_templates").resolve()),
            )
            self.assertEqual(
                records[0]["prompt_generator_args"]["wildcard_root"],
                str((base_dir / "wildcard").resolve()),
            )

    def test_compile_requests_supports_different_generators_per_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            template_root = base_dir / "prompt_templates"
            wildcard_root = base_dir / "wildcards"
            template_root.mkdir(parents=True, exist_ok=True)
            wildcard_root.mkdir(parents=True, exist_ok=True)
            (template_root / "image2.txt").write_text("masterpiece, __character__, __style__\n", encoding="utf-8")
            (wildcard_root / "character.txt").write_text("saber\n", encoding="utf-8")
            (wildcard_root / "style.txt").write_text("portrait\n", encoding="utf-8")

            task = {
                "version": 1,
                "task_name": "mixed_prompt_generators",
                "global_seed": 42,
                "session_count": 1,
                "images": [
                    {
                        "image_name": "image1",
                        "workflow_name": "SdxlEaseLoraWorkflow",
                        "ckpt": "model.safetensors",
                        "lora_stack_config": {},
                        "prompt_generator": {
                            "name": "placeholder_generator",
                            "args": {
                                "mode": "character_portrait",
                                "character": "Altria Saber",
                                "quality_preset": "clean_anime",
                            },
                        },
                        "sample": {
                            "generation_seed_control": "image_index_seed",
                            "steps": 20,
                            "cfg": 7.0,
                            "width": 512,
                            "height": 512,
                        },
                    },
                    {
                        "image_name": "image2",
                        "workflow_name": "SdxlEaseLoraWorkflow",
                        "ckpt": "model.safetensors",
                        "lora_stack_config": {},
                        "prompt_generator": {
                            "name": "wildcard_template_generator",
                            "args": {
                                "seed_control": "session_seed",
                                "template": "image2",
                                "negative_prompt": "bad anatomy",
                            },
                        },
                        "sample": {
                            "generation_seed_control": "image_index_seed",
                            "steps": 24,
                            "cfg": 6.5,
                            "width": 512,
                            "height": 512,
                        },
                    },
                ],
            }
            task_yaml_path = base_dir / "task.yaml"
            task_yaml_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")

            records, manifest, _ = compile_requests(task, task_yaml_path)

            self.assertEqual(len(records), 2)
            self.assertEqual(manifest["request_count"], 2)
            self.assertEqual(records[0]["prompt_generator_name"], "placeholder_generator")
            self.assertEqual(records[1]["prompt_generator_name"], "wildcard_template_generator")
            self.assertEqual(records[1]["positive_prompt"], "masterpiece, saber, portrait")
            self.assertEqual(records[1]["negative_prompt"], "bad anatomy")
            self.assertEqual(
                records[1]["prompt_generator_args"]["template_root"],
                str((base_dir / "prompt_templates").resolve()),
            )
            self.assertEqual(
                records[1]["prompt_generator_args"]["wildcard_root"],
                str((base_dir / "wildcards").resolve()),
            )


if __name__ == "__main__":
    unittest.main()
