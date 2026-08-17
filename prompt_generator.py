from __future__ import annotations

import abc
import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Type

from layout_paths import PROMPT_LISTS_DIR, PROMPT_TEMPLATES_DIR, WILDCARD_DIR


@dataclass(frozen=True)
class PromptBundle:
    positive_prompt: str
    negative_prompt: str
    prompt_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ExpansionSegment:
    text: str
    source_path: Path | None


class PromptGeneratorBase(abc.ABC):
    NAME = "base"
    VERSION = "v1"

    @abc.abstractmethod
    def generate(self, args: Mapping[str, Any], seed: int) -> PromptBundle:
        raise NotImplementedError


def load_prompt_list_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


class PlaceholderPromptGenerator(PromptGeneratorBase):
    """Deterministic WD14-like test prompt generator.

    It builds a comma-separated tag prompt using:
    1) fixed tags derived from structured args
    2) a deterministic random subset sampled from a 20-tag test pool
    """

    NAME = "placeholder_generator"
    VERSION = "v2"

    TEST_TAG_POOL = [
        "1girl",
        "solo",
        "looking_at_viewer",
        "smile",
        "blonde_hair",
        "long_hair",
        "blue_eyes",
        "dress",
        "hair_ornament",
        "upper_body",
        "portrait",
        "simple_background",
        "white_background",
        "outdoors",
        "sky",
        "day",
        "weapon",
        "armor",
        "cape",
        "boots",
    ]

    def _normalize_tag(self, value: Any) -> str:
        return str(value).strip().replace(" ", "_")

    def _append_if_present(self, tags: list[str], args: Mapping[str, Any], key: str) -> None:
        value = args.get(key)
        if value in (None, ""):
            return
        tags.append(self._normalize_tag(value))

    def generate(self, args: Mapping[str, Any], seed: int) -> PromptBundle:
        rng = random.Random(int(seed))
        negative_prompt = str(args.get("negative_prompt", "bad anatomy, worst quality"))

        fixed_tags: list[str] = []

        # Structured args -> WD14-like tags
        self._append_if_present(fixed_tags, args, "character")
        self._append_if_present(fixed_tags, args, "franchise")
        self._append_if_present(fixed_tags, args, "subject")
        self._append_if_present(fixed_tags, args, "shot")
        self._append_if_present(fixed_tags, args, "background")
        self._append_if_present(fixed_tags, args, "costume")
        self._append_if_present(fixed_tags, args, "style")
        self._append_if_present(fixed_tags, args, "quality_preset")

        extra_positive = args.get("extra_positive")
        if extra_positive not in (None, ""):
            if isinstance(extra_positive, (list, tuple)):
                for item in extra_positive:
                    if item not in (None, ""):
                        fixed_tags.append(self._normalize_tag(item))
            else:
                for item in str(extra_positive).split(","):
                    item = item.strip()
                    if item:
                        fixed_tags.append(self._normalize_tag(item))

        # Optional mode tag for testing traceability
        mode = str(args.get("mode", "generic")).strip()
        if mode:
            fixed_tags.append(f"mode_{self._normalize_tag(mode)}")

        # Deduplicate while preserving order
        dedup_fixed_tags: list[str] = []
        seen = set()
        for tag in fixed_tags:
            if tag and tag not in seen:
                dedup_fixed_tags.append(tag)
                seen.add(tag)

        # Sample from 20-tag pool
        requested_tag_count = args.get("tag_count")
        if requested_tag_count is None:
            sampled_count = rng.randint(8, 14)
        else:
            sampled_count = max(1, min(int(requested_tag_count), len(self.TEST_TAG_POOL)))

        available_pool = [tag for tag in self.TEST_TAG_POOL if tag not in seen]
        sampled_tags = rng.sample(available_pool, k=min(sampled_count, len(available_pool)))

        positive_tags = dedup_fixed_tags + sampled_tags
        positive_prompt = ", ".join(positive_tags)

        return PromptBundle(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            prompt_metadata={
                "generator_name": self.NAME,
                "generator_version": self.VERSION,
                "mode": mode,
                "seed_used": int(seed),
                "fixed_tags": dedup_fixed_tags,
                "sampled_tags": sampled_tags,
                "sampled_tag_count": len(sampled_tags),
                "tag_pool_size": len(self.TEST_TAG_POOL),
            },
        )


class WildcardTemplatePromptGenerator(PromptGeneratorBase):
    """Template-based prompt generator with Automatic1111/ComfyUI-style wildcards."""

    NAME = "wildcard_template_generator"
    VERSION = "v2"
    DEFAULT_NEGATIVE_PROMPT = "bad anatomy, worst quality"
    DEFAULT_TEMPLATE_ROOT = str(PROMPT_TEMPLATES_DIR)
    DEFAULT_WILDCARD_ROOT = str(WILDCARD_DIR)
    DEFAULT_MAX_EXPANSION_DEPTH = 32
    WILDCARD_PATTERN = re.compile(r"__([^\r\n]+?)__")
    VALID_SEED_CONTROL_NAMES = frozenset({"global_seed", "session_seed", "image_index_seed"})

    def generate(self, args: Mapping[str, Any], seed: int) -> PromptBundle:
        seed_control = self._resolve_seed_control(args, "seed_control")
        template_identifier = self._resolve_template_identifier(args)
        template_root = self._resolve_template_root(args)
        template_path = self._resolve_template_path(template_identifier, template_root=template_root)
        template_text, template_line_index, template_line_count = self._load_template_text(template_path, args=args, seed=seed)
        negative_template = self._resolve_template_arg(
            args,
            "negative_template",
            "negative_prompt_template",
            "negative_prompt",
            default=self.DEFAULT_NEGATIVE_PROMPT,
        )
        wildcard_root = self._resolve_wildcard_root(args)
        max_expansion_depth = max(1, int(args.get("max_expansion_depth", self.DEFAULT_MAX_EXPANSION_DEPTH)))
        dropout_items, dropout_probs, seed_control_dropout, dropped_segments, dropped_items = self._resolve_dropout_config(
            args
        )
        (
            random_segment_dropout_pair_chance,
            random_segment_dropout_segment_prob,
            seed_control_random_segment_dropout,
        ) = self._resolve_random_segment_dropout_config(args)
        if dropout_items is not None and dropout_probs is not None and seed_control_dropout is not None:
            dropout_seed_value = args.get("_resolved_dropout_seed_value")
            if dropout_seed_value in (None, ""):
                raise ValueError(
                    "Wildcard dropout requires an internal resolved dropout seed value. "
                    "Compile through compile_yaml_to_requests_jsonl.py."
                )
            positive_template, dropped_segments, dropped_items = self._apply_dropout_to_template(
                template_text,
                dropout_items=dropout_items,
                dropout_probs=dropout_probs,
                dropout_seed_value=int(dropout_seed_value),
            )
        else:
            positive_template = template_text

        rng = random.Random(int(seed))
        wildcard_usage: list[str] = []
        wildcard_selection_usage: list[Dict[str, Any]] = []

        positive_prompt = self._expand_template(
            positive_template,
            wildcard_root=wildcard_root,
            rng=rng,
            wildcard_usage=wildcard_usage,
            wildcard_selection_usage=wildcard_selection_usage,
            wildcard_seed=int(seed),
            max_expansion_depth=max_expansion_depth,
            source_path=template_path,
        )
        negative_prompt = self._expand_template(
            negative_template,
            wildcard_root=wildcard_root,
            rng=rng,
            wildcard_usage=wildcard_usage,
            wildcard_selection_usage=wildcard_selection_usage,
            wildcard_seed=int(seed),
            max_expansion_depth=max_expansion_depth,
        )
        random_segment_dropout_metadata = {
            "active": False,
            "applied": False,
            "pair_chance": None,
            "segment_prob": None,
            "target_image_index": None,
            "image_index": None,
            "image_count": None,
            "dropped_segments": [],
        }
        if (
            random_segment_dropout_pair_chance is not None
            and random_segment_dropout_segment_prob is not None
            and seed_control_random_segment_dropout is not None
        ):
            random_segment_dropout_seed_value = args.get("_resolved_random_segment_dropout_seed_value")
            if random_segment_dropout_seed_value in (None, ""):
                raise ValueError(
                    "Random segment dropout requires an internal resolved dropout seed value. "
                    "Compile through compile_yaml_to_requests_jsonl.py."
                )
            positive_prompt, random_segment_dropout_metadata = self._apply_random_segment_dropout_to_prompt(
                positive_prompt,
                pair_chance=float(random_segment_dropout_pair_chance),
                segment_prob=float(random_segment_dropout_segment_prob),
                random_segment_dropout_seed_value=int(random_segment_dropout_seed_value),
                session_id=args.get("_resolved_session_id"),
                image_index=args.get("_resolved_image_index"),
                image_count=args.get("_resolved_image_count"),
            )

        return PromptBundle(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            prompt_metadata={
                "generator_name": self.NAME,
                "generator_version": self.VERSION,
                "seed_used": int(seed),
                "seed_control": seed_control,
                "template_identifier": template_identifier,
                "template_path": str(template_path),
                "template_root": str(template_root),
                "template_line_index": template_line_index,
                "template_line_count": template_line_count,
                "wildcard_root": str(wildcard_root),
                "wildcard_usage": wildcard_usage,
                "wildcard_selection_usage": wildcard_selection_usage,
                "max_expansion_depth": max_expansion_depth,
                "seed_control_dropout": seed_control_dropout,
                "dropped_items": dropped_items,
                "dropped_segments": dropped_segments,
                "seed_control_random_segment_dropout": seed_control_random_segment_dropout,
                "random_segment_dropout": random_segment_dropout_metadata,
            },
        )

    def _resolve_seed_control(self, args: Mapping[str, Any], key: str) -> str:
        value = args.get(key)
        if value in (None, ""):
            raise ValueError(f"Missing required wildcard prompt arg: {key!r}")
        seed_control = str(value)
        if seed_control not in self.VALID_SEED_CONTROL_NAMES:
            raise ValueError(
                f"{key!r} must be one of {sorted(self.VALID_SEED_CONTROL_NAMES)}, got {value!r}"
            )
        return seed_control

    def _resolve_template_identifier(self, args: Mapping[str, Any]) -> str:
        for key in ("template", "template_name", "template_id"):
            value = args.get(key)
            if value not in (None, ""):
                return str(value)
        raise ValueError(
            "Missing required template arg. Expected one of: ('template', 'template_name', 'template_id')"
        )

    def _resolve_template_arg(
        self,
        args: Mapping[str, Any],
        *keys: str,
        default: str | None = None,
    ) -> str:
        for key in keys:
            value = args.get(key)
            if value not in (None, ""):
                return str(value)
        if default is not None:
            return default
        raise ValueError(f"Missing required template arg. Expected one of: {keys}")

    def _resolve_template_root(self, args: Mapping[str, Any]) -> Path:
        for key in ("template_root", "templates_root", "templates_dir"):
            value = args.get(key)
            if value not in (None, ""):
                return Path(str(value))
        return Path(self.DEFAULT_TEMPLATE_ROOT)

    def _resolve_wildcard_root(self, args: Mapping[str, Any]) -> Path:
        for key in ("wildcard_root", "wildcards_root", "wildcards_dir"):
            value = args.get(key)
            if value not in (None, ""):
                return Path(str(value))
        for candidate in (self.DEFAULT_WILDCARD_ROOT, "wildcards"):
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path
        return Path(self.DEFAULT_WILDCARD_ROOT)

    def _resolve_template_path(self, template_identifier: str, *, template_root: Path) -> Path:
        normalized = template_identifier.replace("\\", "/").strip("/")
        if not normalized:
            raise ValueError("Template identifier must not be empty")

        template_path = Path(normalized)
        if any(part in ("", ".", "..") for part in template_path.parts):
            raise ValueError(f"Invalid template identifier {template_identifier!r}")

        if template_path.suffix:
            source_path = template_root / template_path
        else:
            source_path = template_root / template_path.with_suffix(".txt")

        if not source_path.is_file():
            raise ValueError(f"Template file not found for identifier {template_identifier!r}: {source_path}")
        return source_path

    def _load_template_text(
        self,
        template_path: Path,
        *,
        args: Mapping[str, Any],
        seed: int,
    ) -> tuple[str, int | None, int]:
        raw_text = template_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Template file is empty: {template_path}")
        if len(lines) == 1:
            return lines[0], 0, 1

        selected_index = self._select_template_line_index(lines, args=args, seed=seed)
        return lines[selected_index], selected_index, len(lines)

    def _select_template_line_index(self, lines: list[str], *, args: Mapping[str, Any], seed: int) -> int:
        session_id = args.get("_resolved_session_id")
        if session_id in (None, ""):
            return random.Random(int(seed)).randrange(len(lines))

        session_id_text = str(session_id).strip()
        if not session_id_text:
            raise ValueError("_resolved_session_id must not be empty")
        return int(session_id_text, 16) % len(lines)

    def _resolve_dropout_config(
        self,
        args: Mapping[str, Any],
    ) -> tuple[list[str] | None, list[float] | None, str | None, list[str], list[str]]:
        dropout_items = args.get("dropout_items")
        dropout_probs = args.get("dropout_probs")
        seed_control_dropout = args.get("seed_control_dropout")

        if dropout_items in (None, "") and dropout_probs in (None, "") and seed_control_dropout in (None, ""):
            return None, None, None, [], []

        if dropout_items in (None, "") or dropout_probs in (None, "") or seed_control_dropout in (None, ""):
            raise ValueError(
                "Wildcard dropout requires 'dropout_items', 'dropout_probs', and 'seed_control_dropout' together"
            )
        if not isinstance(dropout_items, list) or not isinstance(dropout_probs, list):
            raise ValueError("Wildcard dropout requires 'dropout_items' and 'dropout_probs' to be lists")
        if len(dropout_items) != len(dropout_probs):
            raise ValueError("Wildcard dropout requires 'dropout_items' and 'dropout_probs' to have the same length")

        resolved_items = [str(item).strip() for item in dropout_items]
        resolved_probs: list[float] = []
        for index, probability in enumerate(dropout_probs):
            probability_value = float(probability)
            if probability_value < 0.0 or probability_value > 1.0:
                raise ValueError(
                    f"dropout_probs[{index}] must be within [0.0, 1.0], got {probability!r}"
                )
            resolved_probs.append(probability_value)

        return (
            resolved_items,
            resolved_probs,
            self._resolve_seed_control(args, "seed_control_dropout"),
            [],
            [],
        )

    def _resolve_random_segment_dropout_config(
        self,
        args: Mapping[str, Any],
    ) -> tuple[float | None, float | None, str | None]:
        pair_chance = args.get("random_segment_dropout_pair_chance")
        segment_prob = args.get("random_segment_dropout_segment_prob")
        seed_control_random_segment_dropout = args.get("seed_control_random_segment_dropout")

        if (
            pair_chance in (None, "")
            and segment_prob in (None, "")
            and seed_control_random_segment_dropout in (None, "")
        ):
            return None, None, None

        if (
            pair_chance in (None, "")
            or segment_prob in (None, "")
            or seed_control_random_segment_dropout in (None, "")
        ):
            raise ValueError(
                "Random segment dropout requires 'random_segment_dropout_pair_chance', "
                "'random_segment_dropout_segment_prob', and 'seed_control_random_segment_dropout' together"
            )

        resolved_pair_chance = float(pair_chance)
        resolved_segment_prob = float(segment_prob)
        if resolved_pair_chance < 0.0 or resolved_pair_chance > 1.0:
            raise ValueError(
                "random_segment_dropout_pair_chance must be within [0.0, 1.0], "
                f"got {pair_chance!r}"
            )
        if resolved_segment_prob < 0.0 or resolved_segment_prob > 1.0:
            raise ValueError(
                "random_segment_dropout_segment_prob must be within [0.0, 1.0], "
                f"got {segment_prob!r}"
            )

        return (
            resolved_pair_chance,
            resolved_segment_prob,
            self._resolve_seed_control(args, "seed_control_random_segment_dropout"),
        )

    def _apply_dropout_to_template(
        self,
        template: str,
        *,
        dropout_items: list[str],
        dropout_probs: list[float],
        dropout_seed_value: int,
    ) -> tuple[str, list[str], list[str]]:
        segments = [segment.strip() for segment in str(template).split(",")]
        kept_segments: list[str] = []
        dropped_segments: list[str] = []
        dropped_items: list[str] = []

        for original_segment in segments:
            if not original_segment:
                continue

            segment_should_drop = False
            for index, dropout_item in enumerate(dropout_items):
                if not dropout_item:
                    continue
                if dropout_item not in original_segment:
                    continue
                rng = random.Random(int(dropout_seed_value) + index)
                if rng.random() < dropout_probs[index]:
                    segment_should_drop = True
                    dropped_segments.append(original_segment)
                    dropped_items.append(dropout_item)
                    break

            if not segment_should_drop:
                kept_segments.append(original_segment)

        return ", ".join(kept_segments), dropped_segments, dropped_items

    def _apply_random_segment_dropout_to_prompt(
        self,
        prompt: str,
        *,
        pair_chance: float,
        segment_prob: float,
        random_segment_dropout_seed_value: int,
        session_id: Any,
        image_index: Any,
        image_count: Any,
    ) -> tuple[str, Dict[str, Any]]:
        session_id_text = str(session_id).strip() if session_id not in (None, "") else "preview-session"
        resolved_image_index = int(image_index) if image_index not in (None, "") else 0
        resolved_image_count = int(image_count) if image_count not in (None, "") else 1
        resolved_image_count = max(1, resolved_image_count)

        metadata: Dict[str, Any] = {
            "active": False,
            "applied": False,
            "pair_chance": float(pair_chance),
            "segment_prob": float(segment_prob),
            "target_image_index": None,
            "image_index": resolved_image_index,
            "image_count": resolved_image_count,
            "dropped_segments": [],
        }
        segments = [segment.strip() for segment in str(prompt).split(",") if segment.strip()]
        if not segments:
            return str(prompt), metadata

        activation_roll = self._derive_unit_float(
            int(random_segment_dropout_seed_value),
            session_id_text,
            "random_segment_dropout",
            "activate",
        )
        if activation_roll >= pair_chance:
            return ", ".join(segments), metadata

        metadata["active"] = True
        target_image_index = self._derive_index(
            resolved_image_count,
            int(random_segment_dropout_seed_value),
            session_id_text,
            "random_segment_dropout",
            "target_image",
        )
        metadata["target_image_index"] = target_image_index
        if resolved_image_index != target_image_index:
            return ", ".join(segments), metadata

        kept_segments: list[str] = []
        dropped_segments: list[str] = []
        for segment_index, segment in enumerate(segments):
            drop_roll = self._derive_unit_float(
                int(random_segment_dropout_seed_value),
                session_id_text,
                "random_segment_dropout",
                "segment",
                target_image_index,
                segment_index,
            )
            if drop_roll < segment_prob:
                dropped_segments.append(segment)
                continue
            kept_segments.append(segment)

        if not kept_segments:
            kept_segments.append(segments[0])
            if segments[0] in dropped_segments:
                dropped_segments.remove(segments[0])

        metadata["applied"] = True
        metadata["dropped_segments"] = dropped_segments
        return ", ".join(kept_segments), metadata

    def _derive_unit_float(self, *parts: Any) -> float:
        joined = "||".join(str(part) for part in parts)
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) / float(2**64)

    def _derive_index(self, modulo: int, *parts: Any) -> int:
        if modulo <= 0:
            raise ValueError(f"modulo must be positive, got {modulo!r}")
        joined = "||".join(str(part) for part in parts)
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % int(modulo)

    def _expand_template(
        self,
        template: str,
        *,
        wildcard_root: Path,
        rng: random.Random,
        wildcard_usage: list[str],
        wildcard_selection_usage: list[Dict[str, Any]],
        wildcard_seed: int,
        max_expansion_depth: int,
        source_path: Path | None = None,
    ) -> str:
        segments = [_ExpansionSegment(text=str(template), source_path=source_path)]
        wildcard_occurrence_counts: dict[str, int] = {}
        for depth in range(max_expansion_depth):
            changed = False
            expanded_segments: list[_ExpansionSegment] = []

            for segment in segments:
                expanded_text, choice_replaced = self._expand_choice_syntax(segment.text, rng=rng)
                changed = changed or choice_replaced

                matches = list(self.WILDCARD_PATTERN.finditer(expanded_text))
                if not matches:
                    expanded_segments.append(_ExpansionSegment(text=expanded_text, source_path=segment.source_path))
                    continue

                changed = True
                previous_end = 0
                for match in matches:
                    if match.start() > previous_end:
                        expanded_segments.append(
                            _ExpansionSegment(
                                text=expanded_text[previous_end : match.start()],
                                source_path=segment.source_path,
                            )
                        )

                    token = match.group(1)
                    wildcard_usage.append(token)
                    replacement, replacement_source_path = self._select_wildcard_value(
                        token,
                        wildcard_root=wildcard_root,
                        wildcard_seed=wildcard_seed,
                        wildcard_occurrence_counts=wildcard_occurrence_counts,
                        wildcard_selection_usage=wildcard_selection_usage,
                        referring_source_path=segment.source_path,
                    )
                    expanded_segments.append(
                        _ExpansionSegment(text=replacement, source_path=replacement_source_path)
                    )
                    previous_end = match.end()

                if previous_end < len(expanded_text):
                    expanded_segments.append(
                        _ExpansionSegment(text=expanded_text[previous_end:], source_path=segment.source_path)
                    )

            segments = self._coalesce_segments(expanded_segments)
            if not changed:
                return "".join(segment.text for segment in segments)

        raise ValueError(
            f"Wildcard expansion exceeded max_expansion_depth={max_expansion_depth}. "
            f"Last template: {''.join(segment.text for segment in segments)!r}"
        )

    def _coalesce_segments(self, segments: list[_ExpansionSegment]) -> list[_ExpansionSegment]:
        merged: list[_ExpansionSegment] = []
        for segment in segments:
            if not segment.text:
                continue
            if merged and merged[-1].source_path == segment.source_path:
                previous = merged[-1]
                merged[-1] = _ExpansionSegment(
                    text=previous.text + segment.text,
                    source_path=segment.source_path,
                )
                continue
            merged.append(segment)
        return merged

    def _expand_choice_syntax(self, text: str, *, rng: random.Random) -> tuple[str, bool]:
        result: list[str] = []
        changed = False
        index = 0

        while index < len(text):
            if text[index] != "{":
                result.append(text[index])
                index += 1
                continue

            closing_index = self._find_matching_brace(text, index)
            if closing_index is None:
                result.append(text[index])
                index += 1
                continue

            choice_text = text[index + 1 : closing_index]
            if not self._has_top_level_choice_separator(choice_text):
                result.append(text[index : closing_index + 1])
                index = closing_index + 1
                continue

            options = self._split_choice_options(choice_text)
            selected = rng.choice(options)
            expanded_selected, _ = self._expand_choice_syntax(selected, rng=rng)
            result.append(expanded_selected)
            changed = True
            index = closing_index + 1

        return "".join(result), changed

    def _find_matching_brace(self, text: str, opening_index: int) -> int | None:
        depth = 0
        for index in range(opening_index, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def _split_choice_options(self, text: str) -> list[str]:
        options: list[str] = []
        current: list[str] = []
        depth = 0

        for char in text:
            if char == "{":
                depth += 1
            elif char == "}":
                depth = max(0, depth - 1)

            if char == "|" and depth == 0:
                options.append("".join(current))
                current = []
                continue

            current.append(char)

        options.append("".join(current))
        return options

    def _has_top_level_choice_separator(self, text: str) -> bool:
        depth = 0
        for char in text:
            if char == "{":
                depth += 1
            elif char == "}":
                depth = max(0, depth - 1)
            elif char == "|" and depth == 0:
                return True
        return False

    def _select_wildcard_value(
        self,
        token: str,
        *,
        wildcard_root: Path,
        wildcard_seed: int,
        wildcard_occurrence_counts: dict[str, int],
        wildcard_selection_usage: list[Dict[str, Any]],
        referring_source_path: Path | None = None,
    ) -> tuple[str, Path]:
        normalized = token.replace("\\", "/").strip("/")
        if not normalized:
            raise ValueError("Wildcard token must not be empty")

        wildcard_path = Path(normalized)
        if any(part in ("", ".", "..") for part in wildcard_path.parts):
            raise ValueError(f"Invalid wildcard token {token!r}")

        expected_path = wildcard_root / wildcard_path.with_suffix(".txt")
        source_path = self._resolve_wildcard_source_path(
            wildcard_root,
            wildcard_path.with_suffix(".txt"),
            referring_source_path=referring_source_path,
        )
        if source_path is None:
            raise ValueError(f"Wildcard file not found for token {token!r}: {expected_path}")

        choices = []
        for raw_line in source_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            choices.append(line)

        if not choices:
            raise ValueError(f"Wildcard file is empty after filtering comments/blank lines: {source_path}")

        resolved_token_key = self._resolved_wildcard_key(wildcard_root, source_path)
        occurrence_index = wildcard_occurrence_counts.get(resolved_token_key, 0)
        wildcard_occurrence_counts[resolved_token_key] = occurrence_index + 1
        choice_index = self._derive_index(
            len(choices),
            int(wildcard_seed),
            resolved_token_key,
            occurrence_index,
        )
        selected_value = str(choices[choice_index])
        wildcard_selection_usage.append(
            {
                "token": normalized,
                "resolved_token": resolved_token_key,
                "occurrence_index": occurrence_index,
                "choice_index": choice_index,
                "selected_value": selected_value,
            }
        )
        return selected_value, source_path

    def _resolved_wildcard_key(self, wildcard_root: Path, source_path: Path) -> str:
        wildcard_root_resolved = wildcard_root.resolve()
        source_path_resolved = source_path.resolve()
        try:
            relative_path = source_path_resolved.relative_to(wildcard_root_resolved)
        except ValueError:
            relative_path = source_path
        return relative_path.with_suffix("").as_posix().casefold()

    def _resolve_wildcard_source_path(
        self,
        wildcard_root: Path,
        wildcard_path: Path,
        *,
        referring_source_path: Path | None = None,
    ) -> Path | None:
        source_path = self._resolve_path_from_root(wildcard_root, wildcard_path)
        if source_path is not None:
            return source_path

        if referring_source_path is None:
            return None

        candidate_dir = referring_source_path.parent
        while True:
            try:
                candidate_dir.relative_to(wildcard_root)
            except ValueError:
                break

            if candidate_dir != wildcard_root:
                source_path = self._resolve_path_from_root(candidate_dir, wildcard_path)
                if source_path is not None:
                    return source_path

            if candidate_dir == wildcard_root:
                break
            candidate_dir = candidate_dir.parent

        return None

    def _resolve_path_from_root(self, root: Path, relative_path: Path) -> Path | None:
        exact_path = root / relative_path
        if exact_path.is_file():
            return exact_path
        return self._resolve_case_insensitive_path(root, relative_path)

    def _resolve_case_insensitive_path(self, root: Path, relative_path: Path) -> Path | None:
        current = root
        for part in relative_path.parts:
            try:
                candidates = list(current.iterdir())
            except FileNotFoundError:
                return None

            match = next((candidate for candidate in candidates if candidate.name.casefold() == part.casefold()), None)
            if match is None:
                return None
            current = match

        if current.is_file():
            return current
        return None


class NonWildcardV1PromptGenerator(PromptGeneratorBase):
    """Scene/action-coherent 2-character prompt generator backed by wildcard_engine.

    Unlike ``WildcardTemplatePromptGenerator`` this does not consume .txt wildcard
    templates; instead it delegates positive-prompt construction to the
    tag-algebra engine shipped in ``wildcard_engine`` so that background, action
    and character appearance stay mutually consistent for 2-person scenes.
    """

    NAME = "non_wildcard_v1"
    VERSION = "v1"

    DEFAULT_NEGATIVE_PROMPT = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "jpeg artifacts, signature, watermark, username, blurry"
    )

    _character_cache: Dict[str, list[str]] = {}

    @classmethod
    def _load_characters(cls, path: Path) -> list[str]:
        key = str(path.resolve())
        cached = cls._character_cache.get(key)
        if cached is not None:
            return cached
        with open(path, "r", encoding="utf-8") as handle:
            chars = [line.strip() for line in handle if line.strip()]
        if len(chars) < 2:
            raise ValueError(f"Character list {path} must contain at least 2 entries")
        cls._character_cache[key] = chars
        return chars

    def generate(self, args: Mapping[str, Any], seed: int) -> PromptBundle:
        character_list_arg = args.get("character_list")
        if not character_list_arg:
            raise ValueError("non_wildcard_v1 generator requires 'character_list' arg")
        char_path = Path(str(character_list_arg))
        if not char_path.is_absolute():
            candidates = [Path.cwd() / char_path, WILDCARD_DIR / char_path.name]
            char_path = next((c for c in candidates if c.is_file()), candidates[0])
        characters = self._load_characters(char_path)

        rng = random.Random(seed)
        char1, char2 = rng.sample(characters, 2)

        # Compiler-resolved per-family overrides; empty string is a valid override
        # meaning "no prefix/suffix at all".
        positive_prefix = args.get("positive_prefix")
        positive_suffix = args.get("positive_suffix")

        from wildcard_engine.core.engine import generate_with_debug as _we_gen_debug

        debug = _we_gen_debug(
            seed,
            char1,
            char2,
            quality_prefix=str(positive_prefix) if positive_prefix is not None else None,
            style_suffix=str(positive_suffix) if positive_suffix is not None else None,
        )
        if "error" in debug:
            raise RuntimeError(f"wildcard_engine failed: {debug['error']}")

        positive = str(debug["prompt"])
        negative = str(args.get("negative_prompt", self.DEFAULT_NEGATIVE_PROMPT))

        metadata: Dict[str, Any] = {
            "generator": self.NAME,
            "generator_version": self.VERSION,
            "seed": seed,
            "character_list_path": str(char_path),
            "character_1": char1,
            "character_2": char2,
            "scene_id": debug.get("scene_id"),
            "scene_desc": debug.get("scene_desc"),
            "action_id": debug.get("action_id"),
            "action_desc": debug.get("action_desc"),
            "char1_expr": debug.get("char1_expr"),
            "char2_expr": debug.get("char2_expr"),
            "char1_dressing": debug.get("char1_dressing"),
            "char2_dressing": debug.get("char2_dressing"),
            "collision_ok": debug.get("collision_ok"),
            "resolved_ckpt_family": args.get("_resolved_ckpt_family"),
            "resolved_positive_prefix": positive_prefix,
            "resolved_positive_suffix": positive_suffix,
        }
        return PromptBundle(
            positive_prompt=positive,
            negative_prompt=negative,
            prompt_metadata=metadata,
        )


class NonWildcardV2PromptGenerator(NonWildcardV1PromptGenerator):
    """Versioned alias for the non-wildcard engine with newer family defaults."""

    NAME = "non_wildcard_v2"
    VERSION = "v2"


class PromptListV1PromptGenerator(PromptGeneratorBase):
    """Literal prompt-list generator that selects one exact line per session."""

    NAME = "prompt_list_v1"
    VERSION = "v1"
    DEFAULT_NEGATIVE_PROMPT = "bad anatomy, worst quality"
    DEFAULT_PROMPT_LIST_ROOT = str(PROMPT_LISTS_DIR)

    def generate(self, args: Mapping[str, Any], seed: int) -> PromptBundle:
        prompt_list_identifier = self._resolve_prompt_list_identifier(args)
        prompt_list_root = self._resolve_prompt_list_root(args)
        prompt_list_path = self._resolve_prompt_list_path(prompt_list_identifier, prompt_list_root=prompt_list_root)
        lines = load_prompt_list_lines(prompt_list_path)
        if not lines:
            raise ValueError(f"Prompt list file is empty: {prompt_list_path}")

        session_index = args.get("_resolved_session_index")
        if session_index in (None, ""):
            raise ValueError(
                "prompt_list_v1 requires an internal resolved session index. "
                "Compile through compile_yaml_to_requests_jsonl.py."
            )
        resolved_session_index = int(session_index)
        if resolved_session_index < 0 or resolved_session_index >= len(lines):
            raise ValueError(
                f"Prompt list {prompt_list_path} does not have line index {resolved_session_index}. "
                f"Available line count: {len(lines)}"
            )

        positive_prompt = lines[resolved_session_index]
        negative_prompt = str(args.get("negative_prompt", self.DEFAULT_NEGATIVE_PROMPT))
        return PromptBundle(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            prompt_metadata={
                "generator_name": self.NAME,
                "generator_version": self.VERSION,
                "seed_used": int(seed),
                "prompt_list_identifier": prompt_list_identifier,
                "prompt_list_path": str(prompt_list_path),
                "prompt_list_root": str(prompt_list_root),
                "prompt_line_index": resolved_session_index,
                "prompt_line_count": len(lines),
                "resolved_ckpt_family": args.get("_resolved_ckpt_family"),
            },
        )

    def _resolve_prompt_list_identifier(self, args: Mapping[str, Any]) -> str:
        for key in ("prompt_list", "prompt_list_name", "prompt_list_id"):
            value = args.get(key)
            if value not in (None, ""):
                return str(value)
        raise ValueError("Missing required prompt list arg. Expected one of: ('prompt_list', 'prompt_list_name', 'prompt_list_id')")

    def _resolve_prompt_list_root(self, args: Mapping[str, Any]) -> Path:
        for key in ("prompt_list_root", "prompt_lists_root", "prompt_lists_dir"):
            value = args.get(key)
            if value not in (None, ""):
                return Path(str(value))
        return Path(self.DEFAULT_PROMPT_LIST_ROOT)

    def _resolve_prompt_list_path(self, prompt_list_identifier: str, *, prompt_list_root: Path) -> Path:
        normalized = prompt_list_identifier.replace("\\", "/").strip("/")
        if not normalized:
            raise ValueError("Prompt list identifier must not be empty")

        prompt_list_path = Path(normalized)
        if any(part in ("", ".", "..") for part in prompt_list_path.parts):
            raise ValueError(f"Invalid prompt list identifier {prompt_list_identifier!r}")

        source_path = prompt_list_root / prompt_list_path
        if not source_path.suffix:
            source_path = source_path.with_suffix(".txt")
        if not source_path.is_file():
            raise ValueError(f"Prompt list file not found for identifier {prompt_list_identifier!r}: {source_path}")
        return source_path


PROMPT_GENERATOR_REGISTRY: Dict[str, Type[PromptGeneratorBase]] = {
    PlaceholderPromptGenerator.NAME: PlaceholderPromptGenerator,
    WildcardTemplatePromptGenerator.NAME: WildcardTemplatePromptGenerator,
    NonWildcardV1PromptGenerator.NAME: NonWildcardV1PromptGenerator,
    NonWildcardV2PromptGenerator.NAME: NonWildcardV2PromptGenerator,
    PromptListV1PromptGenerator.NAME: PromptListV1PromptGenerator,
}


def get_prompt_generator(name: str) -> PromptGeneratorBase:
    if name not in PROMPT_GENERATOR_REGISTRY:
        raise ValueError(
            f"Unknown prompt generator {name!r}. Supported generators: {sorted(PROMPT_GENERATOR_REGISTRY)}"
        )
    return PROMPT_GENERATOR_REGISTRY[name]()


def preview_wildcard_template_samples(
    *,
    template: str,
    template_root: str | Path = WildcardTemplatePromptGenerator.DEFAULT_TEMPLATE_ROOT,
    wildcard_root: str | Path = WildcardTemplatePromptGenerator.DEFAULT_WILDCARD_ROOT,
    negative_prompt: str = WildcardTemplatePromptGenerator.DEFAULT_NEGATIVE_PROMPT,
    seed: int = 1,
    count: int = 3,
    max_expansion_depth: int = WildcardTemplatePromptGenerator.DEFAULT_MAX_EXPANSION_DEPTH,
) -> list[PromptBundle]:
    generator = WildcardTemplatePromptGenerator()
    bundles: list[PromptBundle] = []

    for sample_index in range(max(1, int(count))):
        sample_seed = int(seed) + sample_index
        bundle = generator.generate(
            {
                "seed_control": "global_seed",
                "template": str(template),
                "template_root": str(Path(template_root)),
                "wildcard_root": str(Path(wildcard_root)),
                "negative_prompt": str(negative_prompt),
                "max_expansion_depth": int(max_expansion_depth),
            },
            sample_seed,
        )
        bundles.append(bundle)

    return bundles


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview prompt generator outputs from the terminal.")
    subparsers = parser.add_subparsers(dest="command")

    preview_parser = subparsers.add_parser(
        "preview-wildcard-template",
        help="Sample a wildcard template several times and print the expanded prompts.",
    )
    preview_parser.add_argument("--template", required=True, help="Template identifier or filename under template_root.")
    preview_parser.add_argument(
        "--template-root",
        default=WildcardTemplatePromptGenerator.DEFAULT_TEMPLATE_ROOT,
        help="Template root directory. Defaults to ./prompt_templates.",
    )
    preview_parser.add_argument(
        "--wildcard-root",
        default=WildcardTemplatePromptGenerator.DEFAULT_WILDCARD_ROOT,
        help="Wildcard root directory. Defaults to ./wildcard.",
    )
    preview_parser.add_argument(
        "--negative-prompt",
        default=WildcardTemplatePromptGenerator.DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt template or literal negative prompt.",
    )
    preview_parser.add_argument("--seed", type=int, default=1, help="Base seed for the first preview sample.")
    preview_parser.add_argument("--count", type=int, default=3, help="How many preview samples to print.")
    preview_parser.add_argument(
        "--max-expansion-depth",
        type=int,
        default=WildcardTemplatePromptGenerator.DEFAULT_MAX_EXPANSION_DEPTH,
        help="Maximum recursive expansion depth for wildcard/template resolution.",
    )
    preview_parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print prompt metadata for each sample.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command != "preview-wildcard-template":
        parser.print_help()
        return

    bundles = preview_wildcard_template_samples(
        template=args.template,
        template_root=args.template_root,
        wildcard_root=args.wildcard_root,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        count=args.count,
        max_expansion_depth=args.max_expansion_depth,
    )

    for sample_index, bundle in enumerate(bundles, start=1):
        seed_used = int(args.seed) + sample_index - 1
        print(f"Sample {sample_index}")
        print(f"Seed: {seed_used}")
        print(f"Positive: {bundle.positive_prompt}")
        print(f"Negative: {bundle.negative_prompt}")
        if args.show_metadata:
            print("Metadata:")
            print(json.dumps(bundle.prompt_metadata, indent=2, ensure_ascii=False, sort_keys=True))
        if sample_index != len(bundles):
            print()


if __name__ == "__main__":
    main()
