from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class PromptSegment:
    text: str
    bold: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "bold": self.bold,
        }


class PromptBoldingService:
    def __init__(self, patterns: list[re.Pattern[str]]) -> None:
        self._patterns = patterns

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @classmethod
    def from_path(cls, path: str | Path | None) -> "PromptBoldingService":
        if path is None:
            return cls([])
        config_path = Path(path)
        if not config_path.exists():
            return cls([])

        patterns: list[re.Pattern[str]] = []
        with config_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    patterns.append(re.compile(line, re.IGNORECASE))
                except re.error as exc:
                    raise ValueError(f"Invalid regex in {config_path} line {line_number}: {exc}") from exc
        return cls(patterns)

    def build_segments(self, text: str) -> list[dict[str, object]]:
        prompt = str(text or "")
        if not prompt:
            return []
        ranges = self._merged_ranges(prompt)
        if not ranges:
            return [PromptSegment(text=prompt, bold=False).to_dict()]

        segments: list[dict[str, object]] = []
        cursor = 0
        for start, end in ranges:
            if cursor < start:
                segments.append(PromptSegment(text=prompt[cursor:start], bold=False).to_dict())
            segments.append(PromptSegment(text=prompt[start:end], bold=True).to_dict())
            cursor = end
        if cursor < len(prompt):
            segments.append(PromptSegment(text=prompt[cursor:], bold=False).to_dict())
        return segments

    def _merged_ranges(self, text: str) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for pattern in self._patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                if end <= start:
                    continue
                ranges.append((start, end))
        if not ranges:
            return []

        ranges.sort()
        merged: list[tuple[int, int]] = [ranges[0]]
        for start, end in ranges[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
                continue
            merged.append((start, end))
        return merged
