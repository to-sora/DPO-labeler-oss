from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .catalog import PairRecord
from .common import APP_VERSION, DEFECT_TAGS, LabelEventValidationError, normalize_text, utc_timestamp


@dataclass(frozen=True)
class LabelEvent:
    event_id: str
    created_at: str
    client_ts: str
    app_version: str
    dataset_id: str
    session_id: str
    task_key: str
    task_name: str
    task_yaml_name: str
    workflow_name: str
    primary_ckpt: str
    reviewer_username: str
    client_instance_id: str
    review_id: str
    decision: str
    defects_a: tuple[str, ...]
    defects_b: tuple[str, ...]
    display_order: tuple[int, int]
    chosen_image_indices: tuple[int, ...]
    defects_by_image_index: dict[str, tuple[str, ...]]
    note: str

    @property
    def pair_key(self) -> str:
        return f"{self.dataset_id}::{self.session_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "created_at": self.created_at,
            "client_ts": self.client_ts,
            "app_version": self.app_version,
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "task_key": self.task_key,
            "task_name": self.task_name,
            "task_yaml_name": self.task_yaml_name,
            "workflow_name": self.workflow_name,
            "primary_ckpt": self.primary_ckpt,
            "reviewer_username": self.reviewer_username,
            "client_instance_id": self.client_instance_id,
            "review_id": self.review_id,
            "decision": self.decision,
            "defects_a": list(self.defects_a),
            "defects_b": list(self.defects_b),
            "display_order": list(self.display_order),
            "chosen_image_indices": list(self.chosen_image_indices),
            "defects_by_image_index": {
                str(image_index): list(defects)
                for image_index, defects in sorted(self.defects_by_image_index.items(), key=lambda item: int(item[0]))
            },
            "note": self.note,
        }


class LabelStore:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.label_events_path = self.state_dir / "label_events.jsonl"
        self._lock = threading.Lock()
        self._events: list[LabelEvent] = []
        self._latest_by_pair_key: dict[str, LabelEvent] = {}
        self._seen_event_ids: set[str] = set()
        self._version = 0
        self._load_existing_events()

    def get_latest_by_pair_key(self) -> dict[str, LabelEvent]:
        with self._lock:
            return dict(self._latest_by_pair_key)

    def get_latest_snapshot(self) -> tuple[int, dict[str, LabelEvent]]:
        with self._lock:
            return self._version, dict(self._latest_by_pair_key)

    def get_version(self) -> int:
        with self._lock:
            return self._version

    def get_latest_for_pair(self, pair_key: str) -> LabelEvent | None:
        with self._lock:
            return self._latest_by_pair_key.get(pair_key)

    def iter_events(self) -> list[LabelEvent]:
        with self._lock:
            return list(self._events)

    def store_event(
        self,
        pair: PairRecord,
        *,
        reviewer_username: str,
        client_instance_id: str,
        payload: Mapping[str, Any],
    ) -> LabelEvent:
        decision = normalize_text(payload.get("decision"))
        if decision not in {"a_good", "b_good", "both_good", "both_bad", "skip"}:
            raise LabelEventValidationError("decision must be one of: a_good, b_good, both_good, both_bad, skip")
        canonical_indices = tuple(sorted(int(image.image_index) for image in pair.images))
        display_order = self._normalize_display_order(payload.get("display_order"), canonical_indices)
        chosen_image_indices = self._normalize_chosen_image_indices(
            payload.get("chosen_image_indices"),
            decision,
            display_order,
            canonical_indices,
        )
        self._validate_decision_mapping(decision, chosen_image_indices, display_order, canonical_indices)
        defects_a = tuple(self._normalize_defects(payload.get("defects_a")))
        defects_b = tuple(self._normalize_defects(payload.get("defects_b")))
        defects_by_image_index = self._normalize_defects_by_image_index(
            payload.get("defects_by_image_index"),
            display_order,
            canonical_indices,
            defects_a,
            defects_b,
        )

        event = LabelEvent(
            event_id=normalize_text(payload.get("event_id")) or uuid.uuid4().hex,
            created_at=normalize_text(payload.get("created_at")) or utc_timestamp(),
            client_ts=normalize_text(payload.get("client_ts")) or utc_timestamp(),
            app_version=normalize_text(payload.get("app_version")) or APP_VERSION,
            dataset_id=pair.dataset_id,
            session_id=pair.session_id,
            task_key=pair.task_key,
            task_name=pair.task_name,
            task_yaml_name=pair.task_yaml_name,
            workflow_name=pair.workflow_name,
            primary_ckpt=pair.primary_ckpt,
            reviewer_username=reviewer_username,
            client_instance_id=client_instance_id,
            review_id=normalize_text(payload.get("review_id")),
            decision=decision,
            defects_a=defects_a,
            defects_b=defects_b,
            display_order=display_order,
            chosen_image_indices=chosen_image_indices,
            defects_by_image_index=defects_by_image_index,
            note=normalize_text(payload.get("note")),
        )

        with self._lock:
            if event.event_id in self._seen_event_ids:
                return self._latest_by_pair_key.get(event.pair_key, event)
            self.label_events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.label_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._seen_event_ids.add(event.event_id)
            self._events.append(event)
            self._latest_by_pair_key[event.pair_key] = event
            self._version += 1
        return event

    def _load_existing_events(self) -> None:
        if not self.label_events_path.exists():
            return
        for row in self._load_event_rows():
            display_order = self._coerce_display_order(row.get("display_order"))
            defects_a = tuple(self._normalize_defects(row.get("defects_a")))
            defects_b = tuple(self._normalize_defects(row.get("defects_b")))
            event = LabelEvent(
                event_id=normalize_text(row.get("event_id")),
                created_at=normalize_text(row.get("created_at")),
                client_ts=normalize_text(row.get("client_ts")),
                app_version=normalize_text(row.get("app_version")) or APP_VERSION,
                dataset_id=normalize_text(row.get("dataset_id")),
                session_id=normalize_text(row.get("session_id")),
                task_key=normalize_text(row.get("task_key")),
                task_name=normalize_text(row.get("task_name")),
                task_yaml_name=normalize_text(row.get("task_yaml_name")),
                workflow_name=normalize_text(row.get("workflow_name")),
                primary_ckpt=normalize_text(row.get("primary_ckpt")),
                reviewer_username=normalize_text(row.get("reviewer_username")),
                client_instance_id=normalize_text(row.get("client_instance_id")),
                review_id=normalize_text(row.get("review_id")),
                decision=normalize_text(row.get("decision")),
                defects_a=defects_a,
                defects_b=defects_b,
                display_order=display_order,
                chosen_image_indices=self._coerce_chosen_image_indices(
                    row.get("chosen_image_indices"),
                    normalize_text(row.get("decision")),
                    display_order,
                ),
                defects_by_image_index=self._coerce_defects_by_image_index(
                    row.get("defects_by_image_index"),
                    display_order,
                    defects_a,
                    defects_b,
                ),
                note=normalize_text(row.get("note")),
            )
            if not event.event_id or not event.dataset_id or not event.session_id:
                continue
            self._seen_event_ids.add(event.event_id)
            self._events.append(event)
            self._latest_by_pair_key[event.pair_key] = event
            self._version += 1

    def _load_event_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.label_events_path.open("r", encoding="utf-8") as handle:
            raw_lines = handle.readlines()

        non_empty_indices = [index for index, raw_line in enumerate(raw_lines) if raw_line.strip()]
        last_non_empty_index = non_empty_indices[-1] if non_empty_indices else None

        for line_number, raw_line in enumerate(raw_lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if (
                    last_non_empty_index is not None
                    and line_number - 1 == last_non_empty_index
                    and not raw_line.endswith("\n")
                ):
                    break
                raise ValueError(f"Invalid JSON on line {line_number} of {self.label_events_path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object JSONL line in {self.label_events_path} at line {line_number}")
            rows.append(row)
        return rows

    @staticmethod
    def _normalize_defects(value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise LabelEventValidationError("defects must be arrays")
        normalized: list[str] = []
        for defect in value:
            item = normalize_text(defect)
            if item not in DEFECT_TAGS:
                raise LabelEventValidationError(f"Unknown defect tag: {item}")
            if item not in normalized:
                normalized.append(item)
        return normalized

    @staticmethod
    def _normalize_display_order(value: Any, canonical_indices: tuple[int, int]) -> tuple[int, int]:
        if value is None:
            return canonical_indices
        if not isinstance(value, list) or len(value) != 2:
            raise LabelEventValidationError("display_order must be a two-item array")
        normalized = tuple(int(item) for item in value)
        if tuple(sorted(normalized)) != canonical_indices:
            raise LabelEventValidationError("display_order must reference the pair image indices")
        return normalized

    @staticmethod
    def _normalize_chosen_image_indices(
        value: Any,
        decision: str,
        display_order: tuple[int, int],
        canonical_indices: tuple[int, int],
    ) -> tuple[int, ...]:
        if value is None:
            return LabelStore._derive_chosen_image_indices(decision, display_order, canonical_indices)
        if not isinstance(value, list):
            raise LabelEventValidationError("chosen_image_indices must be an array")
        normalized: list[int] = []
        allowed = set(canonical_indices)
        for item in value:
            image_index = int(item)
            if image_index not in allowed:
                raise LabelEventValidationError("chosen_image_indices must reference the pair image indices")
            if image_index not in normalized:
                normalized.append(image_index)
        return tuple(normalized)

    @staticmethod
    def _derive_chosen_image_indices(
        decision: str,
        display_order: tuple[int, int],
        canonical_indices: tuple[int, int],
    ) -> tuple[int, ...]:
        if decision == "a_good":
            return (display_order[0],)
        if decision == "b_good":
            return (display_order[1],)
        if decision == "both_good":
            return canonical_indices
        return ()

    @classmethod
    def _validate_decision_mapping(
        cls,
        decision: str,
        chosen_image_indices: tuple[int, ...],
        display_order: tuple[int, int],
        canonical_indices: tuple[int, int],
    ) -> None:
        expected = cls._derive_chosen_image_indices(decision, display_order, canonical_indices)
        if decision in {"a_good", "b_good"} and chosen_image_indices != expected:
            raise LabelEventValidationError("chosen_image_indices must match the selected A/B decision")
        if decision == "both_good" and tuple(sorted(chosen_image_indices)) != canonical_indices:
            raise LabelEventValidationError("both_good must select both canonical image indices")
        if decision in {"both_bad", "skip"} and chosen_image_indices:
            raise LabelEventValidationError("both_bad and skip may not select any image indices")

    @classmethod
    def _normalize_defects_by_image_index(
        cls,
        value: Any,
        display_order: tuple[int, int],
        canonical_indices: tuple[int, int],
        defects_a: tuple[str, ...],
        defects_b: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        if value is None:
            return {
                str(display_order[0]): defects_a,
                str(display_order[1]): defects_b,
            }
        if not isinstance(value, Mapping):
            raise LabelEventValidationError("defects_by_image_index must be an object")
        allowed = {str(item) for item in canonical_indices}
        normalized: dict[str, tuple[str, ...]] = {str(item): tuple() for item in canonical_indices}
        for key, raw_value in value.items():
            image_index = str(int(key))
            if image_index not in allowed:
                raise LabelEventValidationError("defects_by_image_index must reference the pair image indices")
            normalized[image_index] = tuple(cls._normalize_defects(raw_value))
        return normalized

    @classmethod
    def _coerce_display_order(cls, value: Any) -> tuple[int, int]:
        if isinstance(value, list) and len(value) == 2:
            try:
                normalized = tuple(int(item) for item in value)
            except (TypeError, ValueError):
                normalized = (0, 1)
            if len(set(normalized)) == 2:
                return normalized
        return (0, 1)

    @classmethod
    def _coerce_chosen_image_indices(
        cls,
        value: Any,
        decision: str,
        display_order: tuple[int, int],
    ) -> tuple[int, ...]:
        if isinstance(value, list):
            normalized: list[int] = []
            for item in value:
                try:
                    image_index = int(item)
                except (TypeError, ValueError):
                    continue
                if image_index not in normalized:
                    normalized.append(image_index)
            return tuple(normalized)
        return cls._derive_chosen_image_indices(decision, display_order, tuple(sorted(display_order)))

    @classmethod
    def _coerce_defects_by_image_index(
        cls,
        value: Any,
        display_order: tuple[int, int],
        defects_a: tuple[str, ...],
        defects_b: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        if isinstance(value, Mapping):
            normalized: dict[str, tuple[str, ...]] = {}
            for key, raw_value in value.items():
                try:
                    image_index = str(int(key))
                except (TypeError, ValueError):
                    continue
                normalized[image_index] = tuple(cls._normalize_defects(raw_value))
            if normalized:
                return normalized
        return {
            str(display_order[0]): defects_a,
            str(display_order[1]): defects_b,
        }
