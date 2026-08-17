from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import uuid

from .catalog import CatalogSnapshot, PairRecord
from .common import LabelEventValidationError
from .labels import LabelEvent
from .common import stable_hex, utc_timestamp


@dataclass(frozen=True)
class ReviewSelection:
    review_id: str
    reviewer_username: str
    task_keys: tuple[str, ...]
    mode: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "reviewer_username": self.reviewer_username,
            "task_keys": list(self.task_keys),
            "mode": self.mode,
            "created_at": self.created_at,
        }


class ReviewService:
    def __init__(self) -> None:
        self._selections: dict[str, ReviewSelection] = {}

    def create_review_selection(
        self,
        snapshot: CatalogSnapshot,
        reviewer_username: str,
        task_keys: Sequence[str],
        mode: str,
    ) -> ReviewSelection:
        seen_task_keys: set[str] = set()
        cleaned_task_keys: list[str] = []
        for task_key in task_keys:
            if task_key not in snapshot.tasks_by_key or task_key in seen_task_keys:
                continue
            seen_task_keys.add(task_key)
            cleaned_task_keys.append(task_key)
        if not cleaned_task_keys:
            raise LabelEventValidationError("At least one valid task_key must be selected")
        if mode not in {"sequence", "random"}:
            raise LabelEventValidationError("mode must be 'sequence' or 'random'")
        selection = ReviewSelection(
            review_id=uuid.uuid4().hex,
            reviewer_username=reviewer_username,
            task_keys=tuple(cleaned_task_keys),
            mode=mode,
            created_at=utc_timestamp(),
        )
        self._selections[selection.review_id] = selection
        return selection

    def get_review_selection(self, review_id: str) -> ReviewSelection:
        if review_id not in self._selections:
            raise KeyError("Unknown review session")
        return self._selections[review_id]

    def build_queue(
        self,
        snapshot: CatalogSnapshot,
        selection: ReviewSelection,
        latest_labels_by_pair_key: Mapping[str, LabelEvent],
    ) -> list[dict[str, Any]]:
        task_order = {task_key: index for index, task_key in enumerate(selection.task_keys)}
        pairs = [
            pair
            for pair in snapshot.pairs_by_key.values()
            if pair.task_key in task_order and pair.pair_key not in latest_labels_by_pair_key
        ]

        if selection.mode == "sequence":
            pairs.sort(key=lambda pair: (task_order[pair.task_key], pair.session_index, pair.session_id))
        else:
            subset_hash = stable_hex(*selection.task_keys)
            pairs.sort(
                key=lambda pair: stable_hex(
                    snapshot.review_round_seed,
                    selection.reviewer_username,
                    subset_hash,
                    pair.pair_key,
                )
            )

        queue: list[dict[str, Any]] = []
        for pair in pairs:
            queue.append(
                {
                    "dataset_id": pair.dataset_id,
                    "dataset_display_name": pair.dataset_display_name,
                    "task_key": pair.task_key,
                    "task_name": pair.task_name,
                    "task_yaml_name": pair.task_yaml_name,
                    "session_id": pair.session_id,
                    "session_index": pair.session_index,
                    "workflow_name": pair.workflow_name,
                    "primary_ckpt": pair.primary_ckpt,
                    "is_labeled": False,
                    "latest_decision": None,
                    "latest_reviewer_username": None,
                }
            )
        return queue

    def validate_pair_in_selection(
        self,
        snapshot: CatalogSnapshot,
        selection: ReviewSelection,
        dataset_id: str,
        session_id: str,
    ) -> PairRecord:
        pair_key = f"{dataset_id}::{session_id}"
        pair = snapshot.pairs_by_key.get(pair_key)
        if pair is None or pair.task_key not in selection.task_keys:
            raise KeyError("Pair not available in this review session")
        return pair
