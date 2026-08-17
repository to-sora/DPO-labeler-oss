from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .catalog import CatalogSnapshot, PairRecord
from .common import EXPORT_FILENAMES
from .filters import FilterEngine
from .labels import LabelEvent, LabelStore


class ExportService:
    def __init__(self, filter_engine: FilterEngine) -> None:
        self.filter_engine = filter_engine

    def preview_count(
        self,
        export_type: str,
        filter_ast: Mapping[str, Any],
        snapshot: CatalogSnapshot,
        label_store: LabelStore,
    ) -> int:
        return sum(1 for _ in self.iter_rows(export_type, filter_ast, snapshot, label_store))

    def render_export(
        self,
        export_type: str,
        filter_ast: Mapping[str, Any],
        snapshot: CatalogSnapshot,
        label_store: LabelStore,
    ) -> str:
        lines = [json.dumps(row, ensure_ascii=False) for row in self.iter_rows(export_type, filter_ast, snapshot, label_store)]
        return ("\n".join(lines) + ("\n" if lines else ""))

    def iter_rows(
        self,
        export_type: str,
        filter_ast: Mapping[str, Any],
        snapshot: CatalogSnapshot,
        label_store: LabelStore,
    ) -> Iterable[dict[str, Any]]:
        if export_type == "label-events":
            for event in label_store.iter_events():
                row = event.to_dict()
                if self.filter_engine.matches(filter_ast, self._event_context(event)):
                    yield row
            return

        latest_rows = list(self._iter_latest_rows(snapshot, label_store))
        if export_type == "labels-latest":
            for row in latest_rows:
                if self.filter_engine.matches(filter_ast, self._pair_context(row)):
                    yield row
            return

        if export_type == "preference-pairs":
            for row in latest_rows:
                decision = row["label"]["decision"]
                if decision not in {"a_good", "b_good"}:
                    continue
                chosen_index = self._single_chosen_index(row["label"])
                if chosen_index is None:
                    continue
                if not self.filter_engine.matches(filter_ast, self._pair_context(row)):
                    continue
                rejected_index = self._other_image_index(row["images"], chosen_index)
                if rejected_index is None:
                    continue
                yield self._build_preference_record(row, chosen_index, rejected_index, strict_dpo=False)
            return

        if export_type == "dpo-pairs":
            for row in latest_rows:
                decision = row["label"]["decision"]
                if decision not in {"a_good", "b_good"}:
                    continue
                chosen_index = self._single_chosen_index(row["label"])
                if chosen_index is None:
                    continue
                if not self._is_prompt_compatible(row["images"]):
                    continue
                if not self.filter_engine.matches(filter_ast, self._pair_context(row)):
                    continue
                rejected_index = self._other_image_index(row["images"], chosen_index)
                if rejected_index is None:
                    continue
                yield self._build_preference_record(row, chosen_index, rejected_index, strict_dpo=True)
            return

        raise ValueError(f"Unsupported export type: {export_type}")

    def default_filename(self, export_type: str) -> str:
        return EXPORT_FILENAMES[export_type]

    def _iter_latest_rows(self, snapshot: CatalogSnapshot, label_store: LabelStore) -> Iterable[dict[str, Any]]:
        latest_by_pair_key = label_store.get_latest_by_pair_key()
        for pair in snapshot.pairs_by_key.values():
            latest = latest_by_pair_key.get(pair.pair_key)
            if latest is None:
                continue
            images = [image.to_dict() for image in sorted(pair.images, key=lambda image: int(image.image_index))]
            yield {
                "dataset_id": pair.dataset_id,
                "dataset_display_name": pair.dataset_display_name,
                "task_key": pair.task_key,
                "session_id": pair.session_id,
                "session_index": pair.session_index,
                "task_name": pair.task_name,
                "task_yaml_name": pair.task_yaml_name,
                "workflow_name": pair.workflow_name,
                "primary_ckpt": pair.primary_ckpt,
                "label": latest.to_dict(),
                "images": images,
            }

    def _build_preference_record(
        self,
        row: Mapping[str, Any],
        chosen_index: int,
        rejected_index: int,
        *,
        strict_dpo: bool,
    ) -> dict[str, Any]:
        images = row["images"]
        images_by_index = {int(image["image_index"]): image for image in images}
        chosen = images_by_index[chosen_index]
        rejected = images_by_index[rejected_index]
        return {
            "dataset_id": row["dataset_id"],
            "session_id": row["session_id"],
            "task_key": row["task_key"],
            "task_name": row["task_name"],
            "task_yaml_name": row["task_yaml_name"],
            "workflow_name": row["workflow_name"],
            "primary_ckpt": row["primary_ckpt"],
            "strict_dpo": strict_dpo,
            "decision": row["label"]["decision"],
            "reviewer_username": row["label"]["reviewer_username"],
            "chosen_index": chosen_index,
            "rejected_index": rejected_index,
            "shared_prompt": chosen["positive_prompt"] if self._is_prompt_compatible(images) else None,
            "prompt_a": images[0]["positive_prompt"],
            "prompt_b": images[1]["positive_prompt"],
            "negative_prompt_a": images[0]["negative_prompt"],
            "negative_prompt_b": images[1]["negative_prompt"],
            "chosen_image": chosen,
            "rejected_image": rejected,
            "label": row["label"],
        }

    @staticmethod
    def _single_chosen_index(label: Mapping[str, Any]) -> int | None:
        chosen = label.get("chosen_image_indices")
        if isinstance(chosen, list) and len(chosen) == 1:
            return int(chosen[0])
        return None

    @staticmethod
    def _other_image_index(images: list[Mapping[str, Any]], chosen_index: int) -> int | None:
        candidates = [int(image["image_index"]) for image in images if int(image["image_index"]) != chosen_index]
        if len(candidates) != 1:
            return None
        return candidates[0]

    @staticmethod
    def _is_prompt_compatible(images: list[Mapping[str, Any]]) -> bool:
        if len(images) != 2:
            return False
        return (
            images[0].get("positive_prompt") == images[1].get("positive_prompt")
            and images[0].get("negative_prompt") == images[1].get("negative_prompt")
        )

    @staticmethod
    def _event_context(event: LabelEvent) -> dict[str, Any]:
        return {
            "dataset_id": event.dataset_id,
            "task_key": event.task_key,
            "task_name": event.task_name,
            "task_yaml_name": event.task_yaml_name,
            "reviewer_username": event.reviewer_username,
            "decision": event.decision,
            "is_labeled": True,
            "label_created_at": event.created_at,
            "note": event.note,
            "defects_a": list(event.defects_a),
            "defects_b": list(event.defects_b),
        }

    @staticmethod
    def _pair_context(row: Mapping[str, Any]) -> dict[str, Any]:
        label = row["label"]
        return {
            "dataset_id": row["dataset_id"],
            "task_key": row["task_key"],
            "task_name": row["task_name"],
            "task_yaml_name": row["task_yaml_name"],
            "reviewer_username": label["reviewer_username"],
            "decision": label["decision"],
            "is_labeled": True,
            "label_created_at": label["created_at"],
            "note": label["note"],
            "defects_a": list(label["defects_a"]),
            "defects_b": list(label["defects_b"]),
        }
