from datetime import datetime, timezone

from .manifest import text


class Conflict(ValueError):
    pass


def submission(payload: dict, reviewer_username: str) -> dict:
    choices = payload.get("choices")
    if not isinstance(choices, dict):
        raise ValueError("choices must contain dimension preferences")
    return {"comparison_id": text(payload.get("comparison_id"), "comparison_id"),
            "reviewer_username": text(reviewer_username, "Reviewer"),
            "client_instance_id": text(payload.get("client_instance_id"), "Client"),
            "choices": choices}


def make_events(snapshot: dict, pair: dict, request: dict,
                decisions: list[tuple[str, bool]]) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    events = []
    for d, (decision, inferred) in enumerate(decisions):
        chosen = 0 if decision == "a_good" else 1
        dataset_id = f"{snapshot['task_id']}-dim-{d}"
        events.append({
            "event_id": f"{pair['comparison_id']}-{d}", "created_at": now,
            "client_ts": now, "dataset_id": dataset_id,
            "session_id": pair["comparison_id"], "comparison_id": pair["comparison_id"],
            "task_key": dataset_id, "task_name": snapshot["task_name"],
            "task_yaml_name": "import.yaml", "workflow_name": "imported-images",
            "primary_ckpt": "", "reviewer_username": request["reviewer_username"],
            "client_instance_id": request["client_instance_id"],
            "review_id": snapshot["task_id"], "dimension": snapshot["dimensions"][d],
            "character_name": snapshot["character_name"], "decision": decision,
            "display_order": [0, 1], "chosen_image_indices": [chosen],
            "image_ids": pair["image_ids"], "chosen_image_id": pair["image_ids"][chosen],
            "defects_a": [], "defects_b": [], "defects_by_image_index": {"0": [], "1": []},
            "note": "", "inferred": inferred,
        })
    return events
