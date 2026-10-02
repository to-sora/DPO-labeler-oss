import json
import sqlite3
import os
import uuid
import zipfile
from pathlib import Path

from .database import encode, read


def dpo_record(event: dict, images: list[dict]) -> dict:
    a, b = event["image_ids"]
    chosen = event["chosen_image_indices"][0]
    return {**event, "strict_dpo": False, "chosen_index": chosen,
            "rejected_index": 1 - chosen,
            "chosen": {**images[(a, b)[chosen]], "image_index": chosen},
            "rejected": {**images[(a, b)[1 - chosen]], "image_index": 1 - chosen}}


def bundle(db: sqlite3.Connection, directory: Path, dimension: int | None = None) -> Path:
    snapshot, state = read(db, "snapshot"), read(db, "state")
    dims = snapshot["dimensions"]
    if dimension is not None and dimension not in range(len(dims)):
        raise ValueError("Unknown dimension")
    selected = range(len(dims)) if dimension is None else [dimension]
    events = [event for row in db.execute("SELECT events FROM batches ORDER BY rowid")
              for event in json.loads(row[0])]
    directory.mkdir(exist_ok=True)
    target = directory / ("all-dimensions.zip" if dimension is None else f"dimension-{dimension}.zip")
    temporary = directory / f"{uuid.uuid4().hex}.tmp"
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("import.yaml", read(db, "source_yaml"))
            archive.writestr("task.json", encode({**snapshot,
                "comparisons": state["comparisons"], "exposure": state["exposure"]}))
            for d in selected:
                prefix = f"dimension-{d}/"
                labels = [e for e in events if e["dimension"] == dims[d]]
                ranking = {**state["rankings"][d], "dimension": dims[d],
                           "dataset_id": f"{snapshot['task_id']}-dim-{d}",
                           "images": snapshot["images"]}
                archive.writestr(prefix + "ranking.json", encode(ranking))
                archive.writestr(prefix + "label_events.jsonl",
                                 "".join(encode(e) + "\n" for e in labels))
                archive.writestr(prefix + "dpo_pairs.jsonl", "".join(
                    encode(dpo_record(e, snapshot["images"])) + "\n"
                    for e in labels if not e["inferred"]))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
