import re
import shutil
import sqlite3
import uuid
from pathlib import Path

from .database import initialize, read, task_path, transaction, write
from .export import bundle
from .manifest import parse_manifest
from .state import new_state, prepare_pair, public_state
from .submit import accept


class ImportedTasks:
    def __init__(self, state_dir: Path) -> None:
        self.root = Path(state_dir) / "imported"
        self.root.mkdir(parents=True, exist_ok=True)

    def import_yaml(self, source: str) -> dict:
        snapshot = parse_manifest(source)
        slug = re.sub(r"[^\w-]+", "-", snapshot["character_name"]).strip("-")[:32]
        task_id = f"{slug or 'character'}-{uuid.uuid4().hex}"
        directory = self.root / task_id
        directory.mkdir()
        snapshot.update(task_id=task_id, output_dir=str(directory.resolve()))
        try:
            initialize(directory / "task.sqlite3", snapshot, source, new_state(snapshot))
        except BaseException:
            shutil.rmtree(directory)
            raise
        return self.get(task_id)

    def catalog(self) -> dict:
        tasks, warnings = [], []
        for path in sorted(self.root.glob("*/task.sqlite3")):
            try:
                with transaction(path) as db:
                    snapshot, state = read(db, "snapshot"), read(db, "state")
                tasks.append({k: v for k, v in snapshot.items() if k != "images"} | {
                    "image_count": len(snapshot["images"]),
                    "comparisons": state["comparisons"],
                    "complete": all(r["certified"] for r in state["rankings"])})
            except (ValueError, OSError, sqlite3.DatabaseError) as exc:
                warnings.append(f"{path.parent.name}: {exc}")
        return {"tasks": tasks, "warnings": warnings}

    def get(self, task_id: str) -> dict:
        with transaction(task_path(self.root, task_id)) as db:
            snapshot, state = read(db, "snapshot"), read(db, "state")
            prepare_pair(state)
            write(db, "state", state)
            return public_state(snapshot, state)

    def submit(self, task_id: str, payload: dict) -> dict:
        with transaction(task_path(self.root, task_id)) as db:
            return accept(db, payload)

    def export(self, task_id: str, dimension: int | None = None) -> bytes:
        path = task_path(self.root, task_id)
        with transaction(path) as db:
            return bundle(db, path.parent / "exports", dimension).read_bytes()

    def image(self, task_id: str, index: int) -> Path:
        with transaction(task_path(self.root, task_id)) as db:
            images = read(db, "snapshot")["images"]
        if index not in range(len(images)):
            raise KeyError(index)
        return Path(images[index]["image_path"])
