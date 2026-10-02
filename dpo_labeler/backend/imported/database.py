import json
import re
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@contextmanager
def transaction(path: Path) -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(path, timeout=30)
    try:
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def read(db: sqlite3.Connection, key: str) -> Any:
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        raise ValueError(f"Missing task state: {key}")
    return json.loads(row[0])


def write(db: sqlite3.Connection, key: str, value: object) -> None:
    db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, encode(value)))


def task_path(root: Path, task_id: str) -> Path:
    if not re.fullmatch(r"[\w-]+", task_id):
        raise KeyError(task_id)
    path = root / task_id / "task.sqlite3"
    if not path.is_file():
        raise KeyError(task_id)
    return path


def initialize(path: Path, snapshot: dict, source: str, state: dict) -> None:
    with transaction(path) as db:
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE batches (id TEXT PRIMARY KEY, request TEXT, events TEXT)")
        write(db, "snapshot", snapshot)
        write(db, "source_yaml", source)
        write(db, "state", state)
