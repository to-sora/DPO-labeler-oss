import json
import sqlite3
import uuid

from .database import encode, read
from .manifest import text


def ensure_reviewers(db: sqlite3.Connection) -> None:
    db.execute("CREATE TABLE IF NOT EXISTS reviewers (username TEXT PRIMARY KEY, state TEXT NOT NULL)")
    db.execute("INSERT OR IGNORE INTO meta VALUES ('legacy_batch_limit', "
               "(SELECT COALESCE(MAX(rowid), 0) FROM batches))")


def review_state(db: sqlite3.Connection, reviewer_username: str) -> dict:
    username = text(reviewer_username, "Reviewer")
    ensure_reviewers(db)
    row = db.execute("SELECT state FROM reviewers WHERE username=?", (username,)).fetchone()
    if row is not None:
        return json.loads(row[0])
    state = read(db, "state")
    if state["pending"] is not None:
        state["pending"]["comparison_id"] = uuid.uuid4().hex
    return state


def save_review_state(db: sqlite3.Connection, reviewer_username: str, state: dict) -> None:
    db.execute("INSERT OR REPLACE INTO reviewers VALUES (?, ?)",
               (text(reviewer_username, "Reviewer"), encode(state)))
