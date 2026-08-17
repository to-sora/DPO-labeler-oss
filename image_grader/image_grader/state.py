from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CachedScore:
    model_id: str
    config_hash: str
    ok: bool
    score: float | None
    scale: str
    raw: dict[str, Any]
    error: str | None = None

    def to_output(self, *, cached: bool = True) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "score": self.score,
            "scale": self.scale,
            "raw": self.raw,
            "error": self.error,
            "cached": cached,
        }


class ScoreState:
    def __init__(self, state_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "scores.sqlite3"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def get_score(self, image_id: str, model_id: str, config_hash: str) -> CachedScore | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT model_id, config_hash, ok, score, scale, raw_json, error
                FROM scores
                WHERE image_id = ? AND model_id = ? AND config_hash = ?
                """,
                (image_id, model_id, config_hash),
            ).fetchone()
        if row is None:
            return None
        return CachedScore(
            model_id=str(row["model_id"]),
            config_hash=str(row["config_hash"]),
            ok=bool(row["ok"]),
            score=float(row["score"]) if row["score"] is not None else None,
            scale=str(row["scale"]),
            raw=json.loads(str(row["raw_json"] or "{}")),
            error=str(row["error"]) if row["error"] is not None else None,
        )

    def put_score(
        self,
        *,
        image_id: str,
        model_id: str,
        config_hash: str,
        image_path: str,
        size_bytes: int,
        width: int,
        height: int,
        score: Mapping[str, Any],
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO scores (
                    image_id, model_id, config_hash, image_path, size_bytes, width, height,
                    ok, score, scale, raw_json, error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id, model_id, config_hash) DO UPDATE SET
                    image_path = excluded.image_path,
                    size_bytes = excluded.size_bytes,
                    width = excluded.width,
                    height = excluded.height,
                    ok = excluded.ok,
                    score = excluded.score,
                    scale = excluded.scale,
                    raw_json = excluded.raw_json,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    image_id,
                    model_id,
                    config_hash,
                    image_path,
                    int(size_bytes),
                    int(width),
                    int(height),
                    1 if bool(score.get("ok")) else 0,
                    float(score["score"]) if score.get("score") is not None else None,
                    str(score.get("scale", "0_10")),
                    json.dumps(score.get("raw", {}), ensure_ascii=False, sort_keys=True),
                    str(score["error"]) if score.get("error") is not None else None,
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    image_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    ok INTEGER NOT NULL,
                    score REAL,
                    scale TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (image_id, model_id, config_hash)
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_scores_model ON scores(model_id, config_hash)")
            self._conn.commit()
