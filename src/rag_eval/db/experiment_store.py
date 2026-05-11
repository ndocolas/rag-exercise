from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock


@dataclass
class ExperimentRecord:
    experiment_id: str
    name: str
    status: str
    config: dict
    started_at: float
    updated_at: float
    progress: dict = field(default_factory=dict)
    error: str | None = None


class ExperimentStore:
    """SQLite registry of benchmark runs. Source of truth for /experiments/{id}.

    Background tasks update progress here; on app startup, runs left in
    ``running`` state are marked ``failed`` (orphaned by a previous crash).
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                config TEXT NOT NULL,
                started_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                progress TEXT NOT NULL,
                error TEXT
            )
            """
        )
        self._conn.commit()

    def create(self, experiment_id: str, name: str, config: dict) -> ExperimentRecord:
        now = time.time()
        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            status="queued",
            config=config,
            started_at=now,
            updated_at=now,
            progress={"phase": "queued"},
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO experiments(experiment_id, name, status, config, started_at, "
                "updated_at, progress, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.experiment_id,
                    record.name,
                    record.status,
                    json.dumps(record.config),
                    record.started_at,
                    record.updated_at,
                    json.dumps(record.progress),
                    None,
                ),
            )
            self._conn.commit()
        return record

    def update(
        self,
        experiment_id: str,
        status: str | None = None,
        progress: dict | None = None,
        error: str | None = None,
    ) -> None:
        sets = ["updated_at = ?"]
        params: list = [time.time()]
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if progress is not None:
            sets.append("progress = ?")
            params.append(json.dumps(progress))
        if error is not None:
            sets.append("error = ?")
            params.append(error)
        params.append(experiment_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE experiments SET {', '.join(sets)} WHERE experiment_id = ?", params
            )
            self._conn.commit()

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT experiment_id, name, status, config, started_at, updated_at, "
                "progress, error FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return ExperimentRecord(
            experiment_id=row[0],
            name=row[1],
            status=row[2],
            config=json.loads(row[3]),
            started_at=row[4],
            updated_at=row[5],
            progress=json.loads(row[6]),
            error=row[7],
        )

    def list_recent(self, limit: int = 50) -> list[ExperimentRecord]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT experiment_id, name, status, config, started_at, updated_at, "
                "progress, error FROM experiments ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [
            ExperimentRecord(
                experiment_id=r[0],
                name=r[1],
                status=r[2],
                config=json.loads(r[3]),
                started_at=r[4],
                updated_at=r[5],
                progress=json.loads(r[6]),
                error=r[7],
            )
            for r in rows
        ]

    def mark_orphans_failed(self) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE experiments SET status = 'failed', error = 'orphaned (process restart)' "
                "WHERE status IN ('running', 'queued')"
            )
            self._conn.commit()
            return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
