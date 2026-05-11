from __future__ import annotations

import hashlib
import sqlite3
import struct
from pathlib import Path
from threading import Lock


class EmbeddingCache:
    """SQLite-backed cache keyed by (model, sha256(text)).

    Vectors stored as little-endian float32 blobs. Synchronous (sqlite3 is
    fast enough; lock guards multi-coroutine access via ``asyncio.to_thread``).
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                PRIMARY KEY (model, text_hash)
            )
            """
        )
        self._conn.commit()

    def get_many(self, model: str, texts: list[str]) -> dict[int, list[float]]:
        if not texts:
            return {}
        hashes = [self._hash(t) for t in texts]
        placeholders = ",".join("?" * len(hashes))
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT text_hash, vector, dim FROM embeddings "
                f"WHERE model = ? AND text_hash IN ({placeholders})",
                [model, *hashes],
            )
            rows = {h: (b, d) for h, b, d in cursor.fetchall()}

        result: dict[int, list[float]] = {}
        for idx, h in enumerate(hashes):
            row = rows.get(h)
            if row is None:
                continue
            blob, dim = row
            result[idx] = list(struct.unpack(f"<{dim}f", blob))
        return result

    def put_many(self, model: str, texts: list[str], vectors: list[list[float]]) -> None:
        if not texts:
            return
        rows = []
        for text, vec in zip(texts, vectors, strict=True):
            blob = struct.pack(f"<{len(vec)}f", *vec)
            rows.append((model, self._hash(text), blob, len(vec)))
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings(model, text_hash, vector, dim) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
