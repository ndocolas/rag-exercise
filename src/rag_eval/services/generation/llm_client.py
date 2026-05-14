from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from threading import Lock

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class FuelixLLMClient:
    """Async OpenAI-compatible chat completion client for fuelix.ai with on-disk cache.

    Cache key = sha256(model + temperature + json(messages)) — guarantees that
    identical (pipeline, query) pairs hit cache regardless of where they were
    invoked from (benchmark, /query endpoint, judge calls).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.fuelix.ai/v1",
        cache_path: Path | None = None,
        concurrency: int = 4,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._semaphore = asyncio.Semaphore(concurrency)
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._cache_lock = Lock()
        self._conn: sqlite3.Connection | None = None
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(cache_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    answer TEXT NOT NULL,
                    usage TEXT
                )
                """
            )
            self._conn.commit()

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, messages: list[dict], *, use_cache: bool = True) -> str:
        """Run a chat completion. Set ``use_cache=False`` to force a fresh
        round-trip (skips both the read and the write).
        """
        if use_cache:
            key = self._key(messages)
            cached = self._get(key)
            if cached is not None:
                return cached
        answer = await self._call(messages)
        if use_cache:
            self._put(key, answer)
        return answer

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        reraise=True,
    )
    async def _call(self, messages: list[dict]) -> str:
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                    json={
                        "model": self._model,
                        "messages": messages,
                        "temperature": self._temperature,
                        "max_tokens": self._max_tokens,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

    def _key(self, messages: list[dict]) -> str:
        payload = json.dumps(
            {"model": self._model, "temp": self._temperature, "messages": messages},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get(self, key: str) -> str | None:
        if self._conn is None:
            return None
        with self._cache_lock:
            cursor = self._conn.execute("SELECT answer FROM llm_cache WHERE cache_key = ?", (key,))
            row = cursor.fetchone()
        return row[0] if row else None

    def _put(self, key: str, answer: str) -> None:
        if self._conn is None:
            return
        with self._cache_lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache(cache_key, answer) VALUES (?, ?)",
                (key, answer),
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            with self._cache_lock:
                self._conn.close()
                self._conn = None
