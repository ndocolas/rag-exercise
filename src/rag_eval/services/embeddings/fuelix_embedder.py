from __future__ import annotations

import asyncio

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.services.embeddings.embedder import Embedder


class FuelixEmbedder(Embedder):
    """OpenAI-compatible embeddings via fuelix.ai. Batches inputs and caches results."""

    _MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.fuelix.ai/v1",
        cache: EmbeddingCache | None = None,
        concurrency: int = 8,
        batch_size: int = 100,
        timeout: float = 60.0,
    ):
        if model not in self._MODEL_DIMS:
            raise ValueError(f"Unsupported fuelix embedding model: {model}")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._cache = cache
        self._semaphore = asyncio.Semaphore(concurrency)
        self._batch_size = batch_size
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._MODEL_DIMS[self._model]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []

        if self._cache is not None:
            cached = self._cache.get_many(self._model, texts)
            for idx, vec in cached.items():
                out[idx] = vec
            misses = [(i, t) for i, t in enumerate(texts) if out[i] is None]
        else:
            misses = list(enumerate(texts))

        if misses:
            batches = [
                misses[i : i + self._batch_size] for i in range(0, len(misses), self._batch_size)
            ]
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                results = await asyncio.gather(
                    *(self._embed_batch(client, batch) for batch in batches)
                )
            new_texts: list[str] = []
            new_vecs: list[list[float]] = []
            for batch, vecs in zip(batches, results, strict=True):
                for (idx, text), vec in zip(batch, vecs, strict=True):
                    out[idx] = vec
                    new_texts.append(text)
                    new_vecs.append(vec)
            if self._cache is not None:
                self._cache.put_many(self._model, new_texts, new_vecs)

        return [v for v in out if v is not None]

    async def _embed_batch(
        self, client: httpx.AsyncClient, batch: list[tuple[int, str]]
    ) -> list[list[float]]:
        async with self._semaphore:
            return await self._call(client, [t for _, t in batch])

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        reraise=True,
    )
    async def _call(self, client: httpx.AsyncClient, inputs: list[str]) -> list[list[float]]:
        response = await client.post(
            f"{self._base_url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            json={"model": self._model, "input": inputs, "encoding_format": "float"},
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]
