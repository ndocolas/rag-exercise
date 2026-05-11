from __future__ import annotations

import asyncio
from threading import Lock

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.services.embeddings.embedder import Embedder


class LocalEmbedder(Embedder):
    """Local embedder via fastembed. Runs CPU-bound encode in a thread."""

    _MODEL_DIMS = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-m3": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }

    def __init__(self, model: str, cache: EmbeddingCache | None = None, batch_size: int = 64):
        if model not in self._MODEL_DIMS:
            raise ValueError(f"Unsupported local embedding model: {model}")
        self._model_name = model
        self._cache = cache
        self._batch_size = batch_size
        self._model = None
        self._lock = Lock()

    @property
    def name(self) -> str:
        return self._model_name.replace("/", "__")

    @property
    def dim(self) -> int:
        return self._MODEL_DIMS[self._model_name]

    def _ensure_model(self) -> None:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(model_name=self._model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []

        if self._cache is not None:
            cached = self._cache.get_many(self.name, texts)
            for idx, vec in cached.items():
                out[idx] = vec
            misses = [(i, t) for i, t in enumerate(texts) if out[i] is None]
        else:
            misses = list(enumerate(texts))

        if misses:
            miss_texts = [t for _, t in misses]
            new_vecs = await asyncio.to_thread(self._encode, miss_texts)
            for (idx, _), vec in zip(misses, new_vecs, strict=True):
                out[idx] = vec
            if self._cache is not None:
                self._cache.put_many(self.name, miss_texts, new_vecs)

        return [v for v in out if v is not None]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        assert self._model is not None
        gen = self._model.embed(texts, batch_size=self._batch_size)
        return [vec.tolist() for vec in gen]

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)
