from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract embedder. Async first; subclasses also expose ``embed_sync``
    for callers that run inside ``asyncio.to_thread`` (e.g. semantic chunking).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in pipeline IDs, cache keys and Qdrant collections."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        return asyncio.run(self.embed(texts))
