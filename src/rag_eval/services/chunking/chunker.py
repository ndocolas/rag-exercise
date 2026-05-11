from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

from rag_eval.services.data.fiqa_dataset import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)


class Chunker(ABC):
    """Abstract chunker. Concrete subclasses split documents into chunks.

    Implementations must produce stable, deterministic chunk_ids so that
    embedding caches hit reliably across runs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in pipeline IDs and Qdrant collection names."""

    @abstractmethod
    def chunk(self, documents: Iterable[Document]) -> list[Chunk]: ...

    def _chunk_id(self, doc_id: str, idx: int, parent_idx: int | None = None) -> str:
        if parent_idx is not None:
            return f"{doc_id}::p{parent_idx}::c{idx}"
        return f"{doc_id}::c{idx}"
