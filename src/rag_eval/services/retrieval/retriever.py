from __future__ import annotations

from dataclasses import dataclass

from rag_eval.db.vector_store import QdrantVectorStore, RetrievedChunk
from rag_eval.services.embeddings.embedder import Embedder


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    doc_ids_unique: list[str]


class Retriever:
    """Pairs an embedder with a Qdrant collection. Returns top-k chunks plus
    the unique doc-id ranking (first occurrence wins) used by retrieval metrics.
    """

    def __init__(
        self, embedder: Embedder, store: QdrantVectorStore, collection: str, top_k: int = 10
    ):
        self._embedder = embedder
        self._store = store
        self._collection = collection
        self._top_k = top_k

    @property
    def collection(self) -> str:
        return self._collection

    async def retrieve(self, query_text: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k or self._top_k
        vectors = await self._embedder.embed([query_text])
        chunks = await self._store.search(self._collection, vectors[0], top_k=k)
        return RetrievalResult(chunks=chunks, doc_ids_unique=self._unique_docs(chunks))

    @staticmethod
    def _unique_docs(retrieved: list[RetrievedChunk]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for r in retrieved:
            if r.chunk.doc_id and r.chunk.doc_id not in seen:
                seen.add(r.chunk.doc_id)
                ordered.append(r.chunk.doc_id)
        return ordered
