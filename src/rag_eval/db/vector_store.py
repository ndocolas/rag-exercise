from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from rag_eval.services.chunking.chunker import Chunk

_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class QdrantVectorStore:
    """Async wrapper around AsyncQdrantClient.

    One collection per pipeline. Chunks become Qdrant points whose UUIDs are
    derived deterministically from ``chunk_id`` so re-indexing is idempotent.
    """

    def __init__(self, url: str = "http://localhost:6333"):
        self._client = AsyncQdrantClient(url=url)

    async def ensure_collection(self, collection: str, dim: int, recreate: bool = False) -> None:
        exists = await self._client.collection_exists(collection)
        if exists and recreate:
            await self._client.delete_collection(collection)
            exists = False
        if not exists:
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def upsert(
        self, collection: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        points = [
            PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=vec,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "parent_id": chunk.parent_id,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        await self._upsert_batched(collection, points)

    async def _upsert_batched(
        self, collection: str, points: list[PointStruct], batch_size: int = 256
    ) -> None:
        for i in range(0, len(points), batch_size):
            await self._client.upsert(
                collection_name=collection,
                points=points[i : i + batch_size],
                wait=True,
            )

    async def search(
        self, collection: str, query_vector: list[float], top_k: int = 10
    ) -> list[RetrievedChunk]:
        response = await self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = Chunk(
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                parent_id=payload.get("parent_id"),
                metadata=payload.get("metadata") or {},
            )
            results.append(RetrievedChunk(chunk=chunk, score=float(point.score)))
        return results

    async def collection_size(self, collection: str) -> int:
        info = await self._client.count(collection_name=collection, exact=True)
        return int(info.count)

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))
