from __future__ import annotations

import asyncio
import time
from typing import Literal

import structlog
from fastapi import APIRouter

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.db.vector_store import QdrantVectorStore
from rag_eval.models.schemas import (
    EMBEDDER_LABELS,
    EMBEDDER_TO_PIPELINE,
    EmbedderAlias,
    IndexEmbedderResult,
    IndexResponse,
)
from rag_eval.services.benchmark.pipeline_matrix import PipelineMatrix, PipelineSpec
from rag_eval.services.chunking.fixed_chunker import FixedChunker
from rag_eval.services.data.fiqa_dataset import FiQADataset
from rag_eval.services.embeddings.embedder import Embedder
from rag_eval.services.embeddings.fuelix_embedder import FuelixEmbedder
from rag_eval.services.embeddings.local_embedder import LocalEmbedder
from rag_eval.utils.settings import Settings

logger = structlog.get_logger(__name__)

_DIDACTIC_EMBEDDERS: tuple[EmbedderAlias, ...] = ("openai", "bge-small", "bge-large")


class IndexRouter:
    """`POST /index` — populate Qdrant with the three didactic pipelines.

    Idempotent: skips collections that already have points. Replaces the
    friction of running `/benchmark` just to populate the vector store.

    Design choice: chunking is fixed (`fixed_512_64`) for all three so the only
    moving piece across embedders is the embedder itself — same premise as
    `/compare`.
    """

    def __init__(
        self,
        settings: Settings,
        store: QdrantVectorStore,
        embedding_cache: EmbeddingCache,
    ):
        self._settings = settings
        self._store = store
        self._embedding_cache = embedding_cache
        self._router = APIRouter(tags=["index"])
        self._register_routes()

    @property
    def router(self) -> APIRouter:
        return self._router

    def _register_routes(self) -> None:
        @self._router.post("/index", response_model=IndexResponse)
        async def index() -> IndexResponse:
            return await self._run()

    async def _run(self) -> IndexResponse:
        dataset = FiQADataset(
            data_dir=self._settings.fiqa_data_dir,
            subsample_size=self._settings.subsample_size,
            seed=self._settings.seed,
        )
        data = await dataset.load()
        corpus_size = len(data.corpus)

        chunker = FixedChunker(chunk_size=512, chunk_overlap=64)
        chunks_for_pipeline = await asyncio.to_thread(chunker.chunk, list(data.corpus.values()))

        results: list[IndexEmbedderResult] = []
        for alias in _DIDACTIC_EMBEDDERS:
            spec = self._spec_for_alias(alias)
            t0 = time.perf_counter()
            try:
                status, indexed = await self._index_one(spec, chunks_for_pipeline)
            except Exception as exc:  # noqa: BLE001
                logger.exception("index_failed", embedder=alias, error=str(exc))
                results.append(
                    IndexEmbedderResult(
                        embedder=alias,
                        embedder_label=EMBEDDER_LABELS[alias],
                        pipeline_id=spec.pipeline_id,
                        collection=spec.collection_name(),
                        status="failed",
                        chunks_indexed=0,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            results.append(
                IndexEmbedderResult(
                    embedder=alias,
                    embedder_label=EMBEDDER_LABELS[alias],
                    pipeline_id=spec.pipeline_id,
                    collection=spec.collection_name(),
                    status=status,
                    chunks_indexed=indexed,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            )
        return IndexResponse(corpus_size=corpus_size, embedders=results)

    async def _index_one(
        self, spec: PipelineSpec, chunks: list
    ) -> tuple[Literal["already_indexed", "indexed"], int]:
        embedder = self._make_embedder(spec)
        collection = spec.collection_name()
        await self._store.ensure_collection(collection, embedder.dim, recreate=False)
        existing = await self._store.collection_size(collection)
        if existing > 0:
            return "already_indexed", existing

        batch = 256
        for i in range(0, len(chunks), batch):
            piece = chunks[i : i + batch]
            vectors = await embedder.embed([c.text for c in piece])
            await self._store.upsert(collection, piece, vectors)
        size = await self._store.collection_size(collection)
        logger.info("indexed", pipeline=spec.pipeline_id, collection=collection, points=size)
        return "indexed", size

    @staticmethod
    def _spec_for_alias(alias: EmbedderAlias) -> PipelineSpec:
        pipeline_id = EMBEDDER_TO_PIPELINE[alias]
        for spec in PipelineMatrix.build():
            if spec.pipeline_id == pipeline_id:
                return spec
        raise RuntimeError(f"PipelineMatrix has no entry for {pipeline_id}")

    def _make_embedder(self, spec: PipelineSpec) -> Embedder:
        kind, _, _ = spec.embedder.partition(":")
        if kind == "fuelix":
            return FuelixEmbedder(
                api_key=self._settings.fuelix_api_key,
                base_url=self._settings.fuelix_base_url,
                cache=self._embedding_cache,
                concurrency=self._settings.fuelix_embed_concurrency,
                batch_size=self._settings.fuelix_embed_batch_size,
                **spec.embedder_kwargs,
            )
        if kind == "local":
            return LocalEmbedder(cache=self._embedding_cache, **spec.embedder_kwargs)
        raise ValueError(f"Unknown embedder kind: {kind}")
