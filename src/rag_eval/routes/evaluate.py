from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.db.vector_store import QdrantVectorStore
from rag_eval.models.schemas import (
    EMBEDDER_TO_PIPELINE,
    EmbedderAlias,
    EvaluateJSONResponse,
    EvaluateRequest,
)
from rag_eval.services.benchmark.pipeline_matrix import PipelineMatrix, PipelineSpec
from rag_eval.services.data.fiqa_dataset import FiQADataset
from rag_eval.services.embeddings.embedder import Embedder
from rag_eval.services.embeddings.fuelix_embedder import FuelixEmbedder
from rag_eval.services.embeddings.local_embedder import LocalEmbedder
from rag_eval.services.evaluation.retrieval_benchmark import RetrievalBenchmark
from rag_eval.services.response import EvaluateRenderer, EvaluateRenderInput
from rag_eval.services.retrieval.retriever import Retriever
from rag_eval.utils.settings import Settings

logger = structlog.get_logger(__name__)


class EvaluateRouter:
    """Didactic retrieval benchmark.

    - POST /evaluate — runs one embedder against a sample of FiQA
      queries, compares retrieved doc_ids to qrels, returns 6 plain
      metrics + a 0-100 score. No LLM calls.

    JSON by default. `?format=markdown` returns a small rendered table.
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
        self._dataset = FiQADataset(
            data_dir=settings.fiqa_data_dir,
            subsample_size=settings.subsample_size,
            seed=settings.seed,
        )
        self._benchmark = RetrievalBenchmark(self._dataset)
        self._embedders: dict[str, Embedder] = {}
        self._renderer = EvaluateRenderer()
        self._router = APIRouter(tags=["evaluate"])
        self._register_routes()

    @property
    def router(self) -> APIRouter:
        return self._router

    def _register_routes(self) -> None:
        @self._router.post("/evaluate")
        async def evaluate(req: EvaluateRequest, format: str = Query(default="json")):
            return await self._handle(req, format)

    async def _handle(self, req: EvaluateRequest, format: str):
        spec = self._spec_for_alias(req.embedder)
        await self._require_indexed(spec, embedder_alias=req.embedder)

        embedder = self._get_embedder(spec)
        retriever = Retriever(embedder, self._store, spec.collection_name(), top_k=req.top_k)

        result = await self._benchmark.run(retriever, top_k=req.top_k)

        if format == "markdown":
            render_input = EvaluateRenderInput(
                embedder=req.embedder,
                queries_avaliadas=result.queries_avaliadas,
                hit_rate=result.hit_rate,
                precision_at_10=result.precision_at_10,
                recall_at_10=result.recall_at_10,
                ndcg_at_10=result.ndcg_at_10,
                score=result.score,
            )
            return PlainTextResponse(self._renderer.render(render_input))

        return EvaluateJSONResponse(
            embedder=req.embedder,
            queries_avaliadas=result.queries_avaliadas,
            hit_rate=round(result.hit_rate, 4),
            precision_at_10=round(result.precision_at_10, 4),
            recall_at_10=round(result.recall_at_10, 4),
            ndcg_at_10=round(result.ndcg_at_10, 4),
            score=result.score,
        )

    async def _require_indexed(self, spec: PipelineSpec, *, embedder_alias: EmbedderAlias) -> None:
        collection = spec.collection_name()
        try:
            size = await self._store.collection_size(collection)
        except Exception:
            size = 0
        if size == 0:
            raise HTTPException(
                409,
                f"Collection `{collection}` (embedder `{embedder_alias}`) is not "
                f"indexed. Run `POST /index` to populate Qdrant.",
            )

    @staticmethod
    def _spec_for_alias(alias: EmbedderAlias) -> PipelineSpec:
        pipeline_id = EMBEDDER_TO_PIPELINE[alias]
        for spec in PipelineMatrix.build():
            if spec.pipeline_id == pipeline_id:
                return spec
        raise HTTPException(500, f"PipelineMatrix has no entry for {pipeline_id}")

    def _get_embedder(self, spec: PipelineSpec) -> Embedder:
        if spec.embedder in self._embedders:
            return self._embedders[spec.embedder]
        kind, _, _ = spec.embedder.partition(":")
        if kind == "fuelix":
            embedder: Embedder = FuelixEmbedder(
                api_key=self._settings.fuelix_api_key,
                base_url=self._settings.fuelix_base_url,
                cache=self._embedding_cache,
                **spec.embedder_kwargs,
            )
        else:
            embedder = LocalEmbedder(cache=self._embedding_cache, **spec.embedder_kwargs)
        self._embedders[spec.embedder] = embedder
        return embedder
