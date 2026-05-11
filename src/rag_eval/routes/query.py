from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.db.vector_store import QdrantVectorStore
from rag_eval.models.schemas import QueryRequest, QueryResponse, RetrievedContext
from rag_eval.services.benchmark.pipeline_matrix import PipelineMatrix, PipelineSpec
from rag_eval.services.embeddings.embedder import Embedder
from rag_eval.services.embeddings.fuelix_embedder import FuelixEmbedder
from rag_eval.services.embeddings.local_embedder import LocalEmbedder
from rag_eval.services.generation.llm_client import FuelixLLMClient
from rag_eval.services.generation.rag_prompt import RAGPrompt
from rag_eval.services.pipeline import RAGPipeline
from rag_eval.services.retrieval.retriever import Retriever
from rag_eval.utils.settings import Settings


class QueryRouter:
    """ad-hoc /query endpoint. Reuses already-indexed Qdrant collections; if
    the requested pipeline has no collection yet, returns 409.
    """

    def __init__(
        self,
        settings: Settings,
        store: QdrantVectorStore,
        embedding_cache: EmbeddingCache,
        llm: FuelixLLMClient,
    ):
        self._settings = settings
        self._store = store
        self._embedding_cache = embedding_cache
        self._llm = llm
        self._embedders: dict[str, Embedder] = {}
        self._prompt = RAGPrompt()
        self._router = APIRouter(tags=["query"])
        self._register_routes()

    @property
    def router(self) -> APIRouter:
        return self._router

    def _register_routes(self) -> None:
        @self._router.post("/query", response_model=QueryResponse)
        async def query(req: QueryRequest) -> QueryResponse:
            specs = PipelineMatrix.build()
            spec = self._select(specs, req.pipeline_id)
            if spec is None:
                raise HTTPException(400, f"Unknown pipeline_id: {req.pipeline_id}")

            collection = spec.collection_name()
            try:
                size = await self._store.collection_size(collection)
            except Exception:
                size = 0
            if size == 0:
                raise HTTPException(
                    409,
                    f"Collection {collection} not indexed. Run /experiments/run first.",
                )

            embedder = self._get_embedder(spec)
            retriever = Retriever(embedder, self._store, collection, top_k=req.top_k)
            pipeline = RAGPipeline(
                pipeline_id=spec.pipeline_id,
                embedder=embedder,
                retriever=retriever,
                llm=self._llm,
                prompt=self._prompt,
            )
            answer = await pipeline.answer(req.question, top_k=req.top_k)
            contexts = None
            if req.include_contexts:
                contexts = [
                    RetrievedContext(
                        chunk_id=r.chunk.chunk_id,
                        doc_id=r.chunk.doc_id,
                        text=r.chunk.text,
                        score=r.score,
                    )
                    for r in answer.retrieved
                ]
            return QueryResponse(
                answer=answer.answer,
                pipeline_id=spec.pipeline_id,
                contexts=contexts,
                latency_ms=answer.latency_ms,
            )

    def _select(self, specs: list[PipelineSpec], pipeline_id: str | None) -> PipelineSpec | None:
        if pipeline_id:
            for s in specs:
                if s.pipeline_id == pipeline_id:
                    return s
            return None
        return specs[0] if specs else None

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
