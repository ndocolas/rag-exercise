from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.db.vector_store import QdrantVectorStore
from rag_eval.models.schemas import (
    EMBEDDER_TO_PIPELINE,
    AskJSONResponse,
    AskRequest,
    CompareEmbedderResult,
    CompareJSONResponse,
    CompareRequest,
    EmbedderAlias,
)
from rag_eval.services.benchmark.pipeline_matrix import PipelineMatrix, PipelineSpec
from rag_eval.services.embeddings.embedder import Embedder
from rag_eval.services.embeddings.fuelix_embedder import FuelixEmbedder
from rag_eval.services.embeddings.local_embedder import LocalEmbedder
from rag_eval.services.evaluation.llm_judge import LLMJudge
from rag_eval.services.generation.llm_client import FuelixLLMClient
from rag_eval.services.generation.rag_prompt import RAGPrompt
from rag_eval.services.pipeline import PipelineAnswer, RAGPipeline
from rag_eval.services.response import (
    AskRenderer,
    AskRenderInput,
    CompareRenderer,
    CompareRenderInput,
    EmbedderRun,
)
from rag_eval.services.retrieval.retriever import Retriever
from rag_eval.utils.settings import Settings

logger = structlog.get_logger(__name__)

_NO_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only your "
    "internal knowledge. Do not have access to any external documents."
)

_DIDACTIC_EMBEDDERS: tuple[EmbedderAlias, ...] = ("openai", "bge-small", "bge-large")


@dataclass(frozen=True)
class _Run:
    alias: EmbedderAlias
    spec: PipelineSpec
    answer: PipelineAnswer | None
    error: str | None


class QueryRouter:
    """Didactic RAG endpoints.

    - POST /ask     — single question, minimal JSON: question, answer,
                      retrieved_documents (text-only), and an optional
                      response_without_rag when `with_control=true`.
    - POST /compare — same question across the three didactic embedders;
                      JSON: question + per-embedder { answer,
                      retrieved_documents } or { error }.

    Default response is JSON. `?format=markdown` returns a clean markdown view
    rendered by ``services/response``.
    """

    def __init__(
        self,
        settings: Settings,
        store: QdrantVectorStore,
        embedding_cache: EmbeddingCache,
        llm: FuelixLLMClient,
        judge_llm: FuelixLLMClient,
    ):
        self._settings = settings
        self._store = store
        self._embedding_cache = embedding_cache
        self._llm = llm
        self._judge = LLMJudge(judge_llm)
        self._embedders: dict[str, Embedder] = {}
        self._prompt = RAGPrompt()
        self._ask_renderer = AskRenderer()
        self._compare_renderer = CompareRenderer()
        self._router = APIRouter(tags=["query"])
        self._register_routes()

    @property
    def router(self) -> APIRouter:
        return self._router

    def _register_routes(self) -> None:
        @self._router.post("/ask")
        async def ask(req: AskRequest, format: str = Query(default="json")):
            return await self._handle_ask(req, format)

        @self._router.post("/compare")
        async def compare(req: CompareRequest, format: str = Query(default="json")):
            return await self._handle_compare(req, format)

    async def _handle_ask(self, req: AskRequest, format: str):
        spec = self._spec_for_alias(req.embedder)
        await self._require_indexed(spec, embedder_alias=req.embedder)

        rag_task = asyncio.create_task(
            self._run_rag(spec, req.question, req.top_k, use_cache=False)
        )
        no_rag_task: asyncio.Task[str] | None = None
        if req.with_control:
            no_rag_task = asyncio.create_task(self._run_no_rag(req.question, use_cache=False))

        rag_result = await rag_task
        answer_no_rag: str | None = await no_rag_task if no_rag_task is not None else None

        chunks_texts = [r.chunk.text for r in rag_result.retrieved]

        judge_verdict: str | None = None
        if answer_no_rag is not None:
            judge_verdict = await self._safe_judge(
                self._judge.compare(
                    question=req.question,
                    answer_with_rag=rag_result.answer,
                    answer_without_rag=answer_no_rag,
                    contexts=chunks_texts,
                    use_cache=False,
                )
            )

        if format == "markdown":
            render_input = AskRenderInput(
                question=req.question,
                answer_with_rag=rag_result.answer,
                chunks_texts=chunks_texts,
                answer_no_rag=answer_no_rag,
                judge_verdict=judge_verdict,
            )
            return PlainTextResponse(self._ask_renderer.render(render_input))

        return AskJSONResponse(
            question=req.question,
            response_with_rag=rag_result.answer,
            response_without_rag=answer_no_rag,
            llm_as_judge=judge_verdict,
            retrieved_documents={str(i): text for i, text in enumerate(chunks_texts, start=1)},
        )

    @staticmethod
    async def _safe_judge(coro: Awaitable[str]) -> str | None:
        """Run a judge call and swallow failures so the response still ships.

        Anything from a transport hiccup to an unexpected payload turns into
        ``None`` on the wire — we'd rather degrade the verdict field than
        500 the whole `/ask`.
        """
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_judge_failed", error=f"{type(exc).__name__}: {exc}")
            return None

    async def _handle_compare(self, req: CompareRequest, format: str):
        specs = [self._spec_for_alias(a) for a in _DIDACTIC_EMBEDDERS]
        runs = await asyncio.gather(
            *(
                self._run_one(alias, spec, req.question, req.top_k)
                for alias, spec in zip(_DIDACTIC_EMBEDDERS, specs, strict=True)
            )
        )

        if format == "markdown":
            embedder_runs = [
                EmbedderRun(
                    embedder_alias=r.alias,
                    chunks_texts=[rc.chunk.text for rc in r.answer.retrieved] if r.answer else [],
                    answer=r.answer.answer if r.answer else "",
                    error=r.error,
                )
                for r in runs
            ]
            render_input = CompareRenderInput(question=req.question, runs=embedder_runs)
            return PlainTextResponse(self._compare_renderer.render(render_input))

        results: dict[str, CompareEmbedderResult] = {}
        for r in runs:
            if r.answer is None:
                results[r.alias] = CompareEmbedderResult(error=r.error)
                continue
            results[r.alias] = CompareEmbedderResult(
                answer=r.answer.answer,
                retrieved_documents={
                    str(i): rc.chunk.text for i, rc in enumerate(r.answer.retrieved, start=1)
                },
            )
        return CompareJSONResponse(question=req.question, results=results)

    async def _run_rag(
        self,
        spec: PipelineSpec,
        question: str,
        top_k: int,
        *,
        use_cache: bool = True,
    ) -> PipelineAnswer:
        embedder = self._get_embedder(spec)
        retriever = Retriever(embedder, self._store, spec.collection_name(), top_k=top_k)
        pipeline = RAGPipeline(
            pipeline_id=spec.pipeline_id,
            embedder=embedder,
            retriever=retriever,
            llm=self._llm,
            prompt=self._prompt,
        )
        return await pipeline.answer(question, top_k=top_k, use_cache=use_cache)

    async def _run_no_rag(self, question: str, *, use_cache: bool = True) -> str:
        messages = [
            {"role": "system", "content": _NO_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return await self._llm.complete(messages, use_cache=use_cache)

    async def _run_one(
        self,
        alias: EmbedderAlias,
        spec: PipelineSpec,
        question: str,
        top_k: int,
    ) -> _Run:
        try:
            size = await self._store.collection_size(spec.collection_name())
            if size == 0:
                return _Run(
                    alias=alias,
                    spec=spec,
                    answer=None,
                    error="collection empty — run `POST /index` first",
                )
            answer = await self._run_rag(spec, question, top_k)
            return _Run(alias=alias, spec=spec, answer=answer, error=None)
        except Exception as exc:  # noqa: BLE001
            return _Run(
                alias=alias,
                spec=spec,
                answer=None,
                error=f"{type(exc).__name__}: {exc}",
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
