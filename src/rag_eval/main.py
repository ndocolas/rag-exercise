from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.db.experiment_store import ExperimentStore
from rag_eval.db.vector_store import QdrantVectorStore
from rag_eval.models.schemas import HealthResponse
from rag_eval.routes.dataset import DatasetRouter
from rag_eval.routes.experiments import ExperimentsRouter
from rag_eval.routes.index import IndexRouter
from rag_eval.routes.query import QueryRouter
from rag_eval.services.generation.llm_client import FuelixLLMClient
from rag_eval.utils.settings import Settings, get_settings

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(exception_formatter=structlog.dev.plain_traceback),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class RAGEvalAPI:
    """Builds the FastAPI app, wires dependencies, registers routers, and
    handles startup/shutdown (orphan cleanup, resource teardown).
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._store: QdrantVectorStore | None = None
        self._embedding_cache: EmbeddingCache | None = None
        self._experiment_store: ExperimentStore | None = None
        self._llm: FuelixLLMClient | None = None
        self.app = self._build()

    def _build(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self._startup()
            try:
                yield
            finally:
                await self._shutdown()

        app = FastAPI(
            title="RAG Evaluation Framework",
            version="0.1.0",
            lifespan=lifespan,
        )

        @app.get("/health", response_model=HealthResponse)
        async def health() -> HealthResponse:
            qdrant_ok = await self._qdrant_reachable()
            return HealthResponse(
                status="ok" if qdrant_ok else "degraded",
                qdrant=qdrant_ok,
                fuelix_configured=bool(self._settings.fuelix_api_key),
            )

        return app

    async def _startup(self) -> None:
        s = self._settings
        self._store = QdrantVectorStore(url=s.qdrant_url)
        self._embedding_cache = EmbeddingCache(s.embedding_cache_path)
        self._experiment_store = ExperimentStore(s.experiment_store_path)
        orphans = self._experiment_store.mark_orphans_failed()
        if orphans:
            logger.warning("orphans_marked_failed", count=orphans)

        self._llm = FuelixLLMClient(
            api_key=s.fuelix_api_key,
            model=s.generator_model,
            base_url=s.fuelix_base_url,
            cache_path=s.llm_cache_path,
            concurrency=s.fuelix_llm_concurrency,
            temperature=s.generator_temperature,
            max_tokens=s.generator_max_tokens,
        )

        experiments_router = ExperimentsRouter(
            settings=s,
            store=self._store,
            embedding_cache=self._embedding_cache,
            experiment_store=self._experiment_store,
            llm=self._llm,
        )
        query_router = QueryRouter(
            settings=s,
            store=self._store,
            embedding_cache=self._embedding_cache,
            llm=self._llm,
        )
        index_router = IndexRouter(
            settings=s,
            store=self._store,
            embedding_cache=self._embedding_cache,
        )
        dataset_router = DatasetRouter(settings=s)
        self.app.include_router(experiments_router.router)
        self.app.include_router(experiments_router.benchmark_router)
        self.app.include_router(query_router.router)
        self.app.include_router(index_router.router)
        self.app.include_router(dataset_router.router)

    async def _shutdown(self) -> None:
        if self._store is not None:
            await self._store.close()
        if self._embedding_cache is not None:
            self._embedding_cache.close()
        if self._experiment_store is not None:
            self._experiment_store.close()
        if self._llm is not None:
            self._llm.close()

    async def _qdrant_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self._settings.qdrant_url}/")
                return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False


app = RAGEvalAPI().app
