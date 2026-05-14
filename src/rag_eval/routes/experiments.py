from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException
from ulid import ULID

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.db.experiment_store import ExperimentStore
from rag_eval.db.results_store import ResultsStore
from rag_eval.db.vector_store import QdrantVectorStore
from rag_eval.models.schemas import (
    BenchmarkRanking,
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkSummary,
    BenchmarkWinner,
    ExperimentStatusResponse,
    FlowPhase,
    LatencyEntry,
    PipelineInfo,
    PipelineScore,
    RunExperimentRequest,
    RunExperimentResponse,
)
from rag_eval.services.benchmark.benchmark_runner import BenchmarkRunner, RunOptions
from rag_eval.services.benchmark.digest import build_digest
from rag_eval.services.benchmark.pipeline_matrix import PipelineMatrix, PipelineSpec
from rag_eval.services.benchmark.report_generator import ReportGenerator
from rag_eval.services.data.fiqa_dataset import FiQADataset
from rag_eval.services.generation.llm_client import FuelixLLMClient
from rag_eval.utils.settings import Settings


class ExperimentsRouter:
    """REST surface for experiment orchestration. Owns the FastAPI router."""

    def __init__(
        self,
        settings: Settings,
        store: QdrantVectorStore,
        embedding_cache: EmbeddingCache,
        experiment_store: ExperimentStore,
        llm: FuelixLLMClient,
    ):
        self._settings = settings
        self._store = store
        self._embedding_cache = embedding_cache
        self._experiment_store = experiment_store
        self._llm = llm
        self._tasks: dict[str, asyncio.Task] = {}
        self._router = APIRouter(prefix="/experiments", tags=["experiments"])
        self._benchmark_router = APIRouter(tags=["benchmark"])
        self._register_routes()
        self._register_benchmark_route()

    @property
    def router(self) -> APIRouter:
        return self._router

    @property
    def benchmark_router(self) -> APIRouter:
        return self._benchmark_router

    def _register_routes(self) -> None:
        @self._router.post("/run", response_model=RunExperimentResponse, status_code=202)
        async def run_experiment(req: RunExperimentRequest) -> RunExperimentResponse:
            specs = PipelineMatrix.build(filter_ids=req.pipeline_filter)
            if not specs:
                raise HTTPException(400, "No pipelines matched filter")

            experiment_id = f"exp_{ULID()}"
            config = req.model_dump()
            self._experiment_store.create(experiment_id, name=req.name, config=config)

            queries_count = req.queries_limit or 0
            response = RunExperimentResponse(
                experiment_id=experiment_id,
                status="queued",
                pipelines_count=len(specs),
                queries_count=queries_count,
            )

            self._tasks[experiment_id] = asyncio.create_task(
                self._execute(experiment_id, req, specs)
            )
            return response

        @self._router.get("/{experiment_id}", response_model=ExperimentStatusResponse)
        async def get_experiment(experiment_id: str) -> ExperimentStatusResponse:
            record = self._experiment_store.get(experiment_id)
            if record is None:
                raise HTTPException(404, "Experiment not found")
            return ExperimentStatusResponse(
                experiment_id=record.experiment_id,
                name=record.name,
                status=record.status,
                progress=record.progress,
                started_at=record.started_at,
                updated_at=record.updated_at,
                error=record.error,
            )

        @self._router.get("/{experiment_id}/report")
        async def get_report(experiment_id: str, format: str = "json"):
            record = self._experiment_store.get(experiment_id)
            if record is None:
                raise HTTPException(404, "Experiment not found")
            results_store = ResultsStore(self._settings.results_dir, experiment_id)
            generator = ReportGenerator(results_store, seed=self._settings.seed)
            report = generator.generate()
            if format == "markdown":
                from fastapi.responses import PlainTextResponse

                return PlainTextResponse(report.markdown)
            return report.json_payload

        @self._router.get("", response_model=list[PipelineInfo])
        async def list_pipelines() -> list[PipelineInfo]:
            specs: list[PipelineSpec] = PipelineMatrix.build()
            return [
                PipelineInfo(
                    pipeline_id=s.pipeline_id,
                    chunking=s.chunking,
                    embedder=s.embedder,
                    collection=s.collection_name(),
                )
                for s in specs
            ]

    async def _execute(
        self, experiment_id: str, req: RunExperimentRequest, specs: list[PipelineSpec]
    ) -> None:
        dataset = FiQADataset(
            data_dir=self._settings.fiqa_data_dir,
            subsample_size=req.subsample_size or self._settings.subsample_size,
            seed=self._settings.seed,
        )
        results_store = ResultsStore(self._settings.results_dir, experiment_id)
        runner = BenchmarkRunner(
            settings=self._settings,
            specs=specs,
            dataset=dataset,
            store=self._store,
            embedding_cache=self._embedding_cache,
            results_store=results_store,
            experiment_store=self._experiment_store,
            llm=self._llm,
        )
        options = RunOptions(
            experiment_id=experiment_id,
            name=req.name,
            pipeline_filter=req.pipeline_filter,
            subsample_size=req.subsample_size,
            queries_limit=req.queries_limit,
            top_k=req.top_k,
            judge_model=req.judge_model,
            force_reindex=req.force_reindex,
            force_regen=req.force_regen,
            skip_llm_judges=req.skip_llm_judges,
        )
        try:
            await runner.run(options)
            ReportGenerator(results_store, seed=self._settings.seed).generate()
        finally:
            self._tasks.pop(experiment_id, None)

    def _register_benchmark_route(self) -> None:
        """One-shot synchronous benchmark over all 9 pipelines.

        Unlike ``POST /experiments/run``, this awaits the runner inline and
        returns the digest + markdown report in the same HTTP response, so a
        `.rest` client can read results without polling.
        """

        @self._benchmark_router.post("/benchmark", response_model=BenchmarkResponse)
        async def run_benchmark(req: BenchmarkRequest) -> BenchmarkResponse:
            specs = PipelineMatrix.build(filter_ids=req.pipeline_filter)
            if not specs:
                raise HTTPException(400, "No pipelines matched pipeline_filter")
            experiment_id = f"exp_{ULID()}"
            config = req.model_dump()
            self._experiment_store.create(experiment_id, name=req.name, config=config)

            dataset = FiQADataset(
                data_dir=self._settings.fiqa_data_dir,
                subsample_size=req.subsample_size,
                seed=self._settings.seed,
            )
            results_store = ResultsStore(self._settings.results_dir, experiment_id)
            runner = BenchmarkRunner(
                settings=self._settings,
                specs=specs,
                dataset=dataset,
                store=self._store,
                embedding_cache=self._embedding_cache,
                results_store=results_store,
                experiment_store=self._experiment_store,
                llm=self._llm,
            )
            options = RunOptions(
                experiment_id=experiment_id,
                name=req.name,
                pipeline_filter=req.pipeline_filter,
                subsample_size=req.subsample_size,
                queries_limit=req.queries_limit,
                top_k=req.top_k,
                judge_model=None,
                force_reindex=req.force_reindex,
                force_regen=req.force_regen,
                skip_llm_judges=req.skip_llm_judges,
            )

            t0 = time.monotonic()
            try:
                await runner.run(options)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail={"experiment_id": experiment_id, "error": str(exc)},
                ) from exc

            report = ReportGenerator(results_store, seed=self._settings.seed).generate()
            digest = build_digest(
                results_store.read_aggregate(), results_store.read_per_query()
            )
            duration = time.monotonic() - t0

            winner = (
                BenchmarkWinner(**digest.summary.winner.__dict__)
                if digest.summary.winner is not None
                else None
            )
            summary = BenchmarkSummary(
                winner=winner,
                best_per_chunking={
                    k: PipelineScore(**v.__dict__)
                    for k, v in digest.summary.best_per_chunking.items()
                },
                best_per_embedder={
                    k: PipelineScore(**v.__dict__)
                    for k, v in digest.summary.best_per_embedder.items()
                },
                fastest=(
                    LatencyEntry(**digest.summary.fastest.__dict__)
                    if digest.summary.fastest is not None
                    else None
                ),
                slowest=(
                    LatencyEntry(**digest.summary.slowest.__dict__)
                    if digest.summary.slowest is not None
                    else None
                ),
                takeaways=digest.summary.takeaways,
            )
            ranking = [BenchmarkRanking(**r.__dict__) for r in digest.ranking]

            flow = [
                FlowPhase(
                    phase=ev.phase,
                    detail=ev.detail,
                    duration_ms=round(ev.duration_ms, 2),
                    data=ev.data,
                )
                for ev in runner.flow
            ]
            return BenchmarkResponse(
                experiment_id=experiment_id,
                status="completed",
                duration_seconds=round(duration, 2),
                pipelines_run=[s.pipeline_id for s in specs],
                flow=flow,
                summary=summary,
                ranking=ranking,
                report_markdown=report.markdown,
            )
