from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import structlog

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.db.experiment_store import ExperimentStore
from rag_eval.db.results_store import AggregateResult, QueryResult, ResultsStore
from rag_eval.db.vector_store import QdrantVectorStore
from rag_eval.services.benchmark.pipeline_matrix import PipelineSpec
from rag_eval.services.chunking.chunker import Chunker
from rag_eval.services.chunking.fixed_chunker import FixedChunker
from rag_eval.services.chunking.hierarchical_chunker import HierarchicalChunker
from rag_eval.services.chunking.semantic_chunker import SemanticChunker
from rag_eval.services.data.fiqa_dataset import FiQAData, FiQADataset
from rag_eval.services.embeddings.embedder import Embedder
from rag_eval.services.embeddings.fuelix_embedder import FuelixEmbedder
from rag_eval.services.embeddings.local_embedder import LocalEmbedder
from rag_eval.services.evaluation.deepeval_evaluator import DeepEvalEvaluator
from rag_eval.services.evaluation.failure_analyzer import FailureAnalyzer
from rag_eval.services.evaluation.ragas_evaluator import RagasEvaluator
from rag_eval.services.evaluation.retrieval_evaluator import RetrievalEvaluator
from rag_eval.services.evaluation.statistics_calculator import StatisticsCalculator
from rag_eval.services.generation.llm_client import FuelixLLMClient
from rag_eval.services.generation.rag_prompt import RAGPrompt
from rag_eval.services.pipeline import RAGPipeline
from rag_eval.services.retrieval.retriever import Retriever
from rag_eval.utils.settings import Settings

logger = structlog.get_logger(__name__)


@dataclass
class RunOptions:
    experiment_id: str
    name: str
    pipeline_filter: list[str] | None = None
    subsample_size: int | None = None
    queries_limit: int | None = None
    top_k: int = 10
    judge_model: str | None = None
    force_reindex: bool = False
    force_regen: bool = False
    skip_llm_judges: bool = False
    query_concurrency: int = 8


@dataclass(frozen=True)
class FlowEvent:
    phase: str
    detail: str
    duration_ms: float
    data: dict | None = None


class BenchmarkRunner:
    """Orchestrates the full benchmark: index → retrieve+generate → evaluate → report.

    Single-method public surface (``run``) so background tasks just await it.
    All progress flows through the injected ExperimentStore so the API can poll.
    """

    COMPOSITE_WEIGHTS = {
        "ndcg@10": 0.25,
        "recall@10": 0.15,
        "faithfulness": 0.30,
        "answer_relevancy": 0.20,
        "context_precision": 0.10,
    }

    def __init__(
        self,
        settings: Settings,
        specs: list[PipelineSpec],
        dataset: FiQADataset,
        store: QdrantVectorStore,
        embedding_cache: EmbeddingCache,
        results_store: ResultsStore,
        experiment_store: ExperimentStore,
        llm: FuelixLLMClient,
    ):
        self._settings = settings
        self._specs = specs
        self._dataset = dataset
        self._store = store
        self._embedding_cache = embedding_cache
        self._results_store = results_store
        self._experiment_store = experiment_store
        self._llm = llm
        self._prompt = RAGPrompt()
        self.flow: list[FlowEvent] = []

    def _record(self, phase: str, detail: str, t0: float, data: dict | None = None) -> None:
        self.flow.append(
            FlowEvent(
                phase=phase,
                detail=detail,
                duration_ms=(time.monotonic() - t0) * 1000,
                data=data,
            )
        )

    async def run(self, options: RunOptions) -> None:
        experiment_id = options.experiment_id
        try:
            self._update(experiment_id, "running", phase="loading_data")
            t_load = time.monotonic()
            data = await self._dataset.load()
            queries = list(data.queries.values())
            if options.queries_limit:
                queries = queries[: options.queries_limit]
            logger.info("dataset_loaded", n_corpus=len(data.corpus), n_queries=len(queries))
            self._record(
                "loading_data",
                f"corpus={len(data.corpus)} queries={len(queries)}",
                t_load,
                {"corpus": len(data.corpus), "queries": len(queries)},
            )

            self._update(
                experiment_id, "running", phase="indexing", pipelines_total=len(self._specs)
            )
            t_index_all = time.monotonic()
            embedders: dict[str, Embedder] = {}
            failed: set[str] = set()
            for i, spec in enumerate(self._specs, start=1):
                t_p = time.monotonic()
                try:
                    embedder = self._get_or_make_embedder(spec, embedders)
                    await self._index_pipeline(spec, data, embedder, options.force_reindex)
                    size = await self._store.collection_size(spec.collection_name())
                    self._record(
                        "indexing",
                        f"{spec.pipeline_id} {spec.chunking}+{spec.embedder.split(':',1)[-1]} points={size}",
                        t_p,
                        {"pipeline_id": spec.pipeline_id, "points": size},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "indexing_failed", pipeline=spec.pipeline_id, error=str(exc)
                    )
                    failed.add(spec.pipeline_id)
                    self._record(
                        "indexing_failed",
                        f"{spec.pipeline_id} skipped: {type(exc).__name__}: {str(exc)[:120]}",
                        t_p,
                        {"pipeline_id": spec.pipeline_id, "error": str(exc)[:300]},
                    )
                self._update(
                    experiment_id,
                    "running",
                    phase="indexing",
                    pipelines_completed=i,
                    pipelines_total=len(self._specs),
                )
            self._record(
                "indexing_done",
                f"{len(self._specs)} pipelines indexed",
                t_index_all,
                {"pipelines": len(self._specs)},
            )

            self._update(
                experiment_id,
                "running",
                phase="retrieval_generation",
                pipelines_total=len(self._specs),
            )
            t_rg = time.monotonic()
            for i, spec in enumerate(self._specs, start=1):
                if spec.pipeline_id in failed:
                    self._record(
                        "retrieval_generation_skipped",
                        f"{spec.pipeline_id} skipped (indexing failed)",
                        time.monotonic(),
                        {"pipeline_id": spec.pipeline_id},
                    )
                    continue
                t_p = time.monotonic()
                try:
                    embedder = embedders[spec.embedder]
                    await self._run_queries(spec, embedder, queries, options)
                    self._record(
                        "retrieval_generation",
                        f"{spec.pipeline_id} queries={len(queries)}",
                        t_p,
                        {"pipeline_id": spec.pipeline_id, "queries": len(queries)},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("queries_failed", pipeline=spec.pipeline_id, error=str(exc))
                    failed.add(spec.pipeline_id)
                    self._record(
                        "retrieval_generation_failed",
                        f"{spec.pipeline_id} {type(exc).__name__}: {str(exc)[:120]}",
                        t_p,
                        {"pipeline_id": spec.pipeline_id, "error": str(exc)[:300]},
                    )
                self._update(
                    experiment_id,
                    "running",
                    phase="retrieval_generation",
                    pipelines_completed=i,
                    pipelines_total=len(self._specs),
                )
            self._record(
                "retrieval_generation_done",
                f"{len(self._specs)} pipelines x {len(queries)} queries",
                t_rg,
                {"pipelines": len(self._specs), "queries": len(queries)},
            )

            self._update(
                experiment_id, "running", phase="evaluation", pipelines_total=len(self._specs)
            )
            t_eval = time.monotonic()
            aggregates: list[AggregateResult] = []
            for i, spec in enumerate(self._specs, start=1):
                if spec.pipeline_id in failed:
                    self._record(
                        "evaluation_skipped",
                        f"{spec.pipeline_id} skipped",
                        time.monotonic(),
                        {"pipeline_id": spec.pipeline_id},
                    )
                    continue
                t_p = time.monotonic()
                agg = await self._evaluate_pipeline(spec, data, options)
                aggregates.append(agg)
                self._record(
                    "evaluation",
                    f"{spec.pipeline_id} composite={agg.composite_score:.4f}",
                    t_p,
                    {
                        "pipeline_id": spec.pipeline_id,
                        "composite_score": agg.composite_score,
                        "metrics": agg.metrics,
                    },
                )
                self._update(
                    experiment_id,
                    "running",
                    phase="evaluation",
                    pipelines_completed=i,
                    pipelines_total=len(self._specs),
                )
            self._record(
                "evaluation_done",
                f"{len(aggregates)} pipelines evaluated",
                t_eval,
                {"pipelines": len(aggregates)},
            )

            self._update(experiment_id, "running", phase="reporting")
            t_rep = time.monotonic()
            self._results_store.write_aggregate(aggregates)
            self._record("reporting", "aggregate written", t_rep)
            self._update(experiment_id, "completed", phase="done")

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "benchmark_failed", experiment_id=experiment_id, error=str(exc)
            )
            self._experiment_store.update(experiment_id, status="failed", error=str(exc))
            raise

    def _get_or_make_embedder(self, spec: PipelineSpec, cache: dict[str, Embedder]) -> Embedder:
        if spec.embedder in cache:
            return cache[spec.embedder]
        kind, _, _ = spec.embedder.partition(":")
        if kind == "fuelix":
            embedder: Embedder = FuelixEmbedder(
                api_key=self._settings.fuelix_api_key,
                base_url=self._settings.fuelix_base_url,
                cache=self._embedding_cache,
                concurrency=self._settings.fuelix_embed_concurrency,
                batch_size=self._settings.fuelix_embed_batch_size,
                **spec.embedder_kwargs,
            )
        elif kind == "local":
            embedder = LocalEmbedder(cache=self._embedding_cache, **spec.embedder_kwargs)
        else:
            raise ValueError(f"Unknown embedder kind: {kind}")
        cache[spec.embedder] = embedder
        return embedder

    def _make_chunker(self, spec: PipelineSpec, embedder: Embedder) -> Chunker:
        if spec.chunking == "fixed":
            return FixedChunker(**spec.chunker_kwargs)
        if spec.chunking == "semantic":
            return SemanticChunker(sentence_embedder=embedder, **spec.chunker_kwargs)
        if spec.chunking == "hierarchical":
            return HierarchicalChunker(**spec.chunker_kwargs)
        raise ValueError(f"Unknown chunking: {spec.chunking}")

    async def _index_pipeline(
        self,
        spec: PipelineSpec,
        data: FiQAData,
        embedder: Embedder,
        force: bool,
    ) -> None:
        collection = spec.collection_name()
        await self._store.ensure_collection(collection, embedder.dim, recreate=force)
        if not force:
            existing = await self._store.collection_size(collection)
            if existing > 0:
                logger.info("indexing_skip", collection=collection, points=existing)
                return

        chunker = self._make_chunker(spec, embedder)
        chunks = await asyncio.to_thread(chunker.chunk, list(data.corpus.values()))
        logger.info("chunked", pipeline=spec.pipeline_id, chunks=len(chunks))

        batch = 256
        for i in range(0, len(chunks), batch):
            slice_ = chunks[i : i + batch]
            vectors = await embedder.embed([c.text for c in slice_])
            await self._store.upsert(collection, slice_, vectors)
        logger.info("indexed", pipeline=spec.pipeline_id, collection=collection)

    async def _run_queries(
        self,
        spec: PipelineSpec,
        embedder: Embedder,
        queries: list,
        options: RunOptions,
    ) -> None:
        existing = self._results_store.read_per_query(pipeline_id=spec.pipeline_id)
        already_done: set[str] = set(existing["query_id"]) if not existing.empty else set()
        if options.force_regen:
            already_done = set()

        pending = [q for q in queries if q.query_id not in already_done]
        if not pending:
            return

        retriever = Retriever(embedder, self._store, spec.collection_name(), top_k=options.top_k)
        pipeline = RAGPipeline(
            pipeline_id=spec.pipeline_id,
            embedder=embedder,
            retriever=retriever,
            llm=self._llm,
            prompt=self._prompt,
        )

        sem = asyncio.Semaphore(options.query_concurrency)

        async def process(query) -> QueryResult:
            async with sem:
                resp = await pipeline.answer(query.text, top_k=options.top_k)
            return QueryResult(
                pipeline_id=spec.pipeline_id,
                query_id=query.query_id,
                question=query.text,
                answer=resp.answer,
                retrieved_chunk_ids=[r.chunk.chunk_id for r in resp.retrieved],
                retrieved_doc_ids=resp.doc_ids_unique,
                contexts=[r.chunk.text for r in resp.retrieved],
                scores=[r.score for r in resp.retrieved],
                latency_ms=resp.latency_ms,
            )

        chunk_size = 100
        for i in range(0, len(pending), chunk_size):
            batch_queries = pending[i : i + chunk_size]
            rows = await asyncio.gather(*(process(q) for q in batch_queries))
            self._results_store.write_per_query(rows)
            logger.info(
                "queries_batch",
                pipeline=spec.pipeline_id,
                done=i + len(rows),
                total=len(pending),
            )

    async def _evaluate_pipeline(
        self, spec: PipelineSpec, data: FiQAData, options: RunOptions
    ) -> AggregateResult:
        df = self._results_store.read_per_query(pipeline_id=spec.pipeline_id)
        if df.empty:
            return AggregateResult(
                pipeline_id=spec.pipeline_id,
                chunking=spec.chunking,
                embedder=spec.embedder,
                metrics={},
                failures={},
                composite_score=0.0,
                composite_ci_low=0.0,
                composite_ci_high=0.0,
            )

        retrieved_per_query: dict[str, list[str]] = {
            str(row["query_id"]): list(row["retrieved_doc_ids"]) for _, row in df.iterrows()
        }
        retrieval = RetrievalEvaluator(k=options.top_k).evaluate(retrieved_per_query, data.qrels)

        rows_for_judges = []
        for _, r in df.iterrows():
            qid = str(r["query_id"])
            relevant_ids = data.relevant_doc_ids(qid)
            reference = ""
            for did in retrieved_per_query[qid]:
                if did in relevant_ids and did in data.corpus:
                    reference = data.corpus[did].text
                    break
            rows_for_judges.append(
                {
                    "query_id": qid,
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": list(r["contexts"]),
                    "reference": reference,
                }
            )

        ragas_metrics: dict[str, float] = {}
        deep_metrics: dict[str, float] = {}
        ragas_per_query: dict[str, dict[str, float | None]] = {}

        if not options.skip_llm_judges:
            judge = options.judge_model or self._settings.judge_model
            ragas_eval = RagasEvaluator(
                api_key=self._settings.fuelix_api_key,
                judge_model=judge,
                base_url=self._settings.fuelix_base_url,
            )
            deep_eval = DeepEvalEvaluator(
                api_key=self._settings.fuelix_api_key,
                judge_model=judge,
                base_url=self._settings.fuelix_base_url,
            )
            try:
                ragas_result = await ragas_eval.evaluate(rows_for_judges)
                ragas_metrics = ragas_result.metrics
                ragas_per_query = {
                    r.query_id: {
                        "faithfulness": r.faithfulness,
                        "answer_relevancy": r.answer_relevancy,
                        "context_precision": r.context_precision,
                        "context_recall": r.context_recall,
                    }
                    for r in ragas_result.per_query
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("ragas_failed", pipeline=spec.pipeline_id, error=str(exc))

            try:
                deep_result = await deep_eval.evaluate(rows_for_judges)
                deep_metrics = deep_result.metrics
            except Exception as exc:  # noqa: BLE001
                logger.exception("deepeval_failed", pipeline=spec.pipeline_id, error=str(exc))

        all_metrics = {**retrieval.metrics, **ragas_metrics, **deep_metrics}

        answers: dict[str, str] = {
            str(row["query_id"]): str(row["answer"]) for _, row in df.iterrows()
        }
        analyzer = FailureAnalyzer()
        summary = analyzer.categorize(
            retrieved_docs_per_query=retrieved_per_query,
            qrels=data.qrels,
            answers_per_query=answers,
            ragas_per_query=ragas_per_query,
            top_k=options.top_k,
        )

        stats_calc = StatisticsCalculator(seed=self._settings.seed)
        composite = stats_calc.composite_score(all_metrics, self.COMPOSITE_WEIGHTS)
        per_query_composites = self._per_query_composites(retrieval, ragas_per_query)
        ci_summary = stats_calc.summarize(per_query_composites)

        return AggregateResult(
            pipeline_id=spec.pipeline_id,
            chunking=spec.chunking,
            embedder=spec.embedder,
            metrics=all_metrics,
            failures=summary.distribution,
            composite_score=composite,
            composite_ci_low=ci_summary.ci_low,
            composite_ci_high=ci_summary.ci_high,
        )

    def _per_query_composites(self, retrieval, ragas_per_query) -> list[float]:
        scores = []
        for r in retrieval.per_query:
            faith = (ragas_per_query.get(r.query_id) or {}).get("faithfulness")
            base = 0.4 * r.ndcg_at_k + 0.2 * r.recall_at_k
            if faith is not None:
                base += 0.4 * faith
            scores.append(base)
        return scores

    def _update(self, experiment_id: str, status: str, **progress) -> None:
        progress.setdefault("updated_at", time.time())
        self._experiment_store.update(experiment_id, status=status, progress=progress)
