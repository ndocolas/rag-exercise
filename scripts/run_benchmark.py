"""CLI alternative to POST /experiments/run.

Usage:
    uv run scripts/run_benchmark.py --pipelines P1 P5 --subsample 500 --queries 50
"""

from __future__ import annotations

import argparse
import asyncio

from ulid import ULID

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.db.experiment_store import ExperimentStore
from rag_eval.db.results_store import ResultsStore
from rag_eval.db.vector_store import QdrantVectorStore
from rag_eval.services.benchmark.benchmark_runner import BenchmarkRunner, RunOptions
from rag_eval.services.benchmark.pipeline_matrix import PipelineMatrix
from rag_eval.services.benchmark.report_generator import ReportGenerator
from rag_eval.services.data.fiqa_dataset import FiQADataset
from rag_eval.services.generation.llm_client import FuelixLLMClient
from rag_eval.utils.settings import get_settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG benchmark")
    parser.add_argument("--name", default="cli_run")
    parser.add_argument("--pipelines", nargs="*", default=None, help="P1..P9 filter")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--queries", type=int, default=None, help="limit queries")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--judge", default=None)
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument("--skip-judges", action="store_true", help="skip RAGAS+DeepEval")
    args = parser.parse_args()

    settings = get_settings()
    specs = PipelineMatrix.build(filter_ids=args.pipelines)
    if not specs:
        raise SystemExit("No pipelines matched filter")

    experiment_id = f"exp_{ULID()}"
    print(f"experiment_id={experiment_id}")

    store = QdrantVectorStore(url=settings.qdrant_url)
    embedding_cache = EmbeddingCache(settings.embedding_cache_path)
    experiment_store = ExperimentStore(settings.experiment_store_path)
    experiment_store.create(experiment_id, name=args.name, config=vars(args))
    llm = FuelixLLMClient(
        api_key=settings.fuelix_api_key,
        model=settings.generator_model,
        base_url=settings.fuelix_base_url,
        cache_path=settings.llm_cache_path,
        concurrency=settings.fuelix_llm_concurrency,
        temperature=settings.generator_temperature,
        max_tokens=settings.generator_max_tokens,
    )
    dataset = FiQADataset(
        data_dir=settings.fiqa_data_dir,
        subsample_size=args.subsample or settings.subsample_size,
        seed=settings.seed,
    )
    results_store = ResultsStore(settings.results_dir, experiment_id)

    runner = BenchmarkRunner(
        settings=settings,
        specs=specs,
        dataset=dataset,
        store=store,
        embedding_cache=embedding_cache,
        results_store=results_store,
        experiment_store=experiment_store,
        llm=llm,
    )
    options = RunOptions(
        experiment_id=experiment_id,
        name=args.name,
        pipeline_filter=args.pipelines,
        subsample_size=args.subsample,
        queries_limit=args.queries,
        top_k=args.top_k,
        judge_model=args.judge,
        force_reindex=args.force_reindex,
        force_regen=args.force_regen,
        skip_llm_judges=args.skip_judges,
    )
    try:
        await runner.run(options)
        report = ReportGenerator(results_store, seed=settings.seed).generate()
        print(f"Report written to: {results_store.report_dir}")
        print()
        print(report.markdown)
    finally:
        await store.close()
        embedding_cache.close()
        experiment_store.close()
        llm.close()


if __name__ == "__main__":
    asyncio.run(main())
