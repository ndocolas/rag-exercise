"""Standalone indexing script.

Builds the 9 Qdrant collections without running queries or eval. Useful for
seeding the vector DB ahead of /query traffic.
"""

from __future__ import annotations

import argparse
import asyncio

from rag_eval.db.embedding_cache import EmbeddingCache
from rag_eval.db.experiment_store import ExperimentStore
from rag_eval.db.results_store import ResultsStore
from rag_eval.db.vector_store import QdrantVectorStore
from rag_eval.services.benchmark.benchmark_runner import BenchmarkRunner
from rag_eval.services.benchmark.pipeline_matrix import PipelineMatrix
from rag_eval.services.data.fiqa_dataset import FiQADataset
from rag_eval.services.generation.llm_client import FuelixLLMClient
from rag_eval.utils.settings import get_settings


async def main() -> None:
    parser = argparse.ArgumentParser(description="Index FiQA into 9 Qdrant collections")
    parser.add_argument("--pipelines", nargs="*", default=None)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    specs = PipelineMatrix.build(filter_ids=args.pipelines)
    if not specs:
        raise SystemExit("No pipelines matched filter")

    store = QdrantVectorStore(url=settings.qdrant_url)
    embedding_cache = EmbeddingCache(settings.embedding_cache_path)
    experiment_store = ExperimentStore(settings.experiment_store_path)
    llm = FuelixLLMClient(
        api_key=settings.fuelix_api_key,
        model=settings.generator_model,
        base_url=settings.fuelix_base_url,
        cache_path=settings.llm_cache_path,
    )
    dataset = FiQADataset(
        data_dir=settings.fiqa_data_dir,
        subsample_size=args.subsample or settings.subsample_size,
        seed=settings.seed,
    )
    results_store = ResultsStore(settings.results_dir, "ingest_only")

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
    try:
        data = await dataset.load()
        embedders = {}
        for spec in specs:
            embedder = runner._get_or_make_embedder(spec, embedders)
            await runner._index_pipeline(spec, data, embedder, force=args.force)
            print(f"indexed {spec.pipeline_id} -> {spec.collection_name()}")
    finally:
        await store.close()
        embedding_cache.close()
        experiment_store.close()
        llm.close()


if __name__ == "__main__":
    asyncio.run(main())
