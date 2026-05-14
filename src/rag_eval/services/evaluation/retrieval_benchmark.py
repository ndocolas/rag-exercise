from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

from rag_eval.services.data.fiqa_dataset import FiQADataset
from rag_eval.services.evaluation.retrieval_evaluator import RetrievalEvaluator
from rag_eval.services.retrieval.retriever import Retriever

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetrievalBenchmarkResult:
    """Plain, doc-level retrieval metrics for one embedder.

    `score` is a 0-100 grade averaging the three most readable signals
    (`hit@10`, `recall@10`, `ndcg@10`). Lower-level metrics are exposed
    so callers can dig in if needed.
    """

    queries_avaliadas: int
    hit_rate: float
    precision_at_10: float
    recall_at_10: float
    ndcg_at_10: float
    score: int


class RetrievalBenchmark:
    """Runs a retrieval-only benchmark on the FULL FiQA query set.

    No sampling: every embedder is graded on the same questions so
    `/evaluate` returns comparable scores across the three didactic
    embedders. No LLM calls — only embed + Qdrant search + qrels.
    """

    _CONCURRENCY = 8

    def __init__(self, dataset: FiQADataset):
        self._dataset = dataset

    async def run(
        self,
        retriever: Retriever,
        *,
        top_k: int,
    ) -> RetrievalBenchmarkResult:
        data = await self._dataset.load()
        if not data.queries:
            raise ValueError("FiQA dataset has no evaluable queries (qrels missing).")

        query_ids = list(data.queries.keys())
        retrieved_per_query = await self._retrieve_all(retriever, data, query_ids, top_k)

        aggregate = RetrievalEvaluator(k=top_k).evaluate(retrieved_per_query, data.qrels)

        return self._compose(aggregate, top_k)

    async def _retrieve_all(
        self,
        retriever: Retriever,
        data,
        query_ids: list[str],
        top_k: int,
    ) -> dict[str, list[str]]:
        sem = asyncio.Semaphore(self._CONCURRENCY)

        async def _one(qid: str) -> tuple[str, list[str]]:
            async with sem:
                result = await retriever.retrieve(data.queries[qid].text, top_k=top_k)
                return qid, result.doc_ids_unique

        pairs = await asyncio.gather(*(_one(qid) for qid in query_ids))
        return dict(pairs)

    @staticmethod
    def _compose(aggregate, top_k: int) -> RetrievalBenchmarkResult:
        m = aggregate.metrics
        hit_rate = float(m.get(f"hit@{top_k}", 0.0))
        recall = float(m.get(f"recall@{top_k}", 0.0))
        precision = float(m.get(f"precision@{top_k}", 0.0))
        ndcg = float(m.get(f"ndcg@{top_k}", 0.0))
        # Average the three signals a human reads first; MRR/precision
        # stay in the response but don't bias the headline number.
        score = round(((hit_rate + recall + ndcg) / 3) * 100)
        return RetrievalBenchmarkResult(
            queries_avaliadas=len(aggregate.per_query),
            hit_rate=hit_rate,
            precision_at_10=precision,
            recall_at_10=recall,
            ndcg_at_10=ndcg,
            score=score,
        )
