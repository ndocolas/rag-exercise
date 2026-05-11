from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PerQueryRetrieval:
    query_id: str
    hit_at: dict[int, int]
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float


@dataclass(frozen=True)
class AggregateRetrieval:
    metrics: dict[str, float]
    per_query: list[PerQueryRetrieval]


class RetrievalEvaluator:
    """Computes Recall@k, Precision@k, MRR, nDCG@k, Hit@{1,3,5,10} at the doc level.

    Operates over a list of (query_id, retrieved_doc_ids) and the dataset qrels.
    Doc-level mapping must already be applied upstream — chunks are not visible here.
    """

    def __init__(self, k: int = 10, hit_ks: tuple[int, ...] = (1, 3, 5, 10)):
        self._k = k
        self._hit_ks = hit_ks

    def evaluate(
        self,
        retrieved_per_query: dict[str, list[str]],
        qrels: dict[str, dict[str, int]],
    ) -> AggregateRetrieval:
        per_query: list[PerQueryRetrieval] = []
        for qid, doc_ids in retrieved_per_query.items():
            relevant = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
            if not relevant:
                continue
            top_k = doc_ids[: self._k]
            per_query.append(self._score(qid, top_k, relevant))
        return AggregateRetrieval(metrics=self._aggregate(per_query), per_query=per_query)

    def _score(self, qid: str, top_k: list[str], relevant: set[str]) -> PerQueryRetrieval:
        hits = [1 if d in relevant else 0 for d in top_k]
        retrieved_relevant = sum(hits)

        recall = retrieved_relevant / len(relevant) if relevant else 0.0
        precision = retrieved_relevant / self._k if self._k else 0.0

        mrr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                mrr = 1.0 / rank
                break

        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        ideal_hits = [1] * min(len(relevant), self._k)
        idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        hit_at = {k: int(any(hits[:k])) for k in self._hit_ks}

        return PerQueryRetrieval(
            query_id=qid,
            hit_at=hit_at,
            recall_at_k=recall,
            precision_at_k=precision,
            mrr=mrr,
            ndcg_at_k=ndcg,
        )

    def _aggregate(self, rows: list[PerQueryRetrieval]) -> dict[str, float]:
        if not rows:
            return {}
        n = len(rows)
        agg = {
            f"recall@{self._k}": sum(r.recall_at_k for r in rows) / n,
            f"precision@{self._k}": sum(r.precision_at_k for r in rows) / n,
            "mrr": sum(r.mrr for r in rows) / n,
            f"ndcg@{self._k}": sum(r.ndcg_at_k for r in rows) / n,
        }
        for k in self._hit_ks:
            agg[f"hit@{k}"] = sum(r.hit_at.get(k, 0) for r in rows) / n
        return agg
