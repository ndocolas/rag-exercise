from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureCategory:
    query_id: str
    bucket: str  # success | retrieval_miss | generation_fail | cascade_fail
    sub_category: str  # specific reason inside bucket


@dataclass(frozen=True)
class FailureSummary:
    counts: dict[str, int]
    distribution: dict[str, float]
    sub_distribution: dict[str, float]
    per_query: list[FailureCategory]


class FailureAnalyzer:
    """Categorizes per-query outcomes into retrieval / generation / cascade failures."""

    def __init__(
        self,
        faithfulness_threshold: float = 0.7,
        relevancy_threshold: float = 0.7,
        refusal_phrase: str = "i cannot answer based on the provided context",
    ):
        self._faith_t = faithfulness_threshold
        self._rel_t = relevancy_threshold
        self._refusal = refusal_phrase.lower()

    def categorize(
        self,
        retrieved_docs_per_query: dict[str, list[str]],
        qrels: dict[str, dict[str, int]],
        answers_per_query: dict[str, str],
        ragas_per_query: dict[str, dict[str, float | None]] | None = None,
        top_k: int = 10,
    ) -> FailureSummary:
        ragas_per_query = ragas_per_query or {}
        per_query: list[FailureCategory] = []

        for qid, docs in retrieved_docs_per_query.items():
            relevant = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
            if not relevant:
                continue
            top_docs = docs[:top_k]
            in_corpus_relevant = relevant
            in_topk = bool(set(top_docs) & in_corpus_relevant)

            answer = (answers_per_query.get(qid) or "").strip()
            ragas = ragas_per_query.get(qid, {})
            faith = ragas.get("faithfulness")
            rel = ragas.get("answer_relevancy")
            gen_ok = self._gen_ok(faith, rel)

            if in_topk and gen_ok:
                bucket = "success"
                sub = "ok"
            elif in_topk and not gen_ok:
                bucket = "generation_fail"
                sub = self._gen_subcategory(answer, faith, rel)
            elif not in_topk and gen_ok:
                bucket = "retrieval_miss"
                sub = "no_relevant_in_topk"
            else:
                bucket = "cascade_fail"
                sub = self._gen_subcategory(answer, faith, rel)

            per_query.append(FailureCategory(qid, bucket, sub))

        return self._summarize(per_query)

    def _gen_ok(self, faith: float | None, rel: float | None) -> bool:
        # Conservative: if scores are missing, fall back to "ok" — failure analysis
        # then reflects retrieval-only signal.
        f_ok = faith is None or faith >= self._faith_t
        r_ok = rel is None or rel >= self._rel_t
        return f_ok and r_ok

    def _gen_subcategory(self, answer: str, faith: float | None, rel: float | None) -> str:
        if not answer or self._refusal in answer.lower():
            return "refusal"
        if faith is not None and faith < self._faith_t:
            return "hallucination"
        if rel is not None and rel < self._rel_t:
            return "off_topic"
        return "unknown"

    def _summarize(self, per_query: list[FailureCategory]) -> FailureSummary:
        counts: dict[str, int] = {}
        sub_counts: dict[str, int] = {}
        for row in per_query:
            counts[row.bucket] = counts.get(row.bucket, 0) + 1
            key = f"{row.bucket}:{row.sub_category}"
            sub_counts[key] = sub_counts.get(key, 0) + 1
        n = max(1, len(per_query))
        distribution = {k: v / n for k, v in counts.items()}
        sub_distribution = {k: v / n for k, v in sub_counts.items()}
        return FailureSummary(
            counts=counts,
            distribution=distribution,
            sub_distribution=sub_distribution,
            per_query=per_query,
        )
