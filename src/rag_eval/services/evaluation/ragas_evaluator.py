from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class PerQueryRagas:
    query_id: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


@dataclass(frozen=True)
class AggregateRagas:
    metrics: dict[str, float]
    per_query: list[PerQueryRagas]


class RagasEvaluator:
    """Wraps ragas v0.2 evaluation. Uses fuelix.ai through the openai-compatible
    LangChain ChatOpenAI by pointing ``base_url`` and ``api_key`` at fuelix.

    Reference is optional; if absent, context_recall is skipped.
    """

    def __init__(
        self,
        api_key: str,
        judge_model: str = "gpt-4o",
        base_url: str = "https://api.fuelix.ai/v1",
        embedding_model: str = "text-embedding-3-small",
    ):
        self._api_key = api_key
        self._judge_model = judge_model
        self._base_url = base_url
        self._embedding_model = embedding_model

    async def evaluate(
        self,
        rows: list[dict],
    ) -> AggregateRagas:
        if not rows:
            return AggregateRagas(metrics={}, per_query=[])
        return await asyncio.to_thread(self._evaluate_sync, rows)

    def _evaluate_sync(self, rows: list[dict]) -> AggregateRagas:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from pydantic import SecretStr
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=self._judge_model,
                api_key=SecretStr(self._api_key),
                base_url=self._base_url,
                temperature=0.0,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=self._embedding_model,
                api_key=SecretStr(self._api_key),
                base_url=self._base_url,
            )
        )

        has_reference = any(r.get("reference") for r in rows)
        metrics = [Faithfulness(), ResponseRelevancy()]
        if has_reference:
            metrics.extend([LLMContextPrecisionWithReference(), LLMContextRecall()])

        samples = []
        for r in rows:
            sample = {
                "user_input": r["question"],
                "retrieved_contexts": r["contexts"],
                "response": r["answer"],
            }
            if r.get("reference"):
                sample["reference"] = r["reference"]
            samples.append(sample)

        dataset = EvaluationDataset.from_list(samples)
        result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        df = result.to_pandas()  # type: ignore[attr-defined]

        per_query: list[PerQueryRagas] = []
        for r, (_, row) in zip(rows, df.iterrows(), strict=False):
            per_query.append(
                PerQueryRagas(
                    query_id=r["query_id"],
                    faithfulness=_get(row, "faithfulness"),
                    answer_relevancy=(
                        _get(row, "answer_relevancy") or _get(row, "response_relevancy")
                    ),
                    context_precision=_get(row, "llm_context_precision_with_reference"),
                    context_recall=_get(row, "context_recall"),
                )
            )
        return AggregateRagas(metrics=self._aggregate(per_query), per_query=per_query)

    @staticmethod
    def _aggregate(rows: list[PerQueryRagas]) -> dict[str, float]:
        if not rows:
            return {}

        def mean(getter):
            vals = [v for v in (getter(r) for r in rows) if v is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        return {
            "faithfulness": mean(lambda r: r.faithfulness),
            "answer_relevancy": mean(lambda r: r.answer_relevancy),
            "context_precision": mean(lambda r: r.context_precision),
            "context_recall": mean(lambda r: r.context_recall),
        }


def _get(row, key: str):
    if key not in row:
        return None
    val = row[key]
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None
