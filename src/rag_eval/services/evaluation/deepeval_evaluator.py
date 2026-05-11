from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PerQueryDeepEval:
    query_id: str
    g_eval_correctness: float | None
    hallucination: float | None
    contextual_relevancy: float | None


@dataclass(frozen=True)
class AggregateDeepEval:
    metrics: dict[str, float]
    per_query: list[PerQueryDeepEval]


class DeepEvalEvaluator:
    """Wraps DeepEval. DeepEval uses the openai SDK; we point it at fuelix via
    OPENAI_API_KEY + OPENAI_BASE_URL env vars at evaluation time.
    """

    def __init__(
        self,
        api_key: str,
        judge_model: str = "gpt-4o",
        base_url: str = "https://api.fuelix.ai/v1",
    ):
        self._api_key = api_key
        self._judge_model = judge_model
        self._base_url = base_url

    async def evaluate(self, rows: list[dict]) -> AggregateDeepEval:
        if not rows:
            return AggregateDeepEval(metrics={}, per_query=[])
        return await asyncio.to_thread(self._evaluate_sync, rows)

    def _evaluate_sync(self, rows: list[dict]) -> AggregateDeepEval:
        prev_key = os.environ.get("OPENAI_API_KEY")
        prev_base = os.environ.get("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = self._api_key
        os.environ["OPENAI_BASE_URL"] = self._base_url
        try:
            return self._run(rows)
        finally:
            if prev_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = prev_key
            if prev_base is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = prev_base

    def _run(self, rows: list[dict]) -> AggregateDeepEval:
        from deepeval.metrics import (
            ContextualRelevancyMetric,
            GEval,
            HallucinationMetric,
        )
        from deepeval.test_case import LLMTestCase, SingleTurnParams

        g_eval = GEval(
            name="Correctness",
            criteria=(
                "Determine whether the actual output is factually correct and complete given the "
                "expected output and the retrieved context. Penalize missing key facts."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            model=self._judge_model,
            threshold=0.7,
        )
        hallucination = HallucinationMetric(threshold=0.5, model=self._judge_model)
        relevancy = ContextualRelevancyMetric(threshold=0.5, model=self._judge_model)

        per_query: list[PerQueryDeepEval] = []
        for r in rows:
            ctx_text = "\n\n".join(r.get("contexts", []))
            test_case = LLMTestCase(
                input=r["question"],
                actual_output=r["answer"],
                expected_output=r.get("reference") or "",
                context=[ctx_text] if ctx_text else None,
                retrieval_context=r.get("contexts") or None,
            )
            scores: dict[str, float | None] = {
                "g_eval_correctness": None,
                "hallucination": None,
                "contextual_relevancy": None,
            }
            try:
                g_eval.measure(test_case)
                scores["g_eval_correctness"] = (
                    float(g_eval.score) if g_eval.score is not None else None
                )
            except Exception:
                pass
            try:
                hallucination.measure(test_case)
                scores["hallucination"] = (
                    float(hallucination.score) if hallucination.score is not None else None
                )
            except Exception:
                pass
            try:
                relevancy.measure(test_case)
                scores["contextual_relevancy"] = (
                    float(relevancy.score) if relevancy.score is not None else None
                )
            except Exception:
                pass

            per_query.append(
                PerQueryDeepEval(
                    query_id=r["query_id"],
                    g_eval_correctness=scores["g_eval_correctness"],
                    hallucination=scores["hallucination"],
                    contextual_relevancy=scores["contextual_relevancy"],
                )
            )

        return AggregateDeepEval(metrics=self._aggregate(per_query), per_query=per_query)

    @staticmethod
    def _aggregate(rows: list[PerQueryDeepEval]) -> dict[str, float]:
        if not rows:
            return {}

        def mean(getter):
            vals = [v for v in (getter(r) for r in rows) if v is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        return {
            "g_eval_correctness": mean(lambda r: r.g_eval_correctness),
            "hallucination": mean(lambda r: r.hallucination),
            "contextual_relevancy": mean(lambda r: r.contextual_relevancy),
        }
