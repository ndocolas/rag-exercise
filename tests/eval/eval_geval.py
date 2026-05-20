"""G-Eval evaluation of the agent using 3 financial domain criteria."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent import build_agent
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage

RESULTS_PATH = Path(__file__).parent / "geval_results.json"

QUESTIONS = [
    "What are the risks of investing in bonds?",
    "How does a mutual fund differ from an ETF?",
    "What is the significance of a P/E ratio for stock valuation?",
    "Explain the concept of dollar-cost averaging.",
    "What is 8.5% of 42000?",
    "If I invest 10000 at 6% annual interest compounded yearly, what will I have after 5 years?",
    "What is the percentage gain if a stock goes from 80 to 95?",
    "How many years ago was the year 2000?",
    "What is today's date?",
    "What are index funds and why do investors prefer them over actively managed funds?",
]

CRITERIA = [
    GEval(
        name="Factual accuracy",
        criteria="The response contains only correct financial statements and does not include false information about financial instruments, markets, or calculations.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],  # type: ignore[attr-defined]
    ),
    GEval(
        name="Source citation",
        criteria="When the response uses information from a knowledge base, it cites sources using markers like [1], [2], or similar references.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],  # type: ignore[attr-defined]
    ),
    GEval(
        name="Appropriate confidence",
        criteria="The response avoids making absolute predictions or guarantees about future market movements, prices, or returns.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],  # type: ignore[attr-defined]
    ),
]


async def run_agent_on_question(graph: object, question: str) -> str:
    result = await graph.ainvoke({"messages": [HumanMessage(content=question)]})  # type: ignore[union-attr]
    return result["messages"][-1].content


async def collect_answers() -> list[dict[str, str]]:
    graph = build_agent()
    pairs = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] Running agent for: {q[:60]}...")
        answer = await run_agent_on_question(graph, q)
        pairs.append({"question": q, "answer": answer})
    return pairs


def evaluate_with_geval(pairs: list[dict[str, str]]) -> list[dict]:
    results = []
    for pair in pairs:
        test_case = LLMTestCase(input=pair["question"], actual_output=pair["answer"])
        scores: dict[str, float] = {}
        passed: dict[str, bool] = {}
        for metric in CRITERIA:
            metric.measure(test_case)
            scores[metric.name] = round(metric.score or 0.0, 3)
            passed[metric.name] = metric.is_successful()
        results.append(
            {
                "question": pair["question"],
                "answer": (pair["answer"][:200] + "...") if len(pair["answer"]) > 200 else pair["answer"],
                "scores": scores,
                "passed": passed,
            }
        )
    return results


def print_summary(results: list[dict]) -> None:
    print("\n── G-Eval Results ────────────────────────────────")
    for criterion in CRITERIA:
        scores = [r["scores"][criterion.name] for r in results]
        avg = sum(scores) / len(scores)
        passing = sum(1 for r in results if r["passed"][criterion.name])
        print(f"{criterion.name:<25} avg={avg:.3f}  passed={passing}/{len(results)}")
    print("──────────────────────────────────────────────────")
    print(f"Results saved to: {RESULTS_PATH}")


async def main() -> None:
    pairs = await collect_answers()
    print("\nEvaluating with G-Eval criteria...")
    results = evaluate_with_geval(pairs)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
