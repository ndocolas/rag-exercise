"""Evaluate whether the agent selects the correct tool for each question."""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent import build_agent
from langchain_core.messages import HumanMessage, ToolMessage

CASES_PATH = Path(__file__).parent / "tool_correctness_cases.json"


async def evaluate_case(graph: object, question: str) -> str | None:
    result = await graph.ainvoke({"messages": [HumanMessage(content=question)]})  # type: ignore[union-attr]
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            return msg.name
    return None


async def main() -> None:
    cases = json.loads(CASES_PATH.read_text())
    graph = build_agent()

    correct = 0
    per_tool: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    failures: list[dict[str, str]] = []

    for i, case in enumerate(cases, 1):
        question = case["question"]
        expected = case["expected_tool"]
        got = await evaluate_case(graph, question)
        per_tool[expected]["total"] += 1

        if got == expected:
            correct += 1
            per_tool[expected]["correct"] += 1
        else:
            failures.append({"question": question, "expected": expected, "got": str(got)})

        print(f"[{i:02d}/{len(cases)}] {'✓' if got == expected else '✗'} {expected} | got={got}")

    print("\n── Tool Correctness ──────────────────────────")
    print(f"{'Tool':<20} {'Correct':>7} {'Total':>6} {'Acc':>6}")
    print("─" * 44)
    for tool_name, counts in sorted(per_tool.items()):
        acc = counts["correct"] / counts["total"] * 100
        print(f"{tool_name:<20} {counts['correct']:>7} {counts['total']:>6} {acc:>5.0f}%")
    print("─" * 44)
    overall = correct / len(cases) * 100
    print(f"{'OVERALL':<20} {correct:>7} {len(cases):>6} {overall:>5.0f}%")

    if failures:
        print(f"\n── Failures ({len(failures)}) ──────────────────────────")
        for f in failures:
            print(f"  Q: {f['question'][:70]}")
            print(f"     expected={f['expected']}  got={f['got']}\n")


if __name__ == "__main__":
    asyncio.run(main())
