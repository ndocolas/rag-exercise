"""LangGraph financial agent with RAGTool, CalculatorTool, and DateTool."""
import argparse
import asyncio
import os
import re
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import httpx
import sympy
from dateutil.relativedelta import relativedelta
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr

RAG_ENDPOINT_URL = os.getenv("RAG_ENDPOINT_URL", "http://localhost:8000")
FUELIX_BASE_URL = os.getenv("FUELIX_BASE_URL", "https://api.fuelix.ai/v1")
FUELIX_API_KEY = os.getenv("FUELIX_API_KEY", "")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "claude-sonnet-4-5")


@tool
async def rag_tool(question: str) -> str:
    """Answer financial questions about concepts, definitions, and FiQA dataset analysis.
    Use for: 'what is', 'explain', 'risks of', 'how does X work', qualitative questions."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{RAG_ENDPOINT_URL}/ask",
            json={"question": question, "embedder": "openai", "top_k": 5},
        )
        r.raise_for_status()
        data = r.json()
        return data.get("answer", str(data))


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate mathematical expressions: percentages, compound interest, differences, ratios.
    Use for: 'what is X% of Y', 'calculate', numeric operations.
    Pass the raw math expression, e.g. '15/100 * 3000' or '1000 * (1 + 0.05)**3'."""
    try:
        result = sympy.sympify(expression).evalf()  # type: ignore[union-attr]
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def date_tool(query: str) -> str:
    """Return the current date or compute time differences.
    Use for: 'how many years ago', 'how many days since', 'what year is it', 'current date'.
    Pass the query as-is, e.g. '2010' to compute years since 2010, or 'today' for current date."""
    now = datetime.now()

    # Extract a 4-digit year from the query
    year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", query)
    if year_match:
        target_year = int(year_match.group(1))
        try:
            target_date = datetime(target_year, 1, 1)
            delta = relativedelta(now, target_date)
            return (
                f"From {target_year} to {now.year}: "
                f"{delta.years} years, {delta.months} months, {delta.days} days. "
                f"Today is {now.strftime('%Y-%m-%d')}."
            )
        except ValueError:
            pass

    # No year found — return current date
    return f"Today is {now.strftime('%Y-%m-%d')} ({now.strftime('%A, %B %d, %Y')})."


def build_agent():  # type: ignore[return]
    llm = ChatOpenAI(
        model=GENERATOR_MODEL,
        base_url=FUELIX_BASE_URL,
        api_key=SecretStr(FUELIX_API_KEY),
        temperature=0.0,
    )
    return create_react_agent(llm, [rag_tool, calculator_tool, date_tool])  # type: ignore[return-value]


async def run_agent(question: str) -> str:
    graph = build_agent()
    result = await graph.ainvoke({"messages": [HumanMessage(content=question)]})

    messages = result["messages"]

    # Print tool usage for debug
    for msg in messages:
        if isinstance(msg, ToolMessage):
            print(f"[tool={msg.name}] → {str(msg.content)[:120]}")

    # Return the last AI message
    return messages[-1].content


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial LangGraph agent")
    parser.add_argument("--question", "-q", required=True, help="Question to answer")
    parser.add_argument("--rag-url", default=None, help="RAG endpoint URL (overrides RAG_ENDPOINT_URL)")
    args = parser.parse_args()

    if args.rag_url:
        global RAG_ENDPOINT_URL
        RAG_ENDPOINT_URL = args.rag_url

    if not FUELIX_API_KEY:
        raise SystemExit("FUELIX_API_KEY not set. Add it to .env or export it.")

    answer = asyncio.run(run_agent(args.question))
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
