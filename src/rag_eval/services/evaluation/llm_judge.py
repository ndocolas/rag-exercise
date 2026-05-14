from __future__ import annotations

from rag_eval.services.generation.llm_client import FuelixLLMClient


class LLMJudge:
    """LLM-as-judge that *compares* a RAG answer vs. a no-RAG answer.

    Given the same question answered (a) using only retrieved context
    passages and (b) using only the model's internal knowledge, the
    judge produces a short side-by-side evaluation: faithfulness,
    completeness, hallucinations, and a recommended winner.

    Reuses ``FuelixLLMClient.complete`` so verdicts can be cached on
    disk by ``(model, temperature, messages)``. Pass ``use_cache=False``
    to force a fresh call.
    """

    SYSTEM = (
        "You are a strict evaluator of question-answering systems. "
        "Compare two assistant answers to the same question side by side. "
        "Be concise (4-8 sentences). Highlight strengths, weaknesses, and "
        "any factual errors or hallucinations. End with a one-line verdict: "
        "'Winner: with_rag' / 'Winner: without_rag' / 'Tie'. "
        "Do not rewrite either answer."
    )

    USER_TEMPLATE = (
        "Question:\n{question}\n\n"
        "Retrieved context passages (only available to assistant A):\n{contexts}\n\n"
        "Assistant A — answer using ONLY the retrieved context above:\n"
        "{answer_with_rag}\n\n"
        "Assistant B — answer using ONLY the model's internal knowledge "
        "(no retrieved context was provided):\n{answer_without_rag}\n\n"
        "Compare both answers. Discuss which one is more faithful to the source "
        "material, which is more complete, and which (if any) shows hallucinations. "
        "Then declare the winner."
    )

    def __init__(self, llm: FuelixLLMClient):
        self._llm = llm

    async def compare(
        self,
        question: str,
        answer_with_rag: str,
        answer_without_rag: str,
        contexts: list[str],
        *,
        use_cache: bool = True,
    ) -> str:
        user = self.USER_TEMPLATE.format(
            question=question.strip(),
            contexts=self._format_contexts(contexts),
            answer_with_rag=answer_with_rag.strip(),
            answer_without_rag=answer_without_rag.strip(),
        )
        return await self._llm.complete(
            [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user},
            ],
            use_cache=use_cache,
        )

    @staticmethod
    def _format_contexts(contexts: list[str]) -> str:
        if not contexts:
            return "(no contexts retrieved)"
        return "\n\n".join(f"[{i}] {c.strip()}" for i, c in enumerate(contexts, start=1))
