from __future__ import annotations

from dataclasses import dataclass

from rag_eval.services.response.citation_tracer import CitationTracer


@dataclass(frozen=True)
class AskRenderInput:
    question: str
    answer_with_rag: str
    chunks_texts: list[str]
    answer_no_rag: str | None
    judge_verdict: str | None = None


class AskRenderer:
    """Markdown view of `/ask`. Mirrors the JSON shape: question, RAG
    answer, optional no-RAG answer, comparative judge verdict, then
    retrieved documents at the end.
    """

    def render(self, data: AskRenderInput) -> str:
        out: list[str] = []
        out += ["# Pergunta", "", data.question.strip(), ""]
        out += [
            "# Resposta (com RAG)",
            "",
            CitationTracer.highlight_citations(data.answer_with_rag.strip()),
            "",
        ]
        if data.answer_no_rag is not None:
            out += [
                "# Resposta sem RAG",
                "",
                data.answer_no_rag.strip(),
                "",
            ]
        if data.judge_verdict is not None:
            out += [
                "# Avaliação (LLM-as-judge) — comparação",
                "",
                data.judge_verdict.strip(),
                "",
            ]
        out += self._documents_section(data.chunks_texts)
        return "\n".join(out)

    @staticmethod
    def _documents_section(texts: list[str]) -> list[str]:
        out = ["# Documentos retornados", ""]
        if not texts:
            out += ["(nenhum documento recuperado)", ""]
            return out
        for i, text in enumerate(texts, start=1):
            cleaned = text.strip().replace("\n", " ")
            out.append(f"{i}. {cleaned}")
        out.append("")
        return out
