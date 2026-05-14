from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedderRun:
    embedder_alias: str
    chunks_texts: list[str]
    answer: str
    error: str | None = None


@dataclass(frozen=True)
class CompareRenderInput:
    question: str
    runs: list[EmbedderRun]


class CompareRenderer:
    """Markdown view of `/compare`. Mirrors the JSON shape: question, then one
    section per embedder with answer + retrieved documents (or an error line).
    """

    def render(self, data: CompareRenderInput) -> str:
        out: list[str] = ["# Pergunta", "", data.question.strip(), ""]
        for run in data.runs:
            out += self._embedder_section(run)
        return "\n".join(out)

    @staticmethod
    def _embedder_section(run: EmbedderRun) -> list[str]:
        out = [f"## {run.embedder_alias}", ""]
        if run.error is not None:
            out += [f"Erro: {run.error}", ""]
            return out
        out += ["### Resposta", "", run.answer.strip(), ""]
        out += ["### Documentos retornados", ""]
        if not run.chunks_texts:
            out += ["(nenhum documento recuperado)", ""]
            return out
        for i, text in enumerate(run.chunks_texts, start=1):
            cleaned = text.strip().replace("\n", " ")
            out.append(f"{i}. {cleaned}")
        out.append("")
        return out
