from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluateRenderInput:
    embedder: str
    queries_avaliadas: int
    hit_rate: float
    precision_at_10: float
    recall_at_10: float
    ndcg_at_10: float
    score: int


class EvaluateRenderer:
    """Markdown view of `/evaluate`: short, table-first, no fluff."""

    def render(self, data: EvaluateRenderInput) -> str:
        out: list[str] = []
        out += [f"# Avaliação de retrieval — `{data.embedder}`", ""]
        out += [
            f"**Score: {data.score}/100** "
            f"(média de hit_rate, recall@10, ndcg@10 contra o ground truth do FiQA).",
            "",
            f"Queries avaliadas: **{data.queries_avaliadas}** (sampleadas do dataset FiQA-2018).",
            "",
        ]
        out += [
            "| Métrica | Valor | Significado |",
            "|---|---|---|",
            f"| `hit_rate` | {data.hit_rate:.3f} | "
            "Fração de queries onde pelo menos 1 doc relevante apareceu no top-10. |",
            f"| `precision@10` | {data.precision_at_10:.3f} | "
            "Dos 10 docs devolvidos, quantos eram relevantes (em média). |",
            f"| `recall@10` | {data.recall_at_10:.3f} | "
            "Dos docs relevantes que existiam, quantos foram recuperados. |",
            f"| `ndcg@10` | {data.ndcg_at_10:.3f} | "
            "Qualidade do ranking (0=ruim, 1=ideal). Penaliza relevante embaixo. |",
            "",
        ]
        return "\n".join(out)
