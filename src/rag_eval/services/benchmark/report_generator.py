from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from rag_eval.db.results_store import ResultsStore
from rag_eval.services.evaluation.statistics_calculator import StatisticsCalculator


@dataclass(frozen=True)
class Report:
    markdown: str
    json_payload: dict


class ReportGenerator:
    """Reads aggregate parquet, ranks pipelines, runs paired Wilcoxon between
    top-3, and writes ``report.md`` + ``report.json`` next to the parquet files.
    """

    def __init__(self, results_store: ResultsStore, seed: int = 42):
        self._store = results_store
        self._stats = StatisticsCalculator(seed=seed)

    def generate(self) -> Report:
        agg = self._store.read_aggregate()
        if agg.empty:
            empty = Report(markdown="# Report\n\n(no data)\n", json_payload={"pipelines": []})
            self._write(empty)
            return empty

        ranked = agg.sort_values("composite_score", ascending=False).reset_index(drop=True)
        per_query = self._store.read_per_query()

        top_pipelines = ranked["pipeline_id"].head(3).tolist()
        pairwise = self._pairwise(per_query, top_pipelines)

        json_payload = {
            "ranking": ranked.to_dict(orient="records"),
            "pairwise_top3": [t.__dict__ for t in pairwise],
        }
        markdown = self._render_markdown(ranked, pairwise)
        report = Report(markdown=markdown, json_payload=json_payload)
        self._write(report)
        return report

    def _pairwise(self, per_query: pd.DataFrame, pipelines: list[str]) -> list:
        if per_query.empty or len(pipelines) < 2:
            return []
        pivot: dict[str, dict[str, float]] = {}
        for pid in pipelines:
            sub = per_query[per_query["pipeline_id"] == pid]
            scores = {}
            for _, row in sub.iterrows():
                ctx_count = len(row["contexts"]) if isinstance(row["contexts"], list) else 0
                scores[row["query_id"]] = float(ctx_count)
            pivot[pid] = scores
        return self._stats.pairwise_wilcoxon(pivot, metric="contexts_returned")

    def _render_markdown(self, ranked: pd.DataFrame, pairwise: list) -> str:
        winner = ranked.iloc[0]
        lines = [
            "# RAG Evaluation Report",
            "",
            f"**Winner:** `{winner['pipeline_id']}` "
            f"(chunking=`{winner['chunking']}`, embedder=`{winner['embedder']}`)  ",
            f"**Composite score:** {winner['composite_score']:.4f} "
            f"[CI {winner['composite_ci_low']:.4f}–{winner['composite_ci_high']:.4f}]",
            "",
            "## Ranking",
            "",
        ]
        metric_cols = [c for c in ranked.columns if c.startswith("metric.")]
        failure_cols = [c for c in ranked.columns if c.startswith("failure.")]

        header = ["pipeline_id", "chunking", "embedder", "composite_score", *metric_cols]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for _, row in ranked.iterrows():
            cells = [str(row.get(c, "")) for c in header[:3]]
            cells.append(f"{row['composite_score']:.4f}")
            for c in metric_cols:
                v = row.get(c)
                cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "-")
            lines.append("| " + " | ".join(cells) + " |")

        lines += ["", "## Failure distribution", ""]
        if failure_cols:
            header_f = ["pipeline_id", *failure_cols]
            lines.append("| " + " | ".join(header_f) + " |")
            lines.append("| " + " | ".join("---" for _ in header_f) + " |")
            for _, row in ranked.iterrows():
                cells: list[str] = [str(row["pipeline_id"])]
                for c in failure_cols:
                    v = row.get(c)
                    cells.append(f"{v:.2%}" if isinstance(v, int | float) else "-")
                lines.append("| " + " | ".join(cells) + " |")

        if pairwise:
            lines += ["", "## Paired Wilcoxon (top-3, Bonferroni-adjusted)", ""]
            lines.append("| A | B | metric | statistic | p_adj | effect |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for t in pairwise:
                lines.append(
                    f"| {t.pipeline_a} | {t.pipeline_b} | {t.metric} | "
                    f"{t.statistic:.3f} | {t.p_value_adjusted:.4f} | {t.effect_size:+.3f} |"
                )

        lines += [
            "",
            "## Recommendation",
            "",
            f"Use **`{winner['pipeline_id']}`** "
            f"(chunking=`{winner['chunking']}`, embedder=`{winner['embedder']}`) "
            f"for the collaborative project. Composite {winner['composite_score']:.4f}.",
        ]
        return "\n".join(lines) + "\n"

    def _write(self, report: Report) -> None:
        out_dir = self._store.report_dir
        (out_dir / "report.md").write_text(report.markdown, encoding="utf-8")
        (out_dir / "report.json").write_text(
            json.dumps(report.json_payload, indent=2, default=str), encoding="utf-8"
        )
