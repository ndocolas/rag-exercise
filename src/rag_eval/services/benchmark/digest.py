"""Build a glance-able summary of a benchmark run.

The ``ReportGenerator`` writes a full markdown table + JSON dump for offline
analysis. This module produces a much smaller, structured digest meant to be
returned inline from ``POST /benchmark`` so a `.rest` client can answer
"which embedding / chunking wins?" without opening files.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DigestPipelineScore:
    pipeline_id: str
    composite_score: float


@dataclass(frozen=True)
class DigestWinner:
    pipeline_id: str
    chunking: str
    embedder: str
    composite_score: float
    headline: str


@dataclass(frozen=True)
class DigestLatency:
    pipeline_id: str
    avg_latency_ms: float


@dataclass(frozen=True)
class DigestSummary:
    winner: DigestWinner | None
    best_per_chunking: dict[str, DigestPipelineScore]
    best_per_embedder: dict[str, DigestPipelineScore]
    fastest: DigestLatency | None
    slowest: DigestLatency | None
    takeaways: list[str]


@dataclass(frozen=True)
class DigestRanking:
    rank: int
    pipeline_id: str
    chunking: str
    embedder: str
    composite_score: float
    ndcg_at_10: float | None
    recall_at_10: float | None
    mrr: float | None
    avg_latency_ms: float | None


@dataclass(frozen=True)
class Digest:
    summary: DigestSummary
    ranking: list[DigestRanking]


def _short_embedder(embedder: str) -> str:
    """Drop the ``kind:`` prefix and any ``org/`` path for friendlier strings."""
    after_colon = embedder.split(":", 1)[-1]
    return after_colon.rsplit("/", 1)[-1]


def _short_chunking(chunking: str) -> str:
    return chunking


def _avg_latency_by_pipeline(per_query: pd.DataFrame) -> dict[str, float]:
    """Mean of the ``total`` latency key across rows for each pipeline.

    ``latency_ms`` is a per-row dict with keys retrieve/generate/total. If a
    row is missing the key we skip it rather than blowing up the digest.
    """
    if per_query.empty or "latency_ms" not in per_query.columns:
        return {}
    out: dict[str, list[float]] = {}
    for _, row in per_query.iterrows():
        latency = row["latency_ms"]
        if not isinstance(latency, dict):
            continue
        total = latency.get("total")
        if total is None:
            continue
        out.setdefault(str(row["pipeline_id"]), []).append(float(total))
    return {pid: sum(xs) / len(xs) for pid, xs in out.items() if xs}


def _best_by(df: pd.DataFrame, group_col: str) -> dict[str, DigestPipelineScore]:
    """For each value of ``group_col`` pick the row with the max composite_score.

    Tie-break stably on ``pipeline_id`` so reruns produce the same digest.
    """
    if df.empty:
        return {}
    out: dict[str, DigestPipelineScore] = {}
    for group_value, sub in df.groupby(group_col):
        sub = sub.sort_values(
            ["composite_score", "pipeline_id"], ascending=[False, True]
        )
        top = sub.iloc[0]
        out[str(group_value)] = DigestPipelineScore(
            pipeline_id=str(top["pipeline_id"]),
            composite_score=float(top["composite_score"]),
        )
    return out


def _format_pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f}%"


def _build_takeaways(
    ranked: pd.DataFrame,
    best_chunking: dict[str, DigestPipelineScore],
    best_embedder: dict[str, DigestPipelineScore],
) -> list[str]:
    """Two-to-three deterministic strings derived from group means.

    No LLM. Just describes who wins which dimension and by how much.
    """
    takeaways: list[str] = []
    if ranked.empty:
        return ["No results available."]

    chunking_means = (
        ranked.groupby("chunking")["composite_score"].mean().sort_values(ascending=False)
    )
    if len(chunking_means) >= 2:
        top_ch, top_score = chunking_means.index[0], float(chunking_means.iloc[0])
        deltas = [
            f"{_format_pct((top_score - float(v)) / float(v) if v else 0.0)} vs {k}"
            for k, v in chunking_means.iloc[1:].items()
        ]
        takeaways.append(
            f"Best chunking on average: {_short_chunking(top_ch)} ({', '.join(deltas)})."
        )

    embedder_means = (
        ranked.groupby("embedder")["composite_score"].mean().sort_values(ascending=False)
    )
    if len(embedder_means) >= 2:
        top_em = embedder_means.index[0]
        win_count = sum(1 for v in best_chunking.values() if _embedder_for_pid(ranked, v.pipeline_id) == top_em)
        total = len(best_chunking) or 1
        takeaways.append(
            f"Best embedder on average: {_short_embedder(top_em)} "
            f"(wins {win_count} of {total} chunkings)."
        )

    if best_embedder:
        names = ", ".join(_short_embedder(e) for e in best_embedder)
        takeaways.append(f"Per-embedder champion pipelines: {names}.")

    return takeaways or ["Insufficient data to derive takeaways."]


def _embedder_for_pid(df: pd.DataFrame, pid: str) -> str:
    sub = df[df["pipeline_id"] == pid]
    return str(sub.iloc[0]["embedder"]) if not sub.empty else ""


def _ndcg_col(df: pd.DataFrame) -> str | None:
    """Find the nDCG metric column, robust to ``@10`` vs ``@5``."""
    for col in df.columns:
        if col.startswith("metric.ndcg@"):
            return col
    return None


def _recall_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.startswith("metric.recall@"):
            return col
    return None


def _build_ranking(
    ranked: pd.DataFrame, latency: dict[str, float]
) -> list[DigestRanking]:
    ndcg_col = _ndcg_col(ranked)
    recall_col = _recall_col(ranked)
    out: list[DigestRanking] = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        pid = str(row["pipeline_id"])
        out.append(
            DigestRanking(
                rank=rank,
                pipeline_id=pid,
                chunking=str(row["chunking"]),
                embedder=str(row["embedder"]),
                composite_score=float(row["composite_score"]),
                ndcg_at_10=float(row[ndcg_col]) if ndcg_col and pd.notna(row[ndcg_col]) else None,
                recall_at_10=(
                    float(row[recall_col]) if recall_col and pd.notna(row[recall_col]) else None
                ),
                mrr=float(row["metric.mrr"]) if "metric.mrr" in ranked.columns and pd.notna(row.get("metric.mrr")) else None,
                avg_latency_ms=latency.get(pid),
            )
        )
    return out


def build_digest(aggregate_df: pd.DataFrame, per_query_df: pd.DataFrame) -> Digest:
    """Compose the digest from ``ResultsStore.read_aggregate()`` + ``read_per_query()``.

    Both inputs may be empty (e.g. evaluator skipped everything); we return an
    empty-but-valid digest in that case so the endpoint always has a shape to
    serialize.
    """
    if aggregate_df.empty:
        empty_summary = DigestSummary(
            winner=None,
            best_per_chunking={},
            best_per_embedder={},
            fastest=None,
            slowest=None,
            takeaways=["No results available."],
        )
        return Digest(summary=empty_summary, ranking=[])

    ranked = aggregate_df.sort_values(
        ["composite_score", "pipeline_id"], ascending=[False, True]
    ).reset_index(drop=True)
    latency = _avg_latency_by_pipeline(per_query_df)

    top = ranked.iloc[0]
    runner_up_score = float(ranked.iloc[1]["composite_score"]) if len(ranked) > 1 else None
    headline_bits = [
        f"{_short_chunking(str(top['chunking']))} + "
        f"{_short_embedder(str(top['embedder']))} wins"
    ]
    if runner_up_score is not None and runner_up_score > 0:
        delta = (float(top["composite_score"]) - runner_up_score) / runner_up_score
        headline_bits.append(
            f"by {_format_pct(delta)} over {ranked.iloc[1]['pipeline_id']}"
        )
    winner = DigestWinner(
        pipeline_id=str(top["pipeline_id"]),
        chunking=str(top["chunking"]),
        embedder=str(top["embedder"]),
        composite_score=float(top["composite_score"]),
        headline=" ".join(headline_bits) + ".",
    )

    best_chunking = _best_by(ranked, "chunking")
    best_embedder = _best_by(ranked, "embedder")

    fastest = slowest = None
    if latency:
        f_pid = min(latency, key=latency.get)
        s_pid = max(latency, key=latency.get)
        fastest = DigestLatency(pipeline_id=f_pid, avg_latency_ms=latency[f_pid])
        slowest = DigestLatency(pipeline_id=s_pid, avg_latency_ms=latency[s_pid])

    takeaways = _build_takeaways(ranked, best_chunking, best_embedder)

    summary = DigestSummary(
        winner=winner,
        best_per_chunking=best_chunking,
        best_per_embedder=best_embedder,
        fastest=fastest,
        slowest=slowest,
        takeaways=takeaways,
    )
    ranking = _build_ranking(ranked, latency)
    return Digest(summary=summary, ranking=ranking)
