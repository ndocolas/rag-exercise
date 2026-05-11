from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class QueryResult:
    pipeline_id: str
    query_id: str
    question: str
    answer: str
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    contexts: list[str]
    scores: list[float]
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class AggregateResult:
    pipeline_id: str
    chunking: str
    embedder: str
    metrics: dict[str, float]
    failures: dict[str, float]
    composite_score: float
    composite_ci_low: float
    composite_ci_high: float


class ResultsStore:
    """Writes per-query and aggregate parquet files for one experiment run."""

    def __init__(self, base_dir: Path, run_id: str):
        self._dir = Path(base_dir) / run_id
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def per_query_path(self) -> Path:
        return self._dir / "per_query.parquet"

    @property
    def aggregate_path(self) -> Path:
        return self._dir / "aggregate.parquet"

    @property
    def failures_path(self) -> Path:
        return self._dir / "failures.parquet"

    @property
    def report_dir(self) -> Path:
        return self._dir

    def write_per_query(self, rows: list[QueryResult]) -> None:
        records = []
        for r in rows:
            d = asdict(r)
            d["retrieved_chunk_ids"] = json.dumps(d["retrieved_chunk_ids"])
            d["retrieved_doc_ids"] = json.dumps(d["retrieved_doc_ids"])
            d["contexts"] = json.dumps(d["contexts"])
            d["scores"] = json.dumps(d["scores"])
            d["latency_ms"] = json.dumps(d["latency_ms"])
            records.append(d)
        df = pd.DataFrame(records)
        if self.per_query_path.exists():
            existing = pd.read_parquet(self.per_query_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(self.per_query_path, index=False)

    def read_per_query(self, pipeline_id: str | None = None) -> pd.DataFrame:
        if not self.per_query_path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self.per_query_path)
        if pipeline_id:
            df = df[df["pipeline_id"] == pipeline_id].copy()
        for col in ("retrieved_chunk_ids", "retrieved_doc_ids", "contexts", "scores", "latency_ms"):
            if col in df.columns:
                df[col] = df[col].apply(json.loads)  # type: ignore[union-attr]
        return df  # type: ignore[return-value]

    def write_aggregate(self, rows: list[AggregateResult]) -> None:
        records = []
        for r in rows:
            base = {
                "pipeline_id": r.pipeline_id,
                "chunking": r.chunking,
                "embedder": r.embedder,
                "composite_score": r.composite_score,
                "composite_ci_low": r.composite_ci_low,
                "composite_ci_high": r.composite_ci_high,
            }
            for k, v in r.metrics.items():
                base[f"metric.{k}"] = v
            for k, v in r.failures.items():
                base[f"failure.{k}"] = v
            records.append(base)
        df = pd.DataFrame(records)
        df.to_parquet(self.aggregate_path, index=False)

    def read_aggregate(self) -> pd.DataFrame:
        if not self.aggregate_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.aggregate_path)

    def write_failures(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.failures_path, index=False)
