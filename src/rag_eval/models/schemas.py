from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EmbedderAlias = Literal["openai", "bge-small", "bge-large"]

EMBEDDER_TO_PIPELINE: dict[str, str] = {
    "openai": "P1",
    "bge-small": "P2",
    "bge-large": "P3",
}

EMBEDDER_LABELS: dict[str, str] = {
    "openai": "OpenAI text-embedding-3-small",
    "bge-small": "BGE small (BAAI/bge-small-en-v1.5)",
    "bge-large": "BGE large (BAAI/bge-large-en-v1.5)",
}


class RunExperimentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    subsample_size: int | None = None
    pipeline_filter: list[str] | None = None
    queries_limit: int | None = None
    top_k: int = 10
    judge_model: str | None = None
    force_reindex: bool = False
    force_regen: bool = False
    skip_llm_judges: bool = False


class RunExperimentResponse(BaseModel):
    experiment_id: str
    status: str
    pipelines_count: int
    queries_count: int


class ExperimentStatusResponse(BaseModel):
    experiment_id: str
    name: str
    status: str
    progress: dict
    started_at: float
    updated_at: float
    error: str | None = None


class PipelineInfo(BaseModel):
    pipeline_id: str
    chunking: str
    embedder: str
    collection: str


class AskRequest(BaseModel):
    """Single-question RAG. `embedder` is a friendly alias; chunking is fixed."""

    question: str = Field(..., min_length=1)
    embedder: EmbedderAlias = "openai"
    top_k: int = Field(default=5, ge=1, le=20)
    with_control: bool = True


class CompareRequest(BaseModel):
    """Same question, same chunking, same LLM — only the embedder changes.
    Always runs the three didactic embedders side-by-side.
    """

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AskJSONResponse(BaseModel):
    """Minimal JSON shape: question, answer, retrieved_documents (numbered text-only),
    plus an optional `response_without_rag` when `with_control=true`."""

    question: str
    answer: str
    retrieved_documents: dict[str, str]
    response_without_rag: str | None = None


class CompareEmbedderResult(BaseModel):
    """One side of `/compare`: either (answer + retrieved_documents) or `error`."""

    answer: str | None = None
    retrieved_documents: dict[str, str] | None = None
    error: str | None = None


class CompareJSONResponse(BaseModel):
    question: str
    results: dict[str, CompareEmbedderResult]


class IndexEmbedderResult(BaseModel):
    embedder: EmbedderAlias
    embedder_label: str
    pipeline_id: str
    collection: str
    status: Literal["already_indexed", "indexed", "failed"]
    chunks_indexed: int
    duration_ms: float
    error: str | None = None


class IndexResponse(BaseModel):
    corpus_size: int
    embedders: list[IndexEmbedderResult]


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    fuelix_configured: bool


class BenchmarkRequest(BaseModel):
    """Smoke-sized defaults so the call returns in a few minutes.

    Heavy runs (judges enabled, large subsamples) should still go through
    ``POST /experiments/run`` and the async polling flow.
    """

    name: str = Field(default="full-benchmark", min_length=1, max_length=100)
    subsample_size: int = 500
    queries_limit: int = 20
    top_k: int = 10
    skip_llm_judges: bool = True
    force_reindex: bool = False
    force_regen: bool = False
    pipeline_filter: list[str] | None = None


class PipelineScore(BaseModel):
    pipeline_id: str
    composite_score: float


class BenchmarkWinner(BaseModel):
    pipeline_id: str
    chunking: str
    embedder: str
    composite_score: float
    headline: str


class LatencyEntry(BaseModel):
    pipeline_id: str
    avg_latency_ms: float


class BenchmarkSummary(BaseModel):
    winner: BenchmarkWinner | None
    best_per_chunking: dict[str, PipelineScore]
    best_per_embedder: dict[str, PipelineScore]
    fastest: LatencyEntry | None
    slowest: LatencyEntry | None
    takeaways: list[str]


class BenchmarkRanking(BaseModel):
    rank: int
    pipeline_id: str
    chunking: str
    embedder: str
    composite_score: float
    ndcg_at_10: float | None = None
    recall_at_10: float | None = None
    mrr: float | None = None
    avg_latency_ms: float | None = None


class FlowPhase(BaseModel):
    phase: str
    detail: str
    duration_ms: float
    data: dict | None = None


class BenchmarkResponse(BaseModel):
    experiment_id: str
    status: str
    duration_seconds: float
    pipelines_run: list[str]
    flow: list[FlowPhase]
    summary: BenchmarkSummary
    ranking: list[BenchmarkRanking]
    report_markdown: str
