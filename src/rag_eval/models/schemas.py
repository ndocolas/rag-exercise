from __future__ import annotations

from pydantic import BaseModel, Field


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


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    pipeline_id: str | None = None
    top_k: int = 10
    include_contexts: bool = True


class RetrievedContext(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    pipeline_id: str
    contexts: list[RetrievedContext] | None = None
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    fuelix_configured: bool
