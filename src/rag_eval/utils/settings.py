from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    fuelix_api_key: str = Field(..., min_length=10)
    fuelix_base_url: str = "https://api.fuelix.ai/v1"

    qdrant_url: str = "http://localhost:6333"

    generator_model: str = "claude-sonnet-4-5"
    judge_model: str = "gpt-4o"

    log_level: str = "INFO"

    embedding_cache_path: Path = Path("data/cache/embeddings.sqlite")
    llm_cache_path: Path = Path("data/cache/llm.sqlite")
    experiment_store_path: Path = Path("data/experiments.sqlite")
    results_dir: Path = Path("data/results")
    fiqa_data_dir: Path = Path("data/fiqa")

    subsample_size: int = 10_000
    top_k: int = 10
    seed: int = 42

    fuelix_embed_concurrency: int = 8
    fuelix_embed_batch_size: int = 100
    fuelix_llm_concurrency: int = 4

    chunk_fixed_size: int = 512
    chunk_fixed_overlap: int = 64
    chunk_semantic_percentile: int = 95
    chunk_hierarchical_parent_size: int = 1024
    chunk_hierarchical_child_size: int = 256

    generator_temperature: float = 0.0
    generator_max_tokens: int = 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
