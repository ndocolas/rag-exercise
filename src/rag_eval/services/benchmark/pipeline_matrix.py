from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    chunking: str  # "fixed" | "semantic" | "hierarchical"
    embedder: str  # "fuelix:text-embedding-3-small" | "local:bge-small" | "local:bge-m3"
    chunker_kwargs: dict
    embedder_kwargs: dict

    def collection_name(self) -> str:
        embed_short = self.embedder.split(":", 1)[-1].replace("/", "__")
        return f"fiqa__{self.chunking}__{embed_short}"


class PipelineMatrix:
    """Defines the 9 pipelines in the design spec.

    Order is stable (P1..P9) so reports remain comparable across runs. Filter
    by ID to run a subset.
    """

    CHUNKINGS = (
        ("fixed", {"chunk_size": 512, "chunk_overlap": 64}),
        ("semantic", {"percentile": 95}),
        ("hierarchical", {"parent_size": 1024, "child_size": 256}),
    )

    EMBEDDERS = (
        ("fuelix:text-embedding-3-small", {"model": "text-embedding-3-small"}),
        ("local:BAAI/bge-small-en-v1.5", {"model": "BAAI/bge-small-en-v1.5"}),
        ("local:BAAI/bge-m3", {"model": "BAAI/bge-m3"}),
    )

    @classmethod
    def build(cls, filter_ids: list[str] | None = None) -> list[PipelineSpec]:
        specs: list[PipelineSpec] = []
        idx = 1
        for chunking, ck_kwargs in cls.CHUNKINGS:
            for embedder, em_kwargs in cls.EMBEDDERS:
                pid = f"P{idx}"
                idx += 1
                if filter_ids and pid not in filter_ids:
                    continue
                specs.append(
                    PipelineSpec(
                        pipeline_id=pid,
                        chunking=chunking,
                        embedder=embedder,
                        chunker_kwargs=dict(ck_kwargs),
                        embedder_kwargs=dict(em_kwargs),
                    )
                )
        return specs
