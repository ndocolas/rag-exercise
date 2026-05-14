from __future__ import annotations

import time
from dataclasses import dataclass

from rag_eval.db.vector_store import RetrievedChunk
from rag_eval.services.embeddings.embedder import Embedder
from rag_eval.services.generation.llm_client import FuelixLLMClient
from rag_eval.services.generation.rag_prompt import RAGPrompt
from rag_eval.services.retrieval.retriever import Retriever


@dataclass(frozen=True)
class TraceEvent:
    step: str
    detail: str
    duration_ms: float | None = None
    data: dict | None = None


@dataclass(frozen=True)
class PipelineAnswer:
    answer: str
    retrieved: list[RetrievedChunk]
    doc_ids_unique: list[str]
    latency_ms: dict[str, float]
    trace: list[TraceEvent]


class RAGPipeline:
    """End-to-end RAG runtime composed from injected collaborators.

    Pure composition: no construction logic here. The matrix builder owns
    instantiation; this class only orchestrates retrieve → prompt → generate.
    """

    def __init__(
        self,
        pipeline_id: str,
        embedder: Embedder,
        retriever: Retriever,
        llm: FuelixLLMClient,
        prompt: RAGPrompt,
    ):
        self._pipeline_id = pipeline_id
        self._embedder = embedder
        self._retriever = retriever
        self._llm = llm
        self._prompt = prompt

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    async def answer(self, question: str, top_k: int | None = None) -> PipelineAnswer:
        k = top_k or self._retriever._top_k
        trace: list[TraceEvent] = []
        trace.append(
            TraceEvent(
                step="question",
                detail=f"pipeline={self._pipeline_id} top_k={k}",
                data={"question": question, "pipeline_id": self._pipeline_id, "top_k": k},
            )
        )
        t0 = time.perf_counter()
        result = await self._retriever.retrieve(question, top_k=top_k)
        t1 = time.perf_counter()
        retrieve_ms = (t1 - t0) * 1000
        trace.append(
            TraceEvent(
                step="retrieve",
                detail=(
                    f"collection={self._retriever.collection} "
                    f"hits={len(result.chunks)} unique_docs={len(result.doc_ids_unique)}"
                ),
                duration_ms=retrieve_ms,
                data={
                    "collection": self._retriever.collection,
                    "hits": len(result.chunks),
                    "unique_docs": len(result.doc_ids_unique),
                },
            )
        )
        for rank, rc in enumerate(result.chunks, start=1):
            preview = rc.chunk.text[:200].replace("\n", " ")
            trace.append(
                TraceEvent(
                    step="chunk",
                    detail=(
                        f"#{rank} doc={rc.chunk.doc_id} score={rc.score:.4f} "
                        f"| {preview}{'…' if len(rc.chunk.text) > 200 else ''}"
                    ),
                    data={
                        "rank": rank,
                        "chunk_id": rc.chunk.chunk_id,
                        "doc_id": rc.chunk.doc_id,
                        "score": rc.score,
                        "preview": preview,
                        "chars": len(rc.chunk.text),
                    },
                )
            )
        messages = self._prompt.build(question, result.chunks)
        trace.append(
            TraceEvent(
                step="prompt",
                detail=f"messages={len(messages)} contexts={len(result.chunks)}",
                data={"messages": len(messages), "contexts": len(result.chunks)},
            )
        )
        answer = await self._llm.complete(messages)
        t2 = time.perf_counter()
        generate_ms = (t2 - t1) * 1000
        trace.append(
            TraceEvent(
                step="generate",
                detail=f"answer_chars={len(answer)}",
                duration_ms=generate_ms,
                data={"answer_chars": len(answer)},
            )
        )
        total_ms = (t2 - t0) * 1000
        trace.append(
            TraceEvent(
                step="done",
                detail=f"total={total_ms:.1f}ms",
                duration_ms=total_ms,
            )
        )
        return PipelineAnswer(
            answer=answer,
            retrieved=result.chunks,
            doc_ids_unique=result.doc_ids_unique,
            latency_ms={
                "retrieve": retrieve_ms,
                "generate": generate_ms,
                "total": total_ms,
            },
            trace=trace,
        )
