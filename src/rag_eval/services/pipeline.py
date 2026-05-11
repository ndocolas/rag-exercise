from __future__ import annotations

import time
from dataclasses import dataclass

from rag_eval.db.vector_store import RetrievedChunk
from rag_eval.services.embeddings.embedder import Embedder
from rag_eval.services.generation.llm_client import FuelixLLMClient
from rag_eval.services.generation.rag_prompt import RAGPrompt
from rag_eval.services.retrieval.retriever import Retriever


@dataclass(frozen=True)
class PipelineAnswer:
    answer: str
    retrieved: list[RetrievedChunk]
    doc_ids_unique: list[str]
    latency_ms: dict[str, float]


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
        t0 = time.perf_counter()
        result = await self._retriever.retrieve(question, top_k=top_k)
        t1 = time.perf_counter()
        messages = self._prompt.build(question, result.chunks)
        answer = await self._llm.complete(messages)
        t2 = time.perf_counter()
        return PipelineAnswer(
            answer=answer,
            retrieved=result.chunks,
            doc_ids_unique=result.doc_ids_unique,
            latency_ms={
                "retrieve": (t1 - t0) * 1000,
                "generate": (t2 - t1) * 1000,
                "total": (t2 - t0) * 1000,
            },
        )
