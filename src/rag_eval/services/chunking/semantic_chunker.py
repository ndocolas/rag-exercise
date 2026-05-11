from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from rag_eval.services.chunking.chunker import Chunk, Chunker
from rag_eval.services.data.fiqa_dataset import Document


class SemanticChunker(Chunker):
    """Splits documents at points where adjacent-sentence embedding similarity drops.

    Uses a percentile threshold over the distribution of cosine distances between
    consecutive sentences. Distances above the threshold mark a chunk boundary.
    A precomputed sentence embedder must be supplied; the chunker calls it
    synchronously via ``embed_sync``.
    """

    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

    def __init__(self, sentence_embedder, percentile: int = 95, max_sentences: int = 12):
        self._embedder = sentence_embedder
        self._percentile = percentile
        self._max_sentences = max_sentences

    @property
    def name(self) -> str:
        return f"semantic_p{self._percentile}"

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            text = self._compose(doc)
            sentences = self._split_sentences(text)
            if len(sentences) <= 1:
                chunks.append(Chunk(self._chunk_id(doc.doc_id, 0), doc.doc_id, text))
                continue

            vecs = np.asarray(self._embedder.embed_sync(sentences))
            distances = self._consecutive_distances(vecs)
            threshold = float(np.percentile(distances, self._percentile)) if len(distances) else 0.0

            buckets = self._group(sentences, distances, threshold)
            for idx, bucket in enumerate(buckets):
                chunks.append(
                    Chunk(
                        chunk_id=self._chunk_id(doc.doc_id, idx),
                        doc_id=doc.doc_id,
                        text=" ".join(bucket).strip(),
                    )
                )
        return chunks

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        sentences = cls._SENTENCE_SPLIT.split(text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _consecutive_distances(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normed = vecs / np.clip(norms, 1e-12, None)
        sims = np.sum(normed[:-1] * normed[1:], axis=1)
        return 1.0 - sims

    def _group(
        self, sentences: list[str], distances: np.ndarray, threshold: float
    ) -> list[list[str]]:
        buckets: list[list[str]] = [[sentences[0]]]
        for i, dist in enumerate(distances, start=1):
            cur = buckets[-1]
            if dist >= threshold or len(cur) >= self._max_sentences:
                buckets.append([sentences[i]])
            else:
                cur.append(sentences[i])
        return buckets

    @staticmethod
    def _compose(doc: Document) -> str:
        return f"{doc.title}\n\n{doc.text}".strip() if doc.title else doc.text
