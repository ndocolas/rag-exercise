from __future__ import annotations

from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_eval.services.chunking.chunker import Chunk, Chunker
from rag_eval.services.data.fiqa_dataset import Document


class FixedChunker(Chunker):
    """Token-windowed chunker with overlap. Defaults match the design spec (512/64)."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding: str = "cl100k_base",
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def name(self) -> str:
        return f"fixed_{self._chunk_size}_{self._chunk_overlap}"

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            text = self._compose(doc)
            for idx, piece in enumerate(self._splitter.split_text(text)):
                chunks.append(
                    Chunk(
                        chunk_id=self._chunk_id(doc.doc_id, idx),
                        doc_id=doc.doc_id,
                        text=piece,
                    )
                )
        return chunks

    @staticmethod
    def _compose(doc: Document) -> str:
        return f"{doc.title}\n\n{doc.text}".strip() if doc.title else doc.text
