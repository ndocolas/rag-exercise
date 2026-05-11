from __future__ import annotations

from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_eval.services.chunking.chunker import Chunk, Chunker
from rag_eval.services.data.fiqa_dataset import Document


class HierarchicalChunker(Chunker):
    """Two-level chunker: parent windows + child sub-chunks.

    Children are indexed in the vector store; the chunk metadata carries the
    parent text so the retriever can swap to parent context before generation.
    """

    def __init__(
        self,
        parent_size: int = 1024,
        child_size: int = 256,
        parent_overlap: int = 128,
        child_overlap: int = 32,
        encoding: str = "cl100k_base",
    ):
        self._parent_size = parent_size
        self._child_size = child_size
        self._parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding,
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
        )
        self._child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding,
            chunk_size=child_size,
            chunk_overlap=child_overlap,
        )

    @property
    def name(self) -> str:
        return f"hierarchical_{self._parent_size}_{self._child_size}"

    def chunk(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            text = self._compose(doc)
            parents = self._parent_splitter.split_text(text)
            for p_idx, parent_text in enumerate(parents):
                parent_id = f"{doc.doc_id}::p{p_idx}"
                children = self._child_splitter.split_text(parent_text)
                for c_idx, child_text in enumerate(children):
                    chunks.append(
                        Chunk(
                            chunk_id=self._chunk_id(doc.doc_id, c_idx, parent_idx=p_idx),
                            doc_id=doc.doc_id,
                            text=child_text,
                            parent_id=parent_id,
                            metadata={"parent_text": parent_text},
                        )
                    )
        return chunks

    @staticmethod
    def _compose(doc: Document) -> str:
        return f"{doc.title}\n\n{doc.text}".strip() if doc.title else doc.text
