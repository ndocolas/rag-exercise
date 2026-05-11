from rag_eval.db.vector_store import RetrievedChunk
from rag_eval.services.chunking.chunker import Chunk
from rag_eval.services.retrieval.retriever import Retriever


def test_unique_docs_preserves_order():
    retrieved = [
        RetrievedChunk(Chunk("c1", "doc_A", "x"), 0.9),
        RetrievedChunk(Chunk("c2", "doc_B", "x"), 0.85),
        RetrievedChunk(Chunk("c3", "doc_A", "x"), 0.8),
        RetrievedChunk(Chunk("c4", "doc_C", "x"), 0.7),
    ]
    docs = Retriever._unique_docs(retrieved)
    assert docs == ["doc_A", "doc_B", "doc_C"]
