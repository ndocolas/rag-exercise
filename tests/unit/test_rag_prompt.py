from rag_eval.db.vector_store import RetrievedChunk
from rag_eval.services.chunking.chunker import Chunk
from rag_eval.services.generation.rag_prompt import RAGPrompt


def test_build_messages_with_contexts():
    prompt = RAGPrompt()
    chunks = [
        RetrievedChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="alpha"), score=0.9),
        RetrievedChunk(chunk=Chunk(chunk_id="c2", doc_id="d2", text="beta"), score=0.8),
    ]
    msgs = prompt.build("What?", chunks)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "[1] alpha" in msgs[1]["content"]
    assert "[2] beta" in msgs[1]["content"]
    assert "What?" in msgs[1]["content"]


def test_uses_parent_text_when_present():
    prompt = RAGPrompt()
    chunks = [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id="c1",
                doc_id="d1",
                text="child snippet",
                metadata={"parent_text": "full parent passage"},
            ),
            score=0.9,
        )
    ]
    msgs = prompt.build("Q?", chunks)
    assert "full parent passage" in msgs[1]["content"]
    assert "child snippet" not in msgs[1]["content"]
