from rag_eval.services.chunking.fixed_chunker import FixedChunker
from rag_eval.services.data.fiqa_dataset import Document


def test_short_doc_single_chunk():
    chunker = FixedChunker(chunk_size=512, chunk_overlap=64)
    doc = Document(doc_id="d1", text="Short text.")
    chunks = chunker.chunk([doc])
    assert len(chunks) == 1
    assert chunks[0].doc_id == "d1"
    assert chunks[0].chunk_id == "d1::c0"


def test_long_doc_splits():
    chunker = FixedChunker(chunk_size=50, chunk_overlap=5)
    text = " ".join([f"sentence number {i}." for i in range(200)])
    doc = Document(doc_id="d1", text=text)
    chunks = chunker.chunk([doc])
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"d1::c{i}"
        assert c.doc_id == "d1"


def test_deterministic_chunk_ids():
    chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
    doc = Document(doc_id="d42", text="lorem ipsum dolor sit amet " * 50)
    a = chunker.chunk([doc])
    b = chunker.chunk([doc])
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_name():
    assert FixedChunker(512, 64).name == "fixed_512_64"
