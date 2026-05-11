import tempfile
from pathlib import Path

import pytest

from rag_eval.db.embedding_cache import EmbeddingCache


def test_put_and_get_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cache = EmbeddingCache(Path(tmp) / "cache.sqlite")
        texts = ["hello", "world"]
        vecs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        cache.put_many("model-x", texts, vecs)
        out = cache.get_many("model-x", texts)
        assert out[0] == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)
        assert out[1] == pytest.approx([0.4, 0.5, 0.6], rel=1e-6)
        cache.close()


def test_miss_returns_partial():
    with tempfile.TemporaryDirectory() as tmp:
        cache = EmbeddingCache(Path(tmp) / "cache.sqlite")
        cache.put_many("m", ["a"], [[1.0]])
        out = cache.get_many("m", ["a", "b"])
        assert 0 in out
        assert 1 not in out
        cache.close()


def test_model_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        cache = EmbeddingCache(Path(tmp) / "cache.sqlite")
        cache.put_many("m1", ["a"], [[1.0]])
        out = cache.get_many("m2", ["a"])
        assert out == {}
        cache.close()
