import math

from rag_eval.services.evaluation.retrieval_evaluator import RetrievalEvaluator


def test_perfect_retrieval():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    retrieved = {"q1": ["d1", "d2", "d3", "d4", "d5"]}
    result = RetrievalEvaluator(k=5).evaluate(retrieved, qrels)
    m = result.metrics
    assert m["recall@5"] == 1.0
    assert m["precision@5"] == 2 / 5
    assert m["mrr"] == 1.0
    assert m["hit@1"] == 1.0


def test_no_hit():
    qrels = {"q1": {"d99": 1}}
    retrieved = {"q1": ["d1", "d2", "d3"]}
    result = RetrievalEvaluator(k=3).evaluate(retrieved, qrels)
    assert result.metrics["recall@3"] == 0.0
    assert result.metrics["mrr"] == 0.0
    assert result.metrics["hit@1"] == 0.0


def test_mrr_second_position():
    qrels = {"q1": {"d2": 1}}
    retrieved = {"q1": ["d1", "d2", "d3"]}
    result = RetrievalEvaluator(k=3).evaluate(retrieved, qrels)
    assert math.isclose(result.metrics["mrr"], 0.5)


def test_ndcg_position_dependent():
    qrels = {"q1": {"d1": 1}}
    top_first = {"q1": ["d1", "d2", "d3"]}
    top_third = {"q1": ["dx", "dy", "d1"]}
    e = RetrievalEvaluator(k=3)
    a = e.evaluate(top_first, qrels).metrics["ndcg@3"]
    b = e.evaluate(top_third, qrels).metrics["ndcg@3"]
    assert a > b
