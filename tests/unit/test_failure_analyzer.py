from rag_eval.services.evaluation.failure_analyzer import FailureAnalyzer


def test_success_when_retrieval_and_gen_ok():
    fa = FailureAnalyzer()
    summary = fa.categorize(
        retrieved_docs_per_query={"q1": ["d1"]},
        qrels={"q1": {"d1": 1}},
        answers_per_query={"q1": "answer"},
        ragas_per_query={"q1": {"faithfulness": 0.9, "answer_relevancy": 0.9}},
    )
    assert summary.per_query[0].bucket == "success"


def test_retrieval_miss():
    fa = FailureAnalyzer()
    summary = fa.categorize(
        retrieved_docs_per_query={"q1": ["d99"]},
        qrels={"q1": {"d1": 1}},
        answers_per_query={"q1": "answer"},
        ragas_per_query={"q1": {"faithfulness": 0.9, "answer_relevancy": 0.9}},
    )
    assert summary.per_query[0].bucket == "retrieval_miss"


def test_generation_fail_hallucination():
    fa = FailureAnalyzer()
    summary = fa.categorize(
        retrieved_docs_per_query={"q1": ["d1"]},
        qrels={"q1": {"d1": 1}},
        answers_per_query={"q1": "answer"},
        ragas_per_query={"q1": {"faithfulness": 0.2, "answer_relevancy": 0.9}},
    )
    cat = summary.per_query[0]
    assert cat.bucket == "generation_fail"
    assert cat.sub_category == "hallucination"


def test_cascade_fail():
    fa = FailureAnalyzer()
    summary = fa.categorize(
        retrieved_docs_per_query={"q1": ["dX"]},
        qrels={"q1": {"d1": 1}},
        answers_per_query={"q1": "answer"},
        ragas_per_query={"q1": {"faithfulness": 0.1, "answer_relevancy": 0.1}},
    )
    assert summary.per_query[0].bucket == "cascade_fail"


def test_refusal_subcategory():
    fa = FailureAnalyzer()
    summary = fa.categorize(
        retrieved_docs_per_query={"q1": ["d1"]},
        qrels={"q1": {"d1": 1}},
        answers_per_query={"q1": "I cannot answer based on the provided context."},
        ragas_per_query={"q1": {"faithfulness": 0.9, "answer_relevancy": 0.1}},
    )
    cat = summary.per_query[0]
    assert cat.bucket == "generation_fail"
    assert cat.sub_category == "refusal"
