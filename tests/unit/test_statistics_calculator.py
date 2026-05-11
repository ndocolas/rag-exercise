import math

import numpy as np

from rag_eval.services.evaluation.statistics_calculator import StatisticsCalculator


def test_summarize_basic():
    calc = StatisticsCalculator(n_bootstrap=200, seed=1)
    s = calc.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(s.mean, 3.0)
    assert math.isclose(s.median, 3.0)
    assert s.ci_low <= s.mean <= s.ci_high


def test_summarize_filters_nan():
    calc = StatisticsCalculator(n_bootstrap=100, seed=1)
    s = calc.summarize([float("nan"), 1.0, 2.0])
    assert math.isclose(s.mean, 1.5)


def test_pairwise_wilcoxon_detects_difference():
    calc = StatisticsCalculator(seed=1)
    rng = np.random.default_rng(0)
    queries = [f"q{i}" for i in range(50)]
    a = {q: 0.8 + rng.normal(0, 0.05) for q in queries}
    b = {q: 0.5 + rng.normal(0, 0.05) for q in queries}
    tests = calc.pairwise_wilcoxon({"A": a, "B": b}, metric="x")
    assert len(tests) == 1
    assert tests[0].p_value < 0.01


def test_composite_score():
    calc = StatisticsCalculator()
    score = calc.composite_score({"x": 0.8, "y": 0.4}, {"x": 1.0, "y": 1.0})
    assert math.isclose(score, 0.6)
