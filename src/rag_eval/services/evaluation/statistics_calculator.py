from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    median: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairwiseTest:
    pipeline_a: str
    pipeline_b: str
    metric: str
    statistic: float
    p_value: float
    p_value_adjusted: float
    effect_size: float


class StatisticsCalculator:
    """Bootstrap CIs and paired Wilcoxon tests with Bonferroni correction."""

    def __init__(self, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42):
        self._n_bootstrap = n_bootstrap
        self._ci = ci
        self._seed = seed

    def summarize(self, values: list[float]) -> MetricSummary:
        clean = [v for v in values if v is not None and not _is_nan(v)]
        if not clean:
            nan = float("nan")
            return MetricSummary(nan, nan, nan, nan, nan)
        arr = np.asarray(clean, dtype=float)
        rng = np.random.default_rng(self._seed)
        n = len(arr)
        boot_means = np.empty(self._n_bootstrap, dtype=float)
        for i in range(self._n_bootstrap):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means[i] = sample.mean()
        alpha = (1 - self._ci) / 2
        return MetricSummary(
            mean=float(arr.mean()),
            std=float(arr.std(ddof=1)) if n > 1 else 0.0,
            median=float(np.median(arr)),
            ci_low=float(np.quantile(boot_means, alpha)),
            ci_high=float(np.quantile(boot_means, 1 - alpha)),
        )

    def pairwise_wilcoxon(
        self,
        per_query_values: dict[str, dict[str, float]],
        metric: str,
    ) -> list[PairwiseTest]:
        """``per_query_values[pipeline_id][query_id] = score``. Drops queries
        missing in either pipeline before testing."""
        pipelines = sorted(per_query_values.keys())
        tests: list[PairwiseTest] = []
        n_pairs = max(1, len(pipelines) * (len(pipelines) - 1) // 2)

        for i, a in enumerate(pipelines):
            for b in pipelines[i + 1 :]:
                shared = sorted(set(per_query_values[a]) & set(per_query_values[b]))
                if len(shared) < 5:
                    continue
                xs = np.asarray([per_query_values[a][q] for q in shared], dtype=float)
                ys = np.asarray([per_query_values[b][q] for q in shared], dtype=float)
                diffs = xs - ys
                if np.allclose(diffs, 0):
                    p, statistic = 1.0, 0.0
                else:
                    try:
                        result = stats.wilcoxon(
                            xs, ys, zero_method="wilcox", alternative="two-sided"
                        )
                        statistic = float(result.statistic)  # type: ignore[attr-defined]
                        p = float(result.pvalue)  # type: ignore[attr-defined]
                    except ValueError:
                        statistic, p = 0.0, 1.0
                effect = float(np.mean(np.sign(diffs)))
                tests.append(
                    PairwiseTest(
                        pipeline_a=a,
                        pipeline_b=b,
                        metric=metric,
                        statistic=statistic,
                        p_value=p,
                        p_value_adjusted=min(1.0, p * n_pairs),
                        effect_size=effect,
                    )
                )
        return tests

    def composite_score(self, metrics: dict[str, float], weights: dict[str, float]) -> float:
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 0.0
        score = 0.0
        for name, w in weights.items():
            value = metrics.get(name)
            if value is None or _is_nan(value):
                continue
            score += w * value
        return score / total_weight


def _is_nan(x: float) -> bool:
    return isinstance(x, float) and x != x
