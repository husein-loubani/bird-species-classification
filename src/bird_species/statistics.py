"""
Uncertainty around accuracies and the paired comparison between two classifiers.

Both wrap statsmodels rather than reimplementing the formulas.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from bird_species.config import CONFIDENCE_LEVEL

# Below this many disagreements the chi-square approximation is poor, so the
# exact binomial version of McNemar's test is used instead.
EXACT_MCNEMAR_BELOW = 25


def accuracy_interval(correct: np.ndarray, level: float = CONFIDENCE_LEVEL) -> tuple[float, float]:
    """
    Wilson interval for an accuracy.

    The Wilson interval stays inside [0, 1] and keeps its coverage near the
    edges, where the normal approximation does not.
    """
    correct = np.asarray(correct, dtype=bool)
    low, high = proportion_confint(
        int(correct.sum()), len(correct), alpha=1 - level, method="wilson"
    )
    return float(low), float(high)


def paired_comparison(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """
    McNemar's test on two classifiers scored on the same photographs.

    Photographs both get right or both get wrong say nothing about which is
    better; the test looks only at the ones they disagree on. Under the null
    hypothesis that both have the same accuracy, a disagreement is equally
    likely to favor either.
    """
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    table = np.array([[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]])
    discordant = int(table[0, 1] + table[1, 0])
    result = mcnemar(table, exact=discordant < EXACT_MCNEMAR_BELOW, correction=True)
    return {
        "only_a_correct": int(table[0, 1]),
        "only_b_correct": int(table[1, 0]),
        "accuracy_difference": float(b.mean() - a.mean()),
        "relative_error_reduction": float((b.mean() - a.mean()) / max(1 - a.mean(), 1e-12)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }
