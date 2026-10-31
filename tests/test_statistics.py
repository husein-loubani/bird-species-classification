"""Tests for the accuracy interval and the paired comparison."""

from __future__ import annotations

import numpy as np
import pytest

from bird_species.statistics import accuracy_interval, paired_comparison


def test_wilson_interval_matches_the_textbook_value():
    low, high = accuracy_interval(np.array([True] * 50 + [False] * 50))
    assert low == pytest.approx(0.4038, abs=1e-4)
    assert high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_interval_stays_inside_zero_and_one():
    low, high = accuracy_interval(np.ones(20, dtype=bool))
    assert 0 < low < 1 and high == pytest.approx(1.0)


def test_paired_comparison_counts_only_disagreements():
    a = np.array([True] * 80 + [False] * 10 + [True] * 2 + [False] * 8)
    b = np.array([True] * 80 + [True] * 10 + [False] * 2 + [False] * 8)
    result = paired_comparison(a, b)
    assert (result["only_a_correct"], result["only_b_correct"]) == (2, 10)
    assert result["accuracy_difference"] == pytest.approx(0.08)
    assert result["relative_error_reduction"] == pytest.approx(0.08 / 0.18)
    # Twelve disagreements is below 25, so the exact binomial test applies:
    # P(X <= 2 | n=12, p=0.5) doubled.
    assert result["p_value"] == pytest.approx(2 * sum(__import__("math").comb(12, k) for k in range(3)) / 2 ** 12)
