"""
tests/test_metrics.py
---------------------
Unit tests for metrics flagged by Gemini audit.
Run with:  python -m pytest tests/ -v

Uses CONTROLLED examples where answers are known exactly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.metrics.metrics import (
    sensitivity_at_k,
    rank_of_first_positive,
    candidate_reduction_at_sensitivity,
    _nnt_at_aggregate_sensitivity,
)


# ── sensitivity_at_k ─────────────────────────────────────────────────────────

class TestSensitivityAtK:

    def test_positive_at_rank1(self):
        s = np.array([0.9, 0.5, 0.3])
        l = np.array([1, 0, 0])
        assert sensitivity_at_k(s, l, k=1) == 1.0

    def test_positive_at_rank2_missed_at_k1(self):
        s = np.array([0.9, 0.8, 0.3])
        l = np.array([0, 1, 0])
        assert sensitivity_at_k(s, l, k=1) == 0.0
        assert sensitivity_at_k(s, l, k=2) == 1.0

    def test_no_positives_returns_none(self):
        assert sensitivity_at_k(np.array([0.9, 0.5]), np.array([0, 0]), k=1) is None

    def test_k_larger_than_n(self):
        s = np.array([0.1, 0.9])
        l = np.array([1, 0])
        assert sensitivity_at_k(s, l, k=100) == 1.0

    def test_multiple_positives_caught_at_k3(self):
        s = np.array([0.9, 0.8, 0.7, 0.6])
        l = np.array([0, 0, 1, 1])
        assert sensitivity_at_k(s, l, k=2) == 0.0
        assert sensitivity_at_k(s, l, k=3) == 1.0


# ── rank_of_first_positive ────────────────────────────────────────────────────

class TestRankOfFirstPositive:

    def test_rank1(self):
        s = np.array([0.9, 0.5, 0.3])
        l = np.array([1, 0, 0])
        assert rank_of_first_positive(s, l) == 1

    def test_rank_last(self):
        s = np.array([0.9, 0.7, 0.1])
        l = np.array([0, 0, 1])
        assert rank_of_first_positive(s, l) == 3

    def test_no_positive(self):
        assert rank_of_first_positive(np.array([0.9, 0.7]),
                                       np.array([0, 0])) is None


# ── NNT: controlled where NNT@80 ≠ NNT@90 ────────────────────────────────────

class TestNNT:
    """
    Controlled setup:
      8 patients: malignant at rank 3   → SE(k=3) = 8/10 = 80%
      1 patient:  malignant at rank 6   → SE(k=6) = 9/10 = 90%
      1 patient:  malignant at rank 9

    Expected:
      NNT@80 = mean rank of 8 caught patients = 3.0
      NNT@90 = mean rank of 9 caught patients = (8*3 + 6) / 9 = 3.33
    """

    @staticmethod
    def _patient_with_mal_at_rank(rank: int, n: int = 15):
        scores = np.linspace(0.9, 0.1, n)
        labels = np.zeros(n, dtype=int)
        order  = np.argsort(scores)[::-1]
        labels[order[rank - 1]] = 1
        assert rank_of_first_positive(scores, labels) == rank
        return scores, labels

    def test_nnt80_lt_nnt90(self):
        groups = [self._patient_with_mal_at_rank(3) for _ in range(8)]
        groups.append(self._patient_with_mal_at_rank(6))
        groups.append(self._patient_with_mal_at_rank(9))
        s_list = [g[0] for g in groups]
        l_list = [g[1] for g in groups]

        nnt80 = _nnt_at_aggregate_sensitivity(s_list, l_list, 0.80)
        nnt90 = _nnt_at_aggregate_sensitivity(s_list, l_list, 0.90)

        assert nnt80 is not None and nnt90 is not None
        assert nnt80 < nnt90, (
            f"NNT@80={nnt80:.3f} should be < NNT@90={nnt90:.3f}"
        )
        assert abs(nnt80 - 3.0) < 0.01, f"NNT@80 expected 3.0, got {nnt80}"
        assert abs(nnt90 - np.mean([3]*8 + [6])) < 0.01, \
            f"NNT@90 expected {np.mean([3]*8+[6]):.3f}, got {nnt90:.3f}"

    def test_perfect_ranker_nnt_equals_1(self):
        s_list = [np.array([0.9, 0.1, 0.1])] * 10
        l_list = [np.array([1, 0, 0])] * 10
        assert _nnt_at_aggregate_sensitivity(s_list, l_list, 0.80) == 1.0
        assert _nnt_at_aggregate_sensitivity(s_list, l_list, 0.90) == 1.0

    def test_no_positives_returns_none(self):
        s_list = [np.array([0.9, 0.1])] * 5
        l_list = [np.array([0, 0])] * 5
        assert _nnt_at_aggregate_sensitivity(s_list, l_list, 0.80) is None

    def test_4_patients_nnt80_equals_nnt90_is_expected(self):
        """
        Documents known behaviour on toy data with 4 positive patients.
        Achievable SE = 0, 25%, 50%, 75%, 100%.
        Both 80% and 90% require 100% → NNT@80 == NNT@90.
        This is CORRECT, not a bug.
        """
        s_list = [np.array([0.9, 0.1, 0.1, 0.1])] * 4
        l_list = [np.array([1, 0, 0, 0])] * 4
        nnt80 = _nnt_at_aggregate_sensitivity(s_list, l_list, 0.80)
        nnt90 = _nnt_at_aggregate_sensitivity(s_list, l_list, 0.90)
        assert nnt80 == nnt90, (
            "With 4 patients, NNT@80==NNT@90 is expected: "
            "achievable SE increments are 0/0.25/0.50/0.75/1.0, "
            "so both targets require SE=1.0."
        )


# ── candidate_reduction ───────────────────────────────────────────────────────

class TestCandidateReduction:

    def test_perfect_ranker(self):
        n = 10
        s = np.linspace(0.9, 0.1, n)
        l = np.zeros(n, dtype=int)
        l[np.argmax(s)] = 1          # malignant at rank 1
        cr = candidate_reduction_at_sensitivity([s], [l], [n], 1.0)
        assert abs(cr - (n - 1) / n) < 1e-6, f"Got {cr}"

    def test_worst_ranker(self):
        n = 10
        s = np.linspace(0.9, 0.1, n)
        l = np.zeros(n, dtype=int)
        l[np.argmin(s)] = 1          # malignant at rank n (last)
        cr = candidate_reduction_at_sensitivity([s], [l], [n], 1.0)
        assert cr == 0.0, f"Expected 0.0, got {cr}"

    def test_method_dependency(self):
        """Two methods with different rankings give different CR."""
        n = 10
        s_good = np.linspace(0.9, 0.1, n)
        l = np.zeros(n, dtype=int); l[np.argmax(s_good)] = 1   # rank 1
        cr_good = candidate_reduction_at_sensitivity([s_good], [l], [n], 1.0)

        s_bad = np.linspace(0.1, 0.9, n)   # inverted: malignant at rank n
        l2 = np.zeros(n, dtype=int); l2[np.argmax(s_bad)] = 1  # rank 1 in bad
        # Give bad scorer low score for the positive
        s_bad2 = np.linspace(0.9, 0.1, n)
        l3 = np.zeros(n, dtype=int); l3[np.argmin(s_bad2)] = 1  # rank n
        cr_bad = candidate_reduction_at_sensitivity([s_bad2], [l3], [n], 1.0)

        assert cr_good > cr_bad, \
            f"Good ranker CR={cr_good:.3f} should exceed bad CR={cr_bad:.3f}"
