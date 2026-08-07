"""
Tests for calibration metrics, matched-budget random baseline, non-inferiority, and Holm correction.
"""
import pytest
import numpy as np

from psafe.calibration import compute_ece, compute_adaptive_ece, evaluate_calibration, compute_calibration_slope_intercept
from psafe.baselines import MatchedBudgetRandomRouter, BM25DisagreementRouter, CostOnlyRouter
from psafe.statistical_tests import evaluate_non_inferiority, holm_bonferroni_correction, cohens_dz, cohens_d_pooled


def test_compute_ece_perfect():
    """Perfect calibration should have near-zero ECE."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ece, bins = compute_ece(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_matched_budget_random_exact_count():
    """MatchedBudgetRandomRouter must escalate exactly target_k queries."""
    router = MatchedBudgetRandomRouter(target_activation_rate=0.40, seed=42)
    n = 100
    qids = [f"q_{i}" for i in range(n)]
    dense = np.random.rand(n)
    hybrid = dense + 0.05

    res = router.evaluate_batch(qids, dense, hybrid, target_k=40, seed=123)
    assert sum(1 for a in res["actions"] if a == 6) == 40
    assert res["hybrid_activation"] == pytest.approx(0.40)


def test_matched_budget_random_multi_seed_ci():
    """Multi-seed matched-budget random evaluation must yield valid 95% CI."""
    router = MatchedBudgetRandomRouter(target_activation_rate=0.50, seed=42)
    n = 50
    qids = [f"q_{i}" for i in range(n)]
    dense = np.full(n, 0.5)
    hybrid = np.full(n, 0.7)

    res = router.evaluate_multi_seed(qids, dense, hybrid, target_k=25, n_repetitions=50)
    assert res["mean_ndcg"] == pytest.approx(0.6, abs=1e-6)
    assert res["ci_95"][0] <= res["mean_ndcg"] <= res["ci_95"][1]


def test_non_inferiority_positive_margin():
    """When system equals reference, non-inferiority should be established for epsilon > 0."""
    n = 100
    sys_scores = np.full(n, 0.70)
    ref_scores = np.full(n, 0.70)

    res = evaluate_non_inferiority(sys_scores, ref_scores, epsilon=0.010, alpha=0.05)
    assert res["non_inferiority_established"] is True
    assert res["ci_lower_bound_95_param"] == pytest.approx(0.0, abs=1e-6)
    assert res["p_value_non_inferiority"] < 0.05


def test_non_inferiority_inferior_case():
    """When system is inferior by more than epsilon, non-inferiority must FAIL."""
    n = 100
    sys_scores = np.full(n, 0.50)
    ref_scores = np.full(n, 0.70)  # delta = -0.20 < -0.010

    res = evaluate_non_inferiority(sys_scores, ref_scores, epsilon=0.010, alpha=0.05)
    assert res["non_inferiority_established"] is False
    assert res["p_value_non_inferiority"] > 0.05


def test_holm_bonferroni_monotonicity():
    """Adjusted p-values must be monotonic and <= 1.0."""
    p_raw = [0.001, 0.01, 0.04, 0.05, 0.10]
    p_adj = holm_bonferroni_correction(p_raw)

    assert len(p_adj) == len(p_raw)
    assert all(0.0 <= p <= 1.0 for p in p_adj)
    for i in range(len(p_adj) - 1):
        assert p_adj[i] <= p_adj[i + 1]
