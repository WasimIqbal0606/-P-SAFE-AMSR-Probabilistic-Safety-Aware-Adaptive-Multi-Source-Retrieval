"""Tests for StatisticalTester — paired tests, bootstrap, Holm-Bonferroni."""
import numpy as np
import pytest
from psafe.statistical_tests import (
    StatisticalTester, cohens_d_pooled, cohens_dz,
    holm_bonferroni_correction, get_significance_label,
)


def test_cohens_d_identical():
    a = np.ones(100)
    assert cohens_d_pooled(a, a) == 0.0


def test_cohens_d_large_effect():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(1.0, 1, 1000)
    d = cohens_d_pooled(a, b)
    assert d > 0.5, f"Expected large effect, got {d}"


def test_cohens_dz():
    deltas = np.array([0.1, 0.2, 0.15, 0.05, 0.25])
    dz = cohens_dz(deltas)
    assert dz > 0


def test_holm_bonferroni():
    p_values = [0.01, 0.04, 0.03]
    corrected = holm_bonferroni_correction(p_values)
    assert len(corrected) == 3
    assert corrected[0] <= 0.05  # Smallest p stays significant
    # Corrected values should be >= original
    for orig, corr in zip(sorted(p_values), sorted(corrected)):
        assert corr >= orig


def test_holm_bonferroni_empty():
    assert holm_bonferroni_correction([]) == []


def test_full_comparison_basic():
    tester = StatisticalTester(n_bootstrap=100, n_permutation=100)
    rng = np.random.default_rng(42)
    baseline = rng.normal(0.5, 0.1, 50)
    system = baseline + rng.normal(0.05, 0.02, 50)
    report = tester.full_comparison(baseline, system, "Dense", "P-SAFE")
    assert "paired_ttest" in report
    assert "wilcoxon" in report
    assert "bootstrap_ci" in report
    assert "permutation_test" in report
    assert "effect_size" in report
    assert report["wins"] + report["ties"] + report["losses"] == 50


def test_full_comparison_with_easy_mask():
    tester = StatisticalTester(n_bootstrap=100, n_permutation=100)
    rng = np.random.default_rng(42)
    baseline = rng.normal(0.5, 0.1, 50)
    system = baseline + 0.03
    easy_mask = baseline > 0.5
    report = tester.full_comparison(baseline, system, easy_mask=easy_mask)
    assert "easy_queries" in report
    assert "hard_queries" in report


def test_pairwise_matrix():
    tester = StatisticalTester(n_bootstrap=50, n_permutation=50)
    rng = np.random.default_rng(42)
    methods = {
        "Dense": rng.normal(0.5, 0.1, 30),
        "Hybrid": rng.normal(0.55, 0.1, 30),
        "P-SAFE": rng.normal(0.54, 0.1, 30),
    }
    result = tester.pairwise_comparison_matrix(methods)
    assert result["n_comparisons"] == 3
    assert "correction" in result
    assert result["correction"] == "holm_bonferroni"


def test_aggregate_multi_seed(tmp_path):
    tester = StatisticalTester()
    seeds = [
        {"seed": 42, "dense_ndcg_mean": 0.50, "psafe_ndcg_mean": 0.53, "mean_delta": 0.03, "p_value": 0.01, "hybrid_activation_rate": 0.3},
        {"seed": 123, "dense_ndcg_mean": 0.51, "psafe_ndcg_mean": 0.54, "mean_delta": 0.03, "p_value": 0.02, "hybrid_activation_rate": 0.35},
    ]
    summary = tester.aggregate_multi_seed(seeds, str(tmp_path))
    assert summary["n_seeds"] == 2
    assert "mean_delta" in summary
    assert summary["mean_delta"]["mean"] == 0.03
    import os
    assert os.path.exists(os.path.join(str(tmp_path), "multi_seed_summary.json"))


def test_significance_labels():
    assert "significant improvement" in get_significance_label(0.01, 0.05)
    assert "not significant" in get_significance_label(0.10, 0.05)
    assert "protection" in get_significance_label(0.50, -0.001)
