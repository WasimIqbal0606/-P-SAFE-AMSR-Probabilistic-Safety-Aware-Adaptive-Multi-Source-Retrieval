"""Tests for metrics — taxonomy classification order."""
import pytest
from psafe.metrics import calculate_extended_metrics


def test_taxonomy_failure():
    """Low quality retention when hybrid helps is under-treatment."""
    m = calculate_extended_metrics(
        dense_ndcg=0.60, hybrid_ndcg=0.65, psafe_ndcg=0.55, oracle_ndcg=0.70,
        always_on_hybrid_lat=100, psafe_lat=50,
        always_on_hybrid_easy_deg=0.01, psafe_easy_deg=0.005,
        always_on_hybrid_hard_gain=0.05, psafe_hard_gain=0.02,
    )
    assert m["taxonomy"] == "Hybrid-beneficial / P-SAFE under-treatment"


def test_taxonomy_protection():
    """No benefit from hybrid: hybrid_gain <= 0.01."""
    m = calculate_extended_metrics(
        dense_ndcg=0.60, hybrid_ndcg=0.605, psafe_ndcg=0.60, oracle_ndcg=0.61,
        always_on_hybrid_lat=100, psafe_lat=5,
        always_on_hybrid_easy_deg=0.01, psafe_easy_deg=0.001,
        always_on_hybrid_hard_gain=0.005, psafe_hard_gain=0.001,
    )
    assert m["taxonomy"] == "Protection / no-benefit"
    assert m["quality_retention_applicable"] is True


def test_taxonomy_recovery():
    """P-SAFE beats hybrid and saves latency: selective escalation."""
    m = calculate_extended_metrics(
        dense_ndcg=0.50, hybrid_ndcg=0.55, psafe_ndcg=0.56, oracle_ndcg=0.60,
        always_on_hybrid_lat=200, psafe_lat=80,
        always_on_hybrid_easy_deg=0.02, psafe_easy_deg=0.005,
        always_on_hybrid_hard_gain=0.08, psafe_hard_gain=0.07,
    )
    assert m["taxonomy"] == "Selective escalation"


def test_quality_retention_clamp():
    """Quality retention should be clamped to [-1, 1.5]."""
    m = calculate_extended_metrics(
        dense_ndcg=0.50, hybrid_ndcg=0.50001, psafe_ndcg=0.80, oracle_ndcg=0.80,
        always_on_hybrid_lat=100, psafe_lat=50,
        always_on_hybrid_easy_deg=0, psafe_easy_deg=0,
        always_on_hybrid_hard_gain=0, psafe_hard_gain=0,
    )
    assert m["quality_retention_vs_best_hybrid"] <= 1.5


def test_no_benefit_qr_not_applicable():
    m = calculate_extended_metrics(
        dense_ndcg=0.50, hybrid_ndcg=0.50, psafe_ndcg=0.80, oracle_ndcg=0.80,
        always_on_hybrid_lat=100, psafe_lat=50,
        always_on_hybrid_easy_deg=0, psafe_easy_deg=0,
        always_on_hybrid_hard_gain=0, psafe_hard_gain=0,
    )
    assert m["quality_retention_vs_best_hybrid"] == 0.0
    assert m["quality_retention_applicable"] is False
    assert m["dataset_regime"] == "no_benefit"


def test_oracle_gap_closed():
    m = calculate_extended_metrics(
        dense_ndcg=0.40, hybrid_ndcg=0.50, psafe_ndcg=0.48, oracle_ndcg=0.55,
        always_on_hybrid_lat=200, psafe_lat=80,
        always_on_hybrid_easy_deg=0.01, psafe_easy_deg=0.005,
        always_on_hybrid_hard_gain=0.05, psafe_hard_gain=0.04,
    )
    expected = (0.48 - 0.40) / (0.55 - 0.40)
    assert abs(m["oracle_gap_closed"] - expected) < 1e-6


def test_latency_saving():
    m = calculate_extended_metrics(
        dense_ndcg=0.50, hybrid_ndcg=0.55, psafe_ndcg=0.54, oracle_ndcg=0.58,
        always_on_hybrid_lat=200, psafe_lat=40,
        always_on_hybrid_easy_deg=0.01, psafe_easy_deg=0.005,
        always_on_hybrid_hard_gain=0.05, psafe_hard_gain=0.04,
    )
    assert abs(m["latency_saving_vs_best_hybrid"] - 0.80) < 1e-6
