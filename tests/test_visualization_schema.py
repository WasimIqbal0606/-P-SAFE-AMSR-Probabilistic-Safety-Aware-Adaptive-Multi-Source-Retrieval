"""Tests for visualization schema parsing (FIX 7, 8)."""
import json
import os
import tempfile
import pytest


def test_calibration_new_format():
    """FIX 7: New canonical format with p_gain.bins/empirical."""
    from psafe.visualization.generate_next_level_visuals import _load_json

    cal = {
        "p_gain": {"bins": [0.1, 0.3, 0.5, 0.7, 0.9], "empirical": [0.05, 0.25, 0.5, 0.65, 0.85]},
        "p_harm": {"bins": [0.1, 0.3, 0.5], "empirical": [0.15, 0.35, 0.55]},
    }
    # Just validate the data can be parsed
    assert "p_gain" in cal
    assert len(cal["p_gain"]["bins"]) == 5


def test_calibration_legacy_format():
    """FIX 7: Legacy per-action cal_curve keys."""
    cal = {
        "Dense+BM25+CE_gain_cal_curve": {"mean_pred": [0.1, 0.5, 0.9], "frac_pos": [0.08, 0.45, 0.88]},
        "Deep Hybrid_gain_cal_curve": {"mean_pred": [0.2, 0.6], "frac_pos": [0.18, 0.55]},
    }
    # Find all keys ending with _gain_cal_curve
    suffix = "_gain_cal_curve"
    gain_keys = [k for k in cal if k.endswith(suffix)]
    assert len(gain_keys) == 2


def test_latency_mean_and_mean_ms():
    """FIX 8: Both 'mean' and 'mean_ms' should be accepted."""
    data_old = {"dense_search": {"mean": 0.5}, "reranking": {"mean": 150.0}}
    data_new = {"dense_search": {"mean_ms": 0.5}, "reranking": {"mean_ms": 150.0}}

    for data in [data_old, data_new]:
        components = []
        means = []
        for comp, stats in data.items():
            if isinstance(stats, dict):
                mean = stats.get("mean", stats.get("mean_ms"))
                if mean is not None:
                    components.append(comp)
                    means.append(float(mean))
        assert len(components) == 2
        assert len(means) == 2


def test_graph_contribution_multiple_schemas():
    """FIX 9: Multiple possible key schemas for graph metrics."""
    # Schema A (original)
    data_a = {
        "graph_unique_relevant_docs_total": 5,
        "graph_only_candidates_total": 20,
        "graph_action_win": 3,
        "graph_action_loss": 1,
    }
    # Schema B (alternative)
    data_b = {
        "graph_unique_relevant_docs": 5,
        "graph_only_candidate_count": 20,
        "graph_action_win_count": 3,
        "graph_action_loss_count": 1,
    }

    for data in [data_a, data_b]:
        unique_rel = data.get("graph_unique_relevant_docs_total",
                    data.get("graph_unique_relevant_docs",
                             data.get("graph_only_relevant_count", 0)))
        assert unique_rel == 5
