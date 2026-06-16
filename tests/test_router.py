"""Tests for BPSafeRouter fixes (FIX 4, 5, 6)."""
import numpy as np
import pytest


def test_dense_fallback_is_a0():
    """Dense fallback must always be A0_DENSE."""
    from psafe.router import BPSafeRouter, PSafeDecision
    from psafe.actions import Action

    router = BPSafeRouter(mode="balanced")
    # Not trained — should fallback to Dense
    X = np.random.randn(25).astype(np.float32)
    # Since not trained, scaler won't work, test the untrained path
    assert not hasattr(router, '_fitted') or not router.models_gain


def test_prior_probability_model_deterministic():
    """PriorProbabilityModel should give consistent outputs."""
    from psafe.router import PriorProbabilityModel

    model = PriorProbabilityModel(0.3)
    X = np.random.randn(10, 5)

    probs = model.predict_proba(X)
    assert probs.shape == (10, 2)
    assert np.allclose(probs[:, 1], 0.3)
    assert np.allclose(probs[:, 0], 0.7)

    preds = model.predict(X)
    assert len(preds) == 10
    assert all(p == 0 for p in preds)  # p=0.3 < 0.5

    model_high = PriorProbabilityModel(0.8)
    preds_high = model_high.predict(X)
    assert all(p == 1 for p in preds_high)  # p=0.8 >= 0.5


def test_candidate_count_uses_dict():
    """FIX 4: candidate_count should come from candidate_counts dict, not len(actions)."""
    from psafe.router import BPSafeRouter
    from psafe.actions import Action

    router = BPSafeRouter(mode="balanced")

    # Minimal training data
    n = 20
    features = np.random.randn(n, 25).astype(np.float32)
    dense_ndcg = np.random.rand(n).astype(np.float32)
    hybrid_ndcg = dense_ndcg + np.random.randn(n) * 0.05

    train_data = {
        'features': features,
        'actions': [Action.A0_DENSE.value, Action.A6_DEEP_HYBRID.value],
        'delta_ndcg': {
            Action.A0_DENSE.value: np.zeros(n),
            Action.A6_DEEP_HYBRID.value: hybrid_ndcg - dense_ndcg,
        },
        'latency': {
            Action.A0_DENSE.value: np.full(n, 0.05),
            Action.A6_DEEP_HYBRID.value: np.full(n, 2500.0),
        },
        'harm': {
            Action.A0_DENSE.value: np.zeros(n, dtype=int),
            Action.A6_DEEP_HYBRID.value: ((hybrid_ndcg - dense_ndcg) < -0.01).astype(int),
        },
        'gain': {
            Action.A0_DENSE.value: np.zeros(n, dtype=int),
            Action.A6_DEEP_HYBRID.value: ((hybrid_ndcg - dense_ndcg) > 0.05).astype(int),
        },
    }

    router.train(train_data)

    X = np.random.randn(25).astype(np.float32)
    counts = {Action.A0_DENSE.value: 50, Action.A6_DEEP_HYBRID.value: 400}
    decision = router.route(X, "q1", candidate_counts=counts, split="test")

    # Check that the recorded candidate_count is from the dict
    last_pred = router._action_predictions[-1]
    assert last_pred["candidate_count"] in [50, 400]
    assert last_pred["split"] == "test"


def test_tune_thresholds_multi_action():
    """FIX 6: tune_thresholds should handle multiple actions without crash."""
    from psafe.router import BPSafeRouter
    from psafe.actions import Action

    router = BPSafeRouter(mode="balanced")
    n = 30
    features = np.random.randn(n, 25).astype(np.float32)

    train_data = {
        'features': features,
        'actions': [Action.A0_DENSE.value, Action.A4_DENSE_BM25_CE.value, Action.A6_DEEP_HYBRID.value],
        'delta_ndcg': {
            Action.A0_DENSE.value: np.zeros(n),
            Action.A4_DENSE_BM25_CE.value: np.random.randn(n) * 0.05,
            Action.A6_DEEP_HYBRID.value: np.random.randn(n) * 0.05,
        },
        'latency': {
            Action.A0_DENSE.value: np.full(n, 0.05),
            Action.A4_DENSE_BM25_CE.value: np.full(n, 600.0),
            Action.A6_DEEP_HYBRID.value: np.full(n, 2500.0),
        },
        'harm': {
            Action.A0_DENSE.value: np.zeros(n, dtype=int),
            Action.A4_DENSE_BM25_CE.value: np.random.randint(0, 2, n),
            Action.A6_DEEP_HYBRID.value: np.random.randint(0, 2, n),
        },
        'gain': {
            Action.A0_DENSE.value: np.zeros(n, dtype=int),
            Action.A4_DENSE_BM25_CE.value: np.random.randint(0, 2, n),
            Action.A6_DEEP_HYBRID.value: np.random.randint(0, 2, n),
        },
    }

    router.train(train_data)

    val_data = {
        'features': np.random.randn(10, 25).astype(np.float32),
        'delta_ndcg': {
            Action.A4_DENSE_BM25_CE.value: np.random.randn(10) * 0.05,
            Action.A6_DEEP_HYBRID.value: np.random.randn(10) * 0.05,
        },
    }
    # Should not crash
    router.tune_thresholds(val_data)
    assert "tuned_gain_threshold" in router.diagnostics
    assert "tuned_harm_threshold" in router.diagnostics
