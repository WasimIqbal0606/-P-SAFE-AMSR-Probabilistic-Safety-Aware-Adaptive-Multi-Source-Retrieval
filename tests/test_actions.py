"""Tests for canonical action system."""
import pytest
from psafe.retrievers.actions import (
    Action, ACTION_NAMES, ACTION_LATENCY_PRIORS, ACTION_REQUIRES_RERANKER,
    ACTION_REQUIRES_GRAPH, ACTION_REQUIRES_BGEM3, ACTION_CE_DEPTH,
    ACTION_IS_HYBRID, ACTION_IS_AVAILABLE,
)


def test_action_count():
    assert len(Action) == 17, f"Expected 17 actions, got {len(Action)}"


def test_action_names_complete():
    for a in Action:
        assert a in ACTION_NAMES, f"Missing ACTION_NAMES for {a}"


def test_a0_is_dense_baseline():
    assert Action.A0_DENSE.value == 0
    assert ACTION_NAMES[Action.A0_DENSE] == "Dense"
    assert not ACTION_IS_HYBRID[Action.A0_DENSE]
    assert not ACTION_REQUIRES_RERANKER[Action.A0_DENSE]
    assert not ACTION_REQUIRES_GRAPH[Action.A0_DENSE]
    assert not ACTION_REQUIRES_BGEM3[Action.A0_DENSE]


def test_latency_priors_all_positive():
    for a in Action:
        assert ACTION_LATENCY_PRIORS[a] > 0, f"Latency for {a} must be positive"


def test_bgem3_actions_require_bgem3():
    for a in [Action.A12_BGEM3_DENSE, Action.A13_BGEM3_SPARSE,
              Action.A14_BGEM3_DENSE_SPARSE, Action.A15_BGEM3_DENSE_SPARSE_MULTI,
              Action.A16_BGEM3_DENSE_SPARSE_CE]:
        assert ACTION_REQUIRES_BGEM3[a], f"{a} should require BGE-M3"


def test_availability_a0_always():
    assert ACTION_IS_AVAILABLE(Action.A0_DENSE, {})
    assert ACTION_IS_AVAILABLE(Action.A0_DENSE, {"has_reranker": False})


def test_availability_ce_requires_reranker():
    assert not ACTION_IS_AVAILABLE(Action.A4_DENSE_BM25_CE, {"has_reranker": False})
    assert ACTION_IS_AVAILABLE(Action.A4_DENSE_BM25_CE, {"has_reranker": True})


def test_availability_bgem3_requires_flag():
    assert not ACTION_IS_AVAILABLE(Action.A12_BGEM3_DENSE, {"has_bgem3": False})
    assert ACTION_IS_AVAILABLE(Action.A12_BGEM3_DENSE, {"has_bgem3": True})
