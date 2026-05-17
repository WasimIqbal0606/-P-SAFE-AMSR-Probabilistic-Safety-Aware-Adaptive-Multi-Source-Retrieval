"""
B-P-SAFE-AMSR — Canonical Action System
All runners and routers MUST import from this file.
"""
from enum import IntEnum
from typing import Dict


class Action(IntEnum):
    A0_DENSE = 0
    A1_DENSE_BM25 = 1
    A2_DENSE_GRAPH = 2
    A3_DENSE_BM25_GRAPH = 3
    A4_DENSE_BM25_CE = 4
    A5_DENSE_BM25_GRAPH_CE = 5
    A6_DEEP_HYBRID = 6
    A7_DENSE_BM25_CE_10 = 7
    A8_DENSE_BM25_CE_20 = 8
    A9_DENSE_BM25_CE_50 = 9
    A10_DENSE_BM25_CE_100 = 10
    A11_DYNAMIC_CE_DEPTH = 11
    A12_BGEM3_DENSE = 12
    A13_BGEM3_SPARSE = 13
    A14_BGEM3_DENSE_SPARSE = 14
    A15_BGEM3_DENSE_SPARSE_MULTI = 15
    A16_BGEM3_DENSE_SPARSE_CE = 16


ACTION_NAMES: Dict[Action, str] = {
    Action.A0_DENSE: "Dense",
    Action.A1_DENSE_BM25: "Dense+BM25",
    Action.A2_DENSE_GRAPH: "Dense+Graph",
    Action.A3_DENSE_BM25_GRAPH: "Dense+BM25+Graph",
    Action.A4_DENSE_BM25_CE: "Dense+BM25+CE",
    Action.A5_DENSE_BM25_GRAPH_CE: "Dense+BM25+Graph+CE",
    Action.A6_DEEP_HYBRID: "Deep Hybrid",
    Action.A7_DENSE_BM25_CE_10: "Dense+BM25+CE@10",
    Action.A8_DENSE_BM25_CE_20: "Dense+BM25+CE@20",
    Action.A9_DENSE_BM25_CE_50: "Dense+BM25+CE@50",
    Action.A10_DENSE_BM25_CE_100: "Dense+BM25+CE@100",
    Action.A11_DYNAMIC_CE_DEPTH: "Dynamic CE Depth",
    Action.A12_BGEM3_DENSE: "BGE-M3 Dense",
    Action.A13_BGEM3_SPARSE: "BGE-M3 Sparse",
    Action.A14_BGEM3_DENSE_SPARSE: "BGE-M3 Dense+Sparse",
    Action.A15_BGEM3_DENSE_SPARSE_MULTI: "BGE-M3 Dense+Sparse+Multi",
    Action.A16_BGEM3_DENSE_SPARSE_CE: "BGE-M3 Dense+Sparse+CE",
}

ACTION_LATENCY_PRIORS: Dict[Action, float] = {
    Action.A0_DENSE: 0.05, Action.A1_DENSE_BM25: 5.0,
    Action.A2_DENSE_GRAPH: 10.0, Action.A3_DENSE_BM25_GRAPH: 15.0,
    Action.A4_DENSE_BM25_CE: 600.0, Action.A5_DENSE_BM25_GRAPH_CE: 1300.0,
    Action.A6_DEEP_HYBRID: 2500.0, Action.A7_DENSE_BM25_CE_10: 150.0,
    Action.A8_DENSE_BM25_CE_20: 300.0, Action.A9_DENSE_BM25_CE_50: 600.0,
    Action.A10_DENSE_BM25_CE_100: 1200.0, Action.A11_DYNAMIC_CE_DEPTH: 500.0,
    Action.A12_BGEM3_DENSE: 2.0, Action.A13_BGEM3_SPARSE: 5.0,
    Action.A14_BGEM3_DENSE_SPARSE: 8.0, Action.A15_BGEM3_DENSE_SPARSE_MULTI: 50.0,
    Action.A16_BGEM3_DENSE_SPARSE_CE: 650.0,
}

ACTION_REQUIRES_RERANKER: Dict[Action, bool] = {a: a in {
    Action.A4_DENSE_BM25_CE, Action.A5_DENSE_BM25_GRAPH_CE, Action.A6_DEEP_HYBRID,
    Action.A7_DENSE_BM25_CE_10, Action.A8_DENSE_BM25_CE_20, Action.A9_DENSE_BM25_CE_50,
    Action.A10_DENSE_BM25_CE_100, Action.A11_DYNAMIC_CE_DEPTH, Action.A16_BGEM3_DENSE_SPARSE_CE,
} for a in Action}

ACTION_REQUIRES_GRAPH: Dict[Action, bool] = {a: a in {
    Action.A2_DENSE_GRAPH, Action.A3_DENSE_BM25_GRAPH,
    Action.A5_DENSE_BM25_GRAPH_CE, Action.A6_DEEP_HYBRID,
} for a in Action}

ACTION_REQUIRES_BGEM3: Dict[Action, bool] = {a: a.value >= 12 for a in Action}

ACTION_CE_DEPTH: Dict[Action, int] = {
    Action.A0_DENSE: 0, Action.A1_DENSE_BM25: 0, Action.A2_DENSE_GRAPH: 0,
    Action.A3_DENSE_BM25_GRAPH: 0, Action.A4_DENSE_BM25_CE: 50,
    Action.A5_DENSE_BM25_GRAPH_CE: 50, Action.A6_DEEP_HYBRID: 100,
    Action.A7_DENSE_BM25_CE_10: 10, Action.A8_DENSE_BM25_CE_20: 20,
    Action.A9_DENSE_BM25_CE_50: 50, Action.A10_DENSE_BM25_CE_100: 100,
    Action.A11_DYNAMIC_CE_DEPTH: -1, Action.A12_BGEM3_DENSE: 0,
    Action.A13_BGEM3_SPARSE: 0, Action.A14_BGEM3_DENSE_SPARSE: 0,
    Action.A15_BGEM3_DENSE_SPARSE_MULTI: 0, Action.A16_BGEM3_DENSE_SPARSE_CE: 50,
}

ACTION_IS_HYBRID: Dict[Action, bool] = {a: a != Action.A0_DENSE for a in Action}


def ACTION_IS_AVAILABLE(action: Action, config: dict) -> bool:
    """Check whether an action is available given current config."""
    if action == Action.A0_DENSE:
        return True
    if ACTION_REQUIRES_RERANKER[action] and not config.get("has_reranker", False):
        return False
    if ACTION_REQUIRES_GRAPH[action] and not config.get("has_graph", False):
        return False
    if ACTION_REQUIRES_BGEM3[action] and not config.get("has_bgem3", False):
        return False
    return True
