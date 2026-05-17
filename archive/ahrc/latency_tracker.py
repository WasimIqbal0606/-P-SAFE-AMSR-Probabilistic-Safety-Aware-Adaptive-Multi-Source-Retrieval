"""
Safe-AMSR-SE v3 — Latency Tracker
Fine-grained per-component timing using time.perf_counter().

Tracks:
  - dense_embedding_time_ms
  - dense_search_time_ms
  - bm25_search_time_ms
  - graph_expansion_time_ms
  - fusion_time_ms
  - cross_encoder_time_ms
  - router_decision_time_ms
  - total_latency_ms

Reports: mean, median/p50, p95, p99 for each component.
"""

import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class QueryLatency:
    """Latency breakdown for a single query."""
    query_id: str = ""
    dense_embedding_ms: float = 0.0
    dense_search_ms: float = 0.0
    bm25_search_ms: float = 0.0
    graph_expansion_ms: float = 0.0
    fusion_ms: float = 0.0
    cross_encoder_ms: float = 0.0
    router_decision_ms: float = 0.0
    total_ms: float = 0.0
    # Extra metadata
    cross_encoder_calls: int = 0
    candidates_reranked: int = 0
    candidates_explored: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "query_id": self.query_id,
            "dense_embedding_ms": self.dense_embedding_ms,
            "dense_search_ms": self.dense_search_ms,
            "bm25_search_ms": self.bm25_search_ms,
            "graph_expansion_ms": self.graph_expansion_ms,
            "fusion_ms": self.fusion_ms,
            "cross_encoder_ms": self.cross_encoder_ms,
            "router_decision_ms": self.router_decision_ms,
            "total_ms": self.total_ms,
            "cross_encoder_calls": self.cross_encoder_calls,
            "candidates_reranked": self.candidates_reranked,
            "candidates_explored": self.candidates_explored,
        }


class LatencyTracker:
    """Collects and aggregates per-component latency measurements."""

    def __init__(self):
        self._records: List[QueryLatency] = []
        self._current: Optional[QueryLatency] = None

    def start_query(self, query_id: str):
        """Begin tracking a new query."""
        self._current = QueryLatency(query_id=query_id)
        self._current._t_start = time.perf_counter()

    def end_query(self):
        """Finalize the current query timing."""
        if self._current is not None:
            self._current.total_ms = (time.perf_counter() - self._current._t_start) * 1000
            self._records.append(self._current)
            self._current = None

    @contextmanager
    def track(self, component: str):
        """Context manager for timing a component."""
        t0 = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - t0) * 1000
        if self._current is not None:
            attr = f"{component}_ms"
            if hasattr(self._current, attr):
                setattr(self._current, attr,
                        getattr(self._current, attr) + elapsed)

    def record_component(self, component: str, elapsed_ms: float):
        """Manually record component timing."""
        if self._current is not None:
            attr = f"{component}_ms"
            if hasattr(self._current, attr):
                setattr(self._current, attr,
                        getattr(self._current, attr) + elapsed_ms)

    def record_metadata(self, key: str, value):
        """Record extra metadata on current query."""
        if self._current is not None and hasattr(self._current, key):
            setattr(self._current, key, value)

    def get_records(self) -> List[QueryLatency]:
        return list(self._records)

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """Compute aggregate latency statistics."""
        if not self._records:
            return {}

        components = [
            "dense_embedding", "dense_search", "bm25_search",
            "graph_expansion", "fusion", "cross_encoder",
            "router_decision", "total",
        ]

        result = {}
        for comp in components:
            attr = f"{comp}_ms"
            values = np.array([getattr(r, attr) for r in self._records])
            if len(values) == 0:
                continue
            result[comp] = {
                "mean_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
                "min_ms": float(np.min(values)),
                "max_ms": float(np.max(values)),
                "std_ms": float(np.std(values)),
            }

        # Cross-encoder specific stats
        ce_calls = np.array([r.cross_encoder_calls for r in self._records])
        cands_reranked = np.array([r.candidates_reranked for r in self._records])
        cands_explored = np.array([r.candidates_explored for r in self._records])

        result["cross_encoder_calls_per_query"] = float(np.mean(ce_calls))
        result["candidates_reranked_per_query"] = float(np.mean(cands_reranked))
        result["average_candidates_explored"] = float(np.mean(cands_explored))

        # Cost per nDCG gain (will be filled externally)
        result["cost_per_ndcg_gain"] = 0.0

        return result

    def compute_cost_per_ndcg_gain(
        self,
        baseline_ndcg: float,
        system_ndcg: float,
    ) -> float:
        """Compute cost_per_ndcg_gain = mean_latency / ndcg_gain."""
        if not self._records:
            return float("inf")
        mean_lat = np.mean([r.total_ms for r in self._records])
        ndcg_gain = system_ndcg - baseline_ndcg
        if ndcg_gain <= 0:
            return float("inf")
        return float(mean_lat / ndcg_gain)

    def format_report(self) -> str:
        """Pretty-print latency breakdown."""
        agg = self.aggregate()
        if not agg:
            return "  No latency data collected."

        lines = ["  Component Latency Breakdown:"]
        lines.append(f"  {'Component':<20} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
        lines.append("  " + "─" * 56)

        for comp in ["dense_embedding", "dense_search", "bm25_search",
                      "graph_expansion", "fusion", "cross_encoder",
                      "router_decision", "total"]:
            if comp in agg:
                s = agg[comp]
                lines.append(
                    f"  {comp:<20} {s['mean_ms']:>7.1f}ms "
                    f"{s['p50_ms']:>7.1f}ms {s['p95_ms']:>7.1f}ms "
                    f"{s['p99_ms']:>7.1f}ms"
                )

        lines.append("")
        lines.append(f"  CE calls/query:          {agg.get('cross_encoder_calls_per_query', 0):.1f}")
        lines.append(f"  Candidates reranked/q:   {agg.get('candidates_reranked_per_query', 0):.1f}")
        lines.append(f"  Candidates explored/q:   {agg.get('average_candidates_explored', 0):.1f}")

        return "\n".join(lines)

    def reset(self):
        self._records.clear()
        self._current = None
