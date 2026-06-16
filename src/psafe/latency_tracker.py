"""
B-P-SAFE-AMSR — Canonical Latency Tracker
Merged from archive/ahrc/latency_tracker.py (rich) + psafe minimal.

Tracks per-component timing with context manager support.
"""
import time
import numpy as np
import json
import os
import csv
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict


@dataclass
class QueryLatency:
    """Latency breakdown for a single query."""
    query_id: str = ""
    dataset: str = ""
    seed: int = 0
    fold: str = ""
    mode: str = ""
    method: str = ""
    selected_action: str = ""
    dense_search_ms: float = 0.0
    bm25_search_ms: float = 0.0
    graph_expansion_ms: float = 0.0
    fusion_ms: float = 0.0
    cross_encoder_ms: float = 0.0
    bge_dense_ms: float = 0.0
    bge_sparse_ms: float = 0.0
    bge_multivector_ms: float = 0.0
    feature_extraction_ms: float = 0.0
    router_decision_ms: float = 0.0
    total_ms: float = 0.0
    cross_encoder_calls: int = 0
    candidates_reranked: int = 0
    candidates_explored: int = 0

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "dataset": self.dataset,
            "seed": self.seed,
            "fold": self.fold,
            "mode": self.mode,
            "method": self.method,
            "selected_action": self.selected_action,
            "dense_search_ms": self.dense_search_ms,
            "bm25_search_ms": self.bm25_search_ms,
            "graph_expansion_ms": self.graph_expansion_ms,
            "fusion_ms": self.fusion_ms,
            "cross_encoder_ms": self.cross_encoder_ms,
            "bge_dense_ms": self.bge_dense_ms,
            "bge_sparse_ms": self.bge_sparse_ms,
            "bge_multivector_ms": self.bge_multivector_ms,
            "feature_extraction_ms": self.feature_extraction_ms,
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
        self._simple_records: Dict[str, List[float]] = defaultdict(list)

    # ── Simple API (backward compatible) ──

    def add(self, component: str, time_ms: float):
        """Simple component timing (backward compatible)."""
        self._simple_records[component].append(time_ms)
        if self._current is not None:
            attr = f"{component}_ms"
            if hasattr(self._current, attr):
                setattr(self._current, attr, getattr(self._current, attr) + time_ms)

    # ── Rich API ──

    def start_query(self, query_id: str, dataset: str = "", method: str = "",
                    selected_action: str = ""):
        """Begin tracking a new query."""
        self._current = QueryLatency(
            query_id=query_id, dataset=dataset,
            method=method, selected_action=selected_action
        )
        self._current._t_start = time.perf_counter()

    def end_query(self):
        """Finalize the current query timing."""
        if self._current is not None:
            if self._current.total_ms == 0.0:
                self._current.total_ms = (time.perf_counter() - self._current._t_start) * 1000
            self._records.append(self._current)
            self._current = None

    @contextmanager
    def track(self, component: str):
        """Context manager for timing a component."""
        t0 = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - t0) * 1000
        self._simple_records[component].append(elapsed)
        if self._current is not None:
            attr = f"{component}_ms"
            if hasattr(self._current, attr):
                setattr(self._current, attr, getattr(self._current, attr) + elapsed)

    def record_metadata(self, key: str, value):
        """Record extra metadata on current query."""
        if self._current is not None and hasattr(self._current, key):
            setattr(self._current, key, value)

    # ── Aggregation ──

    def summarize(self, out_dir: str) -> Dict:
        """Compute and save aggregate latency statistics."""
        os.makedirs(out_dir, exist_ok=True)
        summary = {}

        # From rich records
        if self._records:
            components = [
                "dense_search", "bm25_search", "graph_expansion",
                "fusion", "cross_encoder", "router_decision", "total",
                "bge_dense", "bge_sparse", "bge_multivector",
            ]
            for comp in components:
                attr = f"{comp}_ms"
                values = np.array([getattr(r, attr) for r in self._records])
                if len(values) == 0 or np.all(values == 0):
                    continue
                summary[comp] = self._percentile_stats(values)

            ce_calls = np.array([r.cross_encoder_calls for r in self._records])
            summary["cross_encoder_calls_per_query"] = float(np.mean(ce_calls))
            summary["candidates_reranked_per_query"] = float(
                np.mean([r.candidates_reranked for r in self._records]))

        # From simple records
        for component, times in self._simple_records.items():
            if component not in summary:
                arr = np.array(times)
                summary[component] = self._percentile_stats(arr)

        with open(os.path.join(out_dir, "latency_breakdown.json"), "w") as f:
            json.dump(summary, f, indent=4)

        # Save per-query CSV if we have rich records
        if self._records:
            csv_path = os.path.join(out_dir, "latency_per_query.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._records[0].to_dict().keys())
                writer.writeheader()
                for r in self._records:
                    writer.writerow(r.to_dict())

        return summary

    def _percentile_stats(self, arr: np.ndarray) -> Dict:
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    def reset(self):
        self._records.clear()
        self._current = None
        self._simple_records.clear()
