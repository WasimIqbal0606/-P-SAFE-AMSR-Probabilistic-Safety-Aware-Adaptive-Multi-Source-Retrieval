"""
AHRC — Evaluation Framework
Metrics: Recall@k, nDCG@k, MRR, latency, cost, and cost-performance curves.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    latency_ms: float = 0.0
    candidates_explored: int = 0
    k_used: int = 0


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all queries."""
    method: str = ""
    num_queries: int = 0
    # Quality
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    # Efficiency
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    candidates_mean: float = 0.0
    candidates_total: int = 0
    k_mean: float = 0.0
    # Cost-performance
    cost_per_recall: float = 0.0  # candidates / recall@10


class Evaluator:
    """Compute retrieval quality and efficiency metrics."""

    def __init__(self, k_values: List[int] = None, relevance_threshold: int = None):
        self.k_values = k_values or [1, 3, 5, 10, 20]
        # None = auto-detect from data; int = explicit threshold
        self._relevance_threshold = relevance_threshold

    @staticmethod
    def _detect_threshold(relevance_labels: Dict[str, int]) -> int:
        """
        Auto-detect the relevance threshold.
        If max relevance is 1 (binary dataset, e.g. BEIR SciFact), use >= 1.
        If max relevance is >= 2 (graded, e.g. TREC DL, MS MARCO), use >= 2.
        """
        if not relevance_labels:
            return 1
        max_rel = max(relevance_labels.values())
        return 1 if max_rel <= 1 else 2

    # ── Per-query metrics ──────────────────────────────────────────────

    def evaluate_query(
        self,
        retrieved_indices: np.ndarray,
        relevance_labels: Dict[str, int],
        task_ids: List[str],
        query_id: str = "",
        latency_ms: float = 0.0,
        candidates_explored: int = 0,
    ) -> QueryMetrics:
        """
        Compute metrics for a single query.

        Args:
            retrieved_indices: (k,) integer indices of retrieved items.
            relevance_labels: {task_id: relevance_level (0-3)}.
            task_ids: ordered list mapping index → task_id.
            query_id: query identifier.
            latency_ms: query latency.
            candidates_explored: total candidates considered.
        """
        metrics = QueryMetrics(
            query_id=query_id,
            latency_ms=latency_ms,
            candidates_explored=candidates_explored,
            k_used=len(retrieved_indices),
        )

        # Auto-detect or use explicit threshold
        threshold = (
            self._relevance_threshold
            if self._relevance_threshold is not None
            else self._detect_threshold(relevance_labels)
        )

        # Build relevance vector for retrieved items
        retrieved_rels = []
        for idx in retrieved_indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(task_ids):
                tid = task_ids[idx_int]
                rel = relevance_labels.get(tid, 0)
            else:
                rel = 0
            retrieved_rels.append(rel)

        retrieved_rels = np.array(retrieved_rels, dtype=np.float64)

        # Ground truth: all items meeting the relevance threshold
        relevant_set = {tid for tid, rel in relevance_labels.items() if rel >= threshold}
        highly_relevant = {tid for tid, rel in relevance_labels.items() if rel >= threshold + 1}

        # Compute metrics at each k
        for k in self.k_values:
            if k > len(retrieved_indices):
                # Pad with zeros (not retrieved = not relevant)
                padded = np.zeros(k)
                padded[:len(retrieved_rels)] = retrieved_rels
            else:
                padded = retrieved_rels[:k]

            # Recall@k
            retrieved_at_k = set()
            for i, idx in enumerate(retrieved_indices[:k]):
                idx_int = int(idx)
                if 0 <= idx_int < len(task_ids):
                    retrieved_at_k.add(task_ids[idx_int])

            if len(relevant_set) > 0:
                metrics.recall_at_k[k] = len(retrieved_at_k & relevant_set) / len(relevant_set)
            else:
                metrics.recall_at_k[k] = 0.0

            # nDCG@k
            metrics.ndcg_at_k[k] = self._ndcg(padded[:k], relevance_labels, task_ids, k)

        # MRR
        metrics.mrr = self._mrr(retrieved_indices, relevant_set, task_ids)

        return metrics

    # ── nDCG ───────────────────────────────────────────────────────────

    def _ndcg(
        self,
        retrieved_rels: np.ndarray,
        relevance_labels: Dict[str, int],
        task_ids: List[str],
        k: int,
    ) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        # DCG
        dcg = self._dcg(retrieved_rels[:k])

        # Ideal DCG (sort all relevance labels descending)
        all_rels = sorted(relevance_labels.values(), reverse=True)
        ideal_rels = np.array(all_rels[:k], dtype=np.float64)
        idcg = self._dcg(ideal_rels)

        if idcg == 0:
            return 0.0
        return float(dcg / idcg)

    @staticmethod
    def _dcg(rels: np.ndarray) -> float:
        """Discounted Cumulative Gain."""
        if len(rels) == 0:
            return 0.0
        positions = np.arange(1, len(rels) + 1)
        gains = (2 ** rels - 1) / np.log2(positions + 1)
        return float(np.sum(gains))

    # ── MRR ────────────────────────────────────────────────────────────

    @staticmethod
    def _mrr(
        retrieved_indices: np.ndarray,
        relevant_set: set,
        task_ids: List[str],
    ) -> float:
        """Mean Reciprocal Rank (rank of first relevant result)."""
        for rank, idx in enumerate(retrieved_indices, start=1):
            idx_int = int(idx)
            if 0 <= idx_int < len(task_ids):
                if task_ids[idx_int] in relevant_set:
                    return 1.0 / rank
        return 0.0

    # ── Aggregate ──────────────────────────────────────────────────────

    def aggregate(
        self,
        query_metrics: List[QueryMetrics],
        method: str = "",
    ) -> AggregateMetrics:
        """Aggregate per-query metrics into summary statistics."""
        agg = AggregateMetrics(method=method, num_queries=len(query_metrics))

        if not query_metrics:
            return agg

        # Quality metrics — mean across queries
        for k in self.k_values:
            recalls = [qm.recall_at_k.get(k, 0) for qm in query_metrics]
            ndcgs = [qm.ndcg_at_k.get(k, 0) for qm in query_metrics]
            agg.recall_at_k[k] = float(np.mean(recalls))
            agg.ndcg_at_k[k] = float(np.mean(ndcgs))

        agg.mrr = float(np.mean([qm.mrr for qm in query_metrics]))

        # Efficiency metrics
        latencies = [qm.latency_ms for qm in query_metrics]
        agg.latency_mean_ms = float(np.mean(latencies))
        agg.latency_p50_ms = float(np.median(latencies))
        agg.latency_p95_ms = float(np.percentile(latencies, 95))
        agg.latency_p99_ms = float(np.percentile(latencies, 99))

        candidates = [qm.candidates_explored for qm in query_metrics]
        agg.candidates_mean = float(np.mean(candidates))
        agg.candidates_total = int(np.sum(candidates))

        ks = [qm.k_used for qm in query_metrics]
        agg.k_mean = float(np.mean(ks))

        # Cost-performance ratio
        recall_10 = agg.recall_at_k.get(10, 0)
        if recall_10 > 0:
            agg.cost_per_recall = agg.candidates_mean / recall_10
        else:
            agg.cost_per_recall = float("inf")

        return agg

    # ── Formatting ─────────────────────────────────────────────────────

    @staticmethod
    def format_results(results: Dict[str, AggregateMetrics]) -> str:
        """Format comparison table for console output."""
        header = (
            f"{'Method':<25} "
            f"{'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} "
            f"{'nDCG@10':>8} {'MRR':>6} "
            f"{'Lat(ms)':>8} {'Lat95':>8} "
            f"{'AvgK':>6} {'AvgCand':>8}"
        )
        separator = "─" * len(header)

        lines = [separator, header, separator]

        for method, agg in results.items():
            line = (
                f"{method:<25} "
                f"{agg.recall_at_k.get(1, 0):>6.3f} "
                f"{agg.recall_at_k.get(5, 0):>6.3f} "
                f"{agg.recall_at_k.get(10, 0):>6.3f} "
                f"{agg.recall_at_k.get(20, 0):>6.3f} "
                f"{agg.ndcg_at_k.get(10, 0):>8.4f} "
                f"{agg.mrr:>6.3f} "
                f"{agg.latency_mean_ms:>8.2f} "
                f"{agg.latency_p95_ms:>8.2f} "
                f"{agg.k_mean:>6.1f} "
                f"{agg.candidates_mean:>8.1f}"
            )
            lines.append(line)

        lines.append(separator)
        return "\n".join(lines)

    @staticmethod
    def to_dict(agg: AggregateMetrics) -> dict:
        """Convert AggregateMetrics to JSON-serializable dict."""
        return {
            "method": agg.method,
            "num_queries": agg.num_queries,
            "recall_at_k": {str(k): v for k, v in agg.recall_at_k.items()},
            "ndcg_at_k": {str(k): v for k, v in agg.ndcg_at_k.items()},
            "mrr": agg.mrr,
            "latency_mean_ms": agg.latency_mean_ms,
            "latency_p50_ms": agg.latency_p50_ms,
            "latency_p95_ms": agg.latency_p95_ms,
            "latency_p99_ms": agg.latency_p99_ms,
            "candidates_mean": agg.candidates_mean,
            "candidates_total": agg.candidates_total,
            "k_mean": agg.k_mean,
            "cost_per_recall": agg.cost_per_recall,
        }
