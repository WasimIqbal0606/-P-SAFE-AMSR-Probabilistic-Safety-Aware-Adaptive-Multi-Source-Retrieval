"""
Safe-AMSR-SE v3 — Candidate Pool Evaluation (Upgraded)
Deep candidate pool analysis with Jaccard overlap, extended depth curves,
and relevance-aware source attribution.

Metrics:
  - candidate_recall@{10,20,50,100,200}
  - oracle_recall@100
  - unique_relevant_docs_added_by_{bm25,graph,hybrid}
  - dense_coverage_inside_hybrid_pool
  - true_jaccard_overlap@{10,50,100}
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class CandidatePoolMetrics:
    """Metrics for a single query's candidate pool."""
    query_id: str = ""
    # Candidate recall at different depths
    candidate_recall: Dict[int, float] = field(default_factory=dict)
    # Oracle recall: if we could perfectly rerank, how many relevant docs are in pool?
    oracle_recall: float = 0.0
    # How many relevant docs are NEW (not in dense top-k)?
    relevant_new_docs: Dict[int, int] = field(default_factory=dict)
    # Dense coverage inside hybrid pool (old overlap metric)
    dense_coverage_in_hybrid: Dict[int, float] = field(default_factory=dict)
    # True Jaccard overlap: |Dense@k ∩ Hybrid@k| / |Dense@k ∪ Hybrid@k|
    true_jaccard_overlap: Dict[int, float] = field(default_factory=dict)
    # Total candidates in pool
    pool_size: int = 0
    # Source breakdown
    source_counts: Dict[str, int] = field(default_factory=dict)
    # Relevant docs per source
    relevant_per_source: Dict[str, int] = field(default_factory=dict)
    # Unique relevant docs added by each source
    unique_relevant_by_bm25: int = 0
    unique_relevant_by_graph: int = 0
    unique_relevant_by_hybrid: int = 0


class CandidatePoolEvaluator:
    """Evaluate candidate pool quality before reranking."""

    def __init__(self, eval_depths: List[int] = None):
        self.eval_depths = eval_depths or [10, 20, 50, 100, 200]

    def evaluate_pool(
        self,
        candidate_indices: np.ndarray,
        dense_only_indices: np.ndarray,
        relevance_labels: Dict[str, int],
        task_ids: List[str],
        attribution: Dict[int, List[str]],
        query_id: str = "",
        bm25_only_indices: Optional[np.ndarray] = None,
        graph_only_indices: Optional[np.ndarray] = None,
    ) -> CandidatePoolMetrics:
        """
        Evaluate a candidate pool against ground truth.

        Args:
            candidate_indices: full merged candidate pool (ordered by fusion score).
            dense_only_indices: what dense retrieval alone would have returned.
            relevance_labels: {task_id: relevance_level}.
            task_ids: ordered list mapping index → task_id.
            attribution: {doc_idx: [source_names]}.
            query_id: query identifier.
            bm25_only_indices: BM25-only results for unique contribution analysis.
            graph_only_indices: graph-only expansion results.
        """
        metrics = CandidatePoolMetrics(query_id=query_id)
        metrics.pool_size = len(candidate_indices)

        # Auto-detect threshold (1 for binary, 2 for graded)
        max_rel = max(relevance_labels.values()) if relevance_labels else 1
        threshold = 1 if max_rel <= 1 else 2

        # Ground truth relevant set
        relevant_set = {tid for tid, rel in relevance_labels.items() if rel >= threshold}

        if not relevant_set:
            return metrics

        # ── Dense sets at each depth ──────────────────────────────────
        dense_set_at = {}
        for depth in self.eval_depths:
            dense_at_depth = set()
            for idx in dense_only_indices[:min(depth, len(dense_only_indices))]:
                idx_int = int(idx)
                if 0 <= idx_int < len(task_ids):
                    dense_at_depth.add(task_ids[idx_int])
            dense_set_at[depth] = dense_at_depth

        # ── Candidate recall, Jaccard, coverage at each depth ─────────
        for depth in self.eval_depths:
            pool_at_depth = set()
            for idx in candidate_indices[:min(depth, len(candidate_indices))]:
                idx_int = int(idx)
                if 0 <= idx_int < len(task_ids):
                    pool_at_depth.add(task_ids[idx_int])

            # Candidate recall
            metrics.candidate_recall[depth] = (
                len(pool_at_depth & relevant_set) / len(relevant_set)
            )

            # Dense coverage inside hybrid pool (old metric, renamed)
            dense_at_d = dense_set_at.get(depth, set())
            if dense_at_d:
                metrics.dense_coverage_in_hybrid[depth] = (
                    len(pool_at_depth & dense_at_d) / max(len(dense_at_d), 1)
                )
            else:
                metrics.dense_coverage_in_hybrid[depth] = 0.0

            # True Jaccard overlap: |Dense@k ∩ Hybrid@k| / |Dense@k ∪ Hybrid@k|
            union = pool_at_depth | dense_at_d
            intersection = pool_at_depth & dense_at_d
            metrics.true_jaccard_overlap[depth] = (
                len(intersection) / max(len(union), 1)
            )

            # New relevant docs (in pool but NOT in dense)
            pool_relevant = pool_at_depth & relevant_set
            dense_relevant = dense_at_d & relevant_set
            metrics.relevant_new_docs[depth] = len(pool_relevant - dense_relevant)

        # ── Oracle recall: all relevant docs anywhere in pool ─────────
        full_pool = set()
        for idx in candidate_indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(task_ids):
                full_pool.add(task_ids[idx_int])
        metrics.oracle_recall = len(full_pool & relevant_set) / len(relevant_set)

        # ── Unique relevant contributions per source ──────────────────
        dense_set_full = set()
        for idx in dense_only_indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(task_ids):
                dense_set_full.add(task_ids[idx_int])
        dense_relevant = dense_set_full & relevant_set

        if bm25_only_indices is not None and len(bm25_only_indices) > 0:
            bm25_set = set()
            for idx in bm25_only_indices:
                idx_int = int(idx)
                if 0 <= idx_int < len(task_ids):
                    bm25_set.add(task_ids[idx_int])
            bm25_relevant = bm25_set & relevant_set
            # Unique relevant from BM25: in BM25 but NOT in dense
            metrics.unique_relevant_by_bm25 = len(bm25_relevant - dense_relevant)

        if graph_only_indices is not None and len(graph_only_indices) > 0:
            graph_set = set()
            for idx in graph_only_indices:
                idx_int = int(idx)
                if 0 <= idx_int < len(task_ids):
                    graph_set.add(task_ids[idx_int])
            graph_relevant = graph_set & relevant_set
            metrics.unique_relevant_by_graph = len(graph_relevant - dense_relevant)

        # Unique from hybrid (pool minus dense)
        pool_relevant_full = full_pool & relevant_set
        metrics.unique_relevant_by_hybrid = len(pool_relevant_full - dense_relevant)

        # ── Source attribution stats ──────────────────────────────────
        source_counts: Dict[str, int] = {}
        relevant_per_source: Dict[str, int] = {}
        for idx in candidate_indices:
            idx_int = int(idx)
            sources = attribution.get(idx_int, ["unknown"])
            for src in sources:
                source_counts[src] = source_counts.get(src, 0) + 1
                if 0 <= idx_int < len(task_ids):
                    tid = task_ids[idx_int]
                    if relevance_labels.get(tid, 0) >= threshold:
                        relevant_per_source[src] = relevant_per_source.get(src, 0) + 1

        metrics.source_counts = source_counts
        metrics.relevant_per_source = relevant_per_source

        return metrics

    @staticmethod
    def aggregate_pool_metrics(
        all_metrics: List[CandidatePoolMetrics],
    ) -> Dict:
        """Aggregate pool metrics across queries."""
        if not all_metrics:
            return {}

        depths = set()
        for m in all_metrics:
            depths.update(m.candidate_recall.keys())
        depths = sorted(depths)

        result = {
            "num_queries": len(all_metrics),
            "candidate_recall": {},
            "oracle_recall_mean": float(np.mean([m.oracle_recall for m in all_metrics])),
            "pool_size_mean": float(np.mean([m.pool_size for m in all_metrics])),
            "relevant_new_docs": {},
            "dense_coverage_in_hybrid": {},
            "true_jaccard_overlap": {},
        }

        for d in depths:
            recalls = [m.candidate_recall.get(d, 0) for m in all_metrics]
            result["candidate_recall"][d] = float(np.mean(recalls))

            new_docs = [m.relevant_new_docs.get(d, 0) for m in all_metrics]
            result["relevant_new_docs"][d] = float(np.mean(new_docs))

            coverage = [m.dense_coverage_in_hybrid.get(d, 0) for m in all_metrics]
            result["dense_coverage_in_hybrid"][d] = float(np.mean(coverage))

            jaccard = [m.true_jaccard_overlap.get(d, 0) for m in all_metrics]
            result["true_jaccard_overlap"][d] = float(np.mean(jaccard))

        # Unique relevant doc contributions
        result["unique_relevant_by_bm25_mean"] = float(
            np.mean([m.unique_relevant_by_bm25 for m in all_metrics]))
        result["unique_relevant_by_graph_mean"] = float(
            np.mean([m.unique_relevant_by_graph for m in all_metrics]))
        result["unique_relevant_by_hybrid_mean"] = float(
            np.mean([m.unique_relevant_by_hybrid for m in all_metrics]))
        result["unique_relevant_by_bm25_total"] = int(
            sum(m.unique_relevant_by_bm25 for m in all_metrics))
        result["unique_relevant_by_graph_total"] = int(
            sum(m.unique_relevant_by_graph for m in all_metrics))
        result["unique_relevant_by_hybrid_total"] = int(
            sum(m.unique_relevant_by_hybrid for m in all_metrics))

        # Aggregate source counts
        all_sources: Dict[str, List[int]] = {}
        all_relevant_sources: Dict[str, List[int]] = {}
        for m in all_metrics:
            for src, count in m.source_counts.items():
                all_sources.setdefault(src, []).append(count)
            for src, count in m.relevant_per_source.items():
                all_relevant_sources.setdefault(src, []).append(count)

        result["source_counts_mean"] = {
            src: float(np.mean(counts)) for src, counts in all_sources.items()
        }
        result["relevant_per_source_mean"] = {
            src: float(np.mean(counts)) for src, counts in all_relevant_sources.items()
        }

        return result
