"""
AMSR-SE — Candidate Fusion Engine
Merges candidate sets from multiple retrievers (Dense, BM25, Graph),
with support for both weighted score fusion and Reciprocal Rank Fusion (RRF).

Tracks source attribution per candidate for ablation analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional


class CandidateFusion:
    """Fuse candidates from heterogeneous retrieval sources."""

    @staticmethod
    def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        if method == "minmax":
            min_val, max_val = np.min(scores), np.max(scores)
            if max_val > min_val:
                return (scores - min_val) / (max_val - min_val)
            return np.ones_like(scores)
        return scores

    # ── Reciprocal Rank Fusion ─────────────────────────────────────────

    @staticmethod
    def rrf_fuse(
        source_rankings: Dict[str, Tuple[np.ndarray, np.ndarray]],
        rrf_k: int = 60,
        max_candidates: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[str]]]:
        """
        Reciprocal Rank Fusion across multiple ranked lists.

        RRF_score(doc) = sum(1 / (rrf_k + rank_in_source))

        Args:
            source_rankings: {source_name: (indices, scores)} — each list
                             is assumed to be in descending score order.
            rrf_k: smoothing constant (default 60, standard value).
            max_candidates: maximum number of merged candidates to return.

        Returns:
            fused_indices, fused_scores, attribution_map
            attribution_map: {doc_idx: [source_names that contributed]}
        """
        rrf_scores: Dict[int, float] = {}
        attribution: Dict[int, List[str]] = {}
        source_ranks: Dict[int, Dict[str, int]] = {}  # for diagnostics

        for source_name, (indices, scores) in source_rankings.items():
            if len(indices) == 0:
                continue
            for rank, idx in enumerate(indices):
                idx = int(idx)
                rrf_contribution = 1.0 / (rrf_k + rank + 1)  # rank is 0-indexed

                if idx not in rrf_scores:
                    rrf_scores[idx] = 0.0
                    attribution[idx] = []
                    source_ranks[idx] = {}

                rrf_scores[idx] += rrf_contribution
                attribution[idx].append(source_name)
                source_ranks[idx][source_name] = rank + 1

        if not rrf_scores:
            return np.array([]), np.array([]), {}

        # Sort by RRF score descending
        sorted_items = sorted(rrf_scores.items(), key=lambda x: -x[1])
        sorted_items = sorted_items[:max_candidates]

        final_idx = np.array([item[0] for item in sorted_items])
        final_scores = np.array([item[1] for item in sorted_items])

        return final_idx, final_scores, attribution

    # ── Weighted Score Fusion (legacy) ─────────────────────────────────

    @staticmethod
    def weighted_fuse(
        dense_idx: np.ndarray, dense_scores: np.ndarray,
        bm25_idx: np.ndarray = None, bm25_scores: np.ndarray = None,
        graph_idx: np.ndarray = None, graph_scores: np.ndarray = None,
        weights: Dict[str, float] = None,
        max_candidates: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[str]]]:
        """
        Merge candidates using weighted normalized score addition.
        Returns deduplicated indices, fused scores, and source attribution.
        """
        if weights is None:
            weights = {"dense": 0.5, "bm25": 0.3, "graph": 0.2}

        all_candidates: Dict[int, Dict] = {}

        # Dense candidates
        dense_norm = CandidateFusion.normalize_scores(dense_scores)
        for idx, score in zip(dense_idx, dense_norm):
            idx = int(idx)
            all_candidates[idx] = {
                "score": score * weights["dense"],
                "sources": ["dense"]
            }

        # BM25 candidates
        if bm25_idx is not None and len(bm25_idx) > 0:
            bm25_norm = CandidateFusion.normalize_scores(bm25_scores)
            for idx, score in zip(bm25_idx, bm25_norm):
                idx = int(idx)
                weighted_score = score * weights["bm25"]
                if idx in all_candidates:
                    all_candidates[idx]["score"] += weighted_score
                    all_candidates[idx]["sources"].append("bm25")
                else:
                    all_candidates[idx] = {
                        "score": weighted_score,
                        "sources": ["bm25"]
                    }

        # Graph candidates
        if graph_idx is not None and len(graph_idx) > 0:
            graph_norm = CandidateFusion.normalize_scores(graph_scores)
            for idx, score in zip(graph_idx, graph_norm):
                idx = int(idx)
                weighted_score = score * weights["graph"]
                if idx in all_candidates:
                    all_candidates[idx]["score"] += weighted_score
                    all_candidates[idx]["sources"].append("graph")
                else:
                    all_candidates[idx] = {
                        "score": weighted_score,
                        "sources": ["graph"]
                    }

        if not all_candidates:
            return np.array([]), np.array([]), {}

        # Sort descending, limit to max_candidates
        sorted_items = sorted(all_candidates.items(), key=lambda x: -x[1]["score"])
        sorted_items = sorted_items[:max_candidates]

        final_idx = np.array([item[0] for item in sorted_items])
        final_scores = np.array([item[1]["score"] for item in sorted_items])
        attribution = {item[0]: item[1]["sources"] for item in sorted_items}

        return final_idx, final_scores, attribution

    # ── Convenience alias ──────────────────────────────────────────────

    @staticmethod
    def fuse(
        dense_idx: np.ndarray, dense_scores: np.ndarray,
        bm25_idx: np.ndarray = None, bm25_scores: np.ndarray = None,
        graph_idx: np.ndarray = None, graph_scores: np.ndarray = None,
        weights: Dict[str, float] = None,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        max_candidates: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[str]]]:
        """
        Unified fusion interface. Supports 'rrf' or 'weighted'.
        """
        if fusion_method == "rrf":
            source_rankings = {"dense": (dense_idx, dense_scores)}
            if bm25_idx is not None and len(bm25_idx) > 0:
                source_rankings["bm25"] = (bm25_idx, bm25_scores)
            if graph_idx is not None and len(graph_idx) > 0:
                source_rankings["graph"] = (graph_idx, graph_scores)
            return CandidateFusion.rrf_fuse(
                source_rankings, rrf_k=rrf_k, max_candidates=max_candidates
            )
        else:
            return CandidateFusion.weighted_fuse(
                dense_idx, dense_scores,
                bm25_idx, bm25_scores,
                graph_idx, graph_scores,
                weights=weights,
                max_candidates=max_candidates,
            )
