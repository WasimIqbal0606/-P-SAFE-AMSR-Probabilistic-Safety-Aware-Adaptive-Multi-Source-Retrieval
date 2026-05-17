"""
AMSR-SE — Hybrid Retriever (Publication-Grade)
Full retrieval pipeline orchestrator with deep candidate generation,
RRF fusion, cross-encoder reranking, and per-query telemetry.

Pipeline:
  1. Dense retrieval (k=50)
  2. Uncertainty estimation
  3. Adaptive controller decision (easy/medium/hard routing)
  4. Conditional BM25 retrieval (k=100)
  5. Conditional graph expansion (seed_k=10)
  6. RRF candidate fusion (max 200 candidates)
  7. Cross-encoder reranking (top 50)
  8. Return final top-k

Logs every decision for reproducibility and analysis.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .config import AHRCConfig, UncertaintyLevel
from .index_manager import IndexManager
from .uncertainty_module import UncertaintyEstimator
from .adaptive_controller import AdaptiveController, RetrievalDecision
from .graph_expander import GraphExpander
from .candidate_fusion import CandidateFusion
from .reranker import CrossEncoderReranker
from .baselines import BM25Baseline


@dataclass
class RetrievalResult:
    """Result from a single retrieval query."""
    query_id: str
    retrieved_indices: np.ndarray
    retrieved_scores: np.ndarray
    final_k: int
    # Timing
    dense_time_ms: float = 0.0
    uncertainty_time_ms: float = 0.0
    expansion_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0
    # Decision metadata
    decision: Optional[RetrievalDecision] = None
    candidates_explored: int = 0
    graph_expanded: bool = False
    # Candidate pool telemetry (for pool evaluation)
    dense_only_indices: Optional[np.ndarray] = None
    candidate_pool_indices: Optional[np.ndarray] = None
    candidate_pool_scores: Optional[np.ndarray] = None
    candidate_attribution: Optional[Dict] = None


class HybridRetriever:
    """
    Full adaptive hybrid retrieval pipeline.
    Orchestrates: index → uncertainty → controller → BM25 → graph → RRF → rerank.
    """

    # Retrieval depth constants
    DENSE_K = 50          # Always retrieve 50 from FAISS
    BM25_K = 100          # BM25 retrieval depth
    GRAPH_SEED_K = 10     # Number of dense seeds for graph expansion
    GRAPH_NEIGHBORS = 10  # Max neighbors per graph seed
    MAX_CANDIDATES = 200  # Maximum merged pool size
    RERANK_DEPTH = 50     # Cross-encoder reranks top 50

    def __init__(
        self,
        config: AHRCConfig,
        index_manager: IndexManager,
        graph_expander: GraphExpander,
        all_embeddings: np.ndarray,
        bm25_baseline: Optional[BM25Baseline] = None,
        task_texts: Optional[List[str]] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.cfg = config
        self.index = index_manager
        self.graph = graph_expander
        self.all_embeddings = all_embeddings
        self.bm25_baseline = bm25_baseline
        self.task_texts = task_texts
        self.reranker = reranker

        self.uncertainty = UncertaintyEstimator(config.uncertainty)
        self.controller = AdaptiveController(config.adaptive)

        self._results_log: List[RetrievalResult] = []

    # ── Main retrieval ─────────────────────────────────────────────────

    def retrieve(
        self,
        query_id: str,
        query_embedding: np.ndarray,
        final_k: int = 10,
        task_metadata: Optional[Dict] = None,
    ) -> RetrievalResult:
        """
        Full adaptive retrieval pipeline for a single query.

        Args:
            query_id: query identifier.
            query_embedding: (D,) float32 query vector.
            final_k: desired number of results.
            task_metadata: must include 'query_text' for BM25 and cross-encoder.

        Returns:
            RetrievalResult with indices, scores, timing, and candidate pool data.
        """
        if task_metadata is None:
            task_metadata = {}

        result = RetrievalResult(
            query_id=query_id, final_k=final_k,
            retrieved_indices=np.array([]),
            retrieved_scores=np.array([]),
        )
        t_total = time.perf_counter()

        # ── Step 1: Dense retrieval (always deep) ──────────────────────
        t0 = time.perf_counter()
        distances, indices = self.index.search_single(query_embedding, k=self.DENSE_K)

        # Convert L2 distances to similarity scores
        scores = 1.0 / (1.0 + distances)

        # Filter out -1 indices (padding from FAISS)
        valid_mask = indices >= 0
        dense_indices = indices[valid_mask]
        dense_scores = scores[valid_mask]

        result.dense_time_ms = (time.perf_counter() - t0) * 1000
        result.dense_only_indices = dense_indices.copy()

        if len(dense_indices) == 0:
            result.total_time_ms = (time.perf_counter() - t_total) * 1000
            self._results_log.append(result)
            return result

        # ── Step 2: Uncertainty estimation ─────────────────────────────
        t0 = time.perf_counter()
        check_k = min(10, len(dense_indices))
        graph_degrees = self.graph.get_degrees(dense_indices[:check_k])

        uncertainty_score, uncertainty_level, signals = (
            self.uncertainty.compute_uncertainty(
                query_embedding=query_embedding,
                retrieved_scores=dense_scores[:check_k],
                retrieved_indices=dense_indices[:check_k],
                task_metadata=task_metadata,
                graph_degrees=graph_degrees,
            )
        )
        result.uncertainty_time_ms = (time.perf_counter() - t0) * 1000

        # ── Step 3: Controller decision ────────────────────────────────
        decision = self.controller.decide(
            uncertainty_score=uncertainty_score,
            uncertainty_level=uncertainty_level,
            signals=signals,
        )
        result.decision = decision

        # ── Step 4: Multi-source candidate generation ──────────────────
        t0 = time.perf_counter()

        bm25_indices = np.array([], dtype=np.int64)
        bm25_scores = np.array([], dtype=np.float32)
        graph_indices = np.array([], dtype=np.int64)
        graph_scores = np.array([], dtype=np.float32)

        # BM25 retrieval (medium + hard queries)
        if decision.use_bm25 and self.bm25_baseline and 'query_text' in task_metadata:
            bm25_result = self.bm25_baseline.retrieve(
                task_metadata['query_text'], k=self.BM25_K
            )
            bm25_indices = bm25_result.retrieved_indices
            bm25_scores = bm25_result.retrieved_scores

        # Graph expansion (hard queries only, with gating)
        if decision.enable_graph_expansion and self.graph._built:
            seed_k = min(self.GRAPH_SEED_K, len(dense_indices))
            graph_indices, graph_scores = self.graph.expand(
                seed_indices=dense_indices[:seed_k],
                seed_scores=dense_scores[:seed_k],
                query_embedding=query_embedding,
                all_embeddings=self.all_embeddings,
                hops=decision.graph_hops,
                max_neighbors=self.GRAPH_NEIGHBORS,
            )
            result.graph_expanded = True

        result.expansion_time_ms = (time.perf_counter() - t0) * 1000

        # ── Step 5: RRF Fusion ─────────────────────────────────────────
        t0 = time.perf_counter()

        fused_indices, fused_scores, attribution = CandidateFusion.fuse(
            dense_idx=dense_indices,
            dense_scores=dense_scores,
            bm25_idx=bm25_indices,
            bm25_scores=bm25_scores,
            graph_idx=graph_indices,
            graph_scores=graph_scores,
            fusion_method="rrf",
            rrf_k=60,
            max_candidates=self.MAX_CANDIDATES,
        )

        result.expansion_time_ms += (time.perf_counter() - t0) * 1000
        result.candidates_explored = len(fused_indices)

        # Save candidate pool for evaluation
        result.candidate_pool_indices = fused_indices.copy()
        result.candidate_pool_scores = fused_scores.copy()
        result.candidate_attribution = attribution

        # ── Step 6: Cross-Encoder Reranking (top RERANK_DEPTH) ─────────
        t0 = time.perf_counter()
        rerank_depth = min(self.RERANK_DEPTH, len(fused_indices))

        if rerank_depth > 0 and self.reranker and self.task_texts and task_metadata.get('query_text'):
            to_rerank_indices = fused_indices[:rerank_depth]
            to_rerank_scores = fused_scores[:rerank_depth]

            reranked_indices, reranked_scores = self._cross_encode_rerank(
                query_text=task_metadata['query_text'],
                candidate_indices=to_rerank_indices,
                candidate_scores=to_rerank_scores,
            )

            # Combine reranked top with remaining unranked tail
            final_indices = np.concatenate([reranked_indices, fused_indices[rerank_depth:]])
            final_scores = np.concatenate([reranked_scores, fused_scores[rerank_depth:]])
        else:
            # Fallback: exact bi-encoder reranking
            rerank_depth = min(self.RERANK_DEPTH, len(fused_indices))
            if rerank_depth > 0:
                to_rerank = fused_indices[:rerank_depth]
                candidate_embs = self.all_embeddings[to_rerank.astype(int)]
                exact_scores = np.dot(candidate_embs, query_embedding)
                sort_order = np.argsort(-exact_scores)
                final_indices = np.concatenate([to_rerank[sort_order], fused_indices[rerank_depth:]])
                final_scores = np.concatenate([exact_scores[sort_order], fused_scores[rerank_depth:]])
            else:
                final_indices = fused_indices
                final_scores = fused_scores

        result.rerank_time_ms = (time.perf_counter() - t0) * 1000

        # ── Step 7: Final top-k ────────────────────────────────────────
        final_count = min(final_k, len(final_indices))
        result.retrieved_indices = final_indices[:final_count]
        result.retrieved_scores = final_scores[:final_count]
        result.total_time_ms = (time.perf_counter() - t_total) * 1000

        self._results_log.append(result)
        return result

    # ── Cross-Encoder Reranking ────────────────────────────────────────

    def _cross_encode_rerank(
        self,
        query_text: str,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rerank candidates using the cross-encoder model.
        Returns reranked (indices, scores).
        """
        if len(candidate_indices) == 0:
            return candidate_indices, candidate_scores

        cand_texts = [self.task_texts[int(idx)] for idx in candidate_indices]
        reranked_idx, reranked_scores = self.reranker.rerank(
            query_text, cand_texts, candidate_indices
        )

        # Normalize cross-encoder scores to [0, 1]
        reranked_scores = CandidateFusion.normalize_scores(reranked_scores)

        return reranked_idx, reranked_scores

    # ── Batch retrieval ────────────────────────────────────────────────

    def retrieve_batch(
        self,
        queries: list,
        final_k: int = 10,
    ) -> List[RetrievalResult]:
        """Run retrieval for a batch of Query objects."""
        results = []
        for q in queries:
            metadata = {"category": q.category, "query_text": q.text}
            result = self.retrieve(
                query_id=q.id,
                query_embedding=q.embedding,
                final_k=final_k,
                task_metadata=metadata,
            )
            results.append(result)
        return results

    # ── Statistics ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Aggregate retrieval statistics."""
        if not self._results_log:
            return {"num_queries": 0}

        latencies = [r.total_time_ms for r in self._results_log]
        dense_times = [r.dense_time_ms for r in self._results_log]
        candidates = [r.candidates_explored for r in self._results_log]
        expanded = [r.graph_expanded for r in self._results_log]

        return {
            "num_queries": len(self._results_log),
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_p50_ms": float(np.median(latencies)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "dense_time_mean_ms": float(np.mean(dense_times)),
            "candidates_mean": float(np.mean(candidates)),
            "candidates_total": int(np.sum(candidates)),
            "graph_expansion_rate": sum(expanded) / len(expanded),
            "controller_stats": self.controller.get_decision_stats(),
        }

    def reset(self):
        """Clear logs for new experiment run."""
        self._results_log.clear()
        self.controller.reset()
        self.uncertainty.reset_history()
