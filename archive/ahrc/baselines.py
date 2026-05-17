"""
AHRC — Baselines
Comparison retrieval systems for controlled experiments.

Implements:
  1. BM25 (keyword baseline using rank-bm25)
  2. Dense Fixed-k (static FAISS retrieval)
  3. Dense + Graph Fixed (always expand)
  4. Random baseline
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .config import AHRCConfig
from .index_manager import IndexManager
from .graph_expander import GraphExpander


@dataclass
class BaselineResult:
    """Standardized result from any baseline."""
    query_id: str
    retrieved_indices: np.ndarray
    retrieved_scores: np.ndarray
    total_time_ms: float = 0.0
    candidates_explored: int = 0
    method: str = ""


# ── BM25 Baseline ──────────────────────────────────────────────────────

class BM25Baseline:
    """
    BM25 keyword-based retrieval.
    Uses rank_bm25 library for TF-IDF-style scoring.
    """

    def __init__(self, tasks: list):
        self.tasks = tasks
        self.corpus = [t.description.lower().split() for t in tasks]
        self.bm25 = None
        self._build()

    def _build(self):
        """Build BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus)
        except ImportError:
            print("⚠️  rank_bm25 not installed. BM25 baseline will use fallback.")
            self.bm25 = None

    def retrieve(self, query_text: str, query_id: str = "", k: int = 10) -> BaselineResult:
        """Retrieve top-k by BM25 score."""
        t0 = time.perf_counter()

        if self.bm25 is not None:
            tokenized_query = query_text.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
        else:
            # Fallback: Jaccard similarity
            query_words = set(query_text.lower().split())
            scores = np.array([
                len(query_words.intersection(set(doc))) / max(len(query_words.union(set(doc))), 1)
                for doc in self.corpus
            ])

        top_k_indices = np.argsort(-scores)[:k]
        top_k_scores = scores[top_k_indices]

        elapsed = (time.perf_counter() - t0) * 1000

        return BaselineResult(
            query_id=query_id,
            retrieved_indices=top_k_indices,
            retrieved_scores=top_k_scores.astype(np.float32),
            total_time_ms=elapsed,
            candidates_explored=len(scores),
            method="bm25",
        )


# ── Dense Fixed-k Baseline ────────────────────────────────────────────

class DenseFixedBaseline:
    """
    Static dense retrieval with fixed k.
    No uncertainty, no adaptation, no graph expansion.
    """

    def __init__(self, index_manager: IndexManager, all_embeddings: np.ndarray):
        self.index = index_manager
        self.all_embeddings = all_embeddings

    def retrieve(
        self, query_embedding: np.ndarray, query_id: str = "", k: int = 10
    ) -> BaselineResult:
        """Fixed-k dense retrieval."""
        t0 = time.perf_counter()

        distances, indices = self.index.search_single(query_embedding, k=k)

        # Convert L2 distances to similarity
        scores = 1.0 / (1.0 + distances)
        valid = indices >= 0
        indices = indices[valid]
        scores = scores[valid]

        elapsed = (time.perf_counter() - t0) * 1000

        return BaselineResult(
            query_id=query_id,
            retrieved_indices=indices,
            retrieved_scores=scores,
            total_time_ms=elapsed,
            candidates_explored=k,
            method="dense_fixed",
        )


# ── Dense + Graph Fixed Baseline ──────────────────────────────────────

class DenseGraphFixedBaseline:
    """
    Dense retrieval + always-on graph expansion.
    Always expands, always reranks — no adaptation.
    """

    def __init__(
        self,
        index_manager: IndexManager,
        graph_expander: GraphExpander,
        all_embeddings: np.ndarray,
        expansion_hops: int = 1,
        max_neighbors: int = 20,
    ):
        self.index = index_manager
        self.graph = graph_expander
        self.all_embeddings = all_embeddings
        self.hops = expansion_hops
        self.max_neighbors = max_neighbors

    def retrieve(
        self, query_embedding: np.ndarray, query_id: str = "", k: int = 10
    ) -> BaselineResult:
        """Dense + always-on graph expansion."""
        t0 = time.perf_counter()

        # Dense retrieval
        distances, indices = self.index.search_single(query_embedding, k=k * 2)
        scores = 1.0 / (1.0 + distances)
        valid = indices >= 0
        indices = indices[valid]
        scores = scores[valid]

        # Always expand
        if self.graph._built:
            expanded_indices, expanded_scores = self.graph.expand(
                seed_indices=indices,
                seed_scores=scores,
                query_embedding=query_embedding,
                all_embeddings=self.all_embeddings,
                hops=self.hops,
                max_neighbors=self.max_neighbors,
            )
        else:
            expanded_indices, expanded_scores = indices, scores

        # Rerank with exact scores
        if len(expanded_indices) > 0:
            candidate_embs = self.all_embeddings[expanded_indices.astype(int)]
            exact_scores = np.dot(candidate_embs, query_embedding)
            combined = 0.3 * expanded_scores[:len(exact_scores)] + 0.7 * exact_scores
            sort_order = np.argsort(-combined)
            final_indices = expanded_indices[sort_order][:k]
            final_scores = combined[sort_order][:k]
        else:
            final_indices = expanded_indices[:k]
            final_scores = expanded_scores[:k]

        elapsed = (time.perf_counter() - t0) * 1000

        return BaselineResult(
            query_id=query_id,
            retrieved_indices=final_indices,
            retrieved_scores=final_scores,
            total_time_ms=elapsed,
            candidates_explored=len(expanded_indices),
            method="dense_graph_fixed",
        )


# ── Random Baseline ───────────────────────────────────────────────────

class RandomBaseline:
    """Random retrieval — lower bound."""

    def __init__(self, n_tasks: int):
        self.n = n_tasks

    def retrieve(self, query_id: str = "", k: int = 10) -> BaselineResult:
        """Return k random indices."""
        t0 = time.perf_counter()
        indices = np.random.choice(self.n, size=min(k, self.n), replace=False)
        scores = np.random.uniform(0, 1, size=len(indices)).astype(np.float32)
        elapsed = (time.perf_counter() - t0) * 1000

        return BaselineResult(
            query_id=query_id,
            retrieved_indices=indices,
            retrieved_scores=scores,
            total_time_ms=elapsed,
            candidates_explored=k,
            method="random",
        )
