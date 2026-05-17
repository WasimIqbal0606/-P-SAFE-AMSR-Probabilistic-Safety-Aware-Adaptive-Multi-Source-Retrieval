"""
AHRC — Graph-Based Retrieval Expansion
Conditional expansion through task relationship graph.

When uncertainty is high, expand the candidate set by traversing
graph neighbors of initial dense retrieval results, then merge
and deduplicate.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from .config import AdaptiveConfig


class GraphExpander:
    """Expand retrieval candidates via task relationship graph."""

    def __init__(self, config: AdaptiveConfig):
        self.cfg = config
        # Adjacency list: task_index → set of neighbor indices
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)
        self.degree_cache: Dict[int, int] = {}
        self._built = False

    # ── Graph construction ─────────────────────────────────────────────

    def build_from_tasks(self, tasks: list, id_to_index: Dict[str, int]) -> None:
        """
        Build adjacency list from task neighbor lists.

        Args:
            tasks: list of Task objects with .neighbors field.
            id_to_index: mapping from task_id string → integer index.
        """
        self.adjacency.clear()
        self.degree_cache.clear()

        for task in tasks:
            idx = id_to_index.get(task.id)
            if idx is None:
                continue
            for neighbor_id in task.neighbors:
                neighbor_idx = id_to_index.get(neighbor_id)
                if neighbor_idx is not None:
                    self.adjacency[idx].add(neighbor_idx)
                    self.adjacency[neighbor_idx].add(idx)

        # Cache degrees
        for idx in self.adjacency:
            self.degree_cache[idx] = len(self.adjacency[idx])

        self._built = True

    # ── Expansion ──────────────────────────────────────────────────────

    def expand(
        self,
        seed_indices: np.ndarray,
        seed_scores: np.ndarray,
        query_embedding: np.ndarray,
        all_embeddings: np.ndarray,
        hops: int = 1,
        max_neighbors: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Expand candidate set via graph traversal.

        Args:
            seed_indices: (k,) initial candidate indices from dense retrieval.
            seed_scores: (k,) similarity scores of seeds.
            query_embedding: (D,) query vector.
            all_embeddings: (N, D) full embedding matrix.
            hops: number of graph hops (1 or 2).
            max_neighbors: cap on expanded candidates.

        Returns:
            expanded_indices: merged + deduplicated indices.
            expanded_scores: corresponding similarity scores.
        """
        if not self._built:
            return seed_indices, seed_scores

        # Collect neighbors via BFS
        visited: Set[int] = set(seed_indices.tolist())
        frontier: Set[int] = set(seed_indices.tolist())
        new_candidates: Set[int] = set()

        for hop in range(hops):
            next_frontier: Set[int] = set()
            for node in frontier:
                neighbors = self.adjacency.get(node, set())
                for nb in neighbors:
                    if nb not in visited:
                        next_frontier.add(nb)
                        new_candidates.add(nb)
                        visited.add(nb)

                    if len(new_candidates) >= max_neighbors:
                        break
                if len(new_candidates) >= max_neighbors:
                    break
            frontier = next_frontier
            if not frontier or len(new_candidates) >= max_neighbors:
                break

        if not new_candidates:
            return seed_indices, seed_scores

        # Score new candidates by similarity to query
        new_list = list(new_candidates)[:max_neighbors]
        new_embeddings = all_embeddings[new_list]

        # Inner product (embeddings are L2-normalized)
        new_scores = np.dot(new_embeddings, query_embedding)

        # Merge with seeds
        all_indices = np.concatenate([seed_indices, np.array(new_list, dtype=np.int64)])
        all_scores = np.concatenate([seed_scores, new_scores])

        # Deduplicate (keep highest score per index)
        unique_map: Dict[int, float] = {}
        for idx, score in zip(all_indices, all_scores):
            idx_int = int(idx)
            if idx_int not in unique_map or score > unique_map[idx_int]:
                unique_map[idx_int] = float(score)

        merged_indices = np.array(list(unique_map.keys()), dtype=np.int64)
        merged_scores = np.array(list(unique_map.values()), dtype=np.float32)

        # Sort by score descending
        sort_order = np.argsort(-merged_scores)
        return merged_indices[sort_order], merged_scores[sort_order]

    # ── Utilities ──────────────────────────────────────────────────────

    def get_degrees(self, indices: np.ndarray) -> np.ndarray:
        """Get graph degree for a set of node indices."""
        return np.array(
            [self.degree_cache.get(int(i), 0) for i in indices],
            dtype=np.float32,
        )

    def get_stats(self) -> dict:
        """Graph statistics."""
        if not self._built:
            return {"built": False}

        degrees = list(self.degree_cache.values())
        return {
            "built": True,
            "num_nodes_with_edges": len(self.adjacency),
            "total_edges": sum(len(v) for v in self.adjacency.values()) // 2,
            "avg_degree": float(np.mean(degrees)) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
        }
