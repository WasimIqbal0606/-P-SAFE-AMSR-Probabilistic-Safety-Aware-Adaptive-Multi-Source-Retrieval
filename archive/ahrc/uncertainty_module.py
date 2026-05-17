"""
AHRC — Uncertainty Estimation Module
Multi-signal uncertainty scoring for per-query adaptive decisions.

Signals:
  1. Similarity margin (top-1 score − top-2 score)
  2. Score variance across retrieved candidates
  3. Score entropy (distributional uncertainty)
  4. Graph ambiguity (degree / clustering of candidates)
  5. Historical retrieval confidence (rolling window)

Output: scalar uncertainty ∈ [0, 1] and UncertaintyLevel enum.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque

from .config import UncertaintyConfig, UncertaintyLevel


class UncertaintyEstimator:
    """Compute per-query uncertainty from retrieval signals."""

    def __init__(self, config: UncertaintyConfig):
        self.cfg = config
        # Rolling window of recent uncertainty scores for historical signal
        self._history: deque = deque(maxlen=100)
        # Per-category history for conditional estimation
        self._category_history: Dict[str, deque] = {}

    def compute_uncertainty(
        self,
        query_embedding: np.ndarray,
        retrieved_scores: np.ndarray,
        retrieved_indices: np.ndarray,
        task_metadata: Optional[Dict] = None,
        graph_degrees: Optional[np.ndarray] = None,
    ) -> Tuple[float, UncertaintyLevel, Dict[str, float]]:
        """
        Compute composite uncertainty score.

        Args:
            query_embedding: (D,) query vector.
            retrieved_scores: (k,) similarity scores of top-k results.
            retrieved_indices: (k,) indices of top-k results.
            task_metadata: optional dict with 'category', 'complexity', etc.
            graph_degrees: optional (k,) array of node degrees for candidates.

        Returns:
            uncertainty: float ∈ [0, 1]
            level: UncertaintyLevel enum
            signals: dict of individual signal values
        """
        signals = {}

        if len(retrieved_scores) >= 5:
            sorted_scores = np.sort(retrieved_scores)[::-1]
            raw_margin = float(sorted_scores[0] - sorted_scores[1])
            raw_topk_spread = float(sorted_scores[0] - sorted_scores[4])
            
            # Normalize to [0, 1] assuming typical max margins of 0.05 and 0.10
            norm_margin = min(raw_margin / 0.05, 1.0)
            norm_spread = min(raw_topk_spread / 0.10, 1.0)
        else:
            raw_margin = 0.0
            raw_topk_spread = 0.0
            norm_margin = 0.0
            norm_spread = 0.0

        signals["margin"] = raw_margin
        signals["topk_spread"] = raw_topk_spread
        signals["graph_ambiguity"] = min(self._graph_ambiguity_signal(graph_degrees), 0.5)
        signals["historical"] = self._historical_signal(task_metadata)

        # Baseline formula from user, using normalized values
        base_uncertainty = 1.0 - (0.7 * norm_margin + 0.3 * norm_spread)

        # Add optional weighting if enabled
        uncertainty = base_uncertainty
        if self.cfg.graph_ambiguity_weight > 0:
            uncertainty += self.cfg.graph_ambiguity_weight * signals["graph_ambiguity"]
        if self.cfg.historical_weight > 0:
            uncertainty += self.cfg.historical_weight * (signals["historical"] - 0.5)

        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))

        # Classify level
        if uncertainty < self.cfg.low_threshold:
            level = UncertaintyLevel.LOW
        elif uncertainty > self.cfg.high_threshold:
            level = UncertaintyLevel.HIGH
        else:
            level = UncertaintyLevel.MEDIUM

        # Update history
        self._history.append(uncertainty)
        if task_metadata and "category" in task_metadata:
            cat = task_metadata["category"]
            if cat not in self._category_history:
                self._category_history[cat] = deque(maxlen=50)
            self._category_history[cat].append(uncertainty)

        return uncertainty, level, signals

    # ── Individual signals ─────────────────────────────────────────────

    def _margin_signal(self, scores: np.ndarray) -> float:
        """
        Similarity margin: large margin → confident → low uncertainty.
        Returns uncertainty contribution ∈ [0, 1].
        """
        if len(scores) < 2:
            return 0.5  # can't compute margin

        # Sort descending (scores should already be sorted, but be safe)
        sorted_scores = np.sort(scores)[::-1]
        margin = sorted_scores[0] - sorted_scores[1]

        # Saturate: margin ≥ saturation → uncertainty = 0
        normalized = 1.0 - min(margin / self.cfg.margin_saturation, 1.0)
        return float(normalized)

    def _variance_signal(self, scores: np.ndarray) -> float:
        """
        Score variance: high variance → ambiguous → high uncertainty.
        """
        if len(scores) < 2:
            return 0.5

        variance = float(np.var(scores))
        # Normalize: typical variance range [0, 0.1] → [0, 1]
        normalized = min(variance / 0.1, 1.0)
        return normalized

    def _entropy_signal(self, scores: np.ndarray) -> float:
        """
        Shannon entropy of score distribution.
        High entropy → uniform scores → uncertain.
        """
        if len(scores) < 2:
            return 0.5

        # Convert to probability distribution
        scores_pos = np.maximum(scores, 1e-10)
        probs = scores_pos / scores_pos.sum()

        # Shannon entropy, normalized by max entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(scores))

        if max_entropy == 0:
            return 0.5

        return float(entropy / max_entropy)

    def _graph_ambiguity_signal(self, graph_degrees: Optional[np.ndarray]) -> float:
        """
        Graph ambiguity: candidates with many connections → more expansion
        options → higher uncertainty about which path to take.
        """
        if graph_degrees is None or len(graph_degrees) == 0:
            return 0.5  # no graph info → neutral

        avg_degree = float(np.mean(graph_degrees))
        # High degree → many options → higher ambiguity
        # Normalize: typical avg degree [0, 20] → [0, 1]
        normalized = min(avg_degree / 20.0, 1.0)
        return normalized

    def _historical_signal(self, task_metadata: Optional[Dict]) -> float:
        """
        Historical confidence: if recent queries in this category had
        high uncertainty, expect more uncertainty now.
        """
        if not self._history:
            return 0.5  # no history → neutral

        # Category-specific history (if available)
        if task_metadata and "category" in task_metadata:
            cat = task_metadata["category"]
            if cat in self._category_history and len(self._category_history[cat]) >= 5:
                return float(np.mean(self._category_history[cat]))

        # Fall back to global history
        return float(np.mean(self._history))

    # ── Utilities ──────────────────────────────────────────────────────

    def reset_history(self):
        """Clear all history (e.g., between experiments)."""
        self._history.clear()
        self._category_history.clear()

    def get_stats(self) -> dict:
        """Return estimator statistics."""
        return {
            "history_size": len(self._history),
            "category_count": len(self._category_history),
            "recent_avg_uncertainty": float(np.mean(self._history)) if self._history else None,
            "config": {
                "margin_weight": self.cfg.margin_weight,
                "variance_weight": self.cfg.variance_weight,
                "entropy_weight": self.cfg.entropy_weight,
                "graph_ambiguity_weight": self.cfg.graph_ambiguity_weight,
                "historical_weight": self.cfg.historical_weight,
                "low_threshold": self.cfg.low_threshold,
                "high_threshold": self.cfg.high_threshold,
            },
        }
