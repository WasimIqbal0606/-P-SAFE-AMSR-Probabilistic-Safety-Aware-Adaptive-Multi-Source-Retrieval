"""
AHRC — Adaptive Retrieval Controller
Per-query decision engine that dynamically adjusts retrieval parameters
based on uncertainty signals.

Decisions:
  - k (number of candidates to retrieve)
  - similarity threshold
  - graph expansion (on/off, hops, max neighbors)
  - reranking depth
  - index search parameters (efSearch, nprobe)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .config import AdaptiveConfig, UncertaintyLevel


@dataclass
class RetrievalDecision:
    """Output of the adaptive controller — per-query retrieval plan."""
    k: int = 10
    similarity_threshold: float = 0.65
    enable_graph_expansion: bool = False
    graph_hops: int = 1
    graph_max_neighbors: int = 20
    rerank_depth: int = 15
    use_bm25: bool = False
    bm25_k: int = 0
    ef_search: Optional[int] = None   # HNSW override
    nprobe: Optional[int] = None      # IVF override
    # Metadata for logging
    uncertainty_score: float = 0.0
    uncertainty_level: str = "medium"
    decision_reason: str = ""


class AdaptiveController:
    """
    Decides retrieval parameters per query based on uncertainty.

    Core logic:
      HIGH uncertainty → cast a wider net (more k, looser threshold,
                         enable graph expansion, deeper reranking)
      LOW uncertainty  → narrow retrieval (less k, tight threshold,
                         skip graph expansion, early stop)
    """

    def __init__(self, config: AdaptiveConfig):
        self.cfg = config
        self._decision_log = []

    def decide(
        self,
        uncertainty_score: float,
        uncertainty_level: UncertaintyLevel,
        signals: dict,
    ) -> RetrievalDecision:
        """
        Generate retrieval plan based on uncertainty.

        Args:
            uncertainty_score: float ∈ [0, 1]
            uncertainty_level: LOW / MEDIUM / HIGH
            signals: individual signal values for fine-grained decisions.

        Returns:
            RetrievalDecision with all parameters.
        """
        decision = RetrievalDecision(
            uncertainty_score=uncertainty_score,
            uncertainty_level=uncertainty_level.value,
        )

        if uncertainty_level == UncertaintyLevel.LOW:
            decision = self._low_uncertainty_plan(decision, signals)
        elif uncertainty_level == UncertaintyLevel.HIGH:
            decision = self._high_uncertainty_plan(decision, signals)
        else:
            decision = self._medium_uncertainty_plan(decision, signals)

        # SAFETY RULE: Never reduce candidate pool too early
        decision.k = max(10, decision.k)

        self._decision_log.append(decision)
        return decision

    # ── Strategy implementations ───────────────────────────────────────

    def _low_uncertainty_plan(
        self, decision: RetrievalDecision, signals: dict
    ) -> RetrievalDecision:
        """Confident retrieval — narrow and fast."""
        decision.k = self.cfg.k_min
        decision.similarity_threshold = self.cfg.threshold_tight
        decision.enable_graph_expansion = False
        decision.graph_hops = 0
        decision.graph_max_neighbors = 0
        decision.use_bm25 = False
        decision.bm25_k = 0
        decision.rerank_depth = self.cfg.rerank_depth_min
        # Faster index search
        decision.ef_search = max(32, self.cfg.k_min * 2)
        decision.nprobe = 5
        decision.decision_reason = (
            f"LOW uncertainty ({decision.uncertainty_score:.3f}): "
            f"narrow retrieval k={decision.k}, tight threshold, no expansion"
        )
        return decision

    def _medium_uncertainty_plan(
        self, decision: RetrievalDecision, signals: dict
    ) -> RetrievalDecision:
        """Moderate confidence — balanced retrieval."""
        decision.k = self.cfg.k_default + 10  # Increase k for medium queries
        decision.similarity_threshold = self.cfg.threshold_default
        decision.rerank_depth = self.cfg.rerank_depth_default
        
        decision.use_bm25 = True
        decision.bm25_k = 20

        decision.enable_graph_expansion = False
        decision.graph_hops = 0
        decision.graph_max_neighbors = 0

        decision.ef_search = 64
        decision.nprobe = 10
        decision.decision_reason = (
            f"MEDIUM uncertainty ({decision.uncertainty_score:.3f}): "
            f"balanced dense+bm25 k={decision.k}"
        )
        return decision

    def _high_uncertainty_plan(
        self, decision: RetrievalDecision, signals: dict
    ) -> RetrievalDecision:
        """Uncertain retrieval — cast wide net."""
        decision.k = self.cfg.k_max + 10 # Increase k further for hard queries

        decision.similarity_threshold = self.cfg.threshold_loose
        decision.use_bm25 = True
        decision.bm25_k = 30
        
        # GATING: Only expand graph if the raw margin is genuinely low
        if signals.get("margin", 1.0) < 0.05:
            decision.enable_graph_expansion = True
            decision.graph_hops = self.cfg.graph_expansion_hops
            decision.graph_max_neighbors = self.cfg.graph_max_neighbors
        else:
            decision.enable_graph_expansion = False
            decision.graph_hops = 0
            decision.graph_max_neighbors = 0
            
        decision.rerank_depth = self.cfg.rerank_depth_max

        # More thorough index search
        decision.ef_search = 128
        decision.nprobe = 20

        decision.decision_reason = (
            f"HIGH uncertainty ({decision.uncertainty_score:.3f}): "
            f"dense+bm25+graph, wide k={decision.k}, deep rerank"
        )
        return decision

    # ── Logging ────────────────────────────────────────────────────────

    def get_decision_stats(self) -> dict:
        """Aggregate statistics over all decisions made."""
        if not self._decision_log:
            return {"num_decisions": 0}

        ks = [d.k for d in self._decision_log]
        thresholds = [d.similarity_threshold for d in self._decision_log]
        expansions = [d.enable_graph_expansion for d in self._decision_log]
        reranks = [d.rerank_depth for d in self._decision_log]
        uncertainties = [d.uncertainty_score for d in self._decision_log]

        levels = [d.uncertainty_level for d in self._decision_log]
        level_counts = {
            "low": levels.count("low"),
            "medium": levels.count("medium"),
            "high": levels.count("high"),
        }

        return {
            "num_decisions": len(self._decision_log),
            "k_mean": float(np.mean(ks)),
            "k_min": min(ks),
            "k_max": max(ks),
            "threshold_mean": float(np.mean(thresholds)),
            "graph_expansion_rate": sum(expansions) / len(expansions),
            "rerank_depth_mean": float(np.mean(reranks)),
            "uncertainty_mean": float(np.mean(uncertainties)),
            "level_distribution": level_counts,
        }

    def reset(self):
        """Clear decision log."""
        self._decision_log.clear()
