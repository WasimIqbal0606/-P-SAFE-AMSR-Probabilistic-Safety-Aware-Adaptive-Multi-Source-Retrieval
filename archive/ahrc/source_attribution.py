"""
Safe-AMSR-SE v3 — Source Attribution for Relevant Documents
Computes relevance-aware source attribution for the final top-k results.

For each query's final top-10 results, determines:
  - relevant_from_dense_only
  - relevant_from_bm25_only
  - relevant_from_graph_only
  - relevant_from_dense_bm25_overlap
  - relevant_from_dense_graph_overlap
  - relevant_from_bm25_graph_overlap
  - relevant_from_all_sources
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RelevanceAttribution:
    """Relevance-aware source attribution for a single query."""
    query_id: str = ""
    # Total relevant in final top-k
    total_relevant: int = 0
    # Source-exclusive relevant counts
    relevant_from_dense_only: int = 0
    relevant_from_bm25_only: int = 0
    relevant_from_graph_only: int = 0
    # Overlap relevant counts
    relevant_from_dense_bm25: int = 0
    relevant_from_dense_graph: int = 0
    relevant_from_bm25_graph: int = 0
    relevant_from_all_sources: int = 0
    # Non-relevant (for completeness)
    non_relevant_in_top_k: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "query_id": self.query_id,
            "total_relevant": self.total_relevant,
            "relevant_from_dense_only": self.relevant_from_dense_only,
            "relevant_from_bm25_only": self.relevant_from_bm25_only,
            "relevant_from_graph_only": self.relevant_from_graph_only,
            "relevant_from_dense_bm25": self.relevant_from_dense_bm25,
            "relevant_from_dense_graph": self.relevant_from_dense_graph,
            "relevant_from_bm25_graph": self.relevant_from_bm25_graph,
            "relevant_from_all_sources": self.relevant_from_all_sources,
            "non_relevant_in_top_k": self.non_relevant_in_top_k,
        }


class SourceAttributionAnalyzer:
    """Analyze which retrieval sources contribute relevant documents."""

    def __init__(self, relevance_threshold: int = 1):
        self.threshold = relevance_threshold
        self._records: List[RelevanceAttribution] = []

    def analyze_query(
        self,
        query_id: str,
        final_indices: np.ndarray,
        attribution_map: Dict[int, List[str]],
        relevance_labels: Dict[str, int],
        corpus_ids: List[str],
        top_k: int = 10,
    ) -> RelevanceAttribution:
        """
        Analyze source attribution for relevant documents in final top-k.

        Args:
            query_id: query identifier.
            final_indices: final ranked indices after reranking.
            attribution_map: {doc_idx: [source_names]}.
            relevance_labels: {doc_id: relevance_level}.
            corpus_ids: ordered list mapping index → doc_id.
            top_k: how many final results to analyze.
        """
        result = RelevanceAttribution(query_id=query_id)

        for idx in final_indices[:top_k]:
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= len(corpus_ids):
                continue

            doc_id = corpus_ids[idx_int]
            rel = relevance_labels.get(doc_id, 0)

            if rel < self.threshold:
                result.non_relevant_in_top_k += 1
                continue

            result.total_relevant += 1
            sources = set(attribution_map.get(idx_int, ["unknown"]))

            has_dense = "dense" in sources
            has_bm25 = "bm25" in sources
            has_graph = "graph" in sources

            # Classify by source combination
            if has_dense and has_bm25 and has_graph:
                result.relevant_from_all_sources += 1
            elif has_dense and has_bm25:
                result.relevant_from_dense_bm25 += 1
            elif has_dense and has_graph:
                result.relevant_from_dense_graph += 1
            elif has_bm25 and has_graph:
                result.relevant_from_bm25_graph += 1
            elif has_dense:
                result.relevant_from_dense_only += 1
            elif has_bm25:
                result.relevant_from_bm25_only += 1
            elif has_graph:
                result.relevant_from_graph_only += 1

        self._records.append(result)
        return result

    def aggregate(self) -> Dict[str, float]:
        """Aggregate source attribution across all queries."""
        if not self._records:
            return {}

        fields = [
            "total_relevant", "relevant_from_dense_only",
            "relevant_from_bm25_only", "relevant_from_graph_only",
            "relevant_from_dense_bm25", "relevant_from_dense_graph",
            "relevant_from_bm25_graph", "relevant_from_all_sources",
            "non_relevant_in_top_k",
        ]

        result = {}
        for f in fields:
            values = [getattr(r, f) for r in self._records]
            result[f"{f}_mean"] = float(np.mean(values))
            result[f"{f}_sum"] = int(np.sum(values))

        # Compute source contribution rates (as fraction of total relevant)
        total_rel = sum(r.total_relevant for r in self._records)
        if total_rel > 0:
            result["pct_dense_only"] = sum(r.relevant_from_dense_only for r in self._records) / total_rel
            result["pct_bm25_only"] = sum(r.relevant_from_bm25_only for r in self._records) / total_rel
            result["pct_graph_only"] = sum(r.relevant_from_graph_only for r in self._records) / total_rel
            result["pct_dense_bm25"] = sum(r.relevant_from_dense_bm25 for r in self._records) / total_rel
            result["pct_dense_graph"] = sum(r.relevant_from_dense_graph for r in self._records) / total_rel
            result["pct_bm25_graph"] = sum(r.relevant_from_bm25_graph for r in self._records) / total_rel
            result["pct_all_sources"] = sum(r.relevant_from_all_sources for r in self._records) / total_rel
        else:
            for key in ["pct_dense_only", "pct_bm25_only", "pct_graph_only",
                         "pct_dense_bm25", "pct_dense_graph", "pct_bm25_graph",
                         "pct_all_sources"]:
                result[key] = 0.0

        # Unique contribution: relevant docs ONLY from that source
        result["unique_relevant_from_bm25"] = sum(r.relevant_from_bm25_only for r in self._records)
        result["unique_relevant_from_graph"] = sum(r.relevant_from_graph_only for r in self._records)
        result["unique_relevant_from_dense"] = sum(r.relevant_from_dense_only for r in self._records)

        return result

    def get_records(self) -> List[RelevanceAttribution]:
        return list(self._records)

    def reset(self):
        self._records.clear()
