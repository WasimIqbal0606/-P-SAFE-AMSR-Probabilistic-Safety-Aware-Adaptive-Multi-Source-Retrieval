"""
B-P-SAFE-AMSR — Graph Contribution Metrics
Measures whether graph expansion independently contributes relevant evidence.
"""
import json
import os
import numpy as np
from typing import List, Dict, Optional, Set


def compute_graph_metrics(query_id: str, dense_top50, raw_graph_candidates,
                          relevant_docs, final_top50=None, final_top10=None,
                          selected_action: str = "", corpus_ids=None,
                          input_space: str = "index"):
    """
    Capture graph candidates immediately after graph expansion, before fusion.

    Args:
        dense_top50: list/set of dense top-50 IDs (indices or docids)
        raw_graph_candidates: list/set of graph-expanded candidates
        relevant_docs: list/set of relevant doc IDs
        final_top50: optional list of final fused top-50
        final_top10: optional list of final top-10
        corpus_ids: mapping from index to docid (if input_space="index")
        input_space: "index" or "docid"
    """
    dense_set = set(dense_top50)
    graph_set = set(raw_graph_candidates)
    relevant_set = set(relevant_docs)

    # Convert to common space if needed
    if input_space == "index" and corpus_ids is not None:
        def to_docid(idx):
            idx = int(idx)
            return corpus_ids[idx] if 0 <= idx < len(corpus_ids) else None
        dense_docids = {to_docid(i) for i in dense_set} - {None}
        graph_docids = {to_docid(i) for i in graph_set} - {None}
    else:
        dense_docids = dense_set
        graph_docids = graph_set

    graph_only_raw = graph_docids - dense_docids
    graph_only_raw_relevant = graph_only_raw & relevant_set

    metrics = {
        "query_id": query_id,
        "dense_top50_count": len(dense_set),
        "raw_graph_candidates_count": len(graph_set),
        "graph_only_raw_count": len(graph_only_raw),
        "graph_only_raw_relevant_count": len(graph_only_raw_relevant),
        "graph_action_selected": selected_action,
    }

    # Overlap at top-50
    metrics["dense_graph_overlap_at50"] = (
        len(dense_docids & graph_docids) / len(dense_docids | graph_docids)
        if len(dense_docids | graph_docids) > 0 else 0.0
    )

    # Graph unique relevant rate
    if len(graph_only_raw) > 0:
        metrics["graph_unique_relevant_rate"] = len(graph_only_raw_relevant) / len(graph_only_raw)
    else:
        metrics["graph_unique_relevant_rate"] = 0.0

    # Survival after fusion
    if final_top50 is not None:
        final_50_set = set(final_top50)
        if input_space == "index" and corpus_ids is not None:
            final_50_docids = {to_docid(i) for i in final_50_set} - {None}
        else:
            final_50_docids = final_50_set
        metrics["graph_only_survived_top50_after_fusion"] = len(graph_only_raw & final_50_docids)
    else:
        metrics["graph_only_survived_top50_after_fusion"] = 0

    if final_top10 is not None:
        final_10_set = set(final_top10)
        if input_space == "index" and corpus_ids is not None:
            final_10_docids = {to_docid(i) for i in final_10_set} - {None}
        else:
            final_10_docids = final_10_set
        metrics["graph_only_survived_top10_final"] = len(graph_only_raw & final_10_docids)
    else:
        metrics["graph_only_survived_top10_final"] = 0

    return metrics


def summarize_and_save_graph_contribution(metrics_list: List[Dict], out_dir: str,
                                          action_wins: int = 0, action_losses: int = 0):
    """Save graph contribution metrics and output disclaimer if no contribution."""
    os.makedirs(out_dir, exist_ok=True)

    total_graph_unique_relevant = sum(m.get("graph_only_raw_relevant_count", 0) for m in metrics_list)
    total_graph_only = sum(m.get("graph_only_raw_count", 0) for m in metrics_list)

    summary = {
        "n_queries": len(metrics_list),
        "graph_unique_relevant_docs_total": total_graph_unique_relevant,
        "graph_only_candidates_total": total_graph_only,
        "graph_unique_relevant_rate_macro": (
            float(np.mean([m.get("graph_unique_relevant_rate", 0) for m in metrics_list]))
            if metrics_list else 0.0
        ) if 'np' in dir() else 0.0,
        "graph_action_win": action_wins,
        "graph_action_loss": action_losses,
        "metrics_per_query": metrics_list,
    }

    # Compute macro average without numpy dependency
    if metrics_list:
        rates = [m.get("graph_unique_relevant_rate", 0) for m in metrics_list]
        summary["graph_unique_relevant_rate_macro"] = sum(rates) / len(rates)

    with open(os.path.join(out_dir, "graph_contribution.json"), "w") as f:
        json.dump(summary, f, indent=4)

    if total_graph_unique_relevant == 0:
        disclaimer = "Synthetic kNN graph did not independently contribute relevant evidence on this dataset."
        print(f"Disclaimer: {disclaimer}")
        summary["disclaimer"] = disclaimer

    return summary
