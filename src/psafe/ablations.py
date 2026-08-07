"""
B-P-SAFE-AMSR — Router Ablation Matrix
Controlled evaluation of router components and feature groups.

Investigates:
  Component ablations:
    1. Full B-P-SAFE
    2. Minus P_harm (lambda_harm=0, harm gating disabled)
    3. Minus P_gain (lambda_recovery=0, gain gating disabled)
    4. Minus pred_delta (delta prediction disabled)
    5. Minus Latency & Cost (lambda_latency=0, lambda_candidate=0)
    6. Minus Soft Overrides (pure utility U > 0 without mode overrides)

  Feature-group ablations:
    7. Query-only features
    8. Dense-only features
    9. BM25-only features
    10. Dense + BM25 features
    11. Disagreement features
    12. Graph/multi-source features
    13. Complete feature set (all 25)
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from psafe.actions import Action
from psafe.feature_extractor import FEATURE_NAMES
from psafe.router import BPSafeRouter, PSafeDecision

# Define feature subsets by indices
FEATURE_GROUPS = {
    "query_only": [
        "query_length_tokens", "query_has_number", "query_has_uppercase_abbreviation",
        "query_avg_token_idf", "query_max_token_idf", "lexical_specificity_score"
    ],
    "dense_only": [
        "dense_score_max", "dense_score_mean", "dense_score_std",
        "dense_entropy_norm", "dense_score_gap_1_5", "dense_score_gap_1_10"
    ],
    "bm25_only": [
        "bm25_score_max_norm", "bm25_score_mean_norm", "bm25_score_std",
        "bm25_entropy_norm", "bm25_score_gap_1_5"
    ],
    "dense_plus_bm25": [
        "dense_score_max", "dense_score_mean", "dense_score_std",
        "dense_entropy_norm", "dense_score_gap_1_5", "dense_score_gap_1_10",
        "bm25_score_max_norm", "bm25_score_mean_norm", "bm25_score_std",
        "bm25_entropy_norm", "bm25_score_gap_1_5"
    ],
    "disagreement_only": [
        "bm25_dense_overlap_jaccard_10", "bm25_dense_overlap_jaccard_50",
        "dense_bm25_rank_correlation", "candidate_novelty", "source_disagreement_score"
    ],
    "graph_only": [
        "graph_degree_max", "graph_degree_mean", "graph_degree_zero_frac"
    ],
    "complete": FEATURE_NAMES
}


def run_single_ablation(
    train_features: np.ndarray,
    train_delta: np.ndarray,
    train_latency: np.ndarray,
    train_harm: np.ndarray,
    train_gain: np.ndarray,
    val_features: np.ndarray,
    val_delta: np.ndarray,
    test_features: np.ndarray,
    test_dense_ndcg: np.ndarray,
    test_hybrid_ndcg: np.ndarray,
    test_hybrid_lat: np.ndarray,
    query_ids: List[str],
    mode: str = "balanced",
    feature_names_subset: Optional[List[str]] = None,
    component_overrides: Optional[Dict] = None,
    disable_soft_overrides: bool = False,
    candidate_counts: Optional[Dict] = None
) -> Dict:
    """
    Train and evaluate a specific ablation variant on the same split.
    """
    if candidate_counts is None:
        candidate_counts = {Action.A0_DENSE.value: 50, Action.A6_DEEP_HYBRID.value: 400}
        
    # Feature masking if subset provided
    if feature_names_subset is not None:
        sub_indices = [FEATURE_NAMES.index(fn) for fn in feature_names_subset if fn in FEATURE_NAMES]
        X_tr = train_features[:, sub_indices]
        X_val = val_features[:, sub_indices]
        X_te = test_features[:, sub_indices]
    else:
        X_tr = train_features
        X_val = val_features
        X_te = test_features

    router = BPSafeRouter(mode=mode)
    
    # Apply component overrides to router configuration
    if component_overrides:
        for k, v in component_overrides.items():
            setattr(router, k, v)
            
    train_data = {
        'features': X_tr,
        'actions': [Action.A0_DENSE.value, Action.A6_DEEP_HYBRID.value],
        'delta_ndcg': {
            Action.A0_DENSE.value: np.zeros(len(X_tr)),
            Action.A6_DEEP_HYBRID.value: train_delta
        },
        'latency': {
            Action.A0_DENSE.value: np.full(len(X_tr), 3.0),
            Action.A6_DEEP_HYBRID.value: train_latency
        },
        'harm': {
            Action.A0_DENSE.value: np.zeros(len(X_tr), dtype=int),
            Action.A6_DEEP_HYBRID.value: train_harm
        },
        'gain': {
            Action.A0_DENSE.value: np.zeros(len(X_tr), dtype=int),
            Action.A6_DEEP_HYBRID.value: train_gain
        }
    }
    
    router.train(train_data)
    
    val_data = {
        'features': X_val,
        'delta_ndcg': {
            Action.A0_DENSE.value: np.zeros(len(X_val)),
            Action.A6_DEEP_HYBRID.value: val_delta
        }
    }
    router.tune_thresholds(val_data)
    
    # Evaluate on test set
    n_test = len(X_te)
    routed_ndcg = np.zeros(n_test)
    routed_lat = np.zeros(n_test)
    actions_taken = []
    
    for i in range(n_test):
        decision = router.route(X_te[i], query_ids[i], candidate_counts=candidate_counts, split="test")
        
        # If soft overrides are disabled, enforce pure utility threshold
        if disable_soft_overrides:
            act = Action.A6_DEEP_HYBRID.value if decision.expected_utility > 0 else Action.A0_DENSE.value
        else:
            act = decision.action
            
        actions_taken.append(act)
        if act == Action.A6_DEEP_HYBRID.value:
            routed_ndcg[i] = test_hybrid_ndcg[i]
            routed_lat[i] = test_hybrid_lat[i]
        else:
            routed_ndcg[i] = test_dense_ndcg[i]
            routed_lat[i] = 3.0  # Dense latency
            
    mean_ndcg = float(np.mean(routed_ndcg))
    mean_lat = float(np.mean(routed_lat))
    har = float(np.mean([1 if a == Action.A6_DEEP_HYBRID.value else 0 for a in actions_taken]))
    delta_vs_dense = float(mean_ndcg - np.mean(test_dense_ndcg))
    
    # Easy query harm avoidance
    easy_mask = test_dense_ndcg > 0.5
    if np.any(easy_mask):
        hybrid_easy_deg = -float(np.mean(np.minimum(test_hybrid_ndcg[easy_mask] - test_dense_ndcg[easy_mask], 0)))
        routed_easy_deg = -float(np.mean(np.minimum(routed_ndcg[easy_mask] - test_dense_ndcg[easy_mask], 0)))
        harm_avoidance = float(hybrid_easy_deg - routed_easy_deg)
    else:
        harm_avoidance = 0.0
        
    return {
        "mean_ndcg": mean_ndcg,
        "mean_latency": mean_lat,
        "hybrid_activation_rate": har,
        "delta_vs_dense": delta_vs_dense,
        "harm_avoidance": harm_avoidance,
        "routed_ndcg": routed_ndcg.tolist(),
        "actions_taken": actions_taken,
    }


def evaluate_ablation_matrix(
    train_features: np.ndarray,
    train_delta: np.ndarray,
    train_latency: np.ndarray,
    train_harm: np.ndarray,
    train_gain: np.ndarray,
    val_features: np.ndarray,
    val_delta: np.ndarray,
    test_features: np.ndarray,
    test_dense_ndcg: np.ndarray,
    test_hybrid_ndcg: np.ndarray,
    test_hybrid_lat: np.ndarray,
    query_ids: List[str],
    mode: str = "balanced"
) -> Dict[str, Dict]:
    """
    Run full ablation matrix: Component ablations + Feature group ablations.
    """
    results = {}
    
    # 1. Full B-P-SAFE
    results["Full B-P-SAFE"] = run_single_ablation(
        train_features, train_delta, train_latency, train_harm, train_gain,
        val_features, val_delta, test_features, test_dense_ndcg, test_hybrid_ndcg, test_hybrid_lat,
        query_ids, mode=mode
    )
    full_ndcg = results["Full B-P-SAFE"]["mean_ndcg"]
    
    # 2. Minus P_harm (lambda_harm = 0, harm_threshold = 1.0)
    results["Minus P_harm"] = run_single_ablation(
        train_features, train_delta, train_latency, train_harm, train_gain,
        val_features, val_delta, test_features, test_dense_ndcg, test_hybrid_ndcg, test_hybrid_lat,
        query_ids, mode=mode,
        component_overrides={"lambda_harm": 0.0, "harm_threshold": 1.0}
    )
    
    # 3. Minus P_gain (lambda_recovery = 0, gain_threshold = 0.0)
    results["Minus P_gain"] = run_single_ablation(
        train_features, train_delta, train_latency, train_harm, train_gain,
        val_features, val_delta, test_features, test_dense_ndcg, test_hybrid_ndcg, test_hybrid_lat,
        query_ids, mode=mode,
        component_overrides={"lambda_recovery": 0.0, "gain_threshold": 0.0}
    )
    
    # 4. Minus predicted delta (delta prediction weight = 0 in utility)
    # Simulate by training on zero delta
    results["Minus Delta nDCG"] = run_single_ablation(
        train_features, np.zeros_like(train_delta), train_latency, train_harm, train_gain,
        val_features, np.zeros_like(val_delta), test_features, test_dense_ndcg, test_hybrid_ndcg, test_hybrid_lat,
        query_ids, mode=mode
    )
    
    # 5. Minus Latency & Cost (lambda_latency = 0, lambda_candidate = 0)
    results["Minus Latency & Cost"] = run_single_ablation(
        train_features, train_delta, train_latency, train_harm, train_gain,
        val_features, val_delta, test_features, test_dense_ndcg, test_hybrid_ndcg, test_hybrid_lat,
        query_ids, mode=mode,
        component_overrides={"lambda_latency": 0.0, "lambda_candidate": 0.0}
    )
    
    # 6. Minus Soft Overrides (pure utility)
    results["Minus Soft Overrides"] = run_single_ablation(
        train_features, train_delta, train_latency, train_harm, train_gain,
        val_features, val_delta, test_features, test_dense_ndcg, test_hybrid_ndcg, test_hybrid_lat,
        query_ids, mode=mode,
        disable_soft_overrides=True
    )
    
    # Feature Group Ablations
    for fg_name, fg_features in FEATURE_GROUPS.items():
        disp_name = f"Feature: {fg_name.replace('_', ' ').title()}"
        results[disp_name] = run_single_ablation(
            train_features, train_delta, train_latency, train_harm, train_gain,
            val_features, val_delta, test_features, test_dense_ndcg, test_hybrid_ndcg, test_hybrid_lat,
            query_ids, mode=mode,
            feature_names_subset=fg_features
        )
        
    # Calculate delta vs Full B-P-SAFE for each variant
    for k, v in results.items():
        v["delta_vs_full"] = float(v["mean_ndcg"] - full_ndcg)
        
    return results
