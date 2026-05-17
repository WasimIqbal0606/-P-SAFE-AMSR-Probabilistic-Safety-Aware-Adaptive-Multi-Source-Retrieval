"""
B-P-SAFE-AMSR — Extended Safety & Quality Metrics
Canonical taxonomy and metric computations for all reporting.
"""
import numpy as np


def calculate_extended_metrics(dense_ndcg, hybrid_ndcg, psafe_ndcg, oracle_ndcg,
                               always_on_hybrid_lat, psafe_lat,
                               always_on_hybrid_easy_deg, psafe_easy_deg,
                               always_on_hybrid_hard_gain, psafe_hard_gain,
                               dataset_name="",
                               best_hybrid_name="Deep Hybrid",
                               best_hybrid_ndcg=None,
                               best_hybrid_latency=None):
    """
    Compute extended metrics for a single dataset.
    Uses best_hybrid if provided, otherwise falls back to always-on hybrid.
    """
    metrics = {}

    bh_ndcg = best_hybrid_ndcg if best_hybrid_ndcg is not None else hybrid_ndcg
    bh_lat = best_hybrid_latency if best_hybrid_latency is not None else always_on_hybrid_lat

    metrics['best_hybrid_name'] = best_hybrid_name
    metrics['best_hybrid_ndcg'] = bh_ndcg
    metrics['best_hybrid_latency'] = bh_lat

    # 1. Quality retention vs best hybrid
    hybrid_gain = bh_ndcg - dense_ndcg
    if hybrid_gain > 0:
        qr = (psafe_ndcg - dense_ndcg) / hybrid_gain
    else:
        qr = 0.0
    metrics['quality_retention_vs_best_hybrid'] = float(np.clip(qr, -1.0, 1.5))

    # Legacy alias
    metrics['quality_retention_vs_hybrid'] = metrics['quality_retention_vs_best_hybrid']

    # 2. Latency saving vs best hybrid
    if bh_lat > 0:
        metrics['latency_saving_vs_best_hybrid'] = float(1 - (psafe_lat / bh_lat))
    else:
        metrics['latency_saving_vs_best_hybrid'] = 0.0

    # 3. Harm avoidance
    metrics['harm_avoidance'] = float(always_on_hybrid_easy_deg - psafe_easy_deg)

    # 4. Recovery capture
    if always_on_hybrid_hard_gain > 0:
        metrics['recovery_capture'] = float(psafe_hard_gain / always_on_hybrid_hard_gain)
    else:
        metrics['recovery_capture'] = 0.0

    # 5. Oracle gap
    metrics['oracle_gap'] = float(oracle_ndcg - psafe_ndcg)

    # 6. Oracle gap closed
    denom_oracle = oracle_ndcg - dense_ndcg
    if denom_oracle > 0:
        metrics['oracle_gap_closed'] = float((psafe_ndcg - dense_ndcg) / denom_oracle)
    else:
        metrics['oracle_gap_closed'] = 0.0

    # 7. Safe gain
    metrics['safe_gain'] = float(psafe_hard_gain - psafe_easy_deg)

    # 8. Taxonomy Classification (correct order per spec)
    if psafe_ndcg < dense_ndcg - 0.02:
        taxonomy = "P-SAFE failure"
    elif hybrid_gain <= 0.01:
        taxonomy = "Protection / No-benefit"
    elif metrics['quality_retention_vs_best_hybrid'] < 0.25:
        taxonomy = "Hybrid-beneficial / P-SAFE under-treatment"
    elif psafe_ndcg >= bh_ndcg:
        taxonomy = "Recovery / Selective-win"
    else:
        taxonomy = "Selective escalation"

    metrics['taxonomy'] = taxonomy
    metrics['dataset'] = dataset_name

    return metrics
