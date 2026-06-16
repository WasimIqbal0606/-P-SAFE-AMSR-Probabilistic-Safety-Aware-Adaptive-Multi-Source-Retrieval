"""
B-P-SAFE-AMSR — Experiment Core Module
Shared logic for multi-seed, multi-dataset experiments.
"""
import os, sys, time, json, csv, itertools
import numpy as np
from collections import Counter
from psafe.router import BPSafeRouter
from psafe.actions import Action, ACTION_NAMES
from psafe.metrics import calculate_extended_metrics
from psafe.statistical_tests import StatisticalTester
from psafe.feature_extractor import FeatureExtractor, FEATURE_NAMES
from psafe.latency_tracker import LatencyTracker

REQUIRED_OUTPUTS = [
    "extended_metrics.json", "statistical_tests.json", "safety_metrics.json",
    "action_distribution.json", "action_predictions.csv", "validation_tuning.json",
    "best_router_config.json", "router_mode_config.json", "router_thresholds.json",
    "baseline_competitiveness.json", "router_class_balance.json", "latency_breakdown.json", "per_query_metrics.csv",
    "reproducibility_manifest.json",
]


def get_output_dir(root, dataset, seed, mode, fold=None):
    if fold is None:
        return os.path.join(root, dataset, f"seed_{seed}", mode)
    return os.path.join(root, dataset, f"seed_{seed}", f"fold_{fold}", mode)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def check_consistency(out_dir, dense_ndcg, hybrid_ndcg, psafe_ndcg, action_counts, total_queries):
    """Phase 6: Verify metrics files agree. Returns list of errors."""
    errors = []
    em_path = os.path.join(out_dir, "extended_metrics.json")
    st_path = os.path.join(out_dir, "statistical_tests.json")

    if os.path.exists(em_path) and os.path.exists(st_path):
        with open(em_path) as f: em = json.load(f)
        with open(st_path) as f: st = json.load(f)

        # Check dense_ndcg consistency
        st_dense = st.get("P-SAFE vs Dense", {})
        if abs(st_dense.get("baseline_mean", 0) - dense_ndcg) > 1e-6:
            errors.append(f"dense_ndcg mismatch: extended={dense_ndcg:.6f} vs stat_test={st_dense.get('baseline_mean', 0):.6f}")
        if abs(st_dense.get("system_mean", 0) - psafe_ndcg) > 1e-6:
            errors.append(f"psafe_ndcg mismatch: extended={psafe_ndcg:.6f} vs stat_test={st_dense.get('system_mean', 0):.6f}")

    # Check action_distribution vs action_predictions
    ad_path = os.path.join(out_dir, "action_distribution.json")
    ap_path = os.path.join(out_dir, "action_predictions.csv")
    if os.path.exists(ad_path) and os.path.exists(ap_path):
        with open(ad_path) as f: ad = json.load(f)
        with open(ap_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        csv_counts = Counter(r.get("action_name", r.get("selected_action", "")) for r in rows if r.get("split") == "test")
        ad_dist = ad.get("distribution", {})
        for k in set(list(ad_dist.keys()) + list(csv_counts.keys())):
            if ad_dist.get(k, 0) != csv_counts.get(k, 0):
                errors.append(f"action_distribution[{k}] mismatch: json={ad_dist.get(k,0)} vs csv={csv_counts.get(k,0)}")

    if errors:
        save_json(os.path.join(out_dir, "consistency_errors.json"), {"errors": errors})
    return errors


def build_reproducibility_manifest(dataset, seed, mode, config, nq, nd, split_sizes,
                                    thresholds, lambdas, cache_used, runtime_s):
    import platform
    manifest = {
        "dataset": dataset, "seed": seed, "mode": mode,
        "router": "B-P-SAFE-AMSR",
        "profile": "bge_m3",
        "embedding_model": config.get("model_profiles", {}).get("bge_m3", {}).get("embedding_model", "BAAI/bge-m3"),
        "reranker_model": config.get("model_profiles", {}).get("bge_m3", {}).get("reranker_model", "BAAI/bge-reranker-v2-m3"),
        "device": config.get("hardware", {}).get("device", "auto"),
        "python_version": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "num_queries": nq, "num_docs": nd,
        "split_sizes": split_sizes,
        "thresholds": thresholds,
        "lambda_values": lambdas,
        "cache_used": cache_used,
        "runtime_seconds": round(runtime_s, 2),
    }
    try:
        import torch
        manifest["cuda_available"] = torch.cuda.is_available()
        manifest["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    except ImportError:
        manifest["cuda_available"] = False
        manifest["gpu_name"] = "N/A"
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        manifest["git_commit"] = r.stdout.strip() if r.returncode == 0 else "unavailable"
    except Exception:
        manifest["git_commit"] = "unavailable"
    return manifest


def write_skipped(out_dir, filename, reason):
    save_json(os.path.join(out_dir, filename), {"skipped": True, "reason": reason})


def check_missing_outputs(out_dir):
    """Return list of missing required output files."""
    missing = []
    for f in REQUIRED_OUTPUTS:
        if not os.path.exists(os.path.join(out_dir, f)):
            missing.append(f)
    return missing


def write_skipped_outputs(out_dir, missing_files, reason="not generated"):
    if missing_files:
        save_json(os.path.join(out_dir, "skipped_outputs.json"),
                  {"missing": missing_files, "reason": reason})
