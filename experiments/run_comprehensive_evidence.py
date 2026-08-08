"""Regenerate only retained, source-backed secondary paper evidence.

This script is intentionally lightweight. It never loads retrieval models and
never constructs train/validation data from held-out test artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.tools.build_evidence_tables import bootstrap_ci, paired_t, regularized_beta


ROOT = Path(__file__).resolve().parents[1]
VALIDATED = ROOT / "results" / "validated"
DATASETS = ["scifact", "fiqa", "nfcorpus", "arguana"]
SEEDS = [42, 123, 2026]
MODES = ["balanced", "high_recall"]
BASELINES = [
    "Dense-only", "Always-Hybrid", "Random", "Dense-margin", "Dense-entropy",
    "Regression-only", "Classification-only", "Oracle",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def holm(raw: list[float]) -> list[float]:
    order = np.argsort(np.asarray(raw))
    result = np.zeros(len(raw), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(raw) - rank) * raw[index])
        result[index] = min(1.0, running)
    return result.tolist()


def regenerate_verified_baselines() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for seed in SEEDS:
            path = VALIDATED / dataset / f"seed_{seed}" / "balanced" / "baseline_results.json"
            source = read_json(path)
            missing = [name for name in BASELINES if name not in source]
            if missing:
                raise KeyError(f"{path}: missing verified baselines {missing}")
            output[dataset][str(seed)] = {name: source[name] for name in BASELINES}
    write_json(ROOT / "results/baselines/comprehensive_baseline_results.json", output)
    return output


def matched_scores(dense: np.ndarray, hybrid: np.ndarray, target_k: int) -> np.ndarray:
    scores = np.empty(1000, dtype=float)
    for repetition in range(1000):
        rng = np.random.RandomState(42 + repetition * 1000 + 7)
        selected = rng.permutation(len(dense))[:target_k]
        routed = dense.copy()
        routed[selected] = hybrid[selected]
        scores[repetition] = float(np.mean(routed))
    return scores


def regenerate_matched_random() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for seed in SEEDS:
            output[dataset][str(seed)] = {}
            for mode in MODES:
                directory = VALIDATED / dataset / f"seed_{seed}" / mode
                rows = read_csv(directory / "per_query_metrics.csv")
                actions = read_csv(directory / "action_predictions.csv")
                dense = np.asarray([float(row["dense_ndcg"]) for row in rows])
                hybrid = np.asarray([float(row["hybrid_ndcg"]) for row in rows])
                psafe = float(np.mean([float(row["psafe_ndcg"]) for row in rows]))
                target_k = sum(
                    str(row["selected_action"]) in {"6", "A6_DEEP_HYBRID", "Deep Hybrid"}
                    for row in actions
                )
                scores = matched_scores(dense, hybrid, target_k)
                entry = {
                    "router_name": "Matched-Budget-Random",
                    "target_activation_rate": target_k / len(rows),
                    "target_k": target_k,
                    "n_queries": len(rows),
                    "n_repetitions": 1000,
                    "mean_ndcg": float(np.mean(scores)),
                    "std_ndcg": float(np.std(scores, ddof=1)),
                    "ci_95": [float(x) for x in np.percentile(scores, [2.5, 97.5])],
                    "p5_p95": [float(x) for x in np.percentile(scores, [5.0, 95.0])],
                    "min_ndcg": float(np.min(scores)),
                    "max_ndcg": float(np.max(scores)),
                    "psafe_mean_ndcg": psafe,
                    "delta_psafe_vs_matched_random": float(psafe - np.mean(scores)),
                    "empirical_p_value": float((1 + np.sum(scores >= psafe)) / 1001),
                    "psafe_beats_random_ci": bool(psafe > np.percentile(scores, 97.5)),
                }
                output[dataset][str(seed)][mode] = entry
    write_json(ROOT / "results/baselines/matched_budget_random_results.json", output)
    return output


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    details: list[dict[str, Any]] = []
    error = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1])
        count = int(np.sum(mask))
        accuracy = float(np.mean(y[mask])) if count else 0.0
        confidence = float(np.mean(p[mask])) if count else 0.0
        error += count / len(y) * abs(accuracy - confidence)
        details.append({
            "bin": index,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "mean_predicted": confidence,
            "fraction_positive": accuracy,
        })
    return float(error), details


def adaptive_ece(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    edges = np.percentile(p, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    error = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (p <= edges[index + 1] if index == bins - 1 else p < edges[index + 1])
        count = int(np.sum(mask))
        if count:
            error += count / len(y) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(error)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    positive = int(np.sum(y == 1))
    negative = len(y) - positive
    if not positive or not negative:
        return 0.5
    ranks = average_ranks(p)
    return float((np.sum(ranks[y == 1]) - positive * (positive + 1) / 2) / (positive * negative))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    positive = int(np.sum(y == 1))
    if not positive:
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float(np.sum(precision * sorted_y) / positive)


def calibration_line(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return 1.0, 0.0
    logits = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    design = np.column_stack([np.ones(len(y)), logits])
    beta = np.zeros(2, dtype=float)
    for _ in range(100):
        eta = np.clip(design @ beta, -30, 30)
        fitted = 1 / (1 + np.exp(-eta))
        weights = np.clip(fitted * (1 - fitted), 1e-9, None)
        hessian = design.T @ (weights[:, None] * design)
        gradient = design.T @ (y - fitted)
        step = np.linalg.pinv(hessian) @ gradient
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[1]), float(beta[0])


def calibration_report(y: np.ndarray, p: np.ndarray, label: str) -> dict[str, Any]:
    p = np.clip(p, 0.0, 1.0)
    ece, bins = reliability(y, p)
    slope, intercept = calibration_line(y, p)
    prevalence = float(np.mean(y))
    return {
        "label": label,
        "n_samples": len(y),
        "n_positive": int(np.sum(y)),
        "n_negative": int(np.sum(1 - y)),
        "positive_rate": prevalence,
        "prevalence": prevalence,
        "brier_score": float(np.mean((p - y) ** 2)),
        "ece": ece,
        "adaptive_ece": adaptive_ece(y, p),
        "auroc": auroc(y, p),
        "auprc": auprc(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "reliability_bins": bins,
    }


def regenerate_calibration() -> dict[str, Any]:
    output: dict[str, Any] = {}
    reliability_output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset], reliability_output[dataset] = {}, {}
        for seed in SEEDS:
            output[dataset][str(seed)], reliability_output[dataset][str(seed)] = {}, {}
            for mode in MODES:
                directory = VALIDATED / dataset / f"seed_{seed}" / mode
                metrics = read_csv(directory / "per_query_metrics.csv")
                predictions = read_csv(directory / "action_predictions.csv")
                dense = np.asarray([float(row["dense_ndcg"]) for row in metrics])
                hybrid = np.asarray([float(row["hybrid_ndcg"]) for row in metrics])
                delta = hybrid - dense
                gain = (delta > 0.05).astype(int)
                harm = (delta < -0.01).astype(int)
                p_gain = np.asarray([float(row["p_gain"]) for row in predictions])
                p_harm = np.asarray([float(row["p_harm"]) for row in predictions])
                reports = {
                    "P_gain": calibration_report(gain, p_gain, f"{dataset}_seed{seed}_{mode}_Pgain"),
                    "P_harm": calibration_report(harm, p_harm, f"{dataset}_seed{seed}_{mode}_Pharm"),
                }
                output[dataset][str(seed)][mode] = reports
                reliability_output[dataset][str(seed)][mode] = {
                    key: value["reliability_bins"] for key, value in reports.items()
                }
    write_json(ROOT / "results/calibration/calibration_metrics.json", output)
    write_json(ROOT / "results/calibration/reliability_data.json", reliability_output)
    return output


def t_cdf(value: float, degrees: int) -> float:
    x = degrees / (degrees + value * value)
    tail = 0.5 * regularized_beta(x, degrees / 2.0, 0.5)
    return 1.0 - tail if value >= 0 else tail


def t_quantile(probability: float, degrees: int) -> float:
    low, high = -20.0, 20.0
    for _ in range(100):
        middle = (low + high) / 2
        if t_cdf(middle, degrees) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def regenerate_statistics() -> dict[str, Any]:
    family_dense: list[dict[str, Any]] = []
    family_ni: list[dict[str, Any]] = []
    full: dict[str, Any] = {}
    for dataset in DATASETS:
        full[dataset] = {}
        for mode in MODES:
            rows = read_csv(VALIDATED / dataset / "seed_42" / mode / "per_query_metrics.csv")
            dense = np.asarray([float(row["dense_ndcg"]) for row in rows])
            hybrid = np.asarray([float(row["hybrid_ndcg"]) for row in rows])
            psafe = np.asarray([float(row["psafe_ndcg"]) for row in rows])
            delta_dense = psafe - dense
            _, raw_p = paired_t(dense, psafe)
            ci = bootstrap_ci(delta_dense, 5000)
            dz = float(np.mean(delta_dense) / np.std(delta_dense, ddof=1))
            family_dense.append({
                "dataset": dataset,
                "mode": mode,
                "mean_delta": float(np.mean(delta_dense)),
                "raw_p_value": raw_p,
                "cohens_dz": dz,
                "bootstrap_ci_95": [float(ci[0]), float(ci[1])],
            })
            delta_hybrid = psafe - hybrid
            mean_delta = float(np.mean(delta_hybrid))
            se = float(np.std(delta_hybrid, ddof=1) / math.sqrt(len(delta_hybrid)))
            t_value = (mean_delta + 0.010) / se if se else math.inf
            raw_ni = float(1 - t_cdf(t_value, len(delta_hybrid) - 1)) if se else 0.0
            lower = mean_delta - t_quantile(0.95, len(delta_hybrid) - 1) * se
            family_ni.append({
                "dataset": dataset,
                "mode": mode,
                "mean_delta_vs_hybrid": mean_delta,
                "epsilon": 0.010,
                "ci_lower_bound_95": float(lower),
                "raw_p_value_ni": raw_ni,
            })
            full[dataset][mode] = {"n_queries": len(rows)}
    dense_adjusted = holm([row["raw_p_value"] for row in family_dense])
    ni_adjusted = holm([row["raw_p_value_ni"] for row in family_ni])
    for row, adjusted in zip(family_dense, dense_adjusted):
        row["holm_adjusted_p_value"] = adjusted
        row["significant_after_correction"] = bool(adjusted < 0.05)
    for row, adjusted in zip(family_ni, ni_adjusted):
        row["holm_adjusted_p_value_ni"] = adjusted
        row["non_inferiority_established"] = bool(
            adjusted < 0.05 and row["ci_lower_bound_95"] > -row["epsilon"]
        )
    result = {
        "full_per_run_statistics": full,
        "family_1_dense_improvement_holm": family_dense,
        "family_2_non_inferiority_holm": family_ni,
    }
    write_json(ROOT / "results/statistics/statistical_analysis.json", result)
    return result


def main() -> None:
    regenerate_matched_random()
    regenerate_verified_baselines()
    regenerate_calibration()
    regenerate_statistics()
    print("RETAINED EVIDENCE REGENERATION: PASS")


if __name__ == "__main__":
    main()
