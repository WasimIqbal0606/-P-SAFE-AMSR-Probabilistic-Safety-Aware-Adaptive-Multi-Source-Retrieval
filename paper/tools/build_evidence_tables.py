"""Build manuscript tables and audit validated B-P-SAFE result artifacts.

This script intentionally reads only results/validated. It does not rerun
retrieval or invent missing values. It verifies the query-level evidence used
by the paper, then writes reproducible LaTeX/CSV/JSON artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "validated"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
SUPP = PAPER / "supplementary"

DATASETS = ["scifact", "fiqa", "nfcorpus", "arguana"]
DATASET_LABELS = {
    "scifact": "SciFact",
    "fiqa": "FiQA",
    "nfcorpus": "NFCorpus",
    "arguana": "ArguAna",
}
SEEDS = [42, 123, 2026]
MODES = ["lite", "balanced", "high_recall"]
MODE_LABELS = {"lite": "Lite", "balanced": "Balanced", "high_recall": "High recall"}
PRIMARY_MODES = ["balanced", "high_recall"]

BASELINE_ORDER = [
    "Dense-only",
    "Always-Hybrid",
    "Random",
    "Dense-margin",
    "Dense-entropy",
    "Regression-only",
    "Classification-only",
    "Oracle",
]

REQUIRED_FILES = [
    "extended_metrics.json",
    "statistical_tests.json",
    "per_query_metrics.csv",
    "latency_breakdown.json",
    "latency_per_query.csv",
    "baseline_results.json",
    "baseline_per_query_metrics.csv",
    "baseline_statistical_tests.json",
    "reproducibility_manifest.json",
    "router_mode_config.json",
    "action_predictions.csv",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_dir(dataset: str, seed: int, mode: str) -> Path:
    return RESULTS / dataset / f"seed_{seed}" / mode


def approx_equal(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) <= tol


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function."""
    max_iter = 300
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_t(scores_a: Iterable[float], scores_b: Iterable[float]) -> tuple[float, float]:
    """Two-sided paired t-test; returns t for b-a and its p-value."""
    a = np.asarray(list(scores_a), dtype=float)
    b = np.asarray(list(scores_b), dtype=float)
    if len(a) != len(b):
        raise ValueError("Paired arrays have unequal length")
    delta = b - a
    if len(delta) < 2 or np.std(delta, ddof=1) == 0:
        return 0.0, 1.0
    t_stat = float(np.mean(delta) / (np.std(delta, ddof=1) / math.sqrt(len(delta))))
    df = len(delta) - 1
    x = df / (df + t_stat * t_stat)
    p_value = regularized_beta(x, df / 2.0, 0.5)
    return t_stat, p_value


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        avg = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg
        start = end
    return ranks


def wilcoxon_w(deltas: np.ndarray, eps: float = 1e-8) -> tuple[float, int]:
    nonzero = deltas[np.abs(deltas) > eps]
    if len(nonzero) == 0:
        return 0.0, 0
    ranks = average_ranks(np.abs(nonzero))
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    return min(positive, negative), len(nonzero)


def bootstrap_ci(deltas: np.ndarray, n_bootstrap: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    means = np.empty(n_bootstrap)
    for idx in range(n_bootstrap):
        means[idx] = np.mean(rng.choice(deltas, size=len(deltas), replace=True))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def permutation_p(deltas: np.ndarray, n_permutation: int = 2000) -> float:
    rng = np.random.default_rng(42)
    observed = abs(float(np.mean(deltas)))
    count = 0
    for _ in range(n_permutation):
        trial = abs(float(np.mean(rng.choice([-1, 1], size=len(deltas)) * deltas)))
        if trial >= observed:
            count += 1
    return count / n_permutation


def fmt_float(value: float, digits: int = 4, signed: bool = False) -> str:
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}\\%"


def fmt_mean_sd(values: list[float], scale: float = 1.0, digits: int = 3) -> str:
    scaled = [scale * value for value in values]
    return f"{mean(scaled):.{digits}f} $\\pm$ {stdev(scaled):.{digits}f}"


def fmt_p(value: float) -> str:
    if value >= 0.9995:
        return "1.000"
    if value >= 0.001:
        return f"{value:.3f}"
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10**exponent)
    return f"${mantissa:.2f}\\!\\times\\!10^{{{exponent}}}$"


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def audit_one_run(dataset: str, seed: int, mode: str) -> dict:
    directory = run_dir(dataset, seed, mode)
    missing = [name for name in REQUIRED_FILES if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing {missing}")

    metrics = read_json(directory / "extended_metrics.json")
    tests = read_json(directory / "statistical_tests.json")
    manifest = read_json(directory / "reproducibility_manifest.json")
    rows = read_csv(directory / "per_query_metrics.csv")

    query_ids = [row["query_id"] for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise AssertionError(f"{directory}: duplicate query IDs")
    if len(rows) != manifest["split_sizes"]["test"]:
        raise AssertionError(f"{directory}: test row count does not match manifest")

    dense = np.asarray([float(row["dense_ndcg"]) for row in rows])
    hybrid = np.asarray([float(row["hybrid_ndcg"]) for row in rows])
    psafe = np.asarray([float(row["psafe_ndcg"]) for row in rows])

    for key, observed in [
        ("dense_ndcg", float(np.mean(dense))),
        ("best_hybrid_ndcg", float(np.mean(hybrid))),
        ("psafe_ndcg", float(np.mean(psafe))),
    ]:
        if not approx_equal(metrics[key], observed):
            raise AssertionError(f"{directory}: {key} mismatch")

    comparisons = {
        "P-SAFE vs Dense": (dense, psafe),
        "P-SAFE vs Hybrid": (hybrid, psafe),
    }
    comparison_audit = {}
    for name, (baseline, system) in comparisons.items():
        stored = tests[name]
        deltas = system - baseline
        t_stat, p_value = paired_t(baseline, system)
        w_stat, n_nonzero = wilcoxon_w(deltas)
        ci_low, ci_high = bootstrap_ci(deltas, stored["bootstrap_ci"]["n_bootstrap"])
        perm_p = permutation_p(deltas, stored["permutation_test"]["n_permutations"])

        checks = {
            "mean_delta": approx_equal(float(np.mean(deltas)), stored["mean_delta"]),
            "t_statistic": approx_equal(t_stat, stored["paired_ttest"]["t_statistic"], 1e-9),
            "t_p_value": approx_equal(p_value, stored["paired_ttest"]["p_value"], 1e-9),
            "wilcoxon_statistic": approx_equal(w_stat, stored["wilcoxon"].get("w_statistic", 0.0), 1e-9),
            "wilcoxon_nonzero": n_nonzero == stored["wilcoxon"].get("n_nonzero", 0),
            "bootstrap_low": approx_equal(ci_low, stored["bootstrap_ci"]["ci_low"], 1e-12),
            "bootstrap_high": approx_equal(ci_high, stored["bootstrap_ci"]["ci_high"], 1e-12),
            "permutation_p": approx_equal(perm_p, stored["permutation_test"]["p_value"], 1e-12),
        }
        if not all(checks.values()):
            failed = [key for key, passed in checks.items() if not passed]
            raise AssertionError(f"{directory} {name}: failed checks {failed}")
        comparison_audit[name] = checks

    baseline_rows = read_csv(directory / "baseline_per_query_metrics.csv")
    baseline_query_sets: dict[str, set[str]] = {}
    for row in baseline_rows:
        baseline_query_sets.setdefault(row["baseline"], set()).add(row["query_id"])
    expected_set = set(query_ids)
    for method, method_ids in baseline_query_sets.items():
        if method_ids != expected_set:
            raise AssertionError(f"{directory}: baseline {method} uses different queries")

    return {
        "dataset": dataset,
        "seed": seed,
        "mode": mode,
        "n_queries": len(rows),
        "split_hash": metrics["split_hash"],
        "checks": comparison_audit,
        "status": "PASS",
    }


def build_main_results_table() -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Seed-42 endpoint quality and measured end-to-end latency trade-offs. "
        "LS is relative to always-on Deep Hybrid; HGR is not applicable when Hybrid does not improve on Dense and is reported as 0.}",
        "\\label{tab:main_results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Dataset & Mode & Dense & Hybrid & B-P-SAFE & $\\Delta_D$ & $\\Delta_H$ & LS & HAR & HGR \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for mode in MODES:
            metrics = read_json(run_dir(dataset, 42, mode) / "extended_metrics.json")
            lines.append(
                f"{DATASET_LABELS[dataset]} & {MODE_LABELS[mode]} & "
                f"{metrics['dense_ndcg']:.4f} & {metrics['best_hybrid_ndcg']:.4f} & "
                f"{metrics['psafe_ndcg']:.4f} & "
                f"{fmt_float(metrics['psafe_ndcg'] - metrics['dense_ndcg'], signed=True)} & "
                f"{fmt_float(metrics['psafe_ndcg'] - metrics['best_hybrid_ndcg'], signed=True)} & "
                f"{fmt_pct(metrics['latency_saving_vs_best_hybrid'])} & "
                f"{fmt_pct(metrics['hybrid_activation_rate'])} & "
                f"{metrics['quality_retention_vs_hybrid']:.3f} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace[1pt]")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (TABLES / "main_results.tex").write_text("\n".join(lines), encoding="utf-8")


def build_statistical_table() -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Seed-42 paired validation of B-P-SAFE against Dense. "
        "All tests use the same held-out queries; CIs and permutations use 5,000 and 2,000 draws, respectively.}",
        "\\label{tab:paired_stats}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Dataset & Mode & $n$ & $\\Delta_D$ & $t$ & $p_t$ & $W$ & $p_W$ & 95\\% bootstrap CI & $p_{\\rm perm}$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for mode in PRIMARY_MODES:
            tests = read_json(run_dir(dataset, 42, mode) / "statistical_tests.json")["P-SAFE vs Dense"]
            lines.append(
                f"{DATASET_LABELS[dataset]} & {MODE_LABELS[mode]} & {tests['n_queries']} & "
                f"{fmt_float(tests['mean_delta'], signed=True)} & "
                f"{tests['paired_ttest']['t_statistic']:.3f} & "
                f"{fmt_p(tests['paired_ttest']['p_value'])} & "
                f"{tests['wilcoxon']['w_statistic']:.1f} & "
                f"{fmt_p(tests['wilcoxon']['p_value'])} & "
                f"$[{tests['bootstrap_ci']['ci_low']:+.4f},{tests['bootstrap_ci']['ci_high']:+.4f}]$ & "
                f"{fmt_p(tests['permutation_test']['p_value'])} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace[1pt]")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (TABLES / "paired_stats.tex").write_text("\n".join(lines), encoding="utf-8")


def baseline_arrays(directory: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for row in read_csv(directory / "baseline_per_query_metrics.csv"):
        grouped.setdefault(row["baseline"], []).append((row["query_id"], float(row["ndcg"])))
    output = {}
    for method, pairs in grouped.items():
        pairs.sort(key=lambda pair: pair[0])
        output[method] = ([pair[0] for pair in pairs], np.asarray([pair[1] for pair in pairs]))
    return output


def psafe_array(directory: Path) -> tuple[list[str], np.ndarray]:
    pairs = [(row["query_id"], float(row["psafe_ndcg"])) for row in read_csv(directory / "per_query_metrics.csv")]
    pairs.sort(key=lambda pair: pair[0])
    return [pair[0] for pair in pairs], np.asarray([pair[1] for pair in pairs])


def collect_baseline_rows(dataset: str) -> list[dict]:
    directory = run_dir(dataset, 42, "balanced")
    results = read_json(directory / "baseline_results.json")
    arrays = baseline_arrays(directory)
    dense_ids, dense = arrays["Dense-only"]
    hybrid_ids, hybrid = arrays["Always-Hybrid"]
    if dense_ids != hybrid_ids:
        raise AssertionError(f"{dataset}: endpoint query mismatch")
    hybrid_latency = results["Always-Hybrid"]["mean_latency"]

    rows = []
    for method in BASELINE_ORDER:
        ids, scores = arrays[method]
        if ids != dense_ids:
            raise AssertionError(f"{dataset}: {method} query mismatch")
        _, p_dense = paired_t(dense, scores)
        _, p_hybrid = paired_t(hybrid, scores)
        latency = results[method]["mean_latency"]
        rows.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "mode": "--",
                "method": method,
                "ndcg": float(np.mean(scores)),
                "delta_dense": float(np.mean(scores - dense)),
                "delta_hybrid": float(np.mean(scores - hybrid)),
                "latency": latency,
                "latency_saving": 1.0 - latency / hybrid_latency,
                "activation": results[method]["hybrid_activation"],
                "p_dense": p_dense,
                "p_hybrid": p_hybrid,
            }
        )

    for mode in PRIMARY_MODES:
        psafe_directory = run_dir(dataset, 42, mode)
        ids, scores = psafe_array(psafe_directory)
        if ids != dense_ids:
            raise AssertionError(f"{dataset}: B-P-SAFE {mode} query mismatch")
        metrics = read_json(psafe_directory / "extended_metrics.json")
        _, p_dense = paired_t(dense, scores)
        _, p_hybrid = paired_t(hybrid, scores)
        rows.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "mode": MODE_LABELS[mode],
                "method": "B-P-SAFE",
                "ndcg": float(np.mean(scores)),
                "delta_dense": float(np.mean(scores - dense)),
                "delta_hybrid": float(np.mean(scores - hybrid)),
                "latency": metrics["psafe_latency"],
                "latency_saving": metrics["latency_saving_vs_best_hybrid"],
                "activation": metrics["hybrid_activation_rate"],
                "p_dense": p_dense,
                "p_hybrid": p_hybrid,
            }
        )
    return rows


def build_baseline_tables() -> list[dict]:
    all_rows = [row for dataset in DATASETS for row in collect_baseline_rows(dataset)]
    with (SUPP / "router_baseline_comparison_seed42.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    dataset_pairs = [(DATASETS[:2], "a"), (DATASETS[2:], "b")]
    for datasets, suffix in dataset_pairs:
        lines = [
            "\\begin{table}[!htbp]",
            "\\centering",
            "\\caption{Router baseline comparison on seed 42"
            + (" (part I)." if suffix == "a" else " (part II).")
            + " Baseline-router latency excludes uninstrumented decision overhead; B-P-SAFE latency is measured end-to-end. "
            + "The random router uses a validation-derived activation probability, not matched test activation.}",
            f"\\label{{tab:router_baselines_{suffix}}}",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2.2pt}",
            "\\begin{tabular}{lllrrrrrrrr}",
            "\\toprule",
            "Dataset & Mode & Method & nDCG@10 & $\\Delta_D$ & $\\Delta_H$ & Lat. (ms) & LS & HAR & $p_D$ & $p_H$ \\\\",
            "\\midrule",
        ]
        for dataset in datasets:
            dataset_rows = [row for row in all_rows if row["dataset"] == DATASET_LABELS[dataset]]
            for row in dataset_rows:
                lines.append(
                    f"{row['dataset']} & {row['mode']} & {row['method']} & "
                    f"{row['ndcg']:.4f} & {fmt_float(row['delta_dense'], signed=True)} & "
                    f"{fmt_float(row['delta_hybrid'], signed=True)} & {row['latency']:.1f} & "
                    f"{fmt_pct(row['latency_saving'])} & {fmt_pct(row['activation'])} & "
                    f"{fmt_p(row['p_dense'])} & {fmt_p(row['p_hybrid'])} \\\\"
                )
            if dataset != datasets[-1]:
                lines.append("\\midrule")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
        (TABLES / f"router_baselines_{suffix}.tex").write_text("\n".join(lines), encoding="utf-8")
    return all_rows


def not_significantly_worse(directory: Path) -> bool:
    comparison = read_json(directory / "statistical_tests.json")["P-SAFE vs Hybrid"]
    # A positive delta is not "worse" even if it is statistically significant.
    return comparison["mean_delta"] >= 0 or comparison["paired_ttest"]["p_value"] >= 0.05


def build_multiseed_tables() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for mode in PRIMARY_MODES:
            for seed in SEEDS:
                directory = run_dir(dataset, seed, mode)
                metrics = read_json(directory / "extended_metrics.json")
                rows.append(
                    {
                        "dataset": DATASET_LABELS[dataset],
                        "mode": MODE_LABELS[mode],
                        "seed": seed,
                        "ndcg": metrics["psafe_ndcg"],
                        "latency_saving": metrics["latency_saving_vs_best_hybrid"],
                        "activation": metrics["hybrid_activation_rate"],
                        "hgr": metrics["quality_retention_vs_hybrid"],
                        "ogc": metrics["oracle_gap_closed"],
                        "rc": metrics["recovery_capture"],
                        "ha": metrics["harm_avoidance"],
                        "beats_dense": metrics["psafe_ndcg"] > metrics["dense_ndcg"],
                        "not_sig_worse_hybrid": not_significantly_worse(directory),
                    }
                )

    with (SUPP / "multi_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Three-split sensitivity analysis over seeds 42, 123, and 2026 (mean $\\pm$ sample SD). "
        "Counts report seeds beating Dense and seeds with no significant negative difference from Deep Hybrid.}",
        "\\label{tab:multiseed_summary}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.7pt}",
        "\\begin{tabular}{llrrrrrrrrr}",
        "\\toprule",
        "Dataset & Mode & nDCG@10 & LS (\\%) & HAR (\\%) & HGR & OGC & RC & HA & $>{D}$ & $\\not<_{\\rm sig}H$ \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for mode in PRIMARY_MODES:
            subset = [
                row
                for row in rows
                if row["dataset"] == DATASET_LABELS[dataset] and row["mode"] == MODE_LABELS[mode]
            ]
            summary_lines.append(
                f"{DATASET_LABELS[dataset]} & {MODE_LABELS[mode]} & "
                f"{fmt_mean_sd([row['ndcg'] for row in subset])} & "
                f"{fmt_mean_sd([row['latency_saving'] for row in subset], scale=100, digits=1)} & "
                f"{fmt_mean_sd([row['activation'] for row in subset], scale=100, digits=1)} & "
                f"{fmt_mean_sd([row['hgr'] for row in subset])} & "
                f"{fmt_mean_sd([row['ogc'] for row in subset])} & "
                f"{fmt_mean_sd([row['rc'] for row in subset])} & "
                f"{fmt_mean_sd([row['ha'] for row in subset])} & "
                f"{sum(row['beats_dense'] for row in subset)}/3 & "
                f"{sum(row['not_sig_worse_hybrid'] for row in subset)}/3 \\\\"
            )
        if dataset != DATASETS[-1]:
            summary_lines.append("\\addlinespace[1pt]")
    summary_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (TABLES / "multiseed_summary.tex").write_text("\n".join(summary_lines), encoding="utf-8")

    raw_lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Per-seed routing diagnostics used in Table~\\ref{tab:multiseed_summary}.}",
        "\\label{tab:multiseed_raw}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Dataset & Mode/seed & nDCG@10 & LS & HAR & HGR & OGC & RC & HA \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == DATASET_LABELS[dataset]]
        for row in subset:
            raw_lines.append(
                f"{row['dataset']} & {row['mode']} / {row['seed']} & {row['ndcg']:.4f} & "
                f"{fmt_pct(row['latency_saving'])} & {fmt_pct(row['activation'])} & "
                f"{row['hgr']:.3f} & {row['ogc']:.3f} & {row['rc']:.3f} & {row['ha']:.3f} \\\\"
            )
        if dataset != DATASETS[-1]:
            raw_lines.append("\\midrule")
    raw_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (TABLES / "multiseed_raw.tex").write_text("\n".join(raw_lines), encoding="utf-8")
    return rows


def build_config_table() -> None:
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Saved seed-42 deployment settings. Lite is fixed; balanced and high-recall settings are validation-selected or configured before test evaluation.}",
        "\\label{tab:mode_configs}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\begin{tabular}{llrrrrrrc}",
        "\\toprule",
        "Dataset & Mode & $\\tau_{\\rm gain}$ & $\\tau_{\\rm harm}$ & $\\lambda_{\\rm lat}$ & "
        "$\\lambda_{\\rm harm}$ & $\\lambda_{\\rm rec}$ & $\\lambda_{\\rm cand}$ & LCB flag \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for mode in MODES:
            config = read_json(run_dir(dataset, 42, mode) / "router_mode_config.json")
            lines.append(
                f"{DATASET_LABELS[dataset]} & {MODE_LABELS[mode]} & "
                f"{config['gain_threshold']:.2f} & {config['harm_threshold']:.2f} & "
                f"{config['lambda_latency']:.1e} & {config['lambda_harm']:.2f} & "
                f"{config['lambda_recovery']:.2f} & {config['lambda_candidate']:.1e} & "
                f"{'on' if config['use_lcb_safety'] else 'off'} \\\\"
            )
        if dataset != DATASETS[-1]:
            lines.append("\\addlinespace[1pt]")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (TABLES / "mode_configs.tex").write_text("\n".join(lines), encoding="utf-8")


def build_statistical_validation_note(audits: list[dict]) -> None:
    lines = [
        "# Statistical and Evidence Validation",
        "",
        "Generated by `paper/tools/build_evidence_tables.py` from `results/validated`.",
        "",
        "## Checks applied",
        "",
        "- Required result, manifest, latency, baseline, prediction, and per-query files exist.",
        "- Per-query IDs are unique and identical across paired systems and router baselines.",
        "- Per-query means reproduce `extended_metrics.json`.",
        "- Mean deltas, paired t statistics and p-values, Wilcoxon W and nonzero counts,",
        "  seeded bootstrap confidence intervals, and seeded sign-permutation p-values reproduce",
        "  `statistical_tests.json`.",
        "- Baseline comparisons use the same seed-42 test queries.",
        "",
        "The Wilcoxon p-value itself is traced to SciPy's paired signed-rank implementation in",
        "`src/psafe/statistical_tests.py`; this audit independently reproduces W and the number",
        "of nonzero paired differences.",
        "",
        "## Run inventory",
        "",
        "| Dataset | Seed | Mode | Test queries | Split hash | Status |",
        "|---|---:|---|---:|---|---|",
    ]
    for audit in audits:
        lines.append(
            f"| {DATASET_LABELS[audit['dataset']]} | {audit['seed']} | "
            f"{MODE_LABELS[audit['mode']]} | {audit['n_queries']} | "
            f"`{audit['split_hash']}` | {audit['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "Because multiple datasets, modes, and tests are reported, the manuscript treats",
            "p-values as evidence for paired differences within predefined settings, not as one",
            "family-wise confirmatory test. No statistical test is described as a guarantee.",
            "",
        ]
    )
    (SUPP / "statistical_validation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)

    audits = [
        audit_one_run(dataset, seed, mode)
        for dataset in DATASETS
        for seed in SEEDS
        for mode in MODES
    ]
    build_main_results_table()
    build_statistical_table()
    baseline_rows = build_baseline_tables()
    multiseed_rows = build_multiseed_tables()
    build_config_table()
    build_statistical_validation_note(audits)

    payload = {
        "status": "PASS",
        "datasets": DATASETS,
        "seeds": SEEDS,
        "modes": MODES,
        "audited_runs": len(audits),
        "baseline_rows": len(baseline_rows),
        "multiseed_rows": len(multiseed_rows),
        "run_audits": audits,
    }
    (SUPP / "evidence_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"PASS: audited {len(audits)} runs; wrote {len(baseline_rows)} baseline rows "
        f"and {len(multiseed_rows)} multi-seed rows."
    )


if __name__ == "__main__":
    main()
