"""
B-P-SAFE-AMSR — Next-Level Visualization Generator
All plots are generated ONLY from real experiment output files.
No hardcoded or placeholder data is used.

Required input files (any may be absent; absent → skip + log):
  aggregate_metrics.json, safety_metrics.json, statistical_tests.json,
  action_predictions.csv, per_query_metrics.csv, probability_calibration.json,
  oracle_action_distribution.json, rejected_action_reasons.json,
  graph_contribution.json, latency_breakdown.json, extended_metrics.json
"""
import os
import json
import csv
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-import matplotlib so module can be imported even when display unavailable
_plt = None
_sns = None


def _ensure_mpl():
    global _plt, _sns
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        _plt = plt
        _sns = sns
    return _plt, _sns


# ── helpers ──────────────────────────────────────────────────────────

def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: str) -> Optional[List[dict]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, out_dir: str, name: str, captions: dict, caption_text: str):
    """Save PNG, PDF, SVG and register caption."""
    for ext in ("png", "pdf", "svg"):
        sub = os.path.join(out_dir, f"paper_figures_{ext}")
        os.makedirs(sub, exist_ok=True)
        fig.savefig(os.path.join(sub, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    _plt.close(fig)
    captions[name] = caption_text


# ── individual plot generators ───────────────────────────────────────

def _plot_pareto_quality_latency(metrics_dir, out_dir, captions, skipped):
    plt, sns = _ensure_mpl()
    agg = _load_json(os.path.join(metrics_dir, "aggregate_metrics.json"))
    if agg is None:
        skipped.append({"plot_name": "pareto_quality_latency", "reason": "missing aggregate_metrics.json",
                        "missing_input_file": "aggregate_metrics.json"})
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, data in agg.items():
        ndcg = data.get("ndcg_at_k", {}).get("10", data.get("ndcg_at_k", {}).get(10, None))
        lat = data.get("latency_mean_ms", None)
        if ndcg is not None and lat is not None:
            ax.scatter(max(lat, 0.1), ndcg, s=120, label=method, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms, log)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Quality vs Latency Pareto")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "pareto_quality_latency", captions,
          "Pareto frontier: quality (nDCG@10) vs latency per method.")


def _plot_per_query_delta_waterfall(metrics_dir, out_dir, captions, skipped):
    plt, _ = _ensure_mpl()
    rows = _load_csv(os.path.join(metrics_dir, "per_query_metrics.csv"))
    if rows is None:
        skipped.append({"plot_name": "per_query_delta_waterfall", "reason": "missing per_query_metrics.csv",
                        "missing_input_file": "per_query_metrics.csv"})
        return
    deltas = []
    for r in rows:
        d = r.get("delta_psafe_dense") or r.get("delta_ndcg")
        if d is not None:
            deltas.append(float(d))
    if not deltas:
        skipped.append({"plot_name": "per_query_delta_waterfall", "reason": "no delta column in per_query_metrics.csv",
                        "missing_input_file": "per_query_metrics.csv"})
        return
    deltas = sorted(deltas, reverse=True)
    colors = ["green" if x > 0 else "red" if x < 0 else "gray" for x in deltas]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(deltas)), deltas, color=colors, width=1.0)
    ax.axhline(np.mean(deltas), color="blue", linestyle="--", label=f"Mean: {np.mean(deltas):.4f}")
    ax.set_title("Per-Query Delta Waterfall (P-SAFE vs Dense)")
    ax.set_ylabel("Delta nDCG@10")
    ax.legend()
    _save(fig, out_dir, "per_query_delta_waterfall", captions,
          "Waterfall of per-query nDCG@10 delta (P-SAFE minus Dense), sorted descending.")


def _plot_action_distribution(metrics_dir, out_dir, captions, skipped):
    plt, _ = _ensure_mpl()
    rows = _load_csv(os.path.join(metrics_dir, "action_predictions.csv"))
    if rows is None:
        skipped.append({"plot_name": "action_distribution", "reason": "missing action_predictions.csv",
                        "missing_input_file": "action_predictions.csv"})
        return
    from collections import Counter
    action_field = "action_name" if "action_name" in rows[0] else "selected_action"
    counts = Counter(r.get(action_field, "unknown") for r in rows)
    labels, values = zip(*sorted(counts.items(), key=lambda x: -x[1]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, values, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
    ax.set_xlabel("Count")
    ax.set_title("Router Action Distribution")
    _save(fig, out_dir, "action_distribution", captions,
          "Distribution of actions selected by the B-P-SAFE router.")


def _plot_rejected_action_reasons(metrics_dir, out_dir, captions, skipped):
    plt, _ = _ensure_mpl()
    data = _load_json(os.path.join(metrics_dir, "rejected_action_reasons.json"))
    # Also try router_class_balance.json as fallback
    if data is None:
        rows = _load_csv(os.path.join(metrics_dir, "action_predictions.csv"))
        if rows is None:
            skipped.append({"plot_name": "rejected_action_reasons", "reason": "missing data files",
                            "missing_input_file": "rejected_action_reasons.json"})
            return
        from collections import Counter
        reasons = Counter(r.get("rejected_reason", "") for r in rows if r.get("rejected_reason"))
        if not reasons:
            skipped.append({"plot_name": "rejected_action_reasons", "reason": "no rejections recorded",
                            "missing_input_file": "rejected_action_reasons.json"})
            return
        data = dict(reasons)
    labels, values = zip(*sorted(data.items(), key=lambda x: -x[1]))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels, values)
    ax.set_xlabel("Count")
    ax.set_title("Rejected Action Reasons")
    _save(fig, out_dir, "rejected_action_reasons", captions,
          "Breakdown of why alternative actions were rejected by the safety constraints.")


def _plot_calibration(metrics_dir, out_dir, captions, skipped, key="p_gain"):
    """FIX 7: Support both new (p_gain.bins/empirical) and legacy (per-action cal_curve) formats."""
    plt, _ = _ensure_mpl()
    cal = _load_json(os.path.join(metrics_dir, "probability_calibration.json"))
    if cal is None:
        skipped.append({"plot_name": f"{key}_calibration", "reason": "missing probability_calibration.json",
                        "missing_input_file": "probability_calibration.json"})
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    plotted = False

    # Format A: new canonical format with key directly
    if key in cal:
        entry = cal[key]
        bins = entry.get("bins", entry.get("mean_pred", []))
        empirical = entry.get("empirical", entry.get("frac_pos", []))
        if bins and empirical:
            ax.plot(bins, empirical, "o-", label="Observed")
            plotted = True

    # Format B: legacy per-action keys like "Dense+BM25+CE_gain_cal_curve"
    if not plotted:
        suffix = "_gain_cal_curve" if "gain" in key else "_harm_cal_curve"
        for cal_key, cal_data in cal.items():
            if cal_key.endswith(suffix) and isinstance(cal_data, dict):
                x = cal_data.get("mean_pred", cal_data.get("bins", []))
                y = cal_data.get("frac_pos", cal_data.get("empirical", []))
                if x and y:
                    action_label = cal_key.replace(suffix, "")
                    ax.plot(x, y, "o-", label=action_label, alpha=0.8)
                    plotted = True

    if not plotted:
        plt.close(fig)
        skipped.append({"plot_name": f"{key}_calibration", "reason": f"no calibration data found for {key}",
                        "missing_input_file": "probability_calibration.json"})
        return

    ax.set_xlabel(f"Predicted P({key.replace('p_', '').title()})")
    ax.set_ylabel(f"Empirical P({key.replace('p_', '').title()})")
    ax.set_title(f"Calibration: {key.replace('p_', '').title()}")
    ax.legend(fontsize=8)
    _save(fig, out_dir, f"{key}_calibration", captions,
          f"Calibration plot for {key.replace('p_', '')} probability predictions.")


def _plot_oracle_gap(metrics_dir, out_dir, captions, skipped):
    plt, _ = _ensure_mpl()
    ext = _load_json(os.path.join(metrics_dir, "extended_metrics.json"))
    if ext is None:
        skipped.append({"plot_name": "oracle_gap", "reason": "missing extended_metrics.json",
                        "missing_input_file": "extended_metrics.json"})
        return
    gap = ext.get("oracle_gap", None)
    closed = ext.get("oracle_gap_closed", None)
    if gap is None:
        skipped.append({"plot_name": "oracle_gap", "reason": "oracle_gap not in extended_metrics",
                        "missing_input_file": "extended_metrics.json"})
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Oracle Gap", "Gap Closed"], [gap, closed or 0], color=["#e74c3c", "#2ecc71"])
    ax.set_ylabel("nDCG@10")
    ax.set_title("Oracle Gap Analysis")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.4f}",
                ha="center", va="bottom", fontsize=9)
    _save(fig, out_dir, "oracle_gap", captions,
          "Oracle gap: nDCG difference between P-SAFE and oracle routing, and fraction closed.")


def _plot_graph_contribution(metrics_dir, out_dir, captions, skipped):
    """FIX 9: Support multiple possible key schemas for graph contribution."""
    plt, _ = _ensure_mpl()
    data = _load_json(os.path.join(metrics_dir, "graph_contribution.json"))
    if data is None:
        skipped.append({"plot_name": "graph_contribution", "reason": "missing graph_contribution.json",
                        "missing_input_file": "graph_contribution.json"})
        return
    labels = ["Unique Relevant", "Total Graph-Only", "Win", "Loss"]
    values = [
        data.get("graph_unique_relevant_docs_total",
                 data.get("graph_unique_relevant_docs",
                          data.get("graph_only_relevant_count", 0))),
        data.get("graph_only_candidates_total",
                 data.get("graph_only_candidate_count", 0)),
        data.get("graph_action_win",
                 data.get("graph_action_win_count", 0)),
        data.get("graph_action_loss",
                 data.get("graph_action_loss_count", 0)),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["#2ecc71", "#3498db", "#27ae60", "#e74c3c"])
    ax.set_title("Graph Expansion Contribution")
    ax.set_ylabel("Count")
    _save(fig, out_dir, "graph_contribution", captions,
          "Graph expansion metrics: unique relevant docs found, action wins/losses.")


def _plot_latency_breakdown(metrics_dir, out_dir, captions, skipped):
    """FIX 8: Support both 'mean' and 'mean_ms' field names."""
    plt, _ = _ensure_mpl()
    data = _load_json(os.path.join(metrics_dir, "latency_breakdown.json"))
    if data is None:
        skipped.append({"plot_name": "latency_breakdown", "reason": "missing latency_breakdown.json",
                        "missing_input_file": "latency_breakdown.json"})
        return
    components = []
    means = []
    for comp, stats in data.items():
        if isinstance(stats, dict):
            mean = stats.get("mean", stats.get("mean_ms"))
            if mean is not None:
                components.append(comp)
                means.append(float(mean))
    if not components:
        skipped.append({"plot_name": "latency_breakdown", "reason": "no component means in latency_breakdown",
                        "missing_input_file": "latency_breakdown.json"})
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(components, means, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(components))))
    ax.set_xlabel("Mean Latency (ms)")
    ax.set_title("Latency Breakdown by Component")
    _save(fig, out_dir, "latency_breakdown", captions,
          "Mean latency per pipeline component.")


def _plot_quality_retention_vs_latency_saving(metrics_dir, out_dir, captions, skipped):
    plt, _ = _ensure_mpl()
    ext = _load_json(os.path.join(metrics_dir, "extended_metrics.json"))
    if ext is None:
        skipped.append({"plot_name": "quality_retention_vs_latency_saving",
                        "reason": "missing extended_metrics.json",
                        "missing_input_file": "extended_metrics.json"})
        return
    qr = ext.get("quality_retention_vs_best_hybrid", ext.get("quality_retention_vs_hybrid"))
    ls = ext.get("latency_saving_vs_best_hybrid")
    if qr is None or ls is None:
        skipped.append({"plot_name": "quality_retention_vs_latency_saving",
                        "reason": "missing QR or LS fields",
                        "missing_input_file": "extended_metrics.json"})
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ls, qr, s=200, c="#e67e22", edgecolors="black", zorder=5)
    ax.axhline(0.6, color="red", linestyle="--", alpha=0.4, label="QR=0.6")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.4, label="LS=0.5")
    ax.set_xlabel("Latency Saving vs Hybrid")
    ax.set_ylabel("Quality Retention vs Hybrid")
    ax.set_title("Quality Retention vs Latency Saving")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "quality_retention_vs_latency_saving", captions,
          "Trade-off: quality retained vs latency saved relative to always-on hybrid.")


# ── main entry point ─────────────────────────────────────────────────

def generate_all_real_plots(metrics_dir: str, out_dir: str = None):
    """
    Generate all publication plots from real experiment data.
    Skips any plot whose input files are missing.

    Args:
        metrics_dir: directory containing JSON/CSV experiment outputs
        out_dir: output directory (default: metrics_dir/../visualizations_next_level)
    """
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(metrics_dir), "visualizations_next_level")
    os.makedirs(out_dir, exist_ok=True)

    captions = {}
    skipped = []

    plot_fns = [
        _plot_pareto_quality_latency,
        _plot_per_query_delta_waterfall,
        _plot_action_distribution,
        _plot_rejected_action_reasons,
        lambda md, od, c, s: _plot_calibration(md, od, c, s, "p_gain"),
        lambda md, od, c, s: _plot_calibration(md, od, c, s, "p_harm"),
        _plot_oracle_gap,
        _plot_graph_contribution,
        _plot_latency_breakdown,
        _plot_quality_retention_vs_latency_saving,
    ]

    for fn in plot_fns:
        try:
            fn(metrics_dir, out_dir, captions, skipped)
        except Exception as e:
            name = getattr(fn, "__name__", str(fn))
            logger.warning(f"Plot {name} failed: {e}")
            skipped.append({"plot_name": name, "reason": str(e), "missing_input_file": "unknown"})

    # Save captions
    with open(os.path.join(out_dir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=4)

    # Save skipped
    with open(os.path.join(out_dir, "skipped_plots.json"), "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=4)

    generated = list(captions.keys())
    print(f"[Visualization] Generated {len(generated)} plots, skipped {len(skipped)}")
    for name in generated:
        print(f"  [OK] {name}")
    for s in skipped:
        print(f"  [SKIP] {s['plot_name']}: {s['reason']}")

    return {"generated": generated, "skipped": skipped, "captions": captions}
