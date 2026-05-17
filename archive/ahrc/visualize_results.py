"""
Safe-AMSR-SE v3 — Publication-Grade Visualization
12 research-quality plots for the adaptive retrieval paper.
White background versions for publication. Saves PNG + PDF.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Any, List, Optional

# Publication white theme
THEME = {
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'axes.edgecolor': '#333333', 'grid.color': '#cccccc', 'grid.alpha': 0.5,
    'text.color': '#333333', 'axes.labelcolor': '#333333',
    'xtick.color': '#555555', 'ytick.color': '#555555',
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 14, 'axes.labelsize': 12,
}
plt.rcParams.update(THEME)

PALETTE = {
    'Dense': '#2196F3', 'BM25': '#9E9E9E', 'Full AMSR-SE': '#F44336',
    'Oracle': '#4CAF50', 'Random': '#795548', 'Learned': '#FF9800',
    'Rule': '#9C27B0', 'Cost': '#00BCD4', 'Safe': '#FF9800',
}

def _color(name):
    for k, c in PALETTE.items():
        if k.lower() in name.lower(): return c
    return '#607D8B'

def _save(fig, path):
    for ext in ['.png', '.pdf']:
        p = path.replace('.png', ext)
        fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   📈 {path}")


# ═══════════════════════════════════════════════════════════════
# Plot 1: Pareto Quality vs Latency
# ═══════════════════════════════════════════════════════════════
def plot_pareto_frontier(results: Dict[str, Any], output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 7))
    points = []
    for method, m in results.items():
        if method.startswith('_'): continue
        lat = m.get('latency_mean_ms', 0)
        ndcg = m.get('ndcg_at_k', {}).get('10', m.get('ndcg_at_k', {}).get(10, 0))
        c = _color(method)
        ax.scatter(lat, ndcg, c=c, s=180, zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(method, (lat, ndcg), xytext=(8, -4), textcoords='offset points',
                    fontsize=8, color=c)
        points.append((lat, ndcg))
    # Pareto frontier
    pts = sorted(points, key=lambda x: x[0])
    pareto = [pts[0]]
    for p in pts[1:]:
        if p[1] >= pareto[-1][1]: pareto.append(p)
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.plot(px, py, '--', color='#4CAF50', alpha=0.6, linewidth=2, label='Pareto Frontier')
    ax.set_xlabel("Latency (ms)"); ax.set_ylabel("nDCG@10")
    ax.set_title("Pareto Frontier: Quality vs Latency", pad=15)
    ax.set_xscale('symlog', linthresh=1); ax.legend(); ax.grid(True, ls='--')
    _save(fig, os.path.join(output_dir, "pareto_quality_latency.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 2: Hard Query Recovery
# ═══════════════════════════════════════════════════════════════
def plot_hard_query_recovery(dense_ndcg, system_ndcg, safe_ndcg,
                              easy_mask, output_dir, safe_name="Safe-AMSR-SE"):
    fig, ax = plt.subplots(figsize=(9, 6))
    hard_mask = ~easy_mask
    groups = ['Easy', 'Hard', 'All']
    dense_m = [np.mean(dense_ndcg[easy_mask]), np.mean(dense_ndcg[hard_mask]), np.mean(dense_ndcg)]
    full_m = [np.mean(system_ndcg[easy_mask]), np.mean(system_ndcg[hard_mask]), np.mean(system_ndcg)]
    safe_m = [np.mean(safe_ndcg[easy_mask]), np.mean(safe_ndcg[hard_mask]), np.mean(safe_ndcg)]
    x = np.arange(3); w = 0.25
    ax.bar(x - w, dense_m, w, label='Dense', color='#2196F3', alpha=0.85)
    ax.bar(x, full_m, w, label='Full AMSR-SE', color='#F44336', alpha=0.85)
    ax.bar(x + w, safe_m, w, label=safe_name, color='#FF9800', alpha=0.85)
    for i in range(3):
        d = safe_m[i] - dense_m[i]
        ax.text(x[i]+w, safe_m[i]+0.005, f"{d:+.3f}", ha='center', fontsize=8,
                color='#4CAF50' if d >= 0 else '#F44336', fontweight='bold')
    ax.set_ylabel("nDCG@10"); ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_title("Hard Query Recovery", pad=15); ax.legend(); ax.grid(axis='y', ls='--')
    _save(fig, os.path.join(output_dir, "hard_query_recovery_safe_router.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 3: Safe Gain Plot
# ═══════════════════════════════════════════════════════════════
def plot_safe_gain(safety: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(safety.keys())
    hg = [safety[m]["hard_gain_mean"] for m in methods]
    ed = [safety[m]["easy_degradation_mean"] for m in methods]
    ax.barh(methods, hg, color='#4CAF50', alpha=0.8, label='Hard Gain')
    ax.barh(methods, [-e for e in ed], color='#F44336', alpha=0.8, label='Easy Degradation')
    ax.axvline(0, color='#333', lw=0.8)
    ax.set_xlabel("nDCG@10 Change"); ax.set_title("Safe Gain Analysis", pad=15)
    ax.legend(); _save(fig, os.path.join(output_dir, "safe_gain_plot.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 4: Candidate Recall Depth Curve
# ═══════════════════════════════════════════════════════════════
def plot_candidate_recall_curve(pool_metrics: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    for method, pm in pool_metrics.items():
        cr = pm.get("candidate_recall", {})
        if cr:
            depths = sorted(int(k) for k in cr.keys())
            vals = [cr[d] if isinstance(cr[d], float) else cr[str(d)] for d in depths]
            ax.plot(depths, vals, 'o-', label=method, color=_color(method), lw=2)
    ax.set_xlabel("Depth"); ax.set_ylabel("Candidate Recall")
    ax.set_title("Candidate Recall at Depth", pad=15); ax.legend(); ax.grid(True, ls='--')
    _save(fig, os.path.join(output_dir, "candidate_recall_depth_curve.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 5: Source Attribution (Relevant Top-10)
# ═══════════════════════════════════════════════════════════════
def plot_source_attribution_relevant(attr_data: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ['Dense Only', 'BM25 Only', 'Graph Only',
                   'Dense+BM25', 'Dense+Graph', 'BM25+Graph', 'All Sources']
    keys = ['pct_dense_only', 'pct_bm25_only', 'pct_graph_only',
            'pct_dense_bm25', 'pct_dense_graph', 'pct_bm25_graph', 'pct_all_sources']
    values = [attr_data.get(k, 0) * 100 for k in keys]
    colors = ['#2196F3', '#9C27B0', '#4CAF50', '#FF9800', '#00BCD4', '#E91E63', '#795548']
    bars = ax.barh(categories, values, color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        if v > 0: ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                           f"{v:.1f}%", va='center', fontsize=9)
    ax.set_xlabel("% of Relevant Documents"); ax.set_title("Source Attribution (Relevant Top-10)", pad=15)
    _save(fig, os.path.join(output_dir, "source_attribution_relevant_top10.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 6: True Jaccard Overlap Heatmap
# ═══════════════════════════════════════════════════════════════
def plot_jaccard_heatmap(pool_metrics: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(pool_metrics.keys())
    depths = [10, 50, 100]
    data = np.zeros((len(methods), len(depths)))
    for i, m in enumerate(methods):
        jac = pool_metrics[m].get("true_jaccard_overlap", {})
        for j, d in enumerate(depths):
            data[i, j] = jac.get(d, jac.get(str(d), 0))
    im = ax.imshow(data, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(depths))); ax.set_xticklabels([f"@{d}" for d in depths])
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(depths)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha='center', va='center',
                    fontsize=11, fontweight='bold', color='black' if data[i,j] > 0.5 else 'white')
    ax.set_title("True Jaccard Overlap (Dense@k ∩ Hybrid@k / Dense@k ∪ Hybrid@k)", pad=15)
    fig.colorbar(im, label="Jaccard Index")
    _save(fig, os.path.join(output_dir, "true_jaccard_overlap_heatmap.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 7: Router Confusion Matrix
# ═══════════════════════════════════════════════════════════════
def plot_router_confusion(features, router, dense_ndcg, easy_mask, output_dir):
    if not hasattr(router, 'is_trained') or not router.is_trained: return
    fig, ax = plt.subplots(figsize=(6, 5))
    actual_hard = (~easy_mask).astype(int)
    predicted_hard = np.array([1 if router.route(f).action != 0 else 0 for f in features])
    router.reset()
    from sklearn.metrics import confusion_matrix
    from .safe_router import Action
    cm = confusion_matrix(actual_hard, predicted_hard, labels=[0, 1])
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=18,
                    fontweight='bold', color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred Easy', 'Pred Hard'])
    ax.set_yticklabels(['Actually Easy', 'Actually Hard'])
    ax.set_title("Router Confusion Matrix", pad=15); fig.colorbar(im)
    _save(fig, os.path.join(output_dir, "router_confusion_matrix.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 8: Router Calibration Curve
# ═══════════════════════════════════════════════════════════════
def plot_router_calibration(train_stats: Dict, output_dir: str):
    probs = train_stats.get("val_probs", [])
    labels = train_stats.get("val_labels", [])
    if not probs or not labels: return
    fig, ax = plt.subplots(figsize=(7, 6))
    probs, labels = np.array(probs), np.array(labels)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_true = [], []
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
        if mask.sum() > 0:
            bin_means.append(probs[mask].mean())
            bin_true.append(labels[mask].mean())
    ax.plot([0,1], [0,1], '--', color='gray', label='Perfectly calibrated')
    ax.plot(bin_means, bin_true, 'o-', color='#FF9800', lw=2, label='Router')
    ax.set_xlabel("Predicted Hard Probability"); ax.set_ylabel("True Hard Fraction")
    ax.set_title("Router Calibration Curve", pad=15); ax.legend(); ax.grid(True, ls='--')
    _save(fig, os.path.join(output_dir, "router_calibration_curve.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 9: Win/Tie/Loss
# ═══════════════════════════════════════════════════════════════
def plot_win_tie_loss(stats_report: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    w = stats_report.get("wins", 0); t = stats_report.get("ties", 0)
    lo = stats_report.get("losses", 0); total = w + t + lo
    ax.barh([""], [w], color='#4CAF50', label=f'Wins ({w})')
    ax.barh([""], [t], left=[w], color='#9E9E9E', label=f'Ties ({t})')
    ax.barh([""], [lo], left=[w+t], color='#F44336', label=f'Losses ({lo})')
    ax.set_xlim(0, total); ax.set_xlabel("Queries")
    ax.set_title(f"Win/Tie/Loss: {stats_report.get('comparison','')}", pad=15); ax.legend()
    _save(fig, os.path.join(output_dir, "win_tie_loss_safe_router.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 10: Per-Query Delta Waterfall
# ═══════════════════════════════════════════════════════════════
def plot_delta_waterfall(dense_ndcg, system_ndcg, output_dir, name="Safe-AMSR-SE"):
    deltas = system_ndcg - dense_ndcg
    si = np.argsort(deltas); sd = deltas[si]
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#F44336' if d < -1e-8 else '#4CAF50' if d > 1e-8 else '#9E9E9E' for d in sd]
    ax.bar(range(len(sd)), sd, color=colors, width=1.0)
    ax.axhline(0, color='#333', lw=0.8)
    ax.axhline(np.mean(deltas), color='#FF9800', lw=1.5, ls='--', label=f"Mean Δ={np.mean(deltas):+.4f}")
    ax.set_xlabel("Query (sorted)"); ax.set_ylabel(f"Δ nDCG@10 ({name} − Dense)")
    ax.set_title("Per-Query Improvement Waterfall", pad=15); ax.legend(); ax.grid(axis='y', ls='--')
    _save(fig, os.path.join(output_dir, "per_query_delta_waterfall_safe_router.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 11: Latency Breakdown Stacked Bar
# ═══════════════════════════════════════════════════════════════
def plot_latency_breakdown(latency_data: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    components = ['dense_search', 'bm25_search', 'graph_expansion', 'fusion', 'cross_encoder']
    colors = ['#2196F3', '#9C27B0', '#4CAF50', '#FF9800', '#F44336']
    methods = list(latency_data.keys())
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for comp, col in zip(components, colors):
        vals = []
        for m in methods:
            v = latency_data[m].get(comp, {}).get('mean_ms', 0) if isinstance(latency_data[m].get(comp), dict) else 0
            vals.append(v)
        ax.bar(x, vals, bottom=bottom, label=comp.replace('_',' ').title(), color=col, alpha=0.85)
        bottom += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha='right')
    ax.set_ylabel("Latency (ms)"); ax.set_title("Latency Breakdown", pad=15)
    ax.legend(loc='upper left'); ax.grid(axis='y', ls='--')
    _save(fig, os.path.join(output_dir, "latency_breakdown_stacked_bar.png"))


# ═══════════════════════════════════════════════════════════════
# Plot 12: Cross-Encoder Depth Sweep
# ═══════════════════════════════════════════════════════════════
def plot_ce_depth_sweep(sweep_results: Dict, output_dir: str):
    if not sweep_results: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    depths = sorted(sweep_results.keys())
    ndcg = [sweep_results[d]['ndcg_at_10'] for d in depths]
    lat = [sweep_results[d]['latency_mean_ms'] for d in depths]
    ax1.plot(depths, ndcg, 'o-', color='#2196F3', lw=2, label='nDCG@10')
    ax1.set_xlabel("Rerank Depth"); ax1.set_ylabel("nDCG@10")
    ax1.set_title("Quality vs Rerank Depth", pad=10); ax1.grid(True, ls='--'); ax1.legend()
    ax2.plot(depths, lat, 's-', color='#F44336', lw=2, label='Latency')
    ax2.set_xlabel("Rerank Depth"); ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency vs Rerank Depth", pad=10); ax2.grid(True, ls='--'); ax2.legend()
    fig.suptitle("Cross-Encoder Depth Sweep", fontsize=14, y=1.02)
    _save(fig, os.path.join(output_dir, "cross_encoder_depth_sweep.png"))


# ═══════════════════════════════════════════════════════════════
# Plot: Ablation Component Impact
# ═══════════════════════════════════════════════════════════════
def plot_ablation_impact(ablation_results: Dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    baseline_key = None
    for k in ablation_results:
        if 'Dense' in k and 'Graph' not in k and 'BM25' not in k:
            baseline_key = k; break
    if not baseline_key: return
    base_ndcg = ablation_results[baseline_key].get('ndcg_at_k', {}).get('10', 0)
    methods = [m for m in ablation_results if m != baseline_key and not m.startswith('_')]
    impacts = [ablation_results[m].get('ndcg_at_k', {}).get('10', 0) - base_ndcg for m in methods]
    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in impacts]
    ax.barh(methods, impacts, color=colors, alpha=0.85)
    ax.axvline(0, color='#333', lw=0.8)
    for i, v in enumerate(impacts):
        ax.text(v + 0.0005 * (1 if v >= 0 else -1), i, f"{v:+.4f}",
                va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel("Δ nDCG@10 vs Dense"); ax.set_title("Component Ablation Impact", pad=15)
    ax.grid(axis='x', ls='--')
    _save(fig, os.path.join(output_dir, "ablation_component_impact.png"))
