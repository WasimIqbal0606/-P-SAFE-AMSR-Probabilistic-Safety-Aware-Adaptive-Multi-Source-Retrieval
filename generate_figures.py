"""
P-SAFE-AMSR — Figure Generator for README and Paper
Generates all publication-quality figures from validated result files.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "validated")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = ["scifact", "fiqa", "nfcorpus", "arguana", "trec-covid"]
SEEDS = [42, 123, 2026]
MODES = ["lite", "balanced", "high_recall"]

DATASET_LABELS = {
    "scifact": "SciFact", "fiqa": "FiQA", "nfcorpus": "NFCorpus",
    "arguana": "ArguAna", "trec-covid": "TREC-COVID"
}
MODE_LABELS = {"lite": "Lite", "balanced": "Balanced", "high_recall": "High Recall"}

# ── Premium color palette ────────────────────────────────────────────
BG_COLOR = "#0d1117"
CARD_COLOR = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_ORANGE = "#d29922"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#bc8cff"
ACCENT_TEAL = "#39d2c0"

DATASET_COLORS = {
    "scifact": ACCENT_BLUE, "fiqa": ACCENT_GREEN, "nfcorpus": ACCENT_ORANGE,
    "arguana": ACCENT_RED, "trec-covid": ACCENT_PURPLE
}
MODE_COLORS = {"lite": ACCENT_ORANGE, "balanced": ACCENT_BLUE, "high_recall": ACCENT_GREEN}

TAXONOMY_COLORS = {
    "Selective escalation": ACCENT_GREEN,
    "Protection / No-benefit": ACCENT_RED,
    "Hybrid-beneficial / P-SAFE under-treatment": ACCENT_ORANGE,
    "Recovery / Selective-win": ACCENT_TEAL,
    "Hybrid-dominant / near-hybrid": ACCENT_PURPLE,
    "Quality-cost tradeoff": ACCENT_BLUE,
}

def setup_dark_style():
    plt.rcParams.update({
        'figure.facecolor': BG_COLOR, 'axes.facecolor': CARD_COLOR,
        'axes.edgecolor': GRID_COLOR, 'axes.labelcolor': TEXT_COLOR,
        'text.color': TEXT_COLOR, 'xtick.color': TEXT_COLOR,
        'ytick.color': TEXT_COLOR, 'grid.color': GRID_COLOR,
        'grid.alpha': 0.3, 'font.family': 'sans-serif',
        'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'legend.facecolor': CARD_COLOR, 'legend.edgecolor': GRID_COLOR,
        'legend.fontsize': 9,
    })

# ── Data loader ──────────────────────────────────────────────────────
def load_all_results():
    results = []
    for ds in DATASETS:
        for seed in SEEDS:
            for mode in MODES:
                path = os.path.join(RESULTS_DIR, ds, f"seed_{seed}", mode, "extended_metrics.json")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        data = json.load(f)
                    data["_dataset"] = ds
                    data["_seed"] = seed
                    data["_mode"] = mode
                    results.append(data)
    return results

# ── Figure 1: Quality-Latency Pareto Plot ────────────────────────────
def fig_pareto(results):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each dataset's seed-42 high_recall results as the main points
    for ds in DATASETS:
        ds_results = [r for r in results if r["_dataset"] == ds and r["_mode"] == "high_recall"]
        if not ds_results:
            continue

        # Dense baseline (same across seeds for same dataset, use mean)
        dense_ndcgs = [r["dense_ndcg"] for r in ds_results]
        hybrid_ndcgs = [r["best_hybrid_ndcg"] for r in ds_results]
        psafe_ndcgs = [r["psafe_ndcg"] for r in ds_results]
        latency_savings = [r["latency_saving_vs_best_hybrid"] for r in ds_results]

        mean_dense = np.mean(dense_ndcgs)
        mean_hybrid = np.mean(hybrid_ndcgs)
        mean_psafe = np.mean(psafe_ndcgs)
        mean_ls = np.mean(latency_savings)
        std_psafe = np.std(psafe_ndcgs)
        std_ls = np.std(latency_savings)

        color = DATASET_COLORS[ds]
        label = DATASET_LABELS[ds]

        # Dense: 0% latency saving (it IS the cheap baseline), but also 100% latency saving vs Hybrid
        # For Pareto: x = latency saving vs Hybrid, y = nDCG
        ax.scatter(1.0, mean_dense, marker='v', s=80, color=color, alpha=0.5, zorder=3)
        ax.scatter(0.0, mean_hybrid, marker='s', s=80, color=color, alpha=0.5, zorder=3)
        ax.errorbar(mean_ls, mean_psafe, xerr=std_ls, yerr=std_psafe,
                    fmt='o', markersize=10, color=color, capsize=4, capthick=1.5,
                    ecolor=color, alpha=0.9, zorder=5, label=label)

        # Connect the three points
        ax.plot([1.0, mean_ls, 0.0], [mean_dense, mean_psafe, mean_hybrid],
                '--', color=color, alpha=0.3, linewidth=1)

    ax.set_xlabel("Latency Saving vs Always-Hybrid", fontsize=13, fontweight='bold')
    ax.set_ylabel("nDCG@10", fontsize=13, fontweight='bold')
    ax.set_title("Quality-Latency Pareto Tradeoff (High Recall, Multi-Seed Mean +/- Std)",
                 fontsize=14, fontweight='bold', pad=15)

    # Add legend entries for marker types
    legend_extra = [
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=TEXT_COLOR, markersize=8, label='Dense-only', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=TEXT_COLOR, markersize=8, label='Always-Hybrid', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TEXT_COLOR, markersize=8, label='P-SAFE', linestyle='None'),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + legend_extra, loc='lower left', framealpha=0.9)

    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()  # More latency saving to the right

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "pareto_quality_latency.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] pareto_quality_latency.png")


# ── Figure 2: Hybrid Activation Rate by Dataset ─────────────────────
def fig_activation_rate(results):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(DATASETS))
    width = 0.25

    for i, mode in enumerate(MODES):
        means, stds = [], []
        for ds in DATASETS:
            vals = [r["hybrid_activation_rate"] for r in results
                    if r["_dataset"] == ds and r["_mode"] == mode]
            means.append(np.mean(vals) * 100 if vals else 0)
            stds.append(np.std(vals) * 100 if vals else 0)

        bars = ax.bar(x + i * width - width, means, width, yerr=stds,
                      label=MODE_LABELS[mode], color=MODE_COLORS[mode],
                      alpha=0.85, capsize=3, edgecolor='none', error_kw={'elinewidth': 1.2})

        for bar, val in zip(bars, means):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=11)
    ax.set_ylabel("Hybrid Activation Rate (%)", fontsize=13, fontweight='bold')
    ax.set_title("Hybrid Activation Rate by Dataset and Mode (Multi-Seed Mean +/- Std)",
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.2)
    ax.axhline(y=50, color=ACCENT_RED, linestyle=':', alpha=0.3, linewidth=1)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "hybrid_activation_rate.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] hybrid_activation_rate.png")


# ── Figure 3: Dataset Behavior Taxonomy ──────────────────────────────
def fig_taxonomy(results):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count taxonomy classifications across all seeds/modes
    taxonomy_counts = defaultdict(lambda: defaultdict(int))
    for r in results:
        taxonomy_counts[r["_dataset"]][r["taxonomy"]] += 1

    # Create stacked bar chart
    all_taxonomies = sorted(set(r["taxonomy"] for r in results))
    x = np.arange(len(DATASETS))
    bottom = np.zeros(len(DATASETS))

    for tax in all_taxonomies:
        vals = [taxonomy_counts[ds].get(tax, 0) for ds in DATASETS]
        color = TAXONOMY_COLORS.get(tax, "#8b949e")
        ax.bar(x, vals, 0.6, bottom=bottom, label=tax, color=color, alpha=0.85, edgecolor='none')
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=11)
    ax.set_ylabel("Count (seeds x modes)", fontsize=13, fontweight='bold')
    ax.set_title("Dataset Behavior Taxonomy Distribution",
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.grid(True, axis='y', alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "dataset_taxonomy.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] dataset_taxonomy.png")


# ── Figure 4: Latency Saving vs Quality Retention ────────────────────
def fig_latency_vs_quality(results):
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    for ds in DATASETS:
        for mode in MODES:
            ds_results = [r for r in results if r["_dataset"] == ds and r["_mode"] == mode]
            if not ds_results:
                continue

            ls_vals = [r["latency_saving_vs_best_hybrid"] * 100 for r in ds_results]
            qr_vals = [r["quality_retention_vs_best_hybrid"] * 100 for r in ds_results]

            color = DATASET_COLORS[ds]
            marker = {'lite': 'v', 'balanced': 'D', 'high_recall': 'o'}[mode]
            alpha = {'lite': 0.4, 'balanced': 0.7, 'high_recall': 1.0}[mode]

            ax.scatter(np.mean(ls_vals), np.mean(qr_vals),
                       marker=marker, s=120, color=color, alpha=alpha, zorder=5,
                       edgecolors='white', linewidth=0.5)

    # Add quadrant labels
    ax.axhline(y=50, color=GRID_COLOR, linestyle='--', alpha=0.5)
    ax.axvline(x=30, color=GRID_COLOR, linestyle='--', alpha=0.5)
    ax.text(65, 85, "Ideal: High Quality\n+ High Savings", fontsize=9, color=ACCENT_GREEN, alpha=0.7, ha='center')
    ax.text(10, 85, "Quality-focused\n(Low Savings)", fontsize=9, color=ACCENT_BLUE, alpha=0.7, ha='center')
    ax.text(65, 15, "Cost-focused\n(Low Quality)", fontsize=9, color=ACCENT_ORANGE, alpha=0.7, ha='center')

    # Build combined legend
    ds_patches = [mpatches.Patch(color=DATASET_COLORS[ds], label=DATASET_LABELS[ds]) for ds in DATASETS]
    mode_markers = [
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor=TEXT_COLOR, markersize=8, label='Lite', linestyle='None'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=TEXT_COLOR, markersize=8, label='Balanced', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TEXT_COLOR, markersize=8, label='High Recall', linestyle='None'),
    ]
    ax.legend(handles=ds_patches + mode_markers, loc='lower left', framealpha=0.9, ncol=2, fontsize=9)

    ax.set_xlabel("Latency Saving vs Always-Hybrid (%)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Quality Retention vs Hybrid (%)", fontsize=13, fontweight='bold')
    ax.set_title("Latency Saving vs Quality Retention (Multi-Seed Mean per Config)",
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "latency_vs_quality_retention.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] latency_vs_quality_retention.png")


# ── Figure 5: Architecture Diagram ──────────────────────────────────
def fig_architecture():
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    box_style = dict(boxstyle="round,pad=0.5", facecolor=CARD_COLOR, edgecolor=ACCENT_BLUE, linewidth=1.5)
    box_green = dict(boxstyle="round,pad=0.5", facecolor=CARD_COLOR, edgecolor=ACCENT_GREEN, linewidth=1.5)
    box_orange = dict(boxstyle="round,pad=0.5", facecolor=CARD_COLOR, edgecolor=ACCENT_ORANGE, linewidth=1.5)
    box_red = dict(boxstyle="round,pad=0.5", facecolor=CARD_COLOR, edgecolor=ACCENT_RED, linewidth=1.5)
    box_purple = dict(boxstyle="round,pad=0.5", facecolor=CARD_COLOR, edgecolor=ACCENT_PURPLE, linewidth=2)
    box_teal = dict(boxstyle="round,pad=0.6", facecolor="#1a2332", edgecolor=ACCENT_TEAL, linewidth=2)

    arrow_kw = dict(arrowstyle='->', color=TEXT_COLOR, lw=1.5, connectionstyle="arc3,rad=0.0")
    arrow_kw_curved = dict(arrowstyle='->', color=TEXT_COLOR, lw=1.2, connectionstyle="arc3,rad=0.2")

    # Title
    ax.text(7, 8.5, "P-SAFE-AMSR Architecture", fontsize=18, fontweight='bold',
            ha='center', va='center', color=ACCENT_BLUE)

    # Query input
    ax.text(1.5, 7, "Query", fontsize=13, fontweight='bold', ha='center', va='center', bbox=box_style)

    # Feature Extraction
    ax.text(4.5, 7, "Feature Extractor\n(25 signals)", fontsize=10, ha='center', va='center', bbox=box_orange)
    ax.annotate("", xy=(3.3, 7), xytext=(2.3, 7), arrowprops=arrow_kw)

    # Dense retrieval (always runs)
    ax.text(1.5, 5, "Dense Retrieval\n(FAISS)", fontsize=10, ha='center', va='center', bbox=box_green)
    ax.annotate("", xy=(1.5, 6.3), xytext=(1.5, 5.7), arrowprops=arrow_kw)

    # BM25 retrieval
    ax.text(4.5, 5, "BM25 Retrieval", fontsize=10, ha='center', va='center', bbox=box_green)
    ax.annotate("", xy=(4.5, 6.3), xytext=(4.5, 5.7), arrowprops=arrow_kw)

    # Graph features
    ax.text(1.5, 3.3, "Graph\nExpander", fontsize=9, ha='center', va='center', bbox=box_green)
    ax.annotate("", xy=(1.5, 4.3), xytext=(1.5, 3.9), arrowprops=arrow_kw)

    # Router (central)
    ax.text(7.5, 5, "P-SAFE Router\n\nP(Gain) / P(Harm)\nDelta nDCG / Latency\nUtility Function",
            fontsize=11, ha='center', va='center', bbox=box_purple)
    ax.annotate("", xy=(6.0, 5), xytext=(5.5, 5), arrowprops=arrow_kw)
    ax.annotate("", xy=(6.0, 5.5), xytext=(5.4, 7), arrowprops=arrow_kw_curved)

    # Decision outputs
    ax.text(11, 6.8, "A0: Dense\n(Fast, Cheap)", fontsize=11, fontweight='bold',
            ha='center', va='center', bbox=box_green)
    ax.text(11, 3.2, "A6: Deep Hybrid\n(Dense + BM25 +\nGraph + CrossEncoder)",
            fontsize=10, fontweight='bold', ha='center', va='center', bbox=box_red)

    ax.annotate("", xy=(9.5, 6.8), xytext=(9.0, 5.5), arrowprops=dict(
        arrowstyle='->', color=ACCENT_GREEN, lw=2, connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(9.5, 3.5), xytext=(9.0, 4.5), arrowprops=dict(
        arrowstyle='->', color=ACCENT_RED, lw=2, connectionstyle="arc3,rad=0.2"))

    # Decision labels
    ax.text(9.8, 6.0, "Safe", fontsize=10, fontweight='bold', color=ACCENT_GREEN, ha='center')
    ax.text(9.8, 4.0, "Escalate", fontsize=10, fontweight='bold', color=ACCENT_RED, ha='center')

    # Output
    ax.text(13, 5, "Final\nRanking", fontsize=12, fontweight='bold', ha='center', va='center', bbox=box_teal)
    ax.annotate("", xy=(12.2, 5.5), xytext=(11.8, 6.5), arrowprops=arrow_kw_curved)
    ax.annotate("", xy=(12.2, 4.5), xytext=(11.8, 3.5), arrowprops=arrow_kw_curved)

    # Utility formula box at bottom
    formula_box = dict(boxstyle="round,pad=0.6", facecolor="#0d1117", edgecolor=GRID_COLOR, linewidth=1)
    ax.text(7, 1.2, "U(A6|x) = D_pred - L_lat*Latency - L_harm*P(Harm) - L_cand*CandCount + L_rec*P(Gain)",
            fontsize=10, ha='center', va='center', bbox=formula_box, family='monospace', color=ACCENT_TEAL)
    ax.text(7, 0.5, "Escalate to A6 only when U(A6|x) > 0 and safety constraints pass",
            fontsize=9, ha='center', va='center', color='#8b949e', style='italic')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "architecture_diagram.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] architecture_diagram.png")


# ── Figure 6: Multi-Seed Stability ──────────────────────────────────
def fig_multi_seed_stability(results):
    setup_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: nDCG across seeds
    ax = axes[0]
    x = np.arange(len(DATASETS))
    width = 0.22
    for i, seed in enumerate(SEEDS):
        vals = []
        for ds in DATASETS:
            hr = [r["psafe_ndcg"] for r in results
                  if r["_dataset"] == ds and r["_seed"] == seed and r["_mode"] == "high_recall"]
            vals.append(hr[0] if hr else 0)
        colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE]
        ax.bar(x + i * width - width, vals, width, label=f"Seed {seed}",
               color=colors[i], alpha=0.85, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=10)
    ax.set_ylabel("P-SAFE nDCG@10", fontsize=12, fontweight='bold')
    ax.set_title("nDCG Stability Across Seeds (High Recall)", fontsize=13, fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2)

    # Panel B: Hybrid activation rate across seeds
    ax = axes[1]
    for i, seed in enumerate(SEEDS):
        vals = []
        for ds in DATASETS:
            hr = [r["hybrid_activation_rate"] * 100 for r in results
                  if r["_dataset"] == ds and r["_seed"] == seed and r["_mode"] == "high_recall"]
            vals.append(hr[0] if hr else 0)
        colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE]
        ax.bar(x + i * width - width, vals, width, label=f"Seed {seed}",
               color=colors[i], alpha=0.85, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=10)
    ax.set_ylabel("Hybrid Activation Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Activation Stability Across Seeds (High Recall)", fontsize=13, fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "multi_seed_stability.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] multi_seed_stability.png")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("P-SAFE-AMSR Figure Generator")
    print("=" * 50)
    results = load_all_results()
    print(f"Loaded {len(results)} result files\n")

    print("Generating figures...")
    fig_architecture()
    fig_pareto(results)
    fig_activation_rate(results)
    fig_taxonomy(results)
    fig_latency_vs_quality(results)
    fig_multi_seed_stability(results)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Done.")
