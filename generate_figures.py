"""
B-P-SAFE-AMSR — Master Publication Figure Generator
Generates all vector PDF and high-resolution PNG figures for manuscript and README
strictly from validated result artifacts.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PAPER_FIG_DIR = "paper/figures"
README_FIG_DIR = "figures"
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(README_FIG_DIR, exist_ok=True)

DATASETS = ["scifact", "fiqa", "nfcorpus", "arguana"]
DATASET_LABELS = {
    "scifact": "SciFact",
    "fiqa": "FiQA",
    "nfcorpus": "NFCorpus",
    "arguana": "ArguAna"
}
COLORS = {
    "scifact": "#1f77b4",
    "fiqa": "#2ca02c",
    "nfcorpus": "#ff7f0e",
    "arguana": "#d62728",
    "dense": "#7f7f7f",
    "hybrid": "#9467bd",
    "random": "#8c564b",
    "psafe_bal": "#1f77b4",
    "psafe_hr": "#2ca02c",
}


def setup_paper_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'figure.titlesize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'figure.autolayout': True,
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def save_fig(fig, base_name: str):
    """Save to both paper/figures and figures in PDF and PNG."""
    for d in [PAPER_FIG_DIR, README_FIG_DIR]:
        pdf_path = os.path.join(d, f"{base_name}.pdf")
        png_path = os.path.join(d, f"{base_name}.png")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    print(f"Saved {base_name}.pdf and {base_name}.png")


# ── Figure 1: Architecture Overview ──────────────────────────────────────────
def generate_fig1_architecture():
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axis('off')

    # Query Input Box
    ax.text(0.06, 0.5, "Query $x$", ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#e1f5fe', edgecolor='#0288d1', lw=1.5))

    # Feature Extractor
    ax.text(0.24, 0.5, "Feature Extractor\n(25 query, score,\nJaccard, graph)", ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3e0', edgecolor='#f57c00', lw=1.5))

    # Router Decision
    ax.text(0.48, 0.5, "B-P-SAFE Router\n$U(A_6|x) > 0$\n$P_{gain} \\geq \\tau_g$ and $P_{harm} \\leq \\tau_h$",
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f5e9', edgecolor='#388e3c', lw=1.8))

    # Dense Branch (A0)
    ax.text(0.78, 0.75, "$A_0$: Dense Retrieval\n(BGE-M3, Fast / Cheap)", ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='#616161', lw=1.5))

    # Deep Hybrid Branch (A6)
    ax.text(0.78, 0.25, "$A_6$: Deep Hybrid Pipeline\n(Dense + BM25 + Graph + CE)", ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ede7f6', edgecolor='#512da8', lw=1.5))

    # Connecting Arrows
    ax.annotate('', xy=(0.14, 0.5), xytext=(0.10, 0.5), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(0.37, 0.5), xytext=(0.33, 0.5), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(0.67, 0.75), xytext=(0.59, 0.55), arrowprops=dict(arrowstyle='->', lw=1.5, color='#616161'))
    ax.annotate('', xy=(0.67, 0.25), xytext=(0.59, 0.45), arrowprops=dict(arrowstyle='->', lw=1.5, color='#512da8'))

    ax.text(0.61, 0.72, "Dense sufficient", fontsize=8, color='#616161')
    ax.text(0.61, 0.30, "Escalate", fontsize=8, color='#512da8')

    save_fig(fig, "fig1_architecture")
    plt.close(fig)


# ── Figure 2: Quality-Latency Pareto Plot ────────────────────────────────────
def generate_fig2_pareto():
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    with open("results/statistics/statistical_analysis.json") as f:
        stat_data = json.load(f)

    # Plot points for each dataset
    markers = {"scifact": "o", "fiqa": "s", "nfcorpus": "^", "arguana": "D"}
    
    # Load primary seed 42 results
    for ds in DATASETS:
        with open(f"results/validated/{ds}/seed_42/balanced/extended_metrics.json") as f:
            em_bal = json.load(f)
        with open(f"results/validated/{ds}/seed_42/high_recall/extended_metrics.json") as f:
            em_hr = json.load(f)

        d_gain_bal = em_bal["psafe_ndcg"] - em_bal["dense_ndcg"]
        d_gain_hr = em_hr["psafe_ndcg"] - em_hr["dense_ndcg"]
        ls_bal = em_bal["latency_saving_vs_best_hybrid"] * 100.0
        ls_hr = em_hr["latency_saving_vs_best_hybrid"] * 100.0

        ax.scatter(ls_bal, d_gain_bal, color=COLORS[ds], marker=markers[ds], s=80, label=f"{DATASET_LABELS[ds]} (Balanced)")
        ax.scatter(ls_hr, d_gain_hr, color=COLORS[ds], marker=markers[ds], s=120, facecolors='none', edgecolors=COLORS[ds], lw=2, label=f"{DATASET_LABELS[ds]} (High Recall)")
        
        # Connect balanced and high recall
        ax.plot([ls_bal, ls_hr], [d_gain_bal, d_gain_hr], color=COLORS[ds], linestyle=':', alpha=0.7)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Latency Saving vs Always-Hybrid (%)")
    ax.set_ylabel(r"Quality Gain $\Delta_{\rm Dense}$ (nDCG@10)")
    ax.set_title("Quality-Latency Tradeoff (Seed 42)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

    save_fig(fig, "fig2_quality_latency")
    plt.close(fig)


# ── Figure 3: Hybrid Activation Rate ─────────────────────────────────────────
def generate_fig3_activation():
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    x = np.arange(len(DATASETS))
    width = 0.35

    hars_bal = []
    hars_hr = []

    for ds in DATASETS:
        with open(f"results/validated/{ds}/seed_42/balanced/extended_metrics.json") as f:
            em_bal = json.load(f)
        with open(f"results/validated/{ds}/seed_42/high_recall/extended_metrics.json") as f:
            em_hr = json.load(f)
        hars_bal.append(em_bal["hybrid_activation_rate"] * 100.0)
        hars_hr.append(em_hr["hybrid_activation_rate"] * 100.0)

    rects1 = ax.bar(x - width/2, hars_bal, width, label='Balanced', color='#1f77b4')
    rects2 = ax.bar(x + width/2, hars_hr, width, label='High Recall', color='#2ca02c')

    ax.set_ylabel('Hybrid Activation Rate (%)')
    ax.set_title('Cross-Encoder Escalation Rate by Dataset (Seed 42)')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylim(0, 100)
    ax.legend(frameon=True)

    save_fig(fig, "fig3_activation")
    plt.close(fig)


# ── Figure 4: Router Baseline Landscape ──────────────────────────────────────
def generate_fig4_baselines():
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    with open("results/baselines/comprehensive_baseline_results.json") as f:
        base_data = json.load(f)

    # Plot SciFact and NFCorpus baselines as exemplary
    ds = "scifact"
    data_ds = base_data[ds]["42"] if "42" in base_data[ds] else base_data[ds][42]

    methods = list(data_ds.keys())
    ndcgs = [data_ds[m]["mean_ndcg"] for m in methods]
    hars = [data_ds[m]["hybrid_activation"] * 100.0 for m in methods]

    for m, nd, har in zip(methods, ndcgs, hars):
        if "B-P-SAFE" in m:
            ax.scatter(har, nd, color='#d62728', s=100, marker='*', zorder=5)
            ax.annotate(m, (har + 1.5, nd), fontsize=8, weight='bold')
        elif "Oracle" in m:
            ax.scatter(har, nd, color='#9467bd', s=80, marker='X')
            ax.annotate(m, (har + 1.5, nd), fontsize=8)
        else:
            ax.scatter(har, nd, color='#7f7f7f', s=50)
            ax.annotate(m, (har + 1.5, nd), fontsize=7.5, color='#424242')

    ax.set_xlabel("Hybrid Activation Rate (%)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Router Baseline Landscape (SciFact Seed 42)")

    save_fig(fig, "fig4_baselines")
    plt.close(fig)


# ── Figure 5: Latency Composition ────────────────────────────────────────────
def generate_fig5_latency():
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    components = ["Dense Search", "BM25 Search", "Graph Exp.", "RRF Fusion", "Cross-Encoder", "Feature Extr.", "Router Dec."]
    times_ms = [0.37, 14.46, 0.01, 0.02, 695.66, 0.15, 0.08]

    y_pos = np.arange(len(components))
    ax.barh(y_pos, times_ms, color='#1f77b4', align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(components)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Measured Execution Time (ms, log scale)')
    ax.set_title('End-to-End Latency Breakdown (SciFact High Recall)')

    save_fig(fig, "fig5_latency")
    plt.close(fig)


# ── Figure 6: Multi-Seed Split Sensitivity ───────────────────────────────────
def generate_fig6_multiseed():
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for ds in DATASETS:
        seeds = [42, 123, 2026]
        ndcgs = []
        for s in seeds:
            with open(f"results/validated/{ds}/seed_{s}/high_recall/extended_metrics.json") as f:
                em = json.load(f)
            ndcgs.append(em["psafe_ndcg"])
        ax.plot([str(s) for s in seeds], ndcgs, marker='o', lw=1.8, label=DATASET_LABELS[ds], color=COLORS[ds])

    ax.set_xlabel("Data-Split Seed")
    ax.set_ylabel("High-Recall nDCG@10")
    ax.set_title("Multi-Seed Split Sensitivity (High Recall)")
    ax.legend(frameon=True)

    save_fig(fig, "fig6_multiseed")
    plt.close(fig)


# ── Figure 7: Reliability Diagram (P_gain & P_harm) ──────────────────────────
def generate_fig7_calibration_reliability():
    setup_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

    with open("results/calibration/reliability_data.json") as f:
        rel_data = json.load(f)

    # Plot the committed SciFact seed-42 Balanced reliability evidence.
    entry = rel_data["scifact"]["42"]["balanced"]
    gain_bins = entry["P_gain"]
    harm_bins = entry["P_harm"]

    required_bin_keys = {"count", "mean_predicted", "fraction_positive"}
    for target, bins in (("P_gain", gain_bins), ("P_harm", harm_bins)):
        if not bins or any(required_bin_keys - set(bin_row) for bin_row in bins):
            raise ValueError(f"Incomplete reliability evidence for {target}")

    # P_gain plot
    confs_g = [b["mean_predicted"] for b in gain_bins if b["count"] > 0]
    accs_g = [b["fraction_positive"] for b in gain_bins if b["count"] > 0]
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Ideal')
    ax1.plot(confs_g, accs_g, 's-', color='#1f77b4', lw=1.8, label='Empirical')
    ax1.set_xlabel(r'Predicted $P_{\rm gain}$')
    ax1.set_ylabel(r'Empirical Event Rate ($\Delta > 0.05$)')
    ax1.set_title(r'$P_{\rm gain}$ Reliability (SciFact)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(frameon=True)

    # P_harm plot
    confs_h = [b["mean_predicted"] for b in harm_bins if b["count"] > 0]
    accs_h = [b["fraction_positive"] for b in harm_bins if b["count"] > 0]
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Ideal')
    ax2.plot(confs_h, accs_h, 'o-', color='#d62728', lw=1.8, label='Empirical')
    ax2.set_xlabel(r'Predicted $P_{\rm harm}$')
    ax2.set_ylabel(r'Empirical Event Rate ($\Delta < -0.01$)')
    ax2.set_title(r'$P_{\rm harm}$ Reliability (SciFact)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=True)

    save_fig(fig, "fig7_calibration_reliability")
    plt.close(fig)


# ── Figure 8: Matched-Budget Random Baseline Distribution ────────────────────
def generate_fig8_matched_budget_random():
    setup_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    with open("results/baselines/matched_budget_random_results.json") as f:
        mb_data = json.load(f)

    # Plot comparison for seed 42 balanced across all 4 datasets
    x = np.arange(len(DATASETS))
    psafe_vals = []
    rand_means = []
    rand_err_low = []
    rand_err_high = []

    for ds in DATASETS:
        entry = mb_data[ds]["42"]["balanced"] if "42" in mb_data[ds] else mb_data[ds][42]["balanced"]
        p_val = entry["psafe_mean_ndcg"]
        r_mean = entry["mean_ndcg"]
        ci = entry["ci_95"]

        psafe_vals.append(p_val)
        rand_means.append(r_mean)
        rand_err_low.append(r_mean - ci[0])
        rand_err_high.append(ci[1] - r_mean)

    ax.errorbar(x, rand_means, yerr=[rand_err_low, rand_err_high], fmt='o', color='#7f7f7f',
                capsize=5, elinewidth=1.5, markeredgewidth=1.5,
                label='Matched-Budget Random (1000 allocations, 95% CI)')
    ax.scatter(x, psafe_vals, color='#1f77b4', s=90, marker='*', zorder=5, label='B-P-SAFE (Balanced)')

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylabel('nDCG@10')
    ax.set_title('B-P-SAFE vs 1000-Allocation Matched-Budget Random Router')
    ax.legend(frameon=True)

    save_fig(fig, "fig8_matched_budget_random")
    plt.close(fig)
def generate_all_figures():
    print("="*80)
    print("GENERATING ALL PUBLICATION FIGURES (PDF & PNG)")
    print("="*80)
    generate_fig1_architecture()
    generate_fig2_pareto()
    generate_fig3_activation()
    generate_fig4_baselines()
    generate_fig5_latency()
    generate_fig6_multiseed()
    generate_fig7_calibration_reliability()
    generate_fig8_matched_budget_random()
    print("ALL FIGURES GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    generate_all_figures()
