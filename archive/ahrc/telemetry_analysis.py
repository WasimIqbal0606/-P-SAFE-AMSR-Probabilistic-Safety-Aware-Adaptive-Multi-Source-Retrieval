"""
AHRC — Telemetry Analysis
Performs statistical significance tests and generates advanced publication plots.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set premium dark mode styling
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#111111',
    'figure.facecolor': '#111111',
    'axes.edgecolor': '#333333',
    'grid.color': '#333333',
    'text.color': '#EEEEEE',
    'axes.labelcolor': '#EEEEEE',
    'xtick.color': '#EEEEEE',
    'ytick.color': '#EEEEEE',
    'font.family': 'sans-serif',
})

def analyze_telemetry(results_dir="ahrc_results"):
    ablation_path = os.path.join(results_dir, "ablation_results.json")
    
    if not os.path.exists(ablation_path):
        print(f"File not found: {ablation_path}")
        return

    with open(ablation_path, "r") as f:
        ablation = json.load(f)

    if "_query_metrics" not in ablation:
        print("No query-level metrics found.")
        return

    dense_ndcg = np.array(ablation["_query_metrics"]["dense_ndcg"])
    full_ndcg = np.array(ablation["_query_metrics"]["full_ndcg"])

    # 1. Statistical Significance (Paired T-Test)
    t_stat, p_val = stats.ttest_rel(dense_ndcg, full_ndcg)
    print("\n" + "="*50)
    print("🔬 STATISTICAL SIGNIFICANCE TEST (Paired t-test)")
    print("="*50)
    print(f"Mean Dense nDCG@10: {np.mean(dense_ndcg):.4f}")
    print(f"Mean Full  nDCG@10: {np.mean(full_ndcg):.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_val:.4e}")
    if p_val < 0.05:
        print("✅ The difference is STATISTICALLY SIGNIFICANT (p < 0.05).")
    else:
        print("⚠️  The difference is NOT statistically significant.")

    # 2. Hard Query Breakout Plot (Delta nDCG per query)
    deltas = full_ndcg - dense_ndcg
    sort_idx = np.argsort(deltas)
    sorted_deltas = deltas[sort_idx]
    
    # Determine Easy vs Hard conceptually based on Dense score
    # Easy = Dense nDCG > 0.5, Hard = Dense nDCG <= 0.5
    hard_idx = dense_ndcg <= 0.5
    easy_idx = dense_ndcg > 0.5
    
    print("\n" + "="*50)
    print("🔥 EASY VS HARD QUERY RECOVERY")
    print("="*50)
    print(f"Hard Queries (Dense <= 0.5): {np.sum(hard_idx)}")
    print(f"  - Dense Mean: {np.mean(dense_ndcg[hard_idx]) if np.sum(hard_idx) > 0 else 0:.4f}")
    print(f"  - Full Mean:  {np.mean(full_ndcg[hard_idx]) if np.sum(hard_idx) > 0 else 0:.4f}")
    print(f"Easy Queries (Dense > 0.5):  {np.sum(easy_idx)}")
    print(f"  - Dense Mean: {np.mean(dense_ndcg[easy_idx]) if np.sum(easy_idx) > 0 else 0:.4f}")
    print(f"  - Full Mean:  {np.mean(full_ndcg[easy_idx]) if np.sum(easy_idx) > 0 else 0:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#ef4444' if d < 0 else '#10b981' for d in sorted_deltas]
    ax.bar(np.arange(len(sorted_deltas)), sorted_deltas, color=colors, width=1.0)
    ax.set_title("Per-Query Improvement (Δ nDCG@10)", fontsize=14, pad=15)
    ax.set_xlabel("Query Index (sorted by improvement)", fontsize=12)
    ax.set_ylabel("Δ nDCG@10 (AMSR-SE - Dense)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "query_improvements.png"), dpi=300)
    plt.close()
    
    # 3. Component Impact Ablation Chart
    methods = [m for m in ablation.keys() if m != "_metadata" and m != "_query_metrics" and m != "Dense Only"]
    base_ndcg = np.mean(dense_ndcg)
    impacts = [ablation[m]["ndcg_at_k"]["10"] - base_ndcg for m in methods]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(methods, impacts, color='#3b82f6')
    ax.set_title("Component Ablation (nDCG@10 Gain over Dense)", fontsize=14, pad=15)
    ax.set_xlabel("Δ nDCG@10", fontsize=12)
    for i, v in enumerate(impacts):
        ax.text(v + (0.0001 if v >= 0 else -0.0001), i, f"+{v:.4f}" if v >=0 else f"{v:.4f}", 
                va='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ablation_impact.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    analyze_telemetry()
