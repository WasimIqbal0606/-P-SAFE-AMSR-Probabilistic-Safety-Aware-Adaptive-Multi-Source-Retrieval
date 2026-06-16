"""
WARNING (FIX 10): Non-canonical helper. Do NOT use for final paper figures.
Canonical visualizer: psafe/visualization/generate_next_level_visuals.py

This module hardcodes Dense latency as 0.5ms and is incomplete.
Retained for backward compatibility with early multi-dataset summary plots only.
"""
import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def _check_and_load(file_path):
    if not os.path.exists(file_path):
        print(f"Skipped plot: missing real input data ({file_path})")
        return None
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def plot_multi_dataset_pareto(results_dir, datasets):
    """
    1. Multi-dataset Pareto frontier
    x = latency ms log scale
    y = nDCG@10
    color = dataset
    marker = method
    """
    out_dir = os.path.join(results_dir, "visualizations_next_level", "paper_figures_png")
    os.makedirs(out_dir, exist_ok=True)
    
    summary_path = os.path.join(results_dir, "multi_dataset_summary", "multi_dataset_summary.csv")
    df = _check_and_load(summary_path)
    if df is None: return
    
    plt.figure(figsize=(10, 6))
    
    # We want rows per dataset per method
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({"Dataset": row["dataset"], "Method": "Dense", "Latency (ms)": 0.5, "nDCG@10": row["dense_ndcg"]})
        plot_data.append({"Dataset": row["dataset"], "Method": "Always-on Hybrid", "Latency (ms)": row["hybrid_latency"], "nDCG@10": row["always_on_hybrid_ndcg"]})
        plot_data.append({"Dataset": row["dataset"], "Method": "B-P-SAFE", "Latency (ms)": max(row["psafe_latency"], 1.0), "nDCG@10": row["psafe_ndcg"]})
    
    df_pareto = pd.DataFrame(plot_data)
    
    sns.lineplot(data=df_pareto, x="Latency (ms)", y="nDCG@10", hue="Dataset", alpha=0.3, sort=False, legend=False)
    sns.scatterplot(data=df_pareto, x="Latency (ms)", y="nDCG@10", hue="Dataset", style="Method", s=150)
    
    plt.xscale("log")
    plt.title("Multi-Dataset Pareto: Quality vs Latency")
    
    plt.savefig(os.path.join(out_dir, "multi_dataset_pareto.png"), bbox_inches='tight')
    plt.close()

def plot_quality_vs_latency(results_dir):
    """
    2. Quality-retention vs latency-saving plot
    x = latency saving vs hybrid
    y = quality retention vs hybrid
    """
    out_dir = os.path.join(results_dir, "visualizations_next_level", "paper_figures_png")
    os.makedirs(out_dir, exist_ok=True)
    
    summary_path = os.path.join(results_dir, "multi_dataset_summary", "multi_dataset_summary.csv")
    df = _check_and_load(summary_path)
    if df is None or "quality_retention_vs_hybrid" not in df.columns: return
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="latency_saving", y="quality_retention_vs_hybrid", hue="dataset", s=200)
    plt.axhline(0.6, color='r', linestyle='--', alpha=0.5)
    plt.axvline(0.5, color='r', linestyle='--', alpha=0.5)
    plt.title("Quality-Retention vs Latency-Saving")
    plt.xlabel("Latency Saving vs Always-On Hybrid")
    plt.ylabel("Quality Retention vs Always-On Hybrid")
    plt.savefig(os.path.join(out_dir, "quality_vs_latency.png"), bbox_inches='tight')
    plt.close()

def plot_dataset_behavior_taxonomy(results_dir):
    """
    3. Dataset behavior taxonomy
    x = hybrid gain over Dense
    y = hybrid easy-query harm
    """
    out_dir = os.path.join(results_dir, "visualizations_next_level", "paper_figures_png")
    os.makedirs(out_dir, exist_ok=True)
    
    summary_path = os.path.join(results_dir, "multi_dataset_summary", "multi_dataset_summary.csv")
    df = _check_and_load(summary_path)
    if df is None: return
    
    plt.figure(figsize=(8, 6))
    df["hybrid_gain"] = df["always_on_hybrid_ndcg"] - df["dense_ndcg"]
    
    sns.scatterplot(data=df, x="hybrid_gain", y="easy_harm_hybrid", hue="dataset", s=200)
    plt.title("Dataset Behavior Taxonomy: Hybrid Gain vs Easy-Query Harm")
    plt.xlabel("Always-On Hybrid Gain over Dense (nDCG@10)")
    plt.ylabel("Always-On Hybrid Easy-Query Harm")
    plt.savefig(os.path.join(out_dir, "dataset_behavior_taxonomy.png"), bbox_inches='tight')
    plt.close()

def plot_per_query_waterfall(results_dir, dataset):
    """
    4. Per-query delta waterfall
    """
    out_dir = os.path.join(results_dir, dataset, "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    
    per_query_path = os.path.join(results_dir, dataset, "metrics", "per_query_metrics.csv")
    df = _check_and_load(per_query_path)
    if df is None: return
    
    if "delta_psafe_dense" not in df.columns: return
    
    deltas = df["delta_psafe_dense"].sort_values(ascending=False).values
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in deltas]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(deltas)), deltas, color=colors)
    plt.axhline(deltas.mean(), color='blue', linestyle='--', label=f"Mean: {deltas.mean():.4f}")
    plt.title(f"Per-Query Delta Waterfall (P-SAFE vs Dense) - {dataset}")
    plt.ylabel("Δ nDCG@10")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "per_query_waterfall.png"), bbox_inches='tight')
    plt.close()

# Other plots will be implemented similarly based on real data only.
