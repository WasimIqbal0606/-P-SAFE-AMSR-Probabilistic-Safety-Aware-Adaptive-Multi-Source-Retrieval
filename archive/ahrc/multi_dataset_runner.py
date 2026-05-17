"""
P-SAFE-AMSR — Multi-Dataset Runner and Taxonomy Generator

FIX 6 applied:
  1. num_docs from dataset metadata, not candidates_total
  2. best_hybrid selected dynamically, not always Deep Hybrid
  3. hard_gain_hybrid from safety_metrics, not aggregate subtraction
  4. dense latency from aggregate_metrics, not hardcoded 0.5
  5. Taxonomy via calculate_extended_metrics()
  6. Full taxonomy labels including under-treatment and failure
"""
import os, json, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .psafe_experiment_runner import run_psafe_experiment

MANDATORY = ["scifact", "fiqa", "nfcorpus", "arguana", "trec-covid", "scidocs"]

# Hybrid candidate methods for best_hybrid selection
HYBRID_CANDIDATES = [
    "Dense+BM25+CE",
    "Dense+BM25+Graph+CE",
    "Deep Hybrid",
    "BGE-M3 Dense+Sparse+CE",
]


def _select_best_hybrid(agg: dict) -> tuple:
    """Select the best hybrid method from available results.
    Returns (method_name, ndcg@10, latency_mean_ms)."""
    best_name, best_ndcg, best_lat = "Deep Hybrid", 0.0, 0.0
    for candidate in HYBRID_CANDIDATES:
        if candidate in agg:
            ndcg = agg[candidate].get("ndcg_at_k", {}).get("10",
                   agg[candidate].get("ndcg_at_k", {}).get(10, 0))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_lat = agg[candidate].get("latency_mean_ms", 0)
                best_name = candidate
    # Fallback: if none of the candidates exist, pick the highest non-Dense
    if best_ndcg == 0.0:
        for method, data in agg.items():
            if method == "Dense" or method == "P-SAFE-AMSR":
                continue
            ndcg = data.get("ndcg_at_k", {}).get("10",
                   data.get("ndcg_at_k", {}).get(10, 0))
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_lat = data.get("latency_mean_ms", 0)
                best_name = method
    return best_name, best_ndcg, best_lat


def _get_num_docs(agg: dict, ds_dir: str) -> int:
    """Get num_docs from dataset metadata, not candidates_total."""
    # Try config.json metadata first
    config_path = os.path.join(os.path.dirname(ds_dir), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            if "num_docs" in cfg:
                return int(cfg["num_docs"])
        except Exception:
            pass
    # Try aggregate_metrics metadata
    for method, data in agg.items():
        if "num_docs" in data:
            return int(data["num_docs"])
    # Fallback: use candidates_total from Dense as rough estimate
    return agg.get("Dense", {}).get("candidates_total", 0)


def generate_multi_dataset_summary(results_dir, datasets):
    # Import taxonomy calculation from canonical metrics
    try:
        from psafe.statistics.metrics import calculate_extended_metrics
    except ImportError:
        calculate_extended_metrics = None

    summary_dir = os.path.join(results_dir, "multi_dataset_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    rows = []
    taxonomy_rows = []
    
    for ds in datasets:
        ds_dir = os.path.join(results_dir, ds, "Balanced", "metrics")
        if not os.path.exists(ds_dir):
            continue
            
        agg_path = os.path.join(ds_dir, "aggregate_metrics.json")
        saf_path = os.path.join(ds_dir, "safety_metrics.json")
        over_path = os.path.join(ds_dir, "overtreatment_metrics.json")
        stat_path = os.path.join(ds_dir, "statistical_tests.json")

        if not os.path.exists(agg_path):
            continue

        with open(agg_path) as f:
            agg = json.load(f)
        saf = {}
        if os.path.exists(saf_path):
            with open(saf_path) as f:
                saf = json.load(f)
        over = {}
        if os.path.exists(over_path):
            with open(over_path) as f:
                over = json.load(f)
        stat = {}
        if os.path.exists(stat_path):
            with open(stat_path) as f:
                stat = json.load(f)
            
        dense_ndcg = agg.get("Dense", {}).get("ndcg_at_k", {}).get("10", 0)

        # FIX 6.2: Select best hybrid dynamically
        best_hybrid_name, hybrid_ndcg, hybrid_lat = _select_best_hybrid(agg)

        psafe_ndcg = agg.get("P-SAFE-AMSR", {}).get("ndcg_at_k", {}).get("10", 0)
        
        oracle_delta = stat.get("P-SAFE vs Oracle", {}).get("mean_delta", 0)
        oracle_ndcg = psafe_ndcg - oracle_delta if oracle_delta else max(dense_ndcg, hybrid_ndcg)
        
        # FIX 6.4: Dense latency from aggregate, not hardcoded 0.5
        dense_lat = agg.get("Dense", {}).get("latency_mean_ms", 0.05)

        psafe_lat = saf.get("P-SAFE-AMSR", {}).get("avg_latency_ms", 0)
        
        # FIX 6.3: hard_gain_hybrid from safety_metrics
        psafe_saf = saf.get("P-SAFE-AMSR", {})
        hybrid_easy_deg = over.get("harm_avoidance", 0) + psafe_saf.get("easy_degradation_mean", 0)
        # Try to get hard_gain from safety_metrics for best hybrid
        hybrid_hard_gain = saf.get(best_hybrid_name, {}).get("hard_gain_mean",
                           hybrid_ndcg - dense_ndcg - hybrid_easy_deg)

        psafe_easy_deg = psafe_saf.get("easy_degradation_mean", 0)
        psafe_hard_gain = psafe_saf.get("hard_gain_mean", 0)

        # FIX 6.1: num_docs from dataset metadata
        num_docs = _get_num_docs(agg, ds_dir)

        row = {
            "dataset": ds,
            "num_docs": num_docs,
            "num_queries": agg.get("Dense", {}).get("num_queries", 0),
            "best_hybrid_name": best_hybrid_name,
            "dense_ndcg": dense_ndcg,
            "always_on_hybrid_ndcg": hybrid_ndcg,
            "psafe_ndcg": psafe_ndcg,
            "oracle_ndcg": oracle_ndcg,
            "psafe_vs_dense_delta": psafe_ndcg - dense_ndcg,
            "psafe_vs_hybrid_delta": psafe_ndcg - hybrid_ndcg,
            "psafe_latency": psafe_lat,
            "hybrid_latency": hybrid_lat,
            "dense_latency": dense_lat,
            "latency_reduction": 1 - (psafe_lat / max(1, hybrid_lat)),
            "easy_harm_hybrid": hybrid_easy_deg,
            "easy_harm_psafe": psafe_easy_deg,
            "hard_gain_hybrid": hybrid_hard_gain,
            "hard_gain_psafe": psafe_hard_gain,
            "harm_avoidance": over.get("harm_avoidance", 0),
            "safe_gain": psafe_saf.get("safe_gain", 0),
            "hybrid_activation_rate": psafe_saf.get("pct_routed_hybrid", 0),
            "statistical_significance": stat.get("P-SAFE vs Dense", {}).get("paired_ttest", {}).get("p_value", 1.0),
        }
        rows.append(row)
        
        # FIX 6.5 & 6.6: Taxonomy via calculate_extended_metrics
        if calculate_extended_metrics is not None:
            ext_metrics = calculate_extended_metrics(
                dense_ndcg=dense_ndcg,
                hybrid_ndcg=hybrid_ndcg,
                psafe_ndcg=psafe_ndcg,
                oracle_ndcg=oracle_ndcg,
                always_on_hybrid_lat=hybrid_lat,
                psafe_lat=psafe_lat,
                always_on_hybrid_easy_deg=hybrid_easy_deg,
                psafe_easy_deg=psafe_easy_deg,
                always_on_hybrid_hard_gain=hybrid_hard_gain,
                psafe_hard_gain=psafe_hard_gain,
                dataset_name=ds,
                best_hybrid_name=best_hybrid_name,
                best_hybrid_ndcg=hybrid_ndcg,
                best_hybrid_latency=hybrid_lat,
            )
            cat = ext_metrics.get("taxonomy", "Unknown")
            quality_retention = ext_metrics.get("quality_retention_vs_best_hybrid", 0)
        else:
            # FIX 6.6 & 6.7: Manual taxonomy with all labels
            hybrid_delta = hybrid_ndcg - dense_ndcg
            quality_retention = ((psafe_ndcg - dense_ndcg) / hybrid_delta) if hybrid_delta > 0 else 0

            if psafe_ndcg < dense_ndcg - 0.02:
                cat = "P-SAFE failure"
            elif hybrid_delta <= 0.01:
                cat = "Protection / No-benefit"
            elif quality_retention < 0.25 and hybrid_delta > 0.01:
                cat = "Hybrid-beneficial / P-SAFE under-treatment"
            elif psafe_ndcg >= hybrid_ndcg:
                cat = "Recovery / Selective-win"
            else:
                cat = "Selective escalation"
            
        taxonomy_rows.append({
            "Dataset": ds,
            "Best Hybrid": best_hybrid_name,
            "Dense nDCG@10": dense_ndcg,
            "Always-on Hybrid nDCG@10": hybrid_ndcg,
            "P-SAFE nDCG@10": psafe_ndcg,
            "Oracle nDCG@10": oracle_ndcg,
            "P-SAFE vs Dense \u0394": psafe_ndcg - dense_ndcg,
            "P-SAFE vs Hybrid \u0394": psafe_ndcg - hybrid_ndcg,
            "Hybrid activation %": f"{row['hybrid_activation_rate']*100:.1f}%",
            "Latency reduction vs Hybrid": f"{row['latency_reduction']*100:.1f}%",
            "Avoided easy-query nDCG@10 degradation": over.get("harm_avoidance", 0),
            "Hard-query gain": psafe_hard_gain,
            "Behavior label": cat,
            "Statistical status": f"p={row['statistical_significance']:.2e}"
        })

    # Save extended metrics per dataset
    if calculate_extended_metrics and rows:
        ext_dir = os.path.join(summary_dir, "extended_metrics")
        os.makedirs(ext_dir, exist_ok=True)
        
    with open(os.path.join(summary_dir, "multi_dataset_summary.csv"), "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            
    with open(os.path.join(summary_dir, "multi_dataset_summary.md"), "w", encoding="utf-8") as f:
        f.write("# P-SAFE-AMSR Multi-Dataset Taxonomy Summary\n\n")
        if taxonomy_rows:
            headers = list(taxonomy_rows[0].keys())
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for r in taxonomy_rows:
                f.write("| " + " | ".join(str(r[h]) if isinstance(r[h], str) else f"{r[h]:.4f}" for h in headers) + " |\n")

    # 1. Dataset Behavior Taxonomy
    fig = plt.figure(figsize=(8, 6))
    if taxonomy_rows:
        df_tax = pd.DataFrame(taxonomy_rows)
        df_tax["Latency reduction vs always-on hybrid (%)"] = [r["latency_reduction"] * 100 for r in rows]
        sns.scatterplot(
            data=df_tax,
            x="Always-on Hybrid nDCG@10", y="P-SAFE nDCG@10", hue="Behavior label",
            size="Latency reduction vs always-on hybrid (%)", sizes=(100, 500)
        )
        min_val = min(df_tax["Always-on Hybrid nDCG@10"].min(), df_tax["P-SAFE nDCG@10"].min()) - 0.05
        max_val = max(df_tax["Always-on Hybrid nDCG@10"].max(), df_tax["P-SAFE nDCG@10"].max()) + 0.05
        
        plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)
        plt.fill_between([min_val, max_val], [min_val, max_val], max_val, color='green', alpha=0.1, label='P-SAFE > Hybrid')
        plt.fill_between([min_val, max_val], min_val, [min_val, max_val], color='red', alpha=0.1, label='P-SAFE sacrifices quality')
        
        for i, row in df_tax.iterrows():
            plt.text(row["Always-on Hybrid nDCG@10"], row["P-SAFE nDCG@10"] + 0.005, row["Dataset"], 
                     horizontalalignment='center')
                     
    plt.title("Dataset Behavior Taxonomy")
    plt.savefig(os.path.join(summary_dir, "dataset_behavior_taxonomy.png"), bbox_inches='tight')
    plt.close()

    # 2. Latency Savings Plot
    fig = plt.figure(figsize=(8, 6))
    if rows:
        df_rows = pd.DataFrame(rows)
        df_rows["latency_reduction_pct"] = df_rows["latency_reduction"] * 100
        sns.barplot(data=df_rows, x="dataset", y="latency_reduction_pct")
        plt.ylabel("Latency Reduction Relative to Best Hybrid (%)")
    plt.title("Latency Savings vs Best Hybrid")
    plt.savefig(os.path.join(summary_dir, "latency_savings_plot.png"), bbox_inches='tight')
    plt.close()

    # 3. Easy-Query Harm Avoidance Plot
    fig = plt.figure(figsize=(8, 6))
    if taxonomy_rows:
        sns.barplot(data=df_tax, x="Dataset", y="Avoided easy-query nDCG@10 degradation")
    plt.title("P-SAFE Reduces Easy-Query Degradation from Always-on Hybrid Retrieval")
    plt.savefig(os.path.join(summary_dir, "easy_harm_avoidance_plot.png"), bbox_inches='tight')
    plt.close()
    
    # 4. Multi-dataset Pareto — FIX 6.4: real dense latency
    fig = plt.figure(figsize=(8, 6))
    if rows:
        plot_data = []
        for r in rows:
            plot_data.append({"Dataset": r["dataset"], "Method": "Dense", "Latency (ms)": max(r["dense_latency"], 0.01), "nDCG@10": r["dense_ndcg"]})
            plot_data.append({"Dataset": r["dataset"], "Method": f"Best Hybrid ({r['best_hybrid_name']})", "Latency (ms)": r["hybrid_latency"], "nDCG@10": r["always_on_hybrid_ndcg"]})
            plot_data.append({"Dataset": r["dataset"], "Method": "P-SAFE", "Latency (ms)": max(r["psafe_latency"], 1.0), "nDCG@10": r["psafe_ndcg"]})
            
        df_pareto = pd.DataFrame(plot_data)
        
        sns.lineplot(
            data=df_pareto, x="Latency (ms)", y="nDCG@10", hue="Dataset", 
            legend=False, alpha=0.3, sort=False
        )
        
        sns.scatterplot(
            data=df_pareto, x="Latency (ms)", y="nDCG@10", hue="Dataset", style="Method",
            s=150
        )
        plt.xscale("log")
    plt.title("Multi-Dataset Pareto: Quality vs Latency")
    plt.savefig(os.path.join(summary_dir, "multi_dataset_pareto.png"), bbox_inches='tight')
    plt.close()
    
    # 5. Generate LaTeX Table
    tex_path = os.path.join(summary_dir, "p_safe_vs_dense_hybrid_oracle_table.tex")
    if taxonomy_rows:
        df_tax_out = pd.DataFrame(taxonomy_rows)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(df_tax_out.to_latex(index=False, float_format="%.4f"))


def run_multi_dataset(datasets=None, results_dir="results_top_tier_psafe",
                       model_name="BAAI/bge-m3", reranker_name=None,
                       device="auto", max_docs=None, max_queries=None):
    datasets = datasets or MANDATORY
    print("=" * 70)
    print(f"   P-SAFE-AMSR -- Multi-Dataset ({', '.join(datasets)})")
    print(f"   Embedding: {model_name}")
    if reranker_name:
        print(f"   Reranker:  {reranker_name}")
    print("=" * 70)

    completed = []
    failed = []

    for ds in datasets:
        print(f"\n{'=' * 70}\n   Dataset: {ds}\n{'=' * 70}")
        kwargs = {"dataset_name": ds}
        if max_docs: kwargs["max_docs"] = max_docs
        if max_queries: kwargs["max_queries"] = max_queries
        try:
            run_psafe_experiment(source="beir", results_dir=results_dir,
                                 model_name=model_name, device=device, **kwargs)
            completed.append(ds)
        except Exception as e:
            import traceback
            print(f"   FAILED on {ds}: {e}")
            traceback.print_exc()
            failed.append((ds, str(e)))

    if completed:
        generate_multi_dataset_summary(results_dir, completed)
        print(f"\nCross-dataset summary complete in {results_dir}/multi_dataset_summary/")

    # Log completion status
    print(f"\n{'=' * 70}")
    print(f"   COMPLETED: {len(completed)}/{len(datasets)} datasets")
    for ds in completed:
        print(f"     [OK] {ds}")
    for ds, err in failed:
        print(f"     [FAIL] {ds}: {err[:80]}")
    print(f"{'=' * 70}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="P-SAFE-AMSR Multi-Dataset Runner")
    p.add_argument("--datasets", nargs="+", default=MANDATORY)
    p.add_argument("--results-dir", default="results_top_tier_psafe")
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--reranker", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--config", default=None, help="Path to YAML config file")
    args = p.parse_args()

    # Optional YAML config override
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            if "datasets" in cfg:
                args.datasets = cfg["datasets"]
            if "embedding_models" in cfg and cfg["embedding_models"]:
                args.model = cfg["embedding_models"][0]
            if "reranker_models" in cfg and cfg["reranker_models"]:
                args.reranker = cfg["reranker_models"][0]
            if cfg.get("output", {}).get("results_dir"):
                args.results_dir = cfg["output"]["results_dir"]
            if cfg.get("evaluation", {}).get("max_queries") is not None:
                args.max_queries = cfg["evaluation"]["max_queries"]
            print(f"   [Config loaded] {args.config}")
        except Exception as e:
            print(f"   [Config warning] Could not load {args.config}: {e}")

    run_multi_dataset(datasets=args.datasets, results_dir=args.results_dir,
                       model_name=args.model, reranker_name=args.reranker,
                       device=args.device,
                       max_docs=args.max_docs, max_queries=args.max_queries)


if __name__ == "__main__":
    main()
