"""
AMSR-SE — Publication Experiment Runner
Full experiment suite with candidate pool evaluation, statistical testing,
learned routing comparison, and publication plot generation.

Usage:
    python -m ahrc.run_experiments [--tasks 5000] [--queries 200] [--seed 42]
"""

import os
import json
import time
import copy
import csv
import argparse
import numpy as np
from typing import Dict, List

from .config import AHRCConfig, IndexType
from .dataset_generator import DatasetGenerator, build_dataset
from .index_manager import IndexManager
from .graph_expander import GraphExpander
from .hybrid_retriever import HybridRetriever
from .baselines import BM25Baseline, DenseFixedBaseline, DenseGraphFixedBaseline, RandomBaseline
from .evaluation import Evaluator, AggregateMetrics, QueryMetrics
from .candidate_pool_eval import CandidatePoolEvaluator, CandidatePoolMetrics
from .candidate_fusion import CandidateFusion
from .reranker import CrossEncoderReranker
from .statistical_tests import StatisticalTester


def run_experiments(config: AHRCConfig = None, data_dir: str = "ahrc_data"):
    """Run full publication-grade experiment suite."""

    if config is None:
        config = AHRCConfig()

    results_dir = config.experiment.results_dir
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 80)
    print("   AMSR-SE — Adaptive Multi-Source Retrieval with Selective Expansion")
    print("   Publication Experiment Suite")
    print("=" * 80)
    print()

    # ── Step 1: Dataset ────────────────────────────────────────────────
    if os.path.exists(os.path.join(data_dir, "tasks.json")):
        print("📂 Loading existing dataset...")
        dataset = DatasetGenerator.load(data_dir, config)
    else:
        print("📦 Generating new dataset...")
        dataset = build_dataset(config, data_dir)

    tasks = dataset.tasks
    queries = dataset.queries
    task_ids = [t.id for t in tasks]
    id_to_index = {t.id: i for i, t in enumerate(tasks)}
    all_embeddings = np.array([t.embedding for t in tasks], dtype=np.float32)
    task_texts = [t.description for t in tasks]

    print(f"\n📊 Dataset: {len(tasks):,} tasks, {len(queries)} queries\n")

    # ── Step 2: Build index ────────────────────────────────────────────
    index_mgr = IndexManager(config.index)
    index_mgr.build(all_embeddings)

    # ── Step 3: Build graph ────────────────────────────────────────────
    graph_exp = GraphExpander(config.adaptive)
    graph_exp.build_from_tasks(tasks, id_to_index)

    # ── Step 4: Initialize components ──────────────────────────────────
    evaluator = Evaluator(k_values=config.experiment.eval_k_values, relevance_threshold=1)
    pool_evaluator = CandidatePoolEvaluator(eval_depths=[10, 50, 100])
    stat_tester = StatisticalTester(n_bootstrap=10000, n_permutation=5000)
    eval_k = 10

    bm25_bl = BM25Baseline(tasks)

    print("🧠 Loading Cross-Encoder Reranker...")
    reranker = CrossEncoderReranker(device="cpu")
    reranker.load()

    all_results: Dict[str, AggregateMetrics] = {}
    all_pool_metrics: Dict[str, Dict] = {}
    per_query_data: Dict[str, List[Dict]] = {}

    # ══════════════════════════════════════════════════════════════════
    # Step 5: Run baselines
    # ══════════════════════════════════════════════════════════════════

    # 5a. Dense Fixed-k (THE baseline to beat)
    print("\n🔍 Running Dense Fixed-k Baseline (k=10)...")
    dense_bl = DenseFixedBaseline(index_mgr, all_embeddings)
    dense_metrics = []
    dense_ndcg_per_query = []
    dense_indices_per_query = []  # Store for overlap computation

    for q in queries:
        result = dense_bl.retrieve(q.embedding, query_id=q.id, k=eval_k)
        qm = evaluator.evaluate_query(
            result.retrieved_indices, q.relevance, task_ids,
            q.id, result.total_time_ms, result.candidates_explored,
        )
        dense_metrics.append(qm)
        dense_ndcg_per_query.append(qm.ndcg_at_k.get(10, 0))
        dense_indices_per_query.append(result.retrieved_indices)

    all_results["Dense"] = evaluator.aggregate(dense_metrics, "Dense")
    dense_ndcg_arr = np.array(dense_ndcg_per_query)

    # Also run Dense with k=50 for pool comparison
    print("🔍 Running Dense (k=50) for pool evaluation...")
    dense50_pool_metrics = []
    for qi, q in enumerate(queries):
        result50 = dense_bl.retrieve(q.embedding, query_id=q.id, k=50)
        pm = pool_evaluator.evaluate_pool(
            candidate_indices=result50.retrieved_indices,
            dense_only_indices=dense_indices_per_query[qi],
            relevance_labels=q.relevance,
            task_ids=task_ids,
            attribution={int(idx): ["dense"] for idx in result50.retrieved_indices},
            query_id=q.id,
        )
        dense50_pool_metrics.append(pm)
    all_pool_metrics["Dense"] = CandidatePoolEvaluator.aggregate_pool_metrics(dense50_pool_metrics)

    # 5b. BM25 baseline
    print("📖 Running BM25 Baseline...")
    bm25_metrics = []
    for q in queries:
        result = bm25_bl.retrieve(q.text, query_id=q.id, k=eval_k)
        qm = evaluator.evaluate_query(
            result.retrieved_indices, q.relevance, task_ids,
            q.id, result.total_time_ms, result.candidates_explored,
        )
        bm25_metrics.append(qm)
    all_results["BM25"] = evaluator.aggregate(bm25_metrics, "BM25")

    # ══════════════════════════════════════════════════════════════════
    # Step 6: Run AMSR-SE configurations
    # ══════════════════════════════════════════════════════════════════

    amsr_configs = {
        "AMSR-SE (Dense+BM25)": {"use_graph": False, "use_reranker": False},
        "AMSR-SE (Dense+Graph)": {"use_bm25": False, "use_reranker": False},
        "AMSR-SE (Dense+BM25+Graph)": {"use_reranker": False},
        "Full AMSR-SE": {},
    }

    for name, overrides in amsr_configs.items():
        print(f"\n🧠 Running {name}...")

        # Create retriever with appropriate components
        ret_bm25 = bm25_bl if overrides.get("use_bm25", True) else None
        ret_reranker = reranker if overrides.get("use_reranker", True) else None

        cfg_copy = copy.deepcopy(config)
        if overrides.get("use_graph", True) is False:
            cfg_copy.adaptive.graph_expansion_hops = 0
            cfg_copy.adaptive.graph_max_neighbors = 0

        retriever = HybridRetriever(
            cfg_copy, index_mgr, graph_exp, all_embeddings,
            bm25_baseline=ret_bm25, task_texts=task_texts, reranker=ret_reranker,
        )

        method_metrics = []
        method_ndcg_per_query = []
        method_pool_metrics = []

        for qi, q in enumerate(queries):
            result = retriever.retrieve(
                q.id, q.embedding, final_k=eval_k,
                task_metadata={"category": q.category, "query_text": q.text},
            )

            qm = evaluator.evaluate_query(
                result.retrieved_indices, q.relevance, task_ids,
                q.id, result.total_time_ms, result.candidates_explored,
            )
            method_metrics.append(qm)
            method_ndcg_per_query.append(qm.ndcg_at_k.get(10, 0))

            # Candidate pool evaluation
            if result.candidate_pool_indices is not None:
                pm = pool_evaluator.evaluate_pool(
                    candidate_indices=result.candidate_pool_indices,
                    dense_only_indices=dense_indices_per_query[qi],
                    relevance_labels=q.relevance,
                    task_ids=task_ids,
                    attribution=result.candidate_attribution or {},
                    query_id=q.id,
                )
                method_pool_metrics.append(pm)

        all_results[name] = evaluator.aggregate(method_metrics, name)
        per_query_data[name] = method_ndcg_per_query

        if method_pool_metrics:
            all_pool_metrics[name] = CandidatePoolEvaluator.aggregate_pool_metrics(method_pool_metrics)

        retriever.reset()

    # ══════════════════════════════════════════════════════════════════
    # Step 7: Print results
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("   RESULTS")
    print("=" * 80)
    print()
    print(Evaluator.format_results(all_results))

    # Print candidate pool metrics
    print("\n" + "=" * 80)
    print("   CANDIDATE POOL EVALUATION")
    print("=" * 80)
    for method, pm in all_pool_metrics.items():
        print(f"\n  {method}:")
        print(f"    Pool size:        {pm.get('pool_size_mean', 0):.1f}")
        print(f"    Oracle recall:    {pm.get('oracle_recall_mean', 0):.4f}")
        for d in [10, 50, 100]:
            cr = pm.get('candidate_recall', {}).get(d, 0)
            nd = pm.get('relevant_new_docs', {}).get(d, 0)
            ov = pm.get('overlap_with_dense', {}).get(d, 0)
            print(f"    @{d:3d}: recall={cr:.4f}  new_docs={nd:.1f}  overlap={ov:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # Step 8: Statistical significance testing
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("   STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    # Define easy/hard based on dense nDCG
    easy_mask = dense_ndcg_arr > 0.5

    stat_reports = {}
    if "Full AMSR-SE" in per_query_data:
        system_ndcg = np.array(per_query_data["Full AMSR-SE"])
        report = stat_tester.full_comparison(
            dense_ndcg_arr, system_ndcg,
            baseline_name="Dense", system_name="Full AMSR-SE",
            easy_mask=easy_mask,
        )
        stat_reports["Full AMSR-SE"] = report
        print("\n" + StatisticalTester.format_report(report))

    # ══════════════════════════════════════════════════════════════════
    # Step 9: Generate publication plots
    # ══════════════════════════════════════════════════════════════════

    print("\n📊 Generating publication plots...")
    from . import visualize_results as viz

    # Plot 1: Candidate recall at depth
    if all_pool_metrics:
        viz.plot_candidate_recall_at_depth(all_pool_metrics, plots_dir)

    # Plot 2: Hard query recovery
    if "Full AMSR-SE" in per_query_data:
        viz.plot_hard_query_recovery(
            dense_ndcg_arr, np.array(per_query_data["Full AMSR-SE"]),
            easy_mask, plots_dir,
        )

    # Plot 3: Win/tie/loss
    if stat_reports:
        for name, report in stat_reports.items():
            viz.plot_win_tie_loss(report, plots_dir)

    # Plot 4: Source attribution
    if all_pool_metrics:
        viz.plot_source_attribution(all_pool_metrics, plots_dir)

    # Plot 5: Overlap heatmap
    if all_pool_metrics:
        viz.plot_overlap_heatmap(all_pool_metrics, plots_dir)

    # Plot 6: Pareto frontier
    results_json = {m: Evaluator.to_dict(agg) for m, agg in all_results.items()}
    viz.plot_pareto_frontier(results_json, plots_dir)

    # Plot 7: Per-query delta waterfall
    if "Full AMSR-SE" in per_query_data:
        viz.plot_delta_ndcg_waterfall(
            dense_ndcg_arr, np.array(per_query_data["Full AMSR-SE"]),
            plots_dir,
        )

    # Plot 9: Uncertainty calibration
    # Collect uncertainty scores from the last AMSR-SE run
    # (we need to re-run or extract from controller logs)

    # ══════════════════════════════════════════════════════════════════
    # Step 10: Save everything
    # ══════════════════════════════════════════════════════════════════

    # Aggregate metrics
    results_json["_metadata"] = {
        "num_tasks": len(tasks),
        "num_queries": len(queries),
        "index_type": config.index.index_type.value,
        "eval_k": eval_k,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fusion_method": "rrf",
        "rrf_k": 60,
        "dense_k": HybridRetriever.DENSE_K,
        "bm25_k": HybridRetriever.BM25_K,
        "rerank_depth": HybridRetriever.RERANK_DEPTH,
    }

    with open(os.path.join(results_dir, "experiment_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    # Candidate pool metrics
    pool_json = {m: pm for m, pm in all_pool_metrics.items()}
    with open(os.path.join(results_dir, "candidate_pool_metrics.json"), "w") as f:
        json.dump(pool_json, f, indent=2, default=str)

    # Statistical test reports
    with open(os.path.join(results_dir, "statistical_tests.json"), "w") as f:
        json.dump(stat_reports, f, indent=2)

    # Per-query metrics CSV
    csv_path = os.path.join(results_dir, "per_query_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "method", "ndcg_at_10"])
        for qi, q in enumerate(queries):
            writer.writerow([q.id, "Dense", dense_ndcg_per_query[qi]])
            for method, ndcgs in per_query_data.items():
                writer.writerow([q.id, method, ndcgs[qi]])

    # Experiment config
    config_json = {
        "dense_k": HybridRetriever.DENSE_K,
        "bm25_k": HybridRetriever.BM25_K,
        "graph_seed_k": HybridRetriever.GRAPH_SEED_K,
        "graph_neighbors": HybridRetriever.GRAPH_NEIGHBORS,
        "max_candidates": HybridRetriever.MAX_CANDIDATES,
        "rerank_depth": HybridRetriever.RERANK_DEPTH,
        "fusion_method": "rrf",
        "rrf_k": 60,
        "eval_k": eval_k,
    }
    with open(os.path.join(results_dir, "experiment_config.json"), "w") as f:
        json.dump(config_json, f, indent=2)

    print(f"\n💾 All results saved to {results_dir}/")
    print(f"💾 Plots saved to {plots_dir}/")
    print("\n✅ Publication experiment suite complete!")

    return all_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AMSR-SE Publication Experiment Runner")
    parser.add_argument("--tasks", type=int, default=5000, help="Number of tasks")
    parser.add_argument("--queries", type=int, default=200, help="Number of queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--index", type=str, default="hnsw",
                        choices=["hnsw", "ivf", "ivfpq"], help="Index type")
    parser.add_argument("--data-dir", type=str, default="ahrc_data", help="Data directory")
    args = parser.parse_args()

    config = AHRCConfig()
    config.experiment.num_tasks = args.tasks
    config.experiment.num_queries = args.queries
    config.experiment.random_seed = args.seed
    config.index.index_type = IndexType(args.index)

    run_experiments(config, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
