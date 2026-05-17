"""
AHRC — Ablation Study
Systematically measures the impact of each module:

  1. Dense only (no uncertainty, no graph, no adaptation)
  2. Dense + Uncertainty (uncertainty estimation but no adaptation)
  3. Dense + Graph (always-on graph but no uncertainty)
  4. Dense + Uncertainty + Adaptation (no graph)
  5. Full AHRC (dense + uncertainty + adaptation + graph)

Usage:
    python -m ahrc.ablation_study [--tasks 10000] [--queries 200]
"""

import os
import json
import time
import copy
import argparse
import numpy as np
from typing import Dict, List

from .config import AHRCConfig, IndexType
from .dataset_generator import DatasetGenerator, build_dataset
from .index_manager import IndexManager
from .graph_expander import GraphExpander
from .hybrid_retriever import HybridRetriever
from .baselines import DenseFixedBaseline, DenseGraphFixedBaseline, BM25Baseline
from .evaluation import Evaluator, AggregateMetrics, QueryMetrics
from .reranker import CrossEncoderReranker


def run_ablation(config: AHRCConfig = None, data_dir: str = "ahrc_data"):
    """Run ablation study."""

    if config is None:
        config = AHRCConfig()

    results_dir = config.experiment.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("   AHRC — Ablation Study")
    print("   Measuring impact of each component")
    print("=" * 80)
    print()

    # ── Load / generate data ───────────────────────────────────────────
    if os.path.exists(os.path.join(data_dir, "tasks.json")):
        dataset = DatasetGenerator.load(data_dir, config)
    else:
        dataset = build_dataset(config, data_dir)

    tasks = dataset.tasks
    queries = dataset.queries
    task_ids = [t.id for t in tasks]
    id_to_index = {t.id: i for i, t in enumerate(tasks)}
    all_embeddings = np.array([t.embedding for t in tasks], dtype=np.float32)

    index_mgr = IndexManager(config.index)
    index_mgr.build(all_embeddings)

    graph_exp = GraphExpander(config.adaptive)
    graph_exp.build_from_tasks(tasks, id_to_index)
    
    print("🧠 Loading additional components...")
    bm25_bl = BM25Baseline(tasks)
    task_texts = [t.description for t in tasks]
    reranker = CrossEncoderReranker(device="cpu")
    reranker.load()

    evaluator = Evaluator(k_values=config.experiment.eval_k_values, relevance_threshold=1)
    eval_k = 10

    ablation_results: Dict[str, AggregateMetrics] = {}

    # ── Ablation 1: Dense Only ─────────────────────────────────────────
    print("\n🔬 Ablation 1/5: Dense Only (no uncertainty, no graph, no adaptation)")
    dense_bl = DenseFixedBaseline(index_mgr, all_embeddings)
    metrics = _eval_baseline(dense_bl, queries, evaluator, task_ids, eval_k, mode="dense")
    ablation_results["Dense Only"] = evaluator.aggregate(metrics, "Dense Only")

    # ── Ablation 2: Dense + Graph (always on) ──────────────────────────
    print("🔬 Ablation 2/5: Dense + Graph (always expand, no adaptation)")
    dg_bl = DenseGraphFixedBaseline(index_mgr, graph_exp, all_embeddings)
    metrics = _eval_baseline(dg_bl, queries, evaluator, task_ids, eval_k, mode="dense_graph")
    ablation_results["Dense + Graph"] = evaluator.aggregate(metrics, "Dense + Graph")

    # ── Ablation 3: Dense + BM25 ───────────────────────────────────────
    print("🔬 Ablation 3/5: Dense + BM25 (no Graph, no CrossEncoder)")
    cfg3 = copy.deepcopy(config)
    cfg3.adaptive.graph_expansion_hops = 0
    cfg3.adaptive.graph_max_neighbors = 0
    
    # We use HybridRetriever but don't pass reranker, and graph hops = 0
    ahrc3 = HybridRetriever(cfg3, index_mgr, graph_exp, all_embeddings, bm25_baseline=bm25_bl, task_texts=task_texts)
    metrics3 = []
    for q in queries:
        result = ahrc3.retrieve(q.id, q.embedding, final_k=eval_k, task_metadata={"category": q.category, "query_text": q.text})
        qm = evaluator.evaluate_query(result.retrieved_indices, q.relevance, task_ids, q.id, result.total_time_ms, result.candidates_explored)
        metrics3.append(qm)
    ahrc3.reset()
    ablation_results["Dense + BM25"] = evaluator.aggregate(metrics3, "Dense + BM25")

    # ── Ablation 4: Dense + Graph + BM25 ───────────────────────────────
    print("🔬 Ablation 4/5: Dense + Graph + BM25 (no CrossEncoder)")
    cfg4 = copy.deepcopy(config)
    # We pass everything except reranker
    ahrc4 = HybridRetriever(cfg4, index_mgr, graph_exp, all_embeddings, bm25_baseline=bm25_bl, task_texts=task_texts)
    metrics4 = []
    for q in queries:
        result = ahrc4.retrieve(q.id, q.embedding, final_k=eval_k, task_metadata={"category": q.category, "query_text": q.text})
        qm = evaluator.evaluate_query(result.retrieved_indices, q.relevance, task_ids, q.id, result.total_time_ms, result.candidates_explored)
        metrics4.append(qm)
    ahrc4.reset()
    ablation_results["Dense + Graph + BM25"] = evaluator.aggregate(metrics4, "Dense + Graph + BM25")

    # ── Ablation 5: Full AMSR-SE ───────────────────────────────────────
    print("🔬 Ablation 5/5: Full AMSR-SE (Dense + BM25 + Graph + CrossEncoder)")
    ahrc_full = HybridRetriever(config, index_mgr, graph_exp, all_embeddings, bm25_baseline=bm25_bl, task_texts=task_texts, reranker=reranker)
    metrics5 = []
    for q in queries:
        result = ahrc_full.retrieve(q.id, q.embedding, final_k=eval_k, task_metadata={"category": q.category, "query_text": q.text})
        qm = evaluator.evaluate_query(result.retrieved_indices, q.relevance, task_ids, q.id, result.total_time_ms, result.candidates_explored)
        metrics5.append(qm)
    ablation_results["Full AMSR-SE"] = evaluator.aggregate(metrics5, "Full AMSR-SE")

    # Store per-query nDCG for significance testing and plotting
    dense_ndcg_list = [qm.ndcg_at_k.get(10, 0) for qm in metrics] # metrics from Dense Only
    full_ndcg_list = [qm.ndcg_at_k.get(10, 0) for qm in metrics5]
    ablation_results["_query_metrics"] = {
        "dense_ndcg": dense_ndcg_list,
        "full_ndcg": full_ndcg_list
    }

    # ── Print results ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("   ABLATION RESULTS")
    print("=" * 80)
    print()
    
    printable_results = {k: v for k, v in ablation_results.items() if k != "_query_metrics"}
    print(Evaluator.format_results(printable_results))

    # ── Component impact analysis ──────────────────────────────────────
    print("\n📊 Component Impact Analysis:")
    base = ablation_results["Dense Only"]

    for name, agg in printable_results.items():
        if name == "Dense Only":
            continue

        r10_delta = agg.recall_at_k.get(10, 0) - base.recall_at_k.get(10, 0)
        ndcg_delta = agg.ndcg_at_k.get(10, 0) - base.ndcg_at_k.get(10, 0)
        lat_delta = agg.latency_mean_ms - base.latency_mean_ms
        cand_delta = agg.candidates_mean - base.candidates_mean

        print(f"\n   {name}:")
        print(f"     Recall@10: {r10_delta:+.4f}")
        print(f"     nDCG@10:   {ndcg_delta:+.4f}")
        print(f"     Latency:   {lat_delta:+.2f}ms")
        print(f"     Candidates:{cand_delta:+.1f}")

    # ── Save results ───────────────────────────────────────────────────
    ablation_json = {
        method: Evaluator.to_dict(agg) if isinstance(agg, AggregateMetrics) else agg
        for method, agg in ablation_results.items()
    }
    ablation_json["_metadata"] = {
        "num_tasks": len(tasks),
        "num_queries": len(queries),
        "index_type": config.index.index_type.value,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    path = os.path.join(results_dir, "ablation_results.json")
    with open(path, "w") as f:
        json.dump(ablation_json, f, indent=2)
    print(f"\n💾 Ablation results saved to {path}")

    # Generate ablation plots
    try:
        from .visualize_results import generate_ablation_plots
        generate_ablation_plots(ablation_json, results_dir)
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")

    print("\n✅ Ablation study complete!")
    return ablation_results


def _eval_baseline(baseline, queries, evaluator, task_ids, eval_k, mode="dense"):
    """Helper: evaluate a baseline across all queries."""
    metrics = []
    for q in queries:
        if mode == "dense" or mode == "dense_graph":
            result = baseline.retrieve(q.embedding, query_id=q.id, k=eval_k)
        else:
            result = baseline.retrieve(query_id=q.id, k=eval_k)

        qm = evaluator.evaluate_query(
            retrieved_indices=result.retrieved_indices,
            relevance_labels=q.relevance,
            task_ids=task_ids,
            query_id=q.id,
            latency_ms=result.total_time_ms,
            candidates_explored=result.candidates_explored,
        )
        metrics.append(qm)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="AHRC Ablation Study")
    parser.add_argument("--tasks", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="ahrc_data")
    args = parser.parse_args()

    config = AHRCConfig()
    config.experiment.num_tasks = args.tasks
    config.experiment.num_queries = args.queries
    config.experiment.random_seed = args.seed

    run_ablation(config, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
