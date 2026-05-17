"""
AMSR-SE — Universal Benchmark Runner
Runs the full AMSR-SE pipeline on ANY benchmark loaded via dataset_interface.

Usage:
    # Run on BEIR SciFact (small, fast)
    python -m ahrc.run_benchmark --source beir --dataset scifact

    # Run on BEIR FiQA (medium)
    python -m ahrc.run_benchmark --source beir --dataset fiqa --max-docs 10000

    # Run on MS MARCO dev/small
    python -m ahrc.run_benchmark --source msmarco --max-docs 50000 --max-queries 200

    # Run on existing synthetic
    python -m ahrc.run_benchmark --source synthetic

    # List all available datasets
    python -m ahrc.run_benchmark --list
"""

import os
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Optional

from .dataset_interface import load_benchmark, list_available_datasets, BenchmarkData
from .index_manager import IndexManager
from .graph_expander import GraphExpander
from .hybrid_retriever import HybridRetriever
from .baselines import BM25Baseline, DenseFixedBaseline
from .evaluation import Evaluator, AggregateMetrics, QueryMetrics
from .candidate_pool_eval import CandidatePoolEvaluator
from .candidate_fusion import CandidateFusion
from .reranker import CrossEncoderReranker
from .statistical_tests import StatisticalTester
from .config import AHRCConfig, IndexType


def run_benchmark(
    source: str,
    results_dir: str = "amsr_results",
    model_name: str = "all-MiniLM-L6-v2",
    **loader_kwargs,
):
    """
    Run full AMSR-SE evaluation on a loaded benchmark.

    Args:
        source: dataset source ('synthetic', 'beir', 'msmarco', 'trec-dl').
        results_dir: where to save results.
        model_name: sentence-transformer model for encoding.
        **loader_kwargs: passed to dataset loader.
    """
    print("=" * 80)
    print("   AMSR-SE — Universal Benchmark Runner")
    print("=" * 80)

    # ── Step 1: Load benchmark ─────────────────────────────────────────
    data = load_benchmark(source, **loader_kwargs)

    # Create results subdirectory for this benchmark
    bench_dir = os.path.join(results_dir, data.name.replace("/", "_").replace(" ", "_"))
    plots_dir = os.path.join(bench_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Step 2: Embed corpus and queries ───────────────────────────────
    print(f"\n🧠 Embedding {data.num_docs:,} documents with {model_name}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device='cpu')

    t0 = time.time()
    corpus_embeddings = model.encode(
        data.corpus_texts, batch_size=256,
        show_progress_bar=True, normalize_embeddings=True,
    ).astype(np.float32)
    print(f"   ✅ Corpus embedded in {time.time()-t0:.1f}s (dim={corpus_embeddings.shape[1]})")

    print(f"\n🧠 Embedding {data.num_queries} queries...")
    t0 = time.time()
    query_embeddings = model.encode(
        data.query_texts, batch_size=256,
        show_progress_bar=False, normalize_embeddings=True,
    ).astype(np.float32)
    print(f"   ✅ Queries embedded in {time.time()-t0:.1f}s")

    # ── Step 3: Build FAISS index ──────────────────────────────────────
    config = AHRCConfig()
    config.index.embedding_dim = corpus_embeddings.shape[1]

    # Choose index type based on corpus size
    if data.num_docs > 100000:
        config.index.index_type = IndexType.IVF
        config.index.ivf_nlist = min(int(np.sqrt(data.num_docs)), 1000)
    else:
        config.index.index_type = IndexType.HNSW

    index_mgr = IndexManager(config.index)
    index_mgr.build(corpus_embeddings)

    # ── Step 4: Build task graph (simple kNN proximity) ────────────────
    # For real datasets, we build a kNN graph from embeddings
    print("🔗 Building proximity graph...")
    from .graph_expander import GraphExpander
    graph_exp = GraphExpander(config.adaptive)
    # Build a kNN-based graph by finding nearest neighbors
    _build_knn_graph(graph_exp, corpus_embeddings, data.corpus_ids, k=5)

    # ── Step 5: Initialize components ──────────────────────────────────
    evaluator = Evaluator(k_values=[1, 3, 5, 10, 20], relevance_threshold=1)
    pool_evaluator = CandidatePoolEvaluator(eval_depths=[10, 50, 100])
    stat_tester = StatisticalTester(n_bootstrap=10000, n_permutation=5000)

    # BM25 baseline
    print("📖 Building BM25 index...")
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [text.lower().split() for text in data.corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Cross-encoder
    print("🧠 Loading Cross-Encoder...")
    reranker = CrossEncoderReranker(device="cpu")
    reranker.load()

    eval_k = 10

    # ── Step 6: Run Dense baseline ─────────────────────────────────────
    print("\n🔍 Running Dense Baseline (k=10)...")
    dense_bl = DenseFixedBaseline(index_mgr, corpus_embeddings)
    dense_metrics = []
    dense_ndcg_per_query = []
    dense_indices_per_query = []

    for qi in range(data.num_queries):
        qid = data.query_ids[qi]
        q_emb = query_embeddings[qi]
        qrels = data.qrels.get(qid, {})

        result = dense_bl.retrieve(q_emb, query_id=qid, k=eval_k)
        qm = evaluator.evaluate_query(
            result.retrieved_indices, qrels, data.corpus_ids,
            qid, result.total_time_ms, result.candidates_explored,
        )
        dense_metrics.append(qm)
        dense_ndcg_per_query.append(qm.ndcg_at_k.get(10, 0))
        dense_indices_per_query.append(result.retrieved_indices)

    dense_agg = evaluator.aggregate(dense_metrics, "Dense")
    dense_ndcg_arr = np.array(dense_ndcg_per_query)
    all_results = {"Dense": dense_agg}

    # Dense pool evaluation (k=50)
    print("🔍 Running Dense (k=50) for pool evaluation...")
    dense_pool_metrics = []
    for qi in range(data.num_queries):
        qid = data.query_ids[qi]
        q_emb = query_embeddings[qi]
        qrels = data.qrels.get(qid, {})
        result50 = dense_bl.retrieve(q_emb, query_id=qid, k=50)
        pm = pool_evaluator.evaluate_pool(
            candidate_indices=result50.retrieved_indices,
            dense_only_indices=dense_indices_per_query[qi],
            relevance_labels=qrels,
            task_ids=data.corpus_ids,
            attribution={int(idx): ["dense"] for idx in result50.retrieved_indices},
            query_id=qid,
        )
        dense_pool_metrics.append(pm)
    all_pool_metrics = {"Dense": CandidatePoolEvaluator.aggregate_pool_metrics(dense_pool_metrics)}

    # ── Step 7: Run BM25 baseline ──────────────────────────────────────
    print("📖 Running BM25 Baseline...")
    bm25_metrics = []
    for qi in range(data.num_queries):
        qid = data.query_ids[qi]
        q_text = data.query_texts[qi]
        qrels = data.qrels.get(qid, {})

        tokenized_query = q_text.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(-bm25_scores)[:eval_k]
        top_k_scores = bm25_scores[top_k_idx].astype(np.float32)

        qm = evaluator.evaluate_query(
            top_k_idx, qrels, data.corpus_ids,
            qid, 0.0, len(bm25_scores),
        )
        bm25_metrics.append(qm)

    all_results["BM25"] = evaluator.aggregate(bm25_metrics, "BM25")

    # ── Step 8: Run Full AMSR-SE ───────────────────────────────────────
    print("\n🧠 Running Full AMSR-SE...")

    # Create a BM25Baseline-compatible wrapper
    bm25_wrapper = _BM25Wrapper(bm25, data.corpus_texts)

    retriever = HybridRetriever(
        config, index_mgr, graph_exp, corpus_embeddings,
        bm25_baseline=bm25_wrapper, task_texts=data.corpus_texts, reranker=reranker,
    )

    amsr_metrics = []
    amsr_ndcg_per_query = []
    amsr_pool_metrics = []

    for qi in range(data.num_queries):
        qid = data.query_ids[qi]
        q_emb = query_embeddings[qi]
        q_text = data.query_texts[qi]
        qrels = data.qrels.get(qid, {})

        result = retriever.retrieve(
            qid, q_emb, final_k=eval_k,
            task_metadata={"category": "", "query_text": q_text},
        )

        qm = evaluator.evaluate_query(
            result.retrieved_indices, qrels, data.corpus_ids,
            qid, result.total_time_ms, result.candidates_explored,
        )
        amsr_metrics.append(qm)
        amsr_ndcg_per_query.append(qm.ndcg_at_k.get(10, 0))

        if result.candidate_pool_indices is not None:
            pm = pool_evaluator.evaluate_pool(
                candidate_indices=result.candidate_pool_indices,
                dense_only_indices=dense_indices_per_query[qi],
                relevance_labels=qrels,
                task_ids=data.corpus_ids,
                attribution=result.candidate_attribution or {},
                query_id=qid,
            )
            amsr_pool_metrics.append(pm)

    all_results["Full AMSR-SE"] = evaluator.aggregate(amsr_metrics, "Full AMSR-SE")
    amsr_ndcg_arr = np.array(amsr_ndcg_per_query)

    if amsr_pool_metrics:
        all_pool_metrics["Full AMSR-SE"] = CandidatePoolEvaluator.aggregate_pool_metrics(amsr_pool_metrics)

    # ── Step 9: Print results ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"   RESULTS — {data.name}")
    print("=" * 80)
    print()
    print(Evaluator.format_results(all_results))

    # Pool metrics
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

    # ── Step 10: Statistical tests ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("   STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    easy_mask = dense_ndcg_arr > 0.5
    report = stat_tester.full_comparison(
        dense_ndcg_arr, amsr_ndcg_arr,
        baseline_name="Dense", system_name="Full AMSR-SE",
        easy_mask=easy_mask,
    )
    print("\n" + StatisticalTester.format_report(report))

    # ── Step 11: Generate plots ────────────────────────────────────────
    print("\n📊 Generating publication plots...")
    from . import visualize_results as viz

    if all_pool_metrics:
        viz.plot_candidate_recall_at_depth(all_pool_metrics, plots_dir)
        viz.plot_source_attribution(all_pool_metrics, plots_dir)
        viz.plot_overlap_heatmap(all_pool_metrics, plots_dir)

    viz.plot_hard_query_recovery(dense_ndcg_arr, amsr_ndcg_arr, easy_mask, plots_dir)
    viz.plot_win_tie_loss(report, plots_dir)
    viz.plot_delta_ndcg_waterfall(dense_ndcg_arr, amsr_ndcg_arr, plots_dir)

    results_json = {m: Evaluator.to_dict(agg) for m, agg in all_results.items()}
    viz.plot_pareto_frontier(results_json, plots_dir)

    # ── Step 12: Save everything ───────────────────────────────────────
    results_json["_metadata"] = {
        "benchmark": data.name,
        "num_docs": data.num_docs,
        "num_queries": data.num_queries,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
    }
    with open(os.path.join(bench_dir, "experiment_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    with open(os.path.join(bench_dir, "candidate_pool_metrics.json"), "w") as f:
        json.dump(all_pool_metrics, f, indent=2, default=str)

    with open(os.path.join(bench_dir, "statistical_tests.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 All results saved to {bench_dir}/")
    print("✅ Benchmark complete!")

    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Helper classes
# ═══════════════════════════════════════════════════════════════════════

class _BM25Wrapper:
    """Wraps rank_bm25 to match the BM25Baseline.retrieve() interface."""

    def __init__(self, bm25_index, corpus_texts: List[str]):
        self.bm25 = bm25_index
        self.corpus_texts = corpus_texts

    def retrieve(self, query_text: str, k: int = 10, query_id: str = ""):
        from .baselines import BaselineResult
        tokenized = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized)
        top_k_idx = np.argsort(-scores)[:k]
        top_k_scores = scores[top_k_idx].astype(np.float32)
        return BaselineResult(
            query_id=query_id,
            retrieved_indices=top_k_idx,
            retrieved_scores=top_k_scores,
            total_time_ms=0.0,
            candidates_explored=len(scores),
            method="bm25",
        )


def _build_knn_graph(graph_exp, embeddings, doc_ids, k=5):
    """Build a simple kNN proximity graph from embeddings."""
    import faiss

    n, d = embeddings.shape
    # Use a flat index for exact kNN
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Query for k+1 neighbors (includes self)
    distances, indices = index.search(embeddings, k + 1)

    # Build adjacency list
    graph_exp.adjacency = {}
    graph_exp._built = True

    id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    edge_count = 0
    for i in range(n):
        neighbors = []
        for j in range(1, k + 1):  # Skip self at position 0
            neighbor_idx = int(indices[i, j])
            if 0 <= neighbor_idx < n:
                neighbors.append(neighbor_idx)
                edge_count += 1
        graph_exp.adjacency[i] = neighbors

    print(f"   ✅ kNN graph: {n:,} nodes, {edge_count:,} edges (k={k})")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AMSR-SE Universal Benchmark Runner")
    parser.add_argument("--source", type=str, default="synthetic",
                        choices=["synthetic", "beir", "msmarco", "trec-dl"],
                        help="Dataset source")
    parser.add_argument("--dataset", type=str, default="scifact",
                        help="BEIR dataset name (e.g., scifact, fiqa, nfcorpus)")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Max corpus size")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Max number of queries")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformer model")
    parser.add_argument("--results-dir", type=str, default="amsr_results",
                        help="Output directory")
    parser.add_argument("--list", action="store_true",
                        help="List all available datasets")
    args = parser.parse_args()

    if args.list:
        list_available_datasets()
        return

    kwargs = {}
    if args.source == "beir":
        kwargs["dataset_name"] = args.dataset
    if args.max_docs:
        kwargs["max_docs"] = args.max_docs
    if args.max_queries:
        kwargs["max_queries"] = args.max_queries

    run_benchmark(
        source=args.source,
        results_dir=args.results_dir,
        model_name=args.model,
        **kwargs,
    )


if __name__ == "__main__":
    main()
