"""
Safe-AMSR-SE v4 — Master Experiment Runner
Leakage-safe: train/val/test split. Multi-action routing. Pairwise stats.
"""
import os, json, time, csv
import numpy as np
from typing import Dict, List

from .dataset_interface import load_benchmark
from .index_manager import IndexManager
from .graph_expander import GraphExpander
from .hybrid_retriever import HybridRetriever
from .baselines import DenseFixedBaseline
from .evaluation import Evaluator
from .candidate_fusion import CandidateFusion
from .reranker import CrossEncoderReranker
from .statistical_tests import StatisticalTester
from .feature_extractor import FeatureExtractor, FEATURE_NAMES
from .safe_router import (
    RuleBasedRouter, LearnedRouter, MultiActionRouter, CostAwareRouter,
    OracleRouter, RandomRouter, Action, ACTION_NAMES,
    build_training_data, compute_safety_metrics,
)
from .leakage_safe_split import create_stratified_split
from .source_attribution import SourceAttributionAnalyzer
from .table_generator import TableGenerator
from .router_explainability import RouterExplainer
from .config import AHRCConfig, IndexType


def _build_knn_graph(graph_exp, embeddings, k=5):
    import faiss
    n, d = embeddings.shape
    idx = faiss.IndexFlatIP(d)
    idx.add(embeddings)
    _, indices = idx.search(embeddings, k + 1)
    graph_exp.adjacency = {}
    graph_exp._built = True
    for i in range(n):
        neighbors = [int(indices[i, j]) for j in range(1, k+1) if 0 <= indices[i, j] < n]
        graph_exp.adjacency[i] = set(neighbors)
        graph_exp.degree_cache[i] = len(neighbors)


class _BM25Wrap:
    def __init__(self, bm25, texts):
        self.bm25, self.texts = bm25, texts
    def retrieve(self, query_text, k=10, query_id=""):
        from .baselines import BaselineResult
        import time as _t
        t0 = _t.perf_counter()
        scores = self.bm25.get_scores(query_text.lower().split())
        top = np.argsort(-scores)[:k]
        elapsed = (_t.perf_counter() - t0) * 1000
        return BaselineResult(query_id=query_id, retrieved_indices=top,
                              retrieved_scores=scores[top].astype(np.float32),
                              total_time_ms=elapsed, candidates_explored=len(scores), method="bm25")


def run_safe_experiment(source="beir", results_dir="results_safe_amsr_v4",
                         model_name="all-MiniLM-L6-v2", device="auto", **loader_kwargs):
    print("=" * 70)
    print("   Safe-AMSR-SE v4 — Full Experiment Suite")
    print("=" * 70)

    loader_kwargs.setdefault("dataset_name", "scifact")
    ds_name = loader_kwargs.get("dataset_name", source)
    bench_dir = os.path.join(results_dir, ds_name)
    for sub in ["plots", "metrics", "configs", "tables", "logs", "explainability"]:
        os.makedirs(os.path.join(bench_dir, sub), exist_ok=True)

    # ── Phase 0: Load data & build infrastructure ──
    data = load_benchmark(source, **loader_kwargs)
    from sentence_transformers import SentenceTransformer
    embed_device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available():
            try: torch.zeros(1, device='cuda'); embed_device = 'cuda'
            except: pass
    except: pass

    st_model = SentenceTransformer(model_name, device=embed_device)
    corpus_emb = st_model.encode(data.corpus_texts, batch_size=256, show_progress_bar=True,
                                  normalize_embeddings=True).astype(np.float32)
    query_emb = st_model.encode(data.query_texts, batch_size=256,
                                 normalize_embeddings=True).astype(np.float32)

    config = AHRCConfig()
    config.index.embedding_dim = corpus_emb.shape[1]
    config.index.index_type = IndexType.HNSW
    index_mgr = IndexManager(config.index)
    index_mgr.build(corpus_emb)

    graph_exp = GraphExpander(config.adaptive)
    _build_knn_graph(graph_exp, corpus_emb, k=5)

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([t.lower().split() for t in data.corpus_texts])
    bm25_wrap = _BM25Wrap(bm25, data.corpus_texts)

    reranker = CrossEncoderReranker(device=device)
    reranker.load()

    evaluator = Evaluator(k_values=[1, 3, 5, 10, 20], relevance_threshold=1)
    feat_ext = FeatureExtractor()
    feat_ext.build_idf(data.corpus_texts)
    source_attr = SourceAttributionAnalyzer(relevance_threshold=1)
    dense_bl = DenseFixedBaseline(index_mgr, corpus_emb)
    retriever_full = HybridRetriever(config, index_mgr, graph_exp, corpus_emb,
                                      bm25_baseline=bm25_wrap, task_texts=data.corpus_texts,
                                      reranker=reranker)
    eval_k = 10

    # ── Phase 1: Collect per-query nDCG for all actions ──
    print("\nPhase 1: Collecting per-query metrics (Dense + BM25 + Full)...")
    dense_ndcg_list, full_ndcg_list, bm25_ndcg_list = [], [], []
    dense_metrics_list, full_metrics_list, bm25_metrics_list = [], [], []
    feature_list, all_query_ids = [], []
    dense_latencies, full_latencies = [], []

    for qi in range(data.num_queries):
        qid, qe, qt = data.query_ids[qi], query_emb[qi], data.query_texts[qi]
        qrels = data.qrels.get(qid, {})
        all_query_ids.append(qid)

        t0 = time.perf_counter()
        dr = dense_bl.retrieve(qe, query_id=qid, k=eval_k)
        dense_lat = (time.perf_counter() - t0) * 1000
        dense_latencies.append(dense_lat)
        dm = evaluator.evaluate_query(dr.retrieved_indices, qrels, data.corpus_ids, qid, dense_lat, dr.candidates_explored)
        dense_metrics_list.append(dm)
        dense_ndcg_list.append(dm.ndcg_at_k.get(10, 0))

        t0 = time.perf_counter()
        fr = retriever_full.retrieve(qid, qe, final_k=eval_k,
                                      task_metadata={"category": "", "query_text": qt})
        full_lat = (time.perf_counter() - t0) * 1000
        full_latencies.append(full_lat)
        fm = evaluator.evaluate_query(fr.retrieved_indices, qrels, data.corpus_ids, qid, full_lat, fr.candidates_explored)
        full_metrics_list.append(fm)
        full_ndcg_list.append(fm.ndcg_at_k.get(10, 0))

        if fr.candidate_attribution:
            source_attr.analyze_query(qid, fr.retrieved_indices, fr.candidate_attribution,
                                       qrels, data.corpus_ids, top_k=10)

        # BM25-only
        bm25_r = bm25_wrap.retrieve(qt, k=eval_k, query_id=qid)
        bm = evaluator.evaluate_query(bm25_r.retrieved_indices, qrels, data.corpus_ids, qid, bm25_r.total_time_ms, len(data.corpus_texts))
        bm25_metrics_list.append(bm)
        bm25_ndcg_list.append(bm.ndcg_at_k.get(10, 0))

        # Features (Dense@50 + BM25@100)
        dr50 = dense_bl.retrieve(qe, query_id=qid, k=50)
        bm25_r100 = bm25_wrap.retrieve(qt, k=100, query_id=qid)
        graph_deg = graph_exp.get_degrees(dr50.retrieved_indices[:10])
        feats = feat_ext.extract(query_id=qid, query_embedding=qe, query_text=qt,
                                  dense_indices=dr50.retrieved_indices, dense_scores=dr50.retrieved_scores,
                                  bm25_indices=bm25_r100.retrieved_indices, bm25_scores=bm25_r100.retrieved_scores,
                                  graph_degrees=graph_deg)
        feature_list.append(feats)

    retriever_full.reset()
    dense_ndcg = np.array(dense_ndcg_list)
    full_ndcg = np.array(full_ndcg_list)
    bm25_ndcg_all = np.array(bm25_ndcg_list)
    easy_mask = dense_ndcg > 0.5

    dense_agg = evaluator.aggregate(dense_metrics_list, "Dense")
    full_agg = evaluator.aggregate(full_metrics_list, "Full AMSR-SE")

    print(f"   Dense mean nDCG@10: {np.mean(dense_ndcg):.4f}")
    print(f"   Full mean nDCG@10:  {np.mean(full_ndcg):.4f}")
    print(f"   Easy: {np.sum(easy_mask)} | Hard: {np.sum(~easy_mask)}")

    # ── Phase 2: Leakage-safe split + Train routers ──
    print("\nPhase 2: Leakage-safe train/val/test split + router training...")
    split = create_stratified_split(dense_ndcg, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3)

    train_data = build_training_data(
        [feature_list[i] for i in split.train_idx],
        dense_ndcg[split.train_idx], full_ndcg[split.train_idx],
        [all_query_ids[i] for i in split.train_idx],
        action_ndcg={0: dense_ndcg[split.train_idx], 1: bm25_ndcg_all[split.train_idx], 4: full_ndcg[split.train_idx]},
    )

    # Train binary routers
    lr_router = LearnedRouter(model_type="logistic"); lr_router.train(train_data)
    rf_router = LearnedRouter(model_type="random_forest"); rf_router.train(train_data)
    gb_router = LearnedRouter(model_type="gradient_boost"); gb_router.train(train_data)
    ca_router = CostAwareRouter(model_type="logistic"); ca_router.train(train_data)
    ca_router.train_stats["avg_hard_gain"] = float(np.mean(np.maximum(full_ndcg[split.train_idx] - dense_ndcg[split.train_idx], 0)[~easy_mask[split.train_idx]])) if np.sum(~easy_mask[split.train_idx]) > 0 else 0.05
    ca_router.train_stats["avg_easy_harm"] = float(np.mean(np.maximum(dense_ndcg[split.train_idx] - full_ndcg[split.train_idx], 0)[easy_mask[split.train_idx]])) if np.sum(easy_mask[split.train_idx]) > 0 else 0.03

    # Multi-action router
    ma_router = MultiActionRouter(model_type="gradient_boost")
    ma_router.train(train_data)

    # Tune thresholds on VAL only
    val_X = np.array([feature_list[i].to_array(FEATURE_NAMES) for i in split.val_idx])
    val_dense, val_full = dense_ndcg[split.val_idx], full_ndcg[split.val_idx]
    lr_router.tune_threshold(val_X, val_dense, val_full)
    rf_router.tune_threshold(val_X, val_dense, val_full)
    gb_router.tune_threshold(val_X, val_dense, val_full)
    ca_router.tune_threshold(val_X, val_dense, val_full)

    oracle = OracleRouter()
    oracle.set_oracle_labels(all_query_ids, dense_ndcg, full_ndcg)
    rule_router = RuleBasedRouter()

    # ── Phase 3: Evaluate on TEST set only ──
    print("\nPhase 3: Evaluating on TEST set only (leakage-safe)...")
    test_idx = split.test_idx
    test_dense = dense_ndcg[test_idx]
    test_full = full_ndcg[test_idx]
    test_easy = easy_mask[test_idx]

    all_results = {
        "Dense": Evaluator.to_dict(evaluator.aggregate([dense_metrics_list[i] for i in test_idx], "Dense")),
        "Full AMSR-SE": Evaluator.to_dict(evaluator.aggregate([full_metrics_list[i] for i in test_idx], "Full AMSR-SE")),
    }
    all_ndcg = {"Dense": test_dense, "Full AMSR-SE": test_full}
    all_safety = {}

    # BM25 (already collected in Phase 1)
    all_results["BM25"] = Evaluator.to_dict(evaluator.aggregate([bm25_metrics_list[i] for i in test_idx], "BM25"))
    all_ndcg["BM25"] = bm25_ndcg_all[test_idx]

    routers = {
        "Rule-based": rule_router, "Learned-LR": lr_router, "Learned-RF": rf_router,
        "Learned-GB": gb_router, "Cost-Aware": ca_router, "Multi-Action": ma_router,
        "Oracle": oracle,
    }

    for rname, router in routers.items():
        router.reset()
        routed_ndcg = np.zeros(len(test_idx))
        routed_lat = np.zeros(len(test_idx))
        routed_metrics = []
        hybrid_count = 0
        for ti, qi in enumerate(test_idx):
            decision = router.route(feature_list[qi])
            if decision.action == Action.DENSE_ONLY:
                routed_ndcg[ti] = test_dense[ti]
                routed_lat[ti] = dense_latencies[qi]
                routed_metrics.append(dense_metrics_list[qi])
            else:
                routed_ndcg[ti] = test_full[ti]
                routed_lat[ti] = full_latencies[qi]
                routed_metrics.append(full_metrics_list[qi])
                hybrid_count += 1
        all_results[rname] = Evaluator.to_dict(evaluator.aggregate(routed_metrics, rname))
        all_ndcg[rname] = routed_ndcg
        safety = compute_safety_metrics(test_dense, routed_ndcg, test_easy, rname)
        safety["pct_routed_hybrid"] = hybrid_count / len(test_idx)
        safety["pct_preserved_dense"] = 1 - hybrid_count / len(test_idx)
        safety["avg_latency_ms"] = float(np.mean(routed_lat))
        all_safety[rname] = safety

    # Random router
    lr_hr = all_safety.get("Learned-LR", {}).get("pct_routed_hybrid", 0.3)
    rand_router = RandomRouter(hybrid_rate=lr_hr)
    rand_ndcg = np.zeros(len(test_idx))
    rand_met = []
    for ti, qi in enumerate(test_idx):
        d = rand_router.route(feature_list[qi])
        if d.action == Action.DENSE_ONLY:
            rand_ndcg[ti] = test_dense[ti]; rand_met.append(dense_metrics_list[qi])
        else:
            rand_ndcg[ti] = test_full[ti]; rand_met.append(full_metrics_list[qi])
    all_results["Random"] = Evaluator.to_dict(evaluator.aggregate(rand_met, "Random"))
    all_ndcg["Random"] = rand_ndcg
    all_safety["Random"] = compute_safety_metrics(test_dense, rand_ndcg, test_easy, "Random")
    all_safety["Full AMSR-SE"] = compute_safety_metrics(test_dense, test_full, test_easy, "Full AMSR-SE")
    all_safety["Full AMSR-SE"]["pct_routed_hybrid"] = 1.0

    # ── Print results ──
    print("\n" + "=" * 70)
    print(f"   RESULTS on TEST SET -- {data.name} (n={len(test_idx)})")
    print("=" * 70)
    print(f"  {'Method':<20} {'nDCG@10':>8} {'SafeGain':>9} {'Hybrid%':>8} {'Losses':>7}")
    print("  " + "-" * 55)
    for m in all_results:
        n10 = all_results[m].get("ndcg_at_k", {}).get("10", all_results[m].get("ndcg_at_k", {}).get(10, 0))
        sg = all_safety.get(m, {}).get("safe_gain", 0)
        hr = all_safety.get(m, {}).get("pct_routed_hybrid", 1.0) * 100
        ls = all_safety.get(m, {}).get("easy_degradation_rate", 0) * all_safety.get(m, {}).get("n_easy", 0)
        print(f"  {m:<20} {n10:>8.4f} {sg:>+9.4f} {hr:>7.1f}% {ls:>7.0f}")

    # ── Phase 4: Pairwise statistical tests ──
    print("\nPhase 4: Pairwise statistical tests (Holm-Bonferroni corrected)...")
    stat_tester = StatisticalTester(n_bootstrap=10000, n_permutation=5000)
    pairwise = stat_tester.pairwise_comparison_matrix(all_ndcg, test_easy)
    # Also individual reports vs Dense
    stat_reports = {}
    for m in all_ndcg:
        if m != "Dense":
            stat_reports[m] = stat_tester.full_comparison(test_dense, all_ndcg[m], "Dense", m, test_easy)
            tt = stat_reports[m].get("paired_ttest", {})
            es = stat_reports[m].get("effect_size", {})
            print(f"   {m}: d={stat_reports[m]['mean_delta']:+.4f}, p={tt.get('p_value',1):.4e}, "
                  f"d={es.get('cohens_d',0):.3f} ({es.get('magnitude','?')}), "
                  f"W/T/L={stat_reports[m]['wins']}/{stat_reports[m]['ties']}/{stat_reports[m]['losses']}")

    # ── Phase 5: Explainability ──
    print("\nPhase 5: Router explainability...")
    explainer = RouterExplainer(FEATURE_NAMES)
    explain_dir = os.path.join(bench_dir, "explainability")
    test_X = np.array([feature_list[i].to_array(FEATURE_NAMES) for i in test_idx])
    best_router = gb_router
    try:
        exp_report = explainer.full_analysis(best_router, test_X, test_dense, test_full, test_easy, explain_dir)
        print(f"   Optimal threshold: {exp_report.optimal_threshold:.2f}")
    except Exception as e:
        print(f"   Explainability error: {e}")

    # ── Phase 6: Source attribution ──
    attr_agg = source_attr.aggregate()

    # ── Phase 7: Plots ──
    plots_dir = os.path.join(bench_dir, "plots")
    print(f"\nPhase 7: Generating plots...")
    from . import visualize_results as viz

    try: viz.plot_pareto_frontier(all_results, plots_dir)
    except Exception as e: print(f"   plot error: {e}")

    best_key = "Learned-GB"
    if best_key in all_ndcg:
        try: viz.plot_hard_query_recovery(test_dense, test_full, all_ndcg[best_key], test_easy, plots_dir, best_key)
        except Exception as e: print(f"   plot error: {e}")
        try: viz.plot_delta_waterfall(test_dense, all_ndcg[best_key], plots_dir, best_key)
        except Exception as e: print(f"   plot error: {e}")

    try: viz.plot_safe_gain(all_safety, plots_dir)
    except Exception as e: print(f"   plot error: {e}")
    if attr_agg:
        try: viz.plot_source_attribution_relevant(attr_agg, plots_dir)
        except Exception as e: print(f"   plot error: {e}")
    if best_key in stat_reports:
        try: viz.plot_win_tie_loss(stat_reports[best_key], plots_dir)
        except Exception as e: print(f"   plot error: {e}")
    try: viz.plot_router_calibration(gb_router.train_stats, plots_dir)
    except Exception as e: print(f"   plot error: {e}")

    # ── Phase 8: Save outputs ──
    metrics_dir = os.path.join(bench_dir, "metrics")
    tables_dir = os.path.join(bench_dir, "tables")

    def _save_json(obj, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    _save_json(all_results, os.path.join(metrics_dir, "aggregate_metrics.json"))
    _save_json(all_safety, os.path.join(metrics_dir, "safety_metrics.json"))
    _save_json(stat_reports, os.path.join(metrics_dir, "statistical_tests.json"))
    _save_json(pairwise, os.path.join(metrics_dir, "pairwise_tests.json"))
    router_stats = {n: r.get_stats() for n, r in routers.items()}
    _save_json(router_stats, os.path.join(metrics_dir, "router_stats.json"))
    if attr_agg:
        _save_json(attr_agg, os.path.join(metrics_dir, "source_attribution.json"))
    _save_json(split.summary(), os.path.join(metrics_dir, "split_info.json"))
    _save_json(reranker.get_stats(), os.path.join(metrics_dir, "gpu_latency_metrics.json"))

    with open(os.path.join(metrics_dir, "per_query_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "method", "ndcg_at_10", "split"])
        for ti, qi in enumerate(test_idx):
            qid = all_query_ids[qi]
            for method, ndcgs in all_ndcg.items():
                w.writerow([qid, method, f"{ndcgs[ti]:.6f}", "test"])

    TableGenerator.generate_main_results(all_results, all_safety, stat_reports, tables_dir)
    TableGenerator.generate_router_table(router_stats, all_safety, tables_dir)

    exp_config = {
        "benchmark": data.name, "num_docs": data.num_docs, "num_queries": data.num_queries,
        "num_test": len(test_idx), "model": model_name, "reranker": reranker.model_name,
        "device": reranker.actual_device, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": split.summary(), "seed": 42, "version": "v4",
    }
    _save_json(exp_config, os.path.join(bench_dir, "configs", "config.json"))

    from .report_generator import generate_final_report
    generate_final_report(
        dataset_name=data.name, results=all_results, safety_metrics=all_safety,
        stat_reports=stat_reports, config=exp_config, router_stats=router_stats,
        source_attr=attr_agg, output_path=os.path.join(bench_dir, "final_report.md"),
    )

    print(f"\nResults saved to {bench_dir}/")
    print("Safe-AMSR-SE v4 experiment complete!")
    return all_results


def main():
    import argparse
    p = argparse.ArgumentParser(description="Safe-AMSR-SE v4")
    p.add_argument("--source", default="beir")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--results-dir", default="results_safe_amsr_v4")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    kwargs = {"dataset_name": args.dataset}
    if args.max_docs: kwargs["max_docs"] = args.max_docs
    if args.max_queries: kwargs["max_queries"] = args.max_queries
    run_safe_experiment(source=args.source, results_dir=args.results_dir, device=args.device, **kwargs)

if __name__ == "__main__":
    main()
