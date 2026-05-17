"""
P-SAFE-AMSR — Master Experiment Runner
"""
import os, json, time, csv
import numpy as np
import matplotlib.pyplot as plt

from .dataset_interface import load_benchmark
from .index_manager import IndexManager
from .graph_expander import GraphExpander
from .hybrid_retriever import HybridRetriever, RetrievalResult
from .baselines import DenseFixedBaseline, BM25Baseline
from .evaluation import Evaluator
from .candidate_fusion import CandidateFusion
from .reranker import CrossEncoderReranker
from .statistical_tests import StatisticalTester
from .feature_extractor import FeatureExtractor, FEATURE_NAMES
from .psafe_router import PSafeRouter, PSafeTrainingData, Action, ACTION_NAMES, ACTION_LATENCY
from .safe_router import RuleBasedRouter, RandomRouter, OracleRouter, compute_safety_metrics
from .leakage_safe_split import create_stratified_split
from .table_generator import TableGenerator
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

class ActionSimulator:
    def __init__(self, retriever: HybridRetriever, bm25_baseline, graph_expander, dense_bl):
        self.retriever = retriever
        self.bm25_baseline = bm25_baseline
        self.graph_expander = graph_expander
        self.dense_bl = dense_bl
        
    def simulate_action(self, action: Action, query_id: str, query_embedding: np.ndarray, query_text: str, k: int=10) -> RetrievalResult:
        t_total = time.perf_counter()
        
        # Dense
        dense_res = self.dense_bl.retrieve(query_embedding, query_id=query_id, k=50 if action != Action.A6_DEEP_HYBRID else 100)
        dense_idx, dense_scores = dense_res.retrieved_indices, dense_res.retrieved_scores
        
        if action == Action.A0_DENSE:
            return RetrievalResult(query_id=query_id, retrieved_indices=dense_idx[:k], retrieved_scores=dense_scores[:k], final_k=k, total_time_ms=(time.perf_counter()-t_total)*1000)
            
        bm25_idx, bm25_scores = np.array([]), np.array([])
        if action in [Action.A1_DENSE_BM25, Action.A3_DENSE_BM25_GRAPH, Action.A4_DENSE_BM25_CE, Action.A5_DENSE_BM25_GRAPH_CE, Action.A6_DEEP_HYBRID]:
            b_res = self.bm25_baseline.retrieve(query_text, k=100 if action != Action.A6_DEEP_HYBRID else 200)
            bm25_idx, bm25_scores = b_res.retrieved_indices, b_res.retrieved_scores
            
        graph_idx, graph_scores = np.array([]), np.array([])
        if action in [Action.A2_DENSE_GRAPH, Action.A3_DENSE_BM25_GRAPH, Action.A5_DENSE_BM25_GRAPH_CE, Action.A6_DEEP_HYBRID]:
            seed_k = 10 if action != Action.A6_DEEP_HYBRID else 20
            graph_idx, graph_scores = self.graph_expander.expand(
                dense_idx[:seed_k], dense_scores[:seed_k], query_embedding, self.retriever.all_embeddings, 
                hops=1 if action != Action.A6_DEEP_HYBRID else 2, max_neighbors=10)
                
        fused_idx, fused_scores, _ = CandidateFusion.fuse(
            dense_idx, dense_scores, bm25_idx, bm25_scores, graph_idx, graph_scores, 
            max_candidates=200 if action != Action.A6_DEEP_HYBRID else 400
        )
        
        if action in [Action.A4_DENSE_BM25_CE, Action.A5_DENSE_BM25_GRAPH_CE, Action.A6_DEEP_HYBRID]:
            rerank_depth = 50 if action != Action.A6_DEEP_HYBRID else 100
            rerank_idx = fused_idx[:rerank_depth]
            rerank_scores = fused_scores[:rerank_depth]
            final_idx, final_scores = self.retriever._cross_encode_rerank(query_text, rerank_idx, rerank_scores)
            fused_idx = np.concatenate([final_idx, fused_idx[rerank_depth:]])
            fused_scores = np.concatenate([final_scores, fused_scores[rerank_depth:]])
            
        return RetrievalResult(query_id=query_id, retrieved_indices=fused_idx[:k], retrieved_scores=fused_scores[:k], final_k=k, total_time_ms=(time.perf_counter()-t_total)*1000)

class BM25Wrap:
    def __init__(self, bm25, texts):
        self.bm25, self.texts = bm25, texts
    def retrieve(self, query_text, k=10, query_id=""):
        from .baselines import BaselineResult
        scores = self.bm25.get_scores(query_text.lower().split())
        top = np.argsort(-scores)[:k]
        return BaselineResult(query_id=query_id, retrieved_indices=top, retrieved_scores=scores[top].astype(np.float32), total_time_ms=0, candidates_explored=len(scores), method="bm25")

def run_psafe_experiment(source="beir", results_dir="results_psafe_amsr",
                         model_name="BAAI/bge-m3", device="auto", **loader_kwargs):
    print("=" * 70)
    print("   P-SAFE-AMSR — Probabilistic Safety-Aware Router Experiment")
    print("=" * 70)

    loader_kwargs.setdefault("dataset_name", "scifact")
    ds_name = loader_kwargs.get("dataset_name", source)
    bench_dir = os.path.join(results_dir, ds_name)
    for sub in ["plots", "metrics", "configs", "tables"]:
        os.makedirs(os.path.join(bench_dir, sub), exist_ok=True)

    # Load data
    data = load_benchmark(source, **loader_kwargs)

    # VRAM-safe encoding with caching
    from .vram_safe_encoder import encode_texts
    from .embedding_cache import EmbeddingCache

    cache = EmbeddingCache(os.path.join(results_dir, "cache", "embeddings"))

    corpus_emb = encode_texts(
        model_name=model_name,
        texts=data.corpus_texts,
        dataset_name=ds_name,
        kind="corpus",
        device=device,
        cache=cache,
    )
    query_emb = encode_texts(
        model_name=model_name,
        texts=data.query_texts,
        dataset_name=ds_name,
        kind="query",
        device=device,
        cache=cache,
    )

    config = AHRCConfig()
    config.index.embedding_dim = corpus_emb.shape[1]
    index_mgr = IndexManager(config.index)
    index_mgr.build(corpus_emb)

    graph_exp = GraphExpander(config.adaptive)
    _build_knn_graph(graph_exp, corpus_emb, k=5)

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([t.lower().split() for t in data.corpus_texts])
    bm25_wrap = BM25Wrap(bm25, data.corpus_texts)

    reranker = CrossEncoderReranker(device=device)
    reranker.load()

    evaluator = Evaluator(k_values=[10], relevance_threshold=1)
    feat_ext = FeatureExtractor()
    feat_ext.build_idf(data.corpus_texts)
    dense_bl = DenseFixedBaseline(index_mgr, corpus_emb)
    retriever_full = HybridRetriever(config, index_mgr, graph_exp, corpus_emb, bm25_baseline=bm25_wrap, task_texts=data.corpus_texts, reranker=reranker)
    simulator = ActionSimulator(retriever_full, bm25_wrap, graph_exp, dense_bl)
    
    print("\nPhase 1: Collecting per-query metrics for A0-A6...")
    n_queries = min(data.num_queries, loader_kwargs.get("max_queries", data.num_queries) or data.num_queries)
    action_ndcg = {a.value: np.zeros(n_queries) for a in Action}
    action_latency = {a.value: np.zeros(n_queries) for a in Action}
    feature_list, all_query_ids = [], []
    action_metrics_list = {a.value: [] for a in Action}
    
    for qi in range(n_queries):
        qid, qe, qt = data.query_ids[qi], query_emb[qi], data.query_texts[qi]
        qrels = data.qrels.get(qid, {})
        all_query_ids.append(qid)
        
        # Features
        dr50 = dense_bl.retrieve(qe, query_id=qid, k=50)
        bm25_r100 = bm25_wrap.retrieve(qt, k=100, query_id=qid)
        graph_deg = graph_exp.get_degrees(dr50.retrieved_indices[:10])
        feats = feat_ext.extract(query_id=qid, query_embedding=qe, query_text=qt,
                                  dense_indices=dr50.retrieved_indices, dense_scores=dr50.retrieved_scores,
                                  bm25_indices=bm25_r100.retrieved_indices, bm25_scores=bm25_r100.retrieved_scores,
                                  graph_degrees=graph_deg)
        feature_list.append(feats)
        
        for a in Action:
            res = simulator.simulate_action(a, qid, qe, qt, k=10)
            dm = evaluator.evaluate_query(res.retrieved_indices, qrels, data.corpus_ids, qid, res.total_time_ms, res.candidates_explored)
            action_ndcg[a.value][qi] = dm.ndcg_at_k.get(10, 0)
            action_latency[a.value][qi] = res.total_time_ms
            action_metrics_list[a.value].append(dm)
            
        # Graph contribution tracking
        if qi == 0:
            graph_stats = {"graph_only_candidate_count": 0, "graph_only_relevant_count": 0, "graph_unique_relevant_docs": 0, "graph_final_top10_relevant_docs": 0, "dense_graph_overlap_10": 0, "dense_graph_overlap_50": 0}
        
        g_res = simulator.simulate_action(Action.A2_DENSE_GRAPH, qid, qe, qt, k=50)
        
        # calculate overlap
        d_idx = set(dr50.retrieved_indices[:10])
        g_idx_10 = set(g_res.retrieved_indices[:10])
        g_idx_50 = set(g_res.retrieved_indices[:50])
        graph_stats["dense_graph_overlap_10"] += len(d_idx & g_idx_10) / 10 if len(g_idx_10) else 0
        graph_stats["dense_graph_overlap_50"] += len(set(dr50.retrieved_indices[:50]) & g_idx_50) / 50 if len(g_idx_50) else 0
        
        rel_docs = {i for i, r in qrels.items() if r > 0}
        rel_indices = {i for i, doc_id in enumerate(data.corpus_ids) if doc_id in rel_docs}
        
        graph_only_docs = g_idx_50 - set(dr50.retrieved_indices[:50])
        graph_stats["graph_only_candidate_count"] += len(graph_only_docs)
        
        graph_only_rel = graph_only_docs & rel_indices
        graph_stats["graph_only_relevant_count"] += len(graph_only_rel)
        graph_stats["graph_unique_relevant_docs"] += len(graph_only_rel)
        
        graph_stats["graph_final_top10_relevant_docs"] += len(g_idx_10 & rel_indices)
            
    dense_ndcg = action_ndcg[Action.A0_DENSE.value]
    easy_mask = dense_ndcg > 0.5

    print(f"   Dense mean nDCG@10: {np.mean(dense_ndcg):.4f}")
    
    print("\nPhase 2: Leakage-safe train/val/test split + router training...")
    split = create_stratified_split(dense_ndcg, train_ratio=0.4, val_ratio=0.1, test_ratio=0.5)

    train_data = PSafeTrainingData(
        feature_matrix=np.array([feature_list[i].to_array(FEATURE_NAMES) for i in split.train_idx]),
        action_ndcg={a: action_ndcg[a][split.train_idx] for a in action_ndcg},
        action_latency={a: action_latency[a][split.train_idx] for a in action_latency},
        query_ids=[all_query_ids[i] for i in split.train_idx],
        feature_names=FEATURE_NAMES
    )

    psafe = PSafeRouter()
    psafe.train(train_data)

    print("\nPhase 3: Evaluating on TEST set...")
    test_idx = split.test_idx
    test_dense = dense_ndcg[test_idx]
    test_easy = easy_mask[test_idx]

    base_results = {}
    base_ndcg = {}
    base_safety = {}

    for a in Action:
        base_results[ACTION_NAMES[a]] = Evaluator.to_dict(evaluator.aggregate([action_metrics_list[a.value][i] for i in test_idx], ACTION_NAMES[a]))
        base_ndcg[ACTION_NAMES[a]] = action_ndcg[a.value][test_idx]
        base_safety[ACTION_NAMES[a]] = compute_safety_metrics(test_dense, action_ndcg[a.value][test_idx], test_easy, ACTION_NAMES[a])

    dense_n10 = np.mean(test_dense)
    best_single_action = max(Action, key=lambda a: np.mean(action_ndcg[a.value][test_idx]))
    best_single_n10 = np.mean(action_ndcg[best_single_action.value][test_idx])

    modes = {
        "Loose": {
            "gain_threshold": 0.30, "harm_threshold": 0.60, 
            "lambda_lat": 0.000001, "lambda_harm": 0.02, 
            "use_lcb_safety": False, "min_hybrid_rate": 0.10
        },
        "Balanced": {
            "gain_threshold": 0.40, "harm_threshold": 0.45, 
            "lambda_lat": 0.00001, "lambda_harm": 0.05, 
            "use_lcb_safety": False, "min_hybrid_rate": 0.0
        },
        "Strict": {
            "gain_threshold": 0.60, "harm_threshold": 0.20, 
            "lambda_lat": 0.0001, "lambda_harm": 0.10, 
            "use_lcb_safety": True, "min_hybrid_rate": 0.0
        }
    }

    from .report_generator import generate_final_report

    for mode_name, kwargs in modes.items():
        print(f"\n{'='*40}")
        print(f"   Evaluating {mode_name} P-SAFE")
        print(f"{'='*40}")
        
        mode_dir = os.path.join(bench_dir, mode_name)
        for sub in ["plots", "metrics", "configs", "tables"]: os.makedirs(os.path.join(mode_dir, sub), exist_ok=True)
        
        # Oracle Router evaluation for this mode
        oracle_ndcg = np.zeros(len(test_idx))
        oracle_actions = []
        for ti, qi in enumerate(test_idx):
            best_a = Action.A0_DENSE
            best_util = 0.0
            best_n = test_dense[ti]
            for a in Action:
                a_ndcg = action_ndcg[a.value][qi]
                a_lat = action_latency[a.value][qi]
                delta_n = a_ndcg - test_dense[ti]
                harm_label = 1 if delta_n < -0.01 else 0
                utility = delta_n - kwargs["lambda_lat"] * a_lat - kwargs["lambda_harm"] * harm_label
                if utility > best_util or (utility == best_util and a == Action.A0_DENSE):
                    best_util = utility
                    best_a = a
                    best_n = a_ndcg
            oracle_ndcg[ti] = best_n
            oracle_actions.append(best_a)
            
        oracle_n10 = np.mean(oracle_ndcg)
        oracle_dist = {ACTION_NAMES[a]: oracle_actions.count(a) for a in Action}
        
        psafe = PSafeRouter(**kwargs)
        psafe.train(train_data)
        
        psafe_ndcg = np.zeros(len(test_idx))
        psafe_lat = np.zeros(len(test_idx))
        psafe_met = []
        action_preds = []
        
        hybrid_count = 0
        for ti, qi in enumerate(test_idx):
            decision = psafe.route(feature_list[qi], current_hybrid_rate=hybrid_count / max(ti, 1))
            if decision.action != Action.A0_DENSE: hybrid_count += 1
            psafe_ndcg[ti] = action_ndcg[decision.action.value][qi]
            psafe_lat[ti] = action_latency[decision.action.value][qi]
            psafe_met.append(action_metrics_list[decision.action.value][qi])
            action_preds.append({
                "query_id": all_query_ids[qi],
                "query_text": data.query_texts[qi],
                "dense_ndcg": test_dense[ti],
                "selected_action": ACTION_NAMES[decision.action],
                "selected_action_ndcg": psafe_ndcg[ti],
                "selected_action_latency": psafe_lat[ti],
                "selected_action_delta_ndcg": psafe_ndcg[ti] - test_dense[ti],
                "oracle_action": ACTION_NAMES[oracle_actions[ti]],
                "oracle_action_ndcg": oracle_ndcg[ti],
                "oracle_delta_ndcg": oracle_ndcg[ti] - test_dense[ti],
                "P_gain_for_each_action": str(decision.action_p_gain),
                "P_harm_for_each_action": str(decision.action_p_harm),
                "predicted_delta_ndcg_for_each_action": str(decision.action_pred_delta),
                "predicted_latency_for_each_action": str(decision.action_pred_lat),
                "expected_utility_for_each_action": str(decision.action_utilities),
                "rejected_action_reasons": str(decision.rejected_reasons),
                "final_decision_reason": decision.final_decision_reason
            })
            
        # Overtreatment
        hybrid_ndcg = action_ndcg[Action.A6_DEEP_HYBRID.value][test_idx]
        hybrid_lat = action_latency[Action.A6_DEEP_HYBRID.value][test_idx]
        easy_idx = np.where(test_easy)[0]
        hard_idx = np.where(~test_easy)[0]
        
        hybrid_worse = hybrid_ndcg[easy_idx] < test_dense[easy_idx]
        hybrid_overtreatment_rate = float(np.mean(hybrid_worse)) if len(easy_idx) > 0 else 0.0
        hybrid_overtreatment_severity = float(np.mean(test_dense[easy_idx] - hybrid_ndcg[easy_idx])) if len(easy_idx) > 0 else 0.0
        
        psafe_avoided = (psafe_ndcg[easy_idx] >= test_dense[easy_idx]) & hybrid_worse
        psafe_overtreatment_avoidance_rate = float(np.sum(psafe_avoided) / np.sum(hybrid_worse)) if np.sum(hybrid_worse) > 0 else 1.0
        
        latency_avoidance = float(1.0 - np.mean(psafe_lat) / np.mean(hybrid_lat)) if np.mean(hybrid_lat) > 0 else 0.0
        
        hybrid_easy_deg = -float(np.mean(np.minimum(hybrid_ndcg[easy_idx] - test_dense[easy_idx], 0))) if len(easy_idx) > 0 else 0.0
        psafe_easy_deg = -float(np.mean(np.minimum(psafe_ndcg[easy_idx] - test_dense[easy_idx], 0))) if len(easy_idx) > 0 else 0.0
        harm_avoidance = hybrid_easy_deg - psafe_easy_deg
        
        overtreatment_metrics = {
            "easy_query_count": len(easy_idx),
            "hard_query_count": len(hard_idx),
            "hybrid_overtreatment_rate": hybrid_overtreatment_rate,
            "hybrid_overtreatment_severity": hybrid_overtreatment_severity,
            "psafe_overtreatment_avoidance_rate": psafe_overtreatment_avoidance_rate,
            "latency_avoidance": latency_avoidance,
            "harm_avoidance": harm_avoidance
        }
        
        # Graph metrics
        graph_stats_test = graph_stats.copy()
        for k in ["dense_graph_overlap_10", "dense_graph_overlap_50"]:
            graph_stats_test[k] = graph_stats_test[k] / n_queries
            
        g_actions = [d.action for d in psafe.decisions if "GRAPH" in ACTION_NAMES[d.action]]
        graph_stats_test["graph_action_selected_count"] = len(g_actions)
        
        wins = 0
        losses = 0
        for ti, d in enumerate(psafe.decisions):
            if "GRAPH" in ACTION_NAMES[d.action]:
                qi = test_idx[ti]
                if action_ndcg[d.action.value][qi] > test_dense[ti] + 0.01:
                    wins += 1
                elif action_ndcg[d.action.value][qi] < test_dense[ti] - 0.01:
                    losses += 1
        graph_stats_test["graph_action_win_count"] = wins
        graph_stats_test["graph_action_loss_count"] = losses
            
        all_results = base_results.copy()
        all_results["P-SAFE-AMSR"] = Evaluator.to_dict(evaluator.aggregate(psafe_met, "P-SAFE-AMSR"))
        
        all_ndcg = base_ndcg.copy()
        all_ndcg["P-SAFE-AMSR"] = psafe_ndcg
        
        all_safety = base_safety.copy()
        all_safety["P-SAFE-AMSR"] = compute_safety_metrics(test_dense, psafe_ndcg, test_easy, "P-SAFE-AMSR")
        all_safety["P-SAFE-AMSR"]["avg_latency_ms"] = float(np.mean(psafe_lat))
        
        stats = psafe.get_stats()
        all_safety["P-SAFE-AMSR"]["pct_routed_hybrid"] = stats.get('hybrid_rate', 0.0)
        
        saf = all_safety["P-SAFE-AMSR"]
        stats = psafe.get_stats()
        
        print(f"Dense nDCG@10: {dense_n10:.4f}")
        print(f"Best single action nDCG@10 ({ACTION_NAMES[best_single_action]}): {best_single_n10:.4f}")
        print(f"P-SAFE nDCG@10: {np.mean(psafe_ndcg):.4f}")
        print(f"Oracle nDCG@10: {oracle_n10:.4f}")
        print(f"Hybrid activation rate: {stats.get('hybrid_rate', 0)*100:.1f}%")
        print(f"Missed hard query rate: {saf.get('missed_hard_query_rate', 0)*100:.1f}%")
        print(f"Easy-query harm: {saf.get('easy_degradation_mean', 0):.4f}")
        print(f"Hard-query gain: {saf.get('hard_gain_mean', 0):.4f}")
        print(f"Safe gain: {saf.get('safe_gain', 0):.4f}")
        print(f"Router action distribution: {stats.get('action_distribution', {})}")
        print(f"Rejected action reasons: {stats.get('rejected_counts', {})}")
        
        metrics_dir = os.path.join(mode_dir, "metrics")
        with open(os.path.join(metrics_dir, "aggregate_metrics.json"), "w", encoding="utf-8") as f: json.dump(all_results, f, indent=2)
        with open(os.path.join(metrics_dir, "safety_metrics.json"), "w", encoding="utf-8") as f: json.dump(all_safety, f, indent=2)
        with open(os.path.join(metrics_dir, "oracle_action_distribution.json"), "w", encoding="utf-8") as f: json.dump(oracle_dist, f, indent=2)
        with open(os.path.join(metrics_dir, "rejected_action_reasons.json"), "w", encoding="utf-8") as f: json.dump(stats.get('rejected_counts', {}), f, indent=2)
        with open(os.path.join(metrics_dir, "overtreatment_metrics.json"), "w", encoding="utf-8") as f: json.dump(overtreatment_metrics, f, indent=2)
        with open(os.path.join(metrics_dir, "graph_contribution.json"), "w", encoding="utf-8") as f: json.dump(graph_stats_test, f, indent=2)
        
        # Probability calibration
        from sklearn.metrics import brier_score_loss
        from sklearn.calibration import calibration_curve
        
        prob_cal = {}
        for a in Action:
            if a == Action.A0_DENSE or a.value not in psafe.models: continue
            m = psafe.models[a.value]
            if not m['p_gain'] or not m['p_harm']: continue
            
            X_test = np.array([feature_list[i].to_array(FEATURE_NAMES) for i in test_idx])
            X_s = psafe.scaler.transform(X_test)
            
            p_g_preds = m['p_gain'].predict_proba(X_s)[:, 1]
            p_h_preds = m['p_harm'].predict_proba(X_s)[:, 1]
            
            delta = action_ndcg[a.value][test_idx] - test_dense
            y_gain = (delta > psafe.epsilon_gain).astype(int)
            y_harm = (delta < -psafe.epsilon_harm).astype(int)
            
            if len(np.unique(y_gain)) > 1:
                prob_cal[f"{ACTION_NAMES[a]}_gain_brier"] = float(brier_score_loss(y_gain, p_g_preds))
                frac_pos, mean_pred = calibration_curve(y_gain, p_g_preds, n_bins=5)
                prob_cal[f"{ACTION_NAMES[a]}_gain_cal_curve"] = {"frac_pos": frac_pos.tolist(), "mean_pred": mean_pred.tolist()}
                prob_cal[f"{ACTION_NAMES[a]}_gain_ece"] = float(np.mean(np.abs(frac_pos - mean_pred)))
            
            if len(np.unique(y_harm)) > 1:
                prob_cal[f"{ACTION_NAMES[a]}_harm_brier"] = float(brier_score_loss(y_harm, p_h_preds))
                frac_pos, mean_pred = calibration_curve(y_harm, p_h_preds, n_bins=5)
                prob_cal[f"{ACTION_NAMES[a]}_harm_cal_curve"] = {"frac_pos": frac_pos.tolist(), "mean_pred": mean_pred.tolist()}
                prob_cal[f"{ACTION_NAMES[a]}_harm_ece"] = float(np.mean(np.abs(frac_pos - mean_pred)))

        with open(os.path.join(metrics_dir, "probability_calibration.json"), "w", encoding="utf-8") as f: json.dump(prob_cal, f, indent=2)
        
        import csv
        with open(os.path.join(metrics_dir, "action_predictions.csv"), "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=action_preds[0].keys())
            writer.writeheader()
            writer.writerows(action_preds)
            
        with open(os.path.join(metrics_dir, "per_query_metrics.csv"), "w", newline='', encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "method", "ndcg_at_10"])
            for ti, qi in enumerate(test_idx):
                w.writerow([all_query_ids[qi], "Dense", f"{test_dense[ti]:.6f}"])
                w.writerow([all_query_ids[qi], "P-SAFE-AMSR", f"{psafe_ndcg[ti]:.6f}"])
                
        # Statistical Tests
        stat_tester = StatisticalTester(n_bootstrap=1000)
        stat_rep_dense = stat_tester.full_comparison(test_dense, psafe_ndcg, "Dense", "P-SAFE-AMSR", test_easy)
        stat_rep_ce = stat_tester.full_comparison(action_ndcg[Action.A4_DENSE_BM25_CE.value][test_idx], psafe_ndcg, "Dense+BM25+CE", "P-SAFE-AMSR", test_easy)
        stat_rep_dh = stat_tester.full_comparison(action_ndcg[Action.A6_DEEP_HYBRID.value][test_idx], psafe_ndcg, "Deep Hybrid", "P-SAFE-AMSR", test_easy)
        stat_rep_oracle = stat_tester.full_comparison(oracle_ndcg, psafe_ndcg, "Oracle", "P-SAFE-AMSR", test_easy)
        
        stat_reports = {
            "P-SAFE vs Dense": stat_rep_dense,
            "P-SAFE vs Dense+BM25+CE": stat_rep_ce,
            "P-SAFE vs Deep Hybrid": stat_rep_dh,
            "P-SAFE vs Oracle": stat_rep_oracle
        }
        
        with open(os.path.join(metrics_dir, "statistical_tests.json"), "w", encoding="utf-8") as f: json.dump(stat_reports, f, indent=2)
        
        with open(os.path.join(metrics_dir, "action_distribution.json"), "w", encoding="utf-8") as f: json.dump(stats.get('action_distribution', {}), f, indent=2)
        
        config_dict = {
            "num_docs": len(data.corpus_texts),
            "num_queries": data.num_queries,
            "n_train": len(split.train_idx),
            "n_val": len(split.val_idx),
            "n_test": len(split.test_idx),
            "model": model_name,
            "reranker": "BAAI/bge-reranker-v2-m3",
            "seed": 42,
            "device": device,
            "mode": mode_name
        }
        
        generate_final_report(data.name, all_results, all_safety, stat_reports, config_dict, {"P-SAFE": stats}, output_path=os.path.join(mode_dir, "final_report.md"))
        
        from .psafe_visualize import generate_all_plots
        plots_dir = os.path.join(mode_dir, "plots")
        generate_all_plots(all_results, all_ndcg, all_safety, psafe, plots_dir, test_dense, action_ndcg[Action.A4_DENSE_BM25_CE.value][test_idx], test_easy)

    print(f"\nResults saved to {bench_dir}/")
    return base_results
    
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="beir")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-queries", type=int, default=100) # Quick run by default
    args = p.parse_args()
    run_psafe_experiment(source=args.source, dataset_name=args.dataset, max_queries=args.max_queries)
