import numpy as np
import scipy.stats as stats
from typing import List, Dict

FEATURE_NAMES = [
    "query_length", "query_max_idf", "query_mean_idf", "query_sum_idf",
    "dense_score_max", "dense_score_mean", "dense_score_std", 
    "dense_entropy_norm", "dense_score_gap_1_5", "dense_score_gap_1_10",
    "bm25_score_max_norm", "bm25_score_mean_norm", "bm25_score_std",
    "bm25_entropy_norm", "bm25_score_gap_1_5",
    "dense_bm25_overlap_10", "dense_bm25_overlap_50",
    "dense_bm25_rank_correlation", "candidate_novelty", "source_disagreement_score",
    "graph_degree_max", "graph_degree_mean", "graph_degree_zero_frac"
]

class QueryFeatures:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def to_array(self, feature_names: List[str]) -> np.ndarray:
        return np.array([getattr(self, fn, 0.0) for fn in feature_names], dtype=np.float32)

RoutingFeatures = QueryFeatures

class FeatureExtractor:
    def __init__(self):
        self.idf_dict = {}
        
    def build_idf(self, corpus_texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=50000)
        try:
            vec.fit(corpus_texts)
            self.idf_dict = dict(zip(vec.get_feature_names_out(), vec.idf_))
        except ValueError:
            pass # fallback if corpus is empty or invalid
        
    def _entropy(self, scores: np.ndarray, norm_k: int = None) -> float:
        if len(scores) == 0: return 0.0
        scores = np.asarray(scores, dtype=np.float64)
        if np.all(scores <= 0): return 0.0
        
        # Softmax
        scores = scores - np.max(scores)
        exp_s = np.exp(scores)
        p = exp_s / np.sum(exp_s)
        
        ent = -np.sum(p * np.log(p + 1e-9))
        
        # Normalize by log(k)
        if norm_k and norm_k > 1:
            ent = ent / np.log(norm_k)
            
        return float(ent)

    def extract(self, query_id: str, query_embedding: np.ndarray, query_text: str,
                dense_indices: np.ndarray, dense_scores: np.ndarray,
                bm25_indices: np.ndarray, bm25_scores: np.ndarray,
                graph_degrees: List[int]) -> QueryFeatures:
        
        # Lexical / IDF
        words = query_text.lower().split()
        q_len = len(words)
        idfs = [self.idf_dict.get(w, 0.0) for w in words]
        q_max_idf = float(np.max(idfs)) if idfs else 0.0
        q_mean_idf = float(np.mean(idfs)) if idfs else 0.0
        q_sum_idf = float(np.sum(idfs)) if idfs else 0.0

        # Dense Features
        k_d = len(dense_scores)
        d_max = float(np.max(dense_scores)) if k_d > 0 else 0.0
        d_mean = float(np.mean(dense_scores)) if k_d > 0 else 0.0
        d_std = float(np.std(dense_scores)) if k_d > 0 else 0.0
        d_ent = self._entropy(dense_scores, norm_k=k_d)
        d_gap15 = float(dense_scores[0] - dense_scores[4]) if k_d >= 5 else 0.0
        d_gap110 = float(dense_scores[0] - dense_scores[9]) if k_d >= 10 else 0.0
        
        # BM25 Features with Normalization
        k_b = len(bm25_scores)
        if k_b > 0:
            b_max_raw = float(np.max(bm25_scores))
            # Rough BM25 normalizer (since BM25 has no upper bound, dividing by max or a constant)
            norm_factor = max(b_max_raw, 1.0)
            b_norm_scores = bm25_scores / norm_factor
            
            b_max = float(np.max(b_norm_scores))
            b_mean = float(np.mean(b_norm_scores))
            b_std = float(np.std(b_norm_scores))
            b_ent = self._entropy(b_norm_scores, norm_k=k_b)
            b_gap15 = float(b_norm_scores[0] - b_norm_scores[4]) if k_b >= 5 else 0.0
        else:
            b_max, b_mean, b_std, b_ent, b_gap15 = 0.0, 0.0, 0.0, 0.0, 0.0
            
        # Overlap & Correlation (Jaccard)
        set_d10 = set(dense_indices[:10])
        set_b10 = set(bm25_indices[:10])
        overlap_10 = len(set_d10 & set_b10) / len(set_d10 | set_b10) if len(set_d10 | set_b10) > 0 else 0.0
        
        set_d50 = set(dense_indices[:50])
        set_b50 = set(bm25_indices[:50])
        overlap_50 = len(set_d50 & set_b50) / len(set_d50 | set_b50) if len(set_d50 | set_b50) > 0 else 0.0
        
        # Rank correlation (Kendall tau)
        shared = list(set_d50 & set_b50)
        rank_corr = 0.0
        if len(shared) > 1:
            d_ranks = [np.where(dense_indices == idx)[0][0] for idx in shared]
            b_ranks = [np.where(bm25_indices == idx)[0][0] for idx in shared]
            tau, _ = stats.kendalltau(d_ranks, b_ranks)
            rank_corr = float(tau) if not np.isnan(tau) else 0.0
            
        candidate_novelty = 1.0 - overlap_50
        
        # Normalized source disagreement to [0, 1]
        # rank_corr is [-1, 1], so (1 - rank_corr) is [0, 2]
        source_disagreement_score = (1.0 - overlap_10) * ((1.0 - rank_corr) / 2.0)
        
        # Graph Features - numpy safe
        graph_degrees = np.asarray(graph_degrees)
        if graph_degrees.size > 0:
            g_max = float(np.max(graph_degrees))
            g_mean = float(np.mean(graph_degrees))
            g_z_frac = float(np.sum(graph_degrees == 0) / len(graph_degrees))
        else:
            g_max, g_mean, g_z_frac = 0.0, 0.0, 0.0

        return QueryFeatures(
            query_length=q_len,
            query_max_idf=q_max_idf,
            query_mean_idf=q_mean_idf,
            query_sum_idf=q_sum_idf,
        
            dense_score_max=d_max,
            dense_score_mean=d_mean,
            dense_score_std=d_std,
            dense_entropy_norm=d_ent,
            dense_score_gap_1_5=d_gap15,
            dense_score_gap_1_10=d_gap110,
            
            bm25_score_max_norm=b_max,
            bm25_score_mean_norm=b_mean,
            bm25_score_std=b_std,
            bm25_entropy_norm=b_ent,
            bm25_score_gap_1_5=b_gap15,
            
            dense_bm25_overlap_10=overlap_10,
            dense_bm25_overlap_50=overlap_50,
            dense_bm25_rank_correlation=rank_corr,
            candidate_novelty=candidate_novelty,
            source_disagreement_score=source_disagreement_score,
            
            graph_degree_max=g_max,
            graph_degree_mean=g_mean,
            graph_degree_zero_frac=g_z_frac
        )
