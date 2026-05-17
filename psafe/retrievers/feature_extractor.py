"""
B-P-SAFE-AMSR — Canonical Feature Extractor
Extracts per-query routing features from dense/BM25/graph signals.
"""
import numpy as np
import scipy.stats as stats
import json
import os
import re
from typing import List, Dict

FEATURE_NAMES = [
    "query_length_tokens", "query_has_number", "query_has_uppercase_abbreviation",
    "query_avg_token_idf", "query_max_token_idf", "lexical_specificity_score",
    "dense_score_max", "dense_score_mean", "dense_score_std",
    "dense_entropy_norm", "dense_score_gap_1_5", "dense_score_gap_1_10",
    "bm25_score_max_norm", "bm25_score_mean_norm", "bm25_score_std",
    "bm25_entropy_norm", "bm25_score_gap_1_5",
    "bm25_dense_overlap_jaccard_10", "bm25_dense_overlap_jaccard_50",
    "dense_bm25_rank_correlation", "candidate_novelty", "source_disagreement_score",
    "graph_degree_max", "graph_degree_mean", "graph_degree_zero_frac",
]


class QueryFeatures:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_array(self, feature_names: List[str]) -> np.ndarray:
        return np.array([getattr(self, fn, 0.0) for fn in feature_names], dtype=np.float32)


# Backward compatibility alias
RoutingFeatures = QueryFeatures


class FeatureExtractor:
    def __init__(self):
        self.idf_dict = {}

    def build_idf(self, corpus_texts):
        """Build IDF dictionary from corpus using TF-IDF vectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=50000)
        try:
            vec.fit(corpus_texts)
            self.idf_dict = dict(zip(vec.get_feature_names_out(), vec.idf_))
        except ValueError:
            pass  # fallback if corpus is empty or invalid

    def save_feature_schema(self, out_dir: str):
        """Save feature schema to JSON."""
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "feature_schema.json"), "w") as f:
            json.dump({
                "feature_names": FEATURE_NAMES,
                "n_features": len(FEATURE_NAMES),
                "idf_vocab_size": len(self.idf_dict),
            }, f, indent=4)

    def _entropy(self, scores: np.ndarray, norm_k: int = None) -> float:
        if len(scores) == 0:
            return 0.0
        scores = np.asarray(scores, dtype=np.float64)
        if np.all(scores <= 0):
            return 0.0
        scores = scores - np.max(scores)
        exp_s = np.exp(scores)
        p = exp_s / np.sum(exp_s)
        ent = -np.sum(p * np.log(p + 1e-9))
        if norm_k and norm_k > 1:
            ent = ent / np.log(norm_k)
        return float(ent)

    def extract(self, query_id: str, query_embedding: np.ndarray, query_text: str,
                dense_indices: np.ndarray, dense_scores: np.ndarray,
                bm25_indices: np.ndarray, bm25_scores: np.ndarray,
                graph_degrees) -> QueryFeatures:

        # ── Query lexical features ──
        words = query_text.lower().split()
        q_len = len(words)
        idfs = [self.idf_dict.get(w, 0.0) for w in words]
        q_avg_idf = float(np.mean(idfs)) if idfs else 0.0
        q_max_idf = float(np.max(idfs)) if idfs else 0.0

        q_has_number = 1.0 if any(c.isdigit() for c in query_text) else 0.0
        q_has_abbrev = 1.0 if re.search(r'[A-Z]{2,}', query_text) else 0.0

        # Lexical specificity: high IDF words / total words
        if idfs and q_max_idf > 0 and self.idf_dict:
            median_idf = float(np.median(list(self.idf_dict.values())))
            high_idf_count = sum(1 for idf in idfs if idf > median_idf)
            lexical_spec = high_idf_count / max(len(idfs), 1)
        else:
            lexical_spec = 0.0

        # ── Dense features ──
        k_d = len(dense_scores)
        d_max = float(np.max(dense_scores)) if k_d > 0 else 0.0
        d_mean = float(np.mean(dense_scores)) if k_d > 0 else 0.0
        d_std = float(np.std(dense_scores)) if k_d > 0 else 0.0
        d_ent = self._entropy(dense_scores, norm_k=k_d)
        d_gap15 = float(dense_scores[0] - dense_scores[4]) if k_d >= 5 else 0.0
        d_gap110 = float(dense_scores[0] - dense_scores[9]) if k_d >= 10 else 0.0

        # ── BM25 features (normalized) ──
        k_b = len(bm25_scores)
        if k_b > 0:
            b_max_raw = float(np.max(bm25_scores))
            norm_factor = max(b_max_raw, 1.0)
            b_norm = bm25_scores / norm_factor
            b_max = float(np.max(b_norm))
            b_mean = float(np.mean(b_norm))
            b_std = float(np.std(b_norm))
            b_ent = self._entropy(b_norm, norm_k=k_b)
            b_gap15 = float(b_norm[0] - b_norm[4]) if k_b >= 5 else 0.0
        else:
            b_max, b_mean, b_std, b_ent, b_gap15 = 0.0, 0.0, 0.0, 0.0, 0.0

        # ── Overlap & Correlation (Jaccard) ──
        set_d10 = set(dense_indices[:10])
        set_b10 = set(bm25_indices[:10])
        union_10 = set_d10 | set_b10
        overlap_10 = len(set_d10 & set_b10) / len(union_10) if len(union_10) > 0 else 0.0

        set_d50 = set(dense_indices[:50])
        set_b50 = set(bm25_indices[:50])
        union_50 = set_d50 | set_b50
        overlap_50 = len(set_d50 & set_b50) / len(union_50) if len(union_50) > 0 else 0.0

        # Rank correlation (Kendall tau)
        shared = list(set_d50 & set_b50)
        rank_corr = 0.0
        if len(shared) > 1:
            d_ranks = [np.where(dense_indices == idx)[0][0] for idx in shared]
            b_ranks = [np.where(bm25_indices == idx)[0][0] for idx in shared]
            tau, _ = stats.kendalltau(d_ranks, b_ranks)
            rank_corr = float(tau) if not np.isnan(tau) else 0.0

        candidate_novelty = 1.0 - overlap_50

        # Source disagreement normalized to [0, 1]
        rank_corr_norm = (rank_corr + 1.0) / 2.0
        source_disagreement = (1.0 - overlap_10) * (1.0 - rank_corr_norm)
        source_disagreement = float(np.clip(source_disagreement, 0.0, 1.0))

        # ── Graph features (numpy safe) ──
        if graph_degrees is not None:
            graph_degrees = np.asarray(graph_degrees)
        else:
            graph_degrees = np.array([])
        if graph_degrees.size > 0:
            g_max = float(np.max(graph_degrees))
            g_mean = float(np.mean(graph_degrees))
            g_z_frac = float(np.sum(graph_degrees == 0) / len(graph_degrees))
        else:
            g_max, g_mean, g_z_frac = 0.0, 0.0, 0.0

        return QueryFeatures(
            query_length_tokens=q_len,
            query_has_number=q_has_number,
            query_has_uppercase_abbreviation=q_has_abbrev,
            query_avg_token_idf=q_avg_idf,
            query_max_token_idf=q_max_idf,
            lexical_specificity_score=lexical_spec,

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

            bm25_dense_overlap_jaccard_10=overlap_10,
            bm25_dense_overlap_jaccard_50=overlap_50,
            dense_bm25_rank_correlation=rank_corr,
            candidate_novelty=candidate_novelty,
            source_disagreement_score=source_disagreement,

            graph_degree_max=g_max,
            graph_degree_mean=g_mean,
            graph_degree_zero_frac=g_z_frac,
        )
