"""
B-P-SAFE-AMSR — BGE-M3 True Multifunction Retriever
Optional module: gracefully skips if FlagEmbedding is not installed.

FIX 1: argpartition boundary guard when top_k == len(scores)
FIX 2: No silent zero-embedding insertion; retry then raise
FIX 3: Robust lexical query handling for dict/list formats
"""
import numpy as np
import json
import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import BGEM3FlagModel  # type: ignore
    HAS_FLAG_EMBEDDING = True
except ImportError:
    HAS_FLAG_EMBEDDING = False


class BGEM3MultiFunctionRetriever:
    """
    Optional true BGE-M3 mode using FlagEmbedding.
    Supports dense, sparse (lexical), and colbert/multivector scoring.
    Fails gracefully if FlagEmbedding is not installed.
    """
    def __init__(self, model_name="BAAI/bge-m3", use_fp16=True, device="cuda",
                 use_dense=True, use_sparse=True, use_multivector=False,
                 top_k=100, candidate_k=200, multivector_rerank_k=100,
                 weights=(1.0, 0.3, 0.0), batch_size=12,
                 allow_zero_embedding_fallback=False):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self.model = None

        # Config options
        self.use_dense = use_dense
        self.use_sparse = use_sparse
        self.use_multivector = use_multivector
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.multivector_rerank_k = multivector_rerank_k
        self.weights = weights
        self.batch_size = batch_size
        self.allow_zero_embedding_fallback = allow_zero_embedding_fallback  # FIX 2

        # Cache
        self._corpus_dense_vecs = None
        self._corpus_lexical_weights = None
        self._corpus_colbert_vecs = None
        self._query_cache = {}
        self._warnings = []  # FIX 2: track warnings

    @staticmethod
    def is_available() -> bool:
        return HAS_FLAG_EMBEDDING

    def load_model(self):
        if not HAS_FLAG_EMBEDDING:
            raise ImportError("FlagEmbedding not installed. BGE-M3 actions unavailable.")
        if self.model is None:
            self.model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16,
                                         device=self.device)

    def encode_corpus(self, texts: List[str], max_length=8192) -> Dict:
        """Encode full corpus with memory-safe batching. Results are cached."""
        self.load_model()
        n = len(texts)
        all_dense = []
        all_lexical = []
        all_colbert = []

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = texts[start:end]
            try:
                out = self.model.encode(
                    batch, batch_size=self.batch_size, max_length=max_length,
                    return_dense=self.use_dense,
                    return_sparse=self.use_sparse,
                    return_colbert_vecs=self.use_multivector,
                )
                if self.use_dense and 'dense_vecs' in out:
                    all_dense.append(out['dense_vecs'])
                if self.use_sparse and 'lexical_weights' in out:
                    all_lexical.extend(out['lexical_weights'])
                if self.use_multivector and 'colbert_vecs' in out:
                    all_colbert.extend(out['colbert_vecs'])
            except Exception as e:
                # FIX 2: Retry with smaller batch, then raise or fallback
                logger.warning(f"BGE-M3 encode batch {start}-{end} failed: {e}. Retrying with half batch...")
                retry_bs = max(1, self.batch_size // 2)
                try:
                    out = self.model.encode(
                        batch, batch_size=retry_bs, max_length=max_length,
                        return_dense=self.use_dense,
                        return_sparse=self.use_sparse,
                        return_colbert_vecs=self.use_multivector,
                    )
                    if self.use_dense and 'dense_vecs' in out:
                        all_dense.append(out['dense_vecs'])
                    if self.use_sparse and 'lexical_weights' in out:
                        all_lexical.extend(out['lexical_weights'])
                    if self.use_multivector and 'colbert_vecs' in out:
                        all_colbert.extend(out['colbert_vecs'])
                except Exception as e2:
                    if self.allow_zero_embedding_fallback:
                        # FIX 2: Only insert zeros if explicitly allowed
                        warning = f"Batch {start}-{end} failed twice ({e2}). Inserting zero vectors (allow_zero_embedding_fallback=True)."
                        logger.warning(warning)
                        self._warnings.append({"batch": f"{start}-{end}", "error": str(e2), "action": "zero_fallback"})
                        if self.use_dense:
                            all_dense.append(np.zeros((end - start, 1024), dtype=np.float32))
                    else:
                        raise RuntimeError(
                            f"BGE-M3 corpus encoding failed for batch {start}-{end} after retry. "
                            f"Error: {e2}. Set allow_zero_embedding_fallback=True to insert zeros (not recommended for research)."
                        )

        if all_dense:
            self._corpus_dense_vecs = np.vstack(all_dense)
        if all_lexical:
            self._corpus_lexical_weights = all_lexical
        if all_colbert:
            self._corpus_colbert_vecs = all_colbert

        return {
            'dense_vecs': self._corpus_dense_vecs,
            'lexical_weights': self._corpus_lexical_weights,
            'colbert_vecs': self._corpus_colbert_vecs,
        }

    def encode_queries(self, texts: List[str], max_length=512) -> Dict:
        """Encode queries."""
        self.load_model()
        return self.model.encode(
            texts, batch_size=self.batch_size, max_length=max_length,
            return_dense=self.use_dense,
            return_sparse=self.use_sparse,
            return_colbert_vecs=self.use_multivector,
        )

    def retrieve_dense(self, query_vec: np.ndarray, k: int = None) -> tuple:
        """Pure dense retrieval from cached corpus vectors."""
        if self._corpus_dense_vecs is None:
            return np.array([]), np.array([])
        k = k or self.top_k
        scores = query_vec @ self._corpus_dense_vecs.T
        if len(scores.shape) > 1:
            scores = scores.flatten()
        # FIX 1: Guard argpartition when top_k >= len(scores)
        top_k = min(k, len(scores))
        if top_k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        kth = max(top_k - 1, 0)
        top_idx = np.argpartition(-scores, kth)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx, scores[top_idx]

    def retrieve_sparse(self, query_lexical, k: int = None) -> tuple:
        """Sparse lexical retrieval."""
        if self._corpus_lexical_weights is None:
            return np.array([]), np.array([])
        k = k or self.top_k
        n = len(self._corpus_lexical_weights)
        scores = np.zeros(n, dtype=np.float32)
        for i, doc_weights in enumerate(self._corpus_lexical_weights):
            score = 0.0
            for token, q_weight in query_lexical.items():
                if token in doc_weights:
                    score += q_weight * doc_weights[token]
            scores[i] = score
        # FIX 1: Guard argpartition when top_k >= n
        top_k = min(k, n)
        if top_k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        kth = max(top_k - 1, 0)
        top_idx = np.argpartition(-scores, kth)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx, scores[top_idx]

    def retrieve_dense_sparse(self, query_dense, query_lexical,
                               k: int = None) -> tuple:
        """Combined dense + sparse retrieval."""
        k = k or self.top_k
        w_d, w_s = self.weights[0], self.weights[1]

        dense_idx, dense_scores = self.retrieve_dense(query_dense, k=self.candidate_k)
        sparse_idx, sparse_scores = self.retrieve_sparse(query_lexical, k=self.candidate_k)

        # Fuse by score
        score_map = {}
        for idx, s in zip(dense_idx, dense_scores):
            score_map[int(idx)] = w_d * float(s)
        for idx, s in zip(sparse_idx, sparse_scores):
            key = int(idx)
            score_map[key] = score_map.get(key, 0.0) + w_s * float(s)

        sorted_items = sorted(score_map.items(), key=lambda x: -x[1])[:k]
        if not sorted_items:
            return np.array([]), np.array([])
        indices = np.array([x[0] for x in sorted_items], dtype=np.int64)
        scores = np.array([x[1] for x in sorted_items], dtype=np.float32)
        return indices, scores

    def rerank_multivector_topk(self, query_colbert, candidate_indices,
                                 k: int = None) -> tuple:
        """Rerank top candidates using ColBERT/multivector scoring."""
        if self._corpus_colbert_vecs is None or self.model is None:
            return candidate_indices, np.ones(len(candidate_indices), dtype=np.float32)

        k = k or self.multivector_rerank_k
        top_idx = candidate_indices[:k]

        try:
            top_colbert = [self._corpus_colbert_vecs[int(i)] for i in top_idx]
            c_scores = self.model.colbert_score([query_colbert], top_colbert)
            if hasattr(c_scores, "__len__") and len(c_scores) > 0:
                if isinstance(c_scores[0], (list, np.ndarray)):
                    c_scores = c_scores[0]
            c_scores = np.array(c_scores, dtype=np.float32)
            sort_order = np.argsort(-c_scores)
            return top_idx[sort_order], c_scores[sort_order]
        except Exception as e:
            logger.warning(f"Multivector rerank failed: {e}")
            return top_idx, np.ones(len(top_idx), dtype=np.float32)

    def retrieve(self, query_embeddings, k: int = None) -> tuple:
        """Full retrieval pipeline: dense+sparse candidate gen, optional multivector rerank."""
        k = k or self.top_k
        q_dense = query_embeddings.get('dense_vecs')
        q_lexical_raw = query_embeddings.get('lexical_weights', {})
        q_colbert = query_embeddings.get('colbert_vecs')

        # FIX 3: Robust lexical query handling — dict, list, or missing
        if isinstance(q_lexical_raw, list):
            q_lex = q_lexical_raw[0] if q_lexical_raw else {}
        elif isinstance(q_lexical_raw, dict):
            q_lex = q_lexical_raw
        else:
            q_lex = {}

        if q_dense is not None and len(q_dense.shape) > 1:
            q_dense = q_dense[0]

        # FIX 3: Guard against missing dense vectors when dense retrieval is requested
        if self.use_dense and q_dense is None:
            raise ValueError("Dense retrieval requested but query dense_vecs missing")

        if self.use_dense and self.use_sparse:
            indices, scores = self.retrieve_dense_sparse(q_dense, q_lex, k=self.candidate_k)
        elif self.use_dense:
            indices, scores = self.retrieve_dense(q_dense, k=self.candidate_k)
        else:
            indices, scores = self.retrieve_sparse(q_lex, k=self.candidate_k)

        # Multivector rerank on top candidates only
        if self.use_multivector and q_colbert is not None and len(indices) > 0:
            q_col = q_colbert[0] if isinstance(q_colbert, list) else q_colbert
            indices, scores = self.rerank_multivector_topk(q_col, indices, k=k)

        return indices[:k], scores[:k] if len(scores) >= k else scores

    @staticmethod
    def save_skipped_baselines(out_dir: str, reason: str = "FlagEmbedding not installed"):
        """Save skipped baselines info when BGE-M3 is unavailable."""
        os.makedirs(out_dir, exist_ok=True)
        skipped = {
            "skipped_actions": [
                "A12_BGEM3_DENSE", "A13_BGEM3_SPARSE",
                "A14_BGEM3_DENSE_SPARSE", "A15_BGEM3_DENSE_SPARSE_MULTI",
                "A16_BGEM3_DENSE_SPARSE_CE",
            ],
            "reason": reason,
        }
        with open(os.path.join(out_dir, "skipped_baselines.json"), "w") as f:
            json.dump(skipped, f, indent=4)
        return skipped

    def save_warnings(self, out_dir: str):
        """FIX 2: Save any encoding warnings to disk."""
        if self._warnings:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "bge_m3_warnings.json"), "w") as f:
                json.dump(self._warnings, f, indent=4)
