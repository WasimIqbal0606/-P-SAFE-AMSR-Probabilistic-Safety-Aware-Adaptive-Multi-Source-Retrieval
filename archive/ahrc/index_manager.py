"""
AHRC — Index Manager
Build and manage FAISS HNSW and IVF indices for sublinear retrieval.

Replaces the brute-force IndexFlatIP with approximate nearest-neighbor
indices that achieve true O(log n) or O(√n) per-query complexity.
"""

import time
import numpy as np
import faiss
from typing import Tuple, Optional, List

from .config import IndexConfig, IndexType


class IndexManager:
    """Build, configure, and query FAISS approximate indices."""

    def __init__(self, config: IndexConfig):
        self.cfg = config
        self.index: Optional[faiss.Index] = None
        self.n_vectors: int = 0
        self.build_time: float = 0.0
        self._is_trained: bool = False

    # ── Index construction ─────────────────────────────────────────────

    def build(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build index from embedding matrix.

        Args:
            embeddings: (N, D) float32 matrix, L2-normalized for IP.

        Returns:
            Constructed FAISS index.
        """
        n, d = embeddings.shape
        self.n_vectors = n
        self.cfg.embedding_dim = d

        print(f"🏗️  Building {self.cfg.index_type.value.upper()} index "
              f"(n={n:,}, d={d})...")
        t0 = time.time()

        if self.cfg.index_type == IndexType.HNSW:
            self.index = self._build_hnsw(embeddings)
        elif self.cfg.index_type == IndexType.IVF:
            self.index = self._build_ivf(embeddings)
        elif self.cfg.index_type == IndexType.IVFPQ:
            self.index = self._build_ivfpq(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.cfg.index_type}")

        self.build_time = time.time() - t0
        self._is_trained = True
        print(f"   ✅ Index built in {self.build_time:.2f}s "
              f"({self.cfg.index_type.value.upper()}, "
              f"ntotal={self.index.ntotal:,})")
        return self.index

    def _build_hnsw(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build HNSW index.
        Complexity: O(log n) per query.
        """
        d = embeddings.shape[1]

        # Inner product via IndexHNSWFlat
        # HNSW works natively with L2; for IP we pre-normalize and use L2
        # (cosine similarity = IP on unit vectors = 1 - L2²/2)
        index = faiss.IndexHNSWFlat(d, self.cfg.hnsw_m)
        index.hnsw.efConstruction = self.cfg.hnsw_ef_construction
        index.hnsw.efSearch = self.cfg.hnsw_ef_search

        # HNSW doesn't need training
        index.add(embeddings)
        return index

    def _build_ivf(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build IVF (Inverted File) index.
        Complexity: O(n / nlist * nprobe) ≈ O(√n) per query with
        nlist = √n, nprobe = √(nlist).
        """
        d = embeddings.shape[1]
        n = embeddings.shape[0]

        # Auto-tune nlist if dataset is small
        nlist = min(self.cfg.ivf_nlist, int(np.sqrt(n)))
        nlist = max(nlist, 1)

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)

        # Train on embeddings
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(self.cfg.ivf_nprobe, nlist)

        return index

    def _build_ivfpq(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build IVF + Product Quantization index.
        Compressed vectors + inverted file for large-scale retrieval.
        """
        d = embeddings.shape[1]
        n = embeddings.shape[0]

        nlist = min(self.cfg.ivf_nlist, int(np.sqrt(n)))
        nlist = max(nlist, 1)
        pq_m = min(self.cfg.pq_m, d)  # sub-quantizers <= dimension

        # Ensure d is divisible by pq_m
        while d % pq_m != 0 and pq_m > 1:
            pq_m -= 1

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, self.cfg.pq_nbits)

        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(self.cfg.ivf_nprobe, nlist)

        return index

    # ── Querying ───────────────────────────────────────────────────────

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index.

        Args:
            query_embeddings: (nq, D) float32 query matrix.
            k: number of neighbors to return.
            ef_search: HNSW efSearch override (higher = more accurate, slower).
            nprobe: IVF nprobe override (higher = more accurate, slower).

        Returns:
            distances: (nq, k) float32
            indices: (nq, k) int64
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Ensure 2D
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # Dynamic parameter overrides
        if ef_search is not None and self.cfg.index_type == IndexType.HNSW:
            self.index.hnsw.efSearch = ef_search

        if nprobe is not None and self.cfg.index_type in (IndexType.IVF, IndexType.IVFPQ):
            self.index.nprobe = nprobe

        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices

    def search_single(
        self,
        query: np.ndarray,
        k: int = 10,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for a single query vector."""
        dists, idxs = self.search(query.reshape(1, -1), k, **kwargs)
        return dists[0], idxs[0]

    # ── Metadata ───────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return index statistics."""
        stats = {
            "index_type": self.cfg.index_type.value,
            "n_vectors": self.n_vectors,
            "embedding_dim": self.cfg.embedding_dim,
            "build_time_s": round(self.build_time, 4),
            "is_trained": self._is_trained,
        }

        if self.cfg.index_type == IndexType.HNSW:
            stats["hnsw_m"] = self.cfg.hnsw_m
            stats["hnsw_ef_construction"] = self.cfg.hnsw_ef_construction
            stats["hnsw_ef_search"] = self.cfg.hnsw_ef_search

        elif self.cfg.index_type in (IndexType.IVF, IndexType.IVFPQ):
            stats["ivf_nlist"] = self.cfg.ivf_nlist
            stats["ivf_nprobe"] = self.cfg.ivf_nprobe

        return stats

    def set_search_params(self, ef_search: int = None, nprobe: int = None):
        """Update search-time parameters for accuracy/speed tradeoff."""
        if ef_search is not None and self.cfg.index_type == IndexType.HNSW:
            self.index.hnsw.efSearch = ef_search
            self.cfg.hnsw_ef_search = ef_search

        if nprobe is not None and self.cfg.index_type in (IndexType.IVF, IndexType.IVFPQ):
            self.index.nprobe = nprobe
            self.cfg.ivf_nprobe = nprobe
