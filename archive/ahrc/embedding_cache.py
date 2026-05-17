"""
P-SAFE-AMSR — Embedding Cache
Avoids re-encoding corpora across runs. Hash-based .npy caching.
"""
import os
import hashlib
import numpy as np
from typing import Optional


class EmbeddingCache:
    """Disk-backed embedding cache keyed on (model_name, dataset_name, n_docs)."""

    def __init__(self, cache_dir: str = "results_top_tier_psafe/cache/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, model_name: str, dataset_name: str, n_docs: int, kind: str) -> str:
        raw = f"{model_name}|{dataset_name}|{n_docs}|{kind}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.npy")

    def get(self, model_name: str, dataset_name: str, n_docs: int,
            kind: str = "corpus") -> Optional[np.ndarray]:
        key = self._key(model_name, dataset_name, n_docs, kind)
        path = self._path(key)
        if os.path.exists(path):
            arr = np.load(path)
            print(f"   [Cache HIT] {kind} embeddings for {model_name}/{dataset_name} "
                  f"({arr.shape[0]} vectors, dim={arr.shape[1]})")
            return arr
        return None

    def put(self, arr: np.ndarray, model_name: str, dataset_name: str,
            n_docs: int, kind: str = "corpus"):
        key = self._key(model_name, dataset_name, n_docs, kind)
        path = self._path(key)
        np.save(path, arr)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   [Cache SAVE] {kind} embeddings -> {path} ({size_mb:.1f} MB)")

    def has(self, model_name: str, dataset_name: str, n_docs: int,
            kind: str = "corpus") -> bool:
        key = self._key(model_name, dataset_name, n_docs, kind)
        return os.path.exists(self._path(key))
"Embedding cache to avoid redundant re-encoding across runs."
