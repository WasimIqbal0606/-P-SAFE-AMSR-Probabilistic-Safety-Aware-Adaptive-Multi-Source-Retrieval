"""
Safe-AMSR-SE v4 — Cross-Encoder Reranker
GPU-accelerated, batched, fp16 cross-encoder reranking.

Features:
  - Automatic GPU detection with CPU fallback
  - FP16 inference for 2x speedup on GPU
  - Batched inference (configurable batch_size)
  - Latency tracking with P50/P95/P99 percentiles
  - GPU memory usage logging
  - Score caching
  - Depth sweep for latency/quality tradeoff analysis
"""

import subprocess
import time
import numpy as np
from typing import List, Tuple, Dict, Optional


class CrossEncoderReranker:
    """Publication-grade cross-encoder reranker with GPU, fp16, and latency tracking."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        batch_size: int = 8,
        use_fp16: bool = True,
        gpu_temp_limit: Optional[float] = None,
        gpu_cooldown_temp: Optional[float] = None,
    ):
        self.model_name = model_name
        self.requested_device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.gpu_temp_limit = gpu_temp_limit
        self.gpu_cooldown_temp = gpu_cooldown_temp
        self.model = None
        self._is_loaded = False
        self.actual_device = "cpu"
        self._score_cache: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._latencies: List[float] = []  # per-query latencies in ms
        self._gpu_mem_mb: float = 0.0

    def _gpu_temperature_c(self) -> Optional[float]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            temps = [float(x.strip()) for x in result.stdout.splitlines() if x.strip()]
            return max(temps) if temps else None
        except Exception:
            return None

    def _cooldown_if_needed(self):
        if self.gpu_temp_limit is None or self.actual_device != "cuda":
            return
        cooldown_c = self.gpu_cooldown_temp if self.gpu_cooldown_temp is not None else max(35.0, self.gpu_temp_limit - 10.0)
        temp = self._gpu_temperature_c()
        # CrossEncoder batches heat the GPU quickly, so pause slightly before the hard limit.
        if temp is None or temp < self.gpu_temp_limit - 1.0:
            return
        print(f"   [CrossEncoder GPU cooldown] temp={temp:.1f} C; waiting until <= {cooldown_c:.1f} C")
        while temp is not None and temp > cooldown_c:
            time.sleep(10)
            temp = self._gpu_temperature_c()

    def _detect_device(self) -> str:
        """Detect best available device."""
        if self.requested_device != "auto" and self.requested_device != "gpu":
            return self.requested_device

        try:
            import torch
            if torch.cuda.is_available():
                # Verify CUDA actually works
                try:
                    test = torch.zeros(1, device="cuda")
                    del test
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"   🎮 GPU detected: {gpu_name}")
                    return "cuda"
                except Exception as e:
                    print(f"   ⚠️  GPU detected but unusable: {e}")
                    return "cpu"
        except ImportError:
            pass
        return "cpu"

    def load(self):
        """Load the cross-encoder model with optional fp16."""
        if self._is_loaded:
            return

        self.actual_device = self._detect_device()

        from sentence_transformers import CrossEncoder
        print(f"Loading CrossEncoder ({self.model_name}) on {self.actual_device}...")
        t0 = time.perf_counter()
        self.model = CrossEncoder(self.model_name, device=self.actual_device)

        # Apply fp16 if on GPU
        if self.use_fp16 and self.actual_device == "cuda":
            try:
                import torch
                self.model.model.half()
                print(f"   FP16 enabled")
            except Exception as e:
                print(f"   FP16 failed, using fp32: {e}")
                self.use_fp16 = False

        elapsed = (time.perf_counter() - t0) * 1000
        self._is_loaded = True
        self._log_gpu_memory()
        print(f"   Loaded in {elapsed:.0f}ms")

    def rerank(
        self,
        query: str,
        candidate_texts: List[str],
        candidate_indices: np.ndarray,
        use_cache: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rerank candidates using cross-encoder with batching.

        Args:
            query: query text.
            candidate_texts: list of candidate document texts.
            candidate_indices: original indices of candidates.
            use_cache: whether to use score caching.

        Returns:
            (reranked_indices, reranked_scores) sorted by score descending.
        """
        if len(candidate_indices) == 0:
            return candidate_indices, np.array([], dtype=np.float32)

        if not self._is_loaded:
            self.load()

        # Check cache for already-scored pairs
        scores = np.zeros(len(candidate_texts), dtype=np.float32)
        uncached_indices = []

        if use_cache:
            for i, text in enumerate(candidate_texts):
                cache_key = f"{hash(query)}_{hash(text)}"
                if cache_key in self._score_cache:
                    scores[i] = self._score_cache[cache_key]
                    self._cache_hits += 1
                else:
                    uncached_indices.append(i)
                    self._cache_misses += 1
        else:
            uncached_indices = list(range(len(candidate_texts)))

        if uncached_indices:
            # Build pairs for uncached
            pairs = [[query, candidate_texts[i]] for i in uncached_indices]

            # Batched inference
            batch_scores = self._batched_predict(pairs)

            for j, i in enumerate(uncached_indices):
                scores[i] = batch_scores[j]
                if use_cache:
                    cache_key = f"{hash(query)}_{hash(candidate_texts[i])}"
                    self._score_cache[cache_key] = batch_scores[j]

        # Sort by scores descending
        sort_order = np.argsort(-scores)
        return candidate_indices[sort_order], scores[sort_order]

    def _batched_predict(self, pairs: List[List[str]]) -> np.ndarray:
        """Run cross-encoder prediction in batches with latency tracking."""
        all_scores = []
        t0 = time.perf_counter()

        for start in range(0, len(pairs), self.batch_size):
            self._cooldown_if_needed()
            batch = pairs[start:start + self.batch_size]
            batch_scores = self.model.predict(batch, show_progress_bar=False)
            all_scores.extend(batch_scores)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latencies.append(elapsed_ms)
        return np.array(all_scores, dtype=np.float32)

    def score_pairs(
        self,
        query: str,
        candidate_texts: List[str],
    ) -> np.ndarray:
        """Score query-document pairs without reranking."""
        if not self._is_loaded:
            self.load()
        pairs = [[query, text] for text in candidate_texts]
        return self._batched_predict(pairs)

    def get_stats(self) -> Dict:
        """Return reranker statistics with latency percentiles."""
        stats = {
            "model_name": self.model_name,
            "device": self.actual_device,
            "batch_size": self.batch_size,
            "fp16": self.use_fp16 and self.actual_device == "cuda",
            "is_loaded": self._is_loaded,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._score_cache),
            "gpu_memory_mb": self._gpu_mem_mb,
        }
        if self._latencies:
            lats = np.array(self._latencies)
            stats.update({
                "latency_mean_ms": float(np.mean(lats)),
                "latency_p50_ms": float(np.percentile(lats, 50)),
                "latency_p95_ms": float(np.percentile(lats, 95)),
                "latency_p99_ms": float(np.percentile(lats, 99)),
                "n_rerank_calls": len(lats),
            })
        return stats

    def _log_gpu_memory(self):
        """Log GPU memory usage."""
        if self.actual_device != "cuda":
            return
        try:
            import torch
            self._gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
            total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
            print(f"   GPU memory: {self._gpu_mem_mb:.0f}MB / {total:.0f}MB")
        except: pass

    def clear_cache(self):
        """Clear the score cache."""
        self._score_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._latencies.clear()


class CrossEncoderDepthSweep:
    """
    Sweep cross-encoder rerank depth to find optimal quality/latency tradeoff.

    Tests depths = [10, 20, 30, 50, 100] and reports:
      - nDCG@10, Recall@10, MRR
      - latency_mean, latency_p95
      - quality_per_ms
      - cost_per_ndcg_gain
    """

    def __init__(
        self,
        depths: List[int] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.depths = depths or [10, 20, 30, 50, 100]
        self.reranker = reranker
        self.results: Dict[int, Dict] = {}

    def sweep(
        self,
        queries: List[Dict],
        candidate_pools: List[Dict],
        evaluator,
        corpus_ids: List[str],
        qrels: Dict,
        baseline_ndcg: float = 0.0,
    ) -> Dict[int, Dict]:
        """
        Run depth sweep.

        Args:
            queries: list of {"id": ..., "text": ..., "embedding": ...}
            candidate_pools: list of {"indices": ..., "scores": ..., "texts": ...}
            evaluator: Evaluator instance
            corpus_ids: ordered doc id list
            qrels: {query_id: {doc_id: relevance}}
            baseline_ndcg: dense baseline nDCG@10 for cost computation

        Returns:
            {depth: {nDCG@10, Recall@10, MRR, latency_mean, ...}}
        """
        if self.reranker is None or not self.reranker._is_loaded:
            print("   ⚠️  Reranker not loaded, skipping depth sweep")
            return {}

        print(f"\n   🔄 Cross-Encoder Depth Sweep: {self.depths}")

        for depth in self.depths:
            print(f"      Depth={depth}...", end=" ", flush=True)
            metrics_list = []
            latencies = []

            for qi, (q, pool) in enumerate(zip(queries, candidate_pools)):
                t0 = time.perf_counter()

                pool_indices = pool["indices"]
                pool_scores = pool["scores"]
                pool_texts = pool["texts"]

                rerank_n = min(depth, len(pool_indices))
                if rerank_n > 0 and pool_texts:
                    to_rerank_idx = pool_indices[:rerank_n]
                    to_rerank_texts = pool_texts[:rerank_n]

                    reranked_idx, reranked_scores = self.reranker.rerank(
                        q["text"], to_rerank_texts, to_rerank_idx
                    )

                    # Combine with unranked tail
                    final_indices = np.concatenate([
                        reranked_idx, pool_indices[rerank_n:]
                    ])
                    final_scores = np.concatenate([
                        reranked_scores, pool_scores[rerank_n:]
                    ])
                else:
                    final_indices = pool_indices
                    final_scores = pool_scores

                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)

                # Evaluate
                query_rels = qrels.get(q["id"], {})
                qm = evaluator.evaluate_query(
                    final_indices[:10], query_rels, corpus_ids,
                    q["id"], elapsed_ms, len(pool_indices),
                )
                metrics_list.append(qm)

            # Aggregate
            agg = evaluator.aggregate(metrics_list, f"CE_depth_{depth}")
            lat_array = np.array(latencies)

            ndcg10 = agg.ndcg_at_k.get(10, 0)
            ndcg_gain = ndcg10 - baseline_ndcg
            mean_lat = float(np.mean(lat_array))

            self.results[depth] = {
                "depth": depth,
                "ndcg_at_10": ndcg10,
                "recall_at_10": agg.recall_at_k.get(10, 0),
                "mrr": agg.mrr,
                "latency_mean_ms": mean_lat,
                "latency_median_ms": float(np.median(lat_array)),
                "latency_p95_ms": float(np.percentile(lat_array, 95)),
                "latency_p99_ms": float(np.percentile(lat_array, 99)),
                "quality_per_ms": ndcg10 / max(mean_lat, 0.001),
                "cost_per_ndcg_gain": mean_lat / max(ndcg_gain, 0.0001) if ndcg_gain > 0 else float("inf"),
                "ndcg_gain_vs_dense": ndcg_gain,
            }

            print(f"nDCG@10={ndcg10:.4f}, lat={mean_lat:.1f}ms, "
                  f"gain={ndcg_gain:+.4f}")

        return self.results

    def find_optimal_depth(self, max_latency_ms: float = 500.0) -> int:
        """Find the smallest depth that preserves most nDCG gain within latency budget."""
        if not self.results:
            return 50  # default

        max_ndcg = max(r["ndcg_at_10"] for r in self.results.values())

        # Find smallest depth where nDCG is >= 95% of max and within latency budget
        for depth in sorted(self.results.keys()):
            r = self.results[depth]
            if (r["ndcg_at_10"] >= 0.95 * max_ndcg and
                    r["latency_mean_ms"] <= max_latency_ms):
                return depth

        # Fallback: smallest depth
        return min(self.results.keys())

    def format_results(self) -> str:
        """Pretty-print sweep results."""
        if not self.results:
            return "  No depth sweep results."

        lines = ["  Cross-Encoder Depth Sweep Results:"]
        lines.append(f"  {'Depth':>6} {'nDCG@10':>8} {'R@10':>7} {'MRR':>6} "
                      f"{'Lat(ms)':>8} {'P95':>8} {'Q/ms':>8} {'$/gain':>10}")
        lines.append("  " + "─" * 72)

        for depth in sorted(self.results.keys()):
            r = self.results[depth]
            cost = r['cost_per_ndcg_gain']
            cost_str = f"{cost:.1f}" if cost < 1e6 else "∞"
            lines.append(
                f"  {depth:>6} {r['ndcg_at_10']:>8.4f} {r['recall_at_10']:>7.4f} "
                f"{r['mrr']:>6.4f} {r['latency_mean_ms']:>8.1f} "
                f"{r['latency_p95_ms']:>8.1f} {r['quality_per_ms']:>8.5f} "
                f"{cost_str:>10}"
            )

        optimal = self.find_optimal_depth()
        lines.append(f"\n  🎯 Optimal depth: {optimal}")

        return "\n".join(lines)
