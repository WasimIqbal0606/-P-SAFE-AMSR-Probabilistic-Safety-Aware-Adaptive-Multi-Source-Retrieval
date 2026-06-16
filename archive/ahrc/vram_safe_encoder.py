"""
P-SAFE-AMSR — VRAM-Safe Encoder
FP16 encoding with thermal-safe batch sizes.

NOTE (FIX 7): This encoder uses SentenceTransformer.encode() which produces
dense embeddings ONLY. It does NOT use FlagEmbedding's dense+sparse+multivector
mode. For true BGE-M3 multifunction retrieval, use psafe.retrievers.bgem3_retriever.

bge_m3_mode = "dense_only"
"""
import os
import gc
import time
import subprocess
import numpy as np
from typing import List, Optional

from .embedding_cache import EmbeddingCache


def _clear_gpu():
    """Free GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except ImportError:
        pass


def _log_gpu_mem(tag: str = ""):
    """Log current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"   [GPU {tag}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    except (ImportError, AttributeError):
        pass


def _gpu_temperature_c():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            return None
        temps = [float(x.strip()) for x in r.stdout.splitlines() if x.strip()]
        return max(temps) if temps else None
    except Exception:
        return None


def _cooldown_if_needed(limit_c=None, cooldown_c=None):
    if limit_c is None:
        return
    cooldown_c = cooldown_c if cooldown_c is not None else max(35.0, limit_c - 10.0)
    while True:
        temp = _gpu_temperature_c()
        if temp is None or temp < limit_c - 5.0:
            return
        print(f"   [GPU cooldown] temp={temp:.1f} C; waiting until <= {cooldown_c:.1f} C")
        while temp is not None and temp > cooldown_c:
            time.sleep(10)
            temp = _gpu_temperature_c()


def encode_texts(
    model_name: str,
    texts: List[str],
    dataset_name: str,
    kind: str = "corpus",
    device: str = "auto",
    cache: Optional[EmbeddingCache] = None,
    normalize: bool = True,
    batch_size: int = 16,
    chunk_size: int = 0,
    gpu_temp_limit: Optional[float] = None,
    gpu_cooldown_temp: Optional[float] = None,
) -> np.ndarray:
    """
    Encode texts with FP16 and thermal-safe batch sizing.
    BGE-M3: batch_size=16, FP16 enabled.
    """
    n = len(texts)

    # Check cache first
    if cache is not None:
        cached = cache.get(model_name, dataset_name, n, kind)
        if cached is not None:
            return cached

    # Determine device
    embed_device = "cpu"
    try:
        import torch
        if device in ("auto", "cuda") and torch.cuda.is_available():
            embed_device = "cuda"
    except ImportError:
        pass

    from sentence_transformers import SentenceTransformer

    batch_size = max(1, int(batch_size))
    chunk_size = max(0, int(chunk_size or 0))
    # FIX 7: Label as dense-only mode
    bge_m3_mode = "dense_only" if "bge-m3" in model_name.lower() else "n/a"

    print(f"   [Encoder] Model: {model_name} (mode: {bge_m3_mode})")
    print(f"   [Encoder] Device: {embed_device}, Batch: {batch_size}, FP16: True, Texts: {n:,}, Kind: {kind}")

    _clear_gpu()
    _log_gpu_mem("before-load")

    st_model = SentenceTransformer(model_name, device=embed_device)

    # Enable FP16 on GPU
    if embed_device == "cuda":
        try:
            import torch
            st_model = st_model.half()
            print("   [Encoder] FP16 enabled")
        except Exception:
            pass

    _log_gpu_mem("after-load")

    # Encode. Chunking lets the runner cool down between chunks instead of
    # running one long opaque GPU call.
    if chunk_size and chunk_size < n:
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            _cooldown_if_needed(gpu_temp_limit, gpu_cooldown_temp)
            print(f"   [Encoder] chunk {start:,}-{end:,} / {n:,}")
            chunks.append(st_model.encode(
                texts[start:end],
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
            ).astype(np.float32))
        embeddings = np.vstack(chunks)
    else:
        _cooldown_if_needed(gpu_temp_limit, gpu_cooldown_temp)
        embeddings = st_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
        ).astype(np.float32)

    _log_gpu_mem("after-encode")

    # Free model from GPU immediately
    del st_model
    _clear_gpu()
    _log_gpu_mem("after-cleanup")

    print(f"   [Encoder] Output shape: {embeddings.shape}")

    # Save to cache
    if cache is not None:
        cache.put(embeddings, model_name, dataset_name, n, kind)

    return embeddings
