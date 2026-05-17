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


def encode_texts(
    model_name: str,
    texts: List[str],
    dataset_name: str,
    kind: str = "corpus",
    device: str = "auto",
    cache: Optional[EmbeddingCache] = None,
    normalize: bool = True,
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

    # BGE-M3 specific: batch=16, FP16
    batch_size = 16
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

    # Encode
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
