"""
AHRC — Adaptive Hybrid Retrieval Controller
Configuration and hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class IndexType(Enum):
    HNSW = "hnsw"
    IVF = "ivf"
    IVFPQ = "ivfpq"


class UncertaintyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class IndexConfig:
    """FAISS index construction parameters."""
    index_type: IndexType = IndexType.HNSW
    embedding_dim: int = 384  # all-MiniLM-L6-v2 default
    # HNSW parameters
    hnsw_m: int = 32          # connections per node
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    # IVF parameters
    ivf_nlist: int = 100      # number of Voronoi cells
    ivf_nprobe: int = 10      # cells to search at query time
    # IVFPQ parameters
    pq_m: int = 48            # sub-quantizers
    pq_nbits: int = 8         # bits per sub-quantizer
    metric: str = "ip"        # inner product (cosine after L2-norm)


@dataclass
class UncertaintyConfig:
    """Uncertainty estimation thresholds."""
    # Signal weights for composite score
    margin_weight: float = 0.30
    variance_weight: float = 0.20
    entropy_weight: float = 0.20
    graph_ambiguity_weight: float = 0.15
    historical_weight: float = 0.15
    # Thresholds for level classification
    low_threshold: float = 0.30
    high_threshold: float = 0.65
    # Margin-specific
    margin_saturation: float = 0.5  # margin above this → low uncertainty


@dataclass
class AdaptiveConfig:
    """Adaptive controller parameters."""
    # k range
    k_min: int = 10
    k_default: int = 10
    k_max: int = 20
    # Similarity thresholds (disabled to prevent destroying recall when using cross-encoders)
    threshold_tight: float = -999.0
    threshold_default: float = -999.0
    threshold_loose: float = -999.0
    # Reranking
    rerank_depth_min: int = 0
    rerank_depth_default: int = 10
    rerank_depth_max: int = 20
    # Graph expansion
    graph_expansion_hops: int = 1
    graph_max_neighbors: int = 20


@dataclass
class ExperimentConfig:
    """Experiment and dataset configuration."""
    # Dataset
    num_tasks: int = 10_000
    num_queries: int = 200
    num_categories: int = 12
    avg_relationships_per_task: float = 3.5
    relevance_levels: int = 4          # 0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant
    # Evaluation
    eval_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    # Reproducibility
    random_seed: int = 42
    # Embedding model
    model_name: str = "all-MiniLM-L6-v2"
    # Output
    results_dir: str = "ahrc_results"


@dataclass
class AHRCConfig:
    """Top-level configuration."""
    index: IndexConfig = field(default_factory=IndexConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
