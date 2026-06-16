"""Tests for FeatureExtractor — feature extraction stability."""
import numpy as np
import pytest
from psafe.feature_extractor import FeatureExtractor, FEATURE_NAMES, QueryFeatures


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 25, f"Expected 25 features, got {len(FEATURE_NAMES)}"


def test_query_features_to_array():
    qf = QueryFeatures(dense_score_max=0.9, bm25_score_max_norm=0.7)
    arr = qf.to_array(FEATURE_NAMES)
    assert arr.shape == (len(FEATURE_NAMES),)
    assert arr.dtype == np.float32


def test_extract_basic():
    ext = FeatureExtractor()
    ext.idf_dict = {"hello": 3.0, "world": 2.5}

    dense_idx = np.arange(50, dtype=np.int64)
    dense_scores = np.linspace(1.0, 0.0, 50).astype(np.float32)
    bm25_idx = np.arange(30, 80, dtype=np.int64)
    bm25_scores = np.linspace(15.0, 0.0, 50).astype(np.float32)
    graph_degrees = [3, 2, 0, 5, 1, 0, 0, 4, 2, 1]

    qf = ext.extract(
        query_id="q1", query_embedding=np.zeros(128),
        query_text="hello world test",
        dense_indices=dense_idx, dense_scores=dense_scores,
        bm25_indices=bm25_idx, bm25_scores=bm25_scores,
        graph_degrees=graph_degrees,
    )

    arr = qf.to_array(FEATURE_NAMES)
    assert arr.shape == (25,)
    assert not np.any(np.isnan(arr)), "NaN detected in features"
    assert not np.any(np.isinf(arr)), "Inf detected in features"


def test_extract_empty_graph():
    ext = FeatureExtractor()
    qf = ext.extract(
        query_id="q2", query_embedding=np.zeros(128),
        query_text="test query",
        dense_indices=np.arange(10), dense_scores=np.ones(10, dtype=np.float32),
        bm25_indices=np.arange(10), bm25_scores=np.ones(10, dtype=np.float32),
        graph_degrees=[],
    )
    assert qf.graph_degree_max == 0.0
    assert qf.graph_degree_mean == 0.0
    assert qf.graph_degree_zero_frac == 0.0


def test_extract_empty_bm25():
    ext = FeatureExtractor()
    qf = ext.extract(
        query_id="q3", query_embedding=np.zeros(128),
        query_text="test",
        dense_indices=np.arange(10), dense_scores=np.ones(10, dtype=np.float32),
        bm25_indices=np.array([]), bm25_scores=np.array([]),
        graph_degrees=[1, 2],
    )
    assert qf.bm25_score_max_norm == 0.0


def test_source_disagreement_bounded():
    """Source disagreement should be in [0, 1]."""
    ext = FeatureExtractor()
    qf = ext.extract(
        query_id="q4", query_embedding=np.zeros(128),
        query_text="divergent sources test",
        dense_indices=np.arange(50), dense_scores=np.linspace(1, 0, 50).astype(np.float32),
        bm25_indices=np.arange(50, 100), bm25_scores=np.linspace(1, 0, 50).astype(np.float32),
        graph_degrees=[0] * 10,
    )
    assert 0.0 <= qf.source_disagreement_score <= 1.0


def test_lexical_features():
    ext = FeatureExtractor()
    ext.idf_dict = {"covid": 6.0, "treatment": 4.0}
    qf = ext.extract(
        query_id="q5", query_embedding=np.zeros(128),
        query_text="COVID-19 treatment EFFICACY",
        dense_indices=np.arange(10), dense_scores=np.ones(10, dtype=np.float32),
        bm25_indices=np.arange(10), bm25_scores=np.ones(10, dtype=np.float32),
        graph_degrees=[1],
    )
    assert qf.query_has_uppercase_abbreviation == 1.0
    assert qf.query_length_tokens == 3
