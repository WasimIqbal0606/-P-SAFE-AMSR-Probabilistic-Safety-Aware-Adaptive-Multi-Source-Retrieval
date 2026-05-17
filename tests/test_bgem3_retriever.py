"""Tests for BGE-M3 retriever fixes (FIX 1, FIX 3)."""
import numpy as np
import pytest


def test_argpartition_when_k_equals_n():
    """FIX 1: argpartition must not crash when top_k == len(scores)."""
    from psafe.retrievers.bgem3_retriever import BGEM3MultiFunctionRetriever

    retriever = BGEM3MultiFunctionRetriever.__new__(BGEM3MultiFunctionRetriever)
    n = 10
    d = 4
    retriever._corpus_dense_vecs = np.random.randn(n, d).astype(np.float32)
    retriever.top_k = n  # k == len(corpus)

    query = np.random.randn(d).astype(np.float32)
    idx, scores = retriever.retrieve_dense(query, k=n)
    assert len(idx) == n
    assert len(scores) == n
    # Scores should be sorted descending
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_argpartition_k_greater_than_n():
    """FIX 1: When k > len(scores), should return all items."""
    from psafe.retrievers.bgem3_retriever import BGEM3MultiFunctionRetriever

    retriever = BGEM3MultiFunctionRetriever.__new__(BGEM3MultiFunctionRetriever)
    n = 5
    d = 4
    retriever._corpus_dense_vecs = np.random.randn(n, d).astype(np.float32)
    retriever.top_k = 100  # much larger than n

    query = np.random.randn(d).astype(np.float32)
    idx, scores = retriever.retrieve_dense(query, k=100)
    assert len(idx) == n  # can't return more than n


def test_argpartition_empty():
    """Edge case: empty corpus."""
    from psafe.retrievers.bgem3_retriever import BGEM3MultiFunctionRetriever

    retriever = BGEM3MultiFunctionRetriever.__new__(BGEM3MultiFunctionRetriever)
    retriever._corpus_dense_vecs = None
    retriever.top_k = 10

    idx, scores = retriever.retrieve_dense(np.random.randn(4).astype(np.float32))
    assert len(idx) == 0
    assert len(scores) == 0


def test_lexical_query_dict():
    """FIX 3: lexical_weights as dict should work."""
    from psafe.retrievers.bgem3_retriever import BGEM3MultiFunctionRetriever

    retriever = BGEM3MultiFunctionRetriever.__new__(BGEM3MultiFunctionRetriever)
    retriever.use_dense = False
    retriever.use_sparse = True
    retriever.use_multivector = False
    retriever.top_k = 5
    retriever.candidate_k = 10
    retriever._corpus_dense_vecs = None
    retriever._corpus_lexical_weights = [{"hello": 1.0}, {"world": 2.0}]
    retriever._corpus_colbert_vecs = None

    q_emb = {"lexical_weights": {"hello": 1.0}}  # dict, not list
    idx, scores = retriever.retrieve(q_emb)
    assert len(idx) <= 5


def test_lexical_query_list():
    """FIX 3: lexical_weights as list should work."""
    from psafe.retrievers.bgem3_retriever import BGEM3MultiFunctionRetriever

    retriever = BGEM3MultiFunctionRetriever.__new__(BGEM3MultiFunctionRetriever)
    retriever.use_dense = False
    retriever.use_sparse = True
    retriever.use_multivector = False
    retriever.top_k = 5
    retriever.candidate_k = 10
    retriever._corpus_dense_vecs = None
    retriever._corpus_lexical_weights = [{"hello": 1.0}, {"world": 2.0}]
    retriever._corpus_colbert_vecs = None

    q_emb = {"lexical_weights": [{"hello": 1.0}]}  # list format
    idx, scores = retriever.retrieve(q_emb)
    assert len(idx) <= 5


def test_lexical_query_empty():
    """FIX 3: Missing lexical_weights should not crash."""
    from psafe.retrievers.bgem3_retriever import BGEM3MultiFunctionRetriever

    retriever = BGEM3MultiFunctionRetriever.__new__(BGEM3MultiFunctionRetriever)
    retriever.use_dense = False
    retriever.use_sparse = True
    retriever.use_multivector = False
    retriever.top_k = 5
    retriever.candidate_k = 10
    retriever._corpus_dense_vecs = None
    retriever._corpus_lexical_weights = [{"hello": 1.0}, {"world": 2.0}]
    retriever._corpus_colbert_vecs = None

    q_emb = {}  # no lexical_weights at all
    idx, scores = retriever.retrieve(q_emb)
    assert len(idx) <= 5
