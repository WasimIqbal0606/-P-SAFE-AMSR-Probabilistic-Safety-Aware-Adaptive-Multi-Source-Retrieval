"""
Safe-AMSR-SE v3 — Graph Ablation Study
Tests whether graph expansion helps or adds noise.

NOTE (FIX 8): This graph ablation currently tests Dense, Dense+kNNGraph,
and Dense+LexicalGraph. BM25/CE graph ablations are handled elsewhere
(see psafe_experiment_runner.py action simulation).

Graph construction strategies:
  1. kNN semantic graph (FAISS inner product)
  2. Lexical-overlap graph (shared token counting)
  3. Hybrid semantic + lexical graph

Expansion strategies:
  4. Personalized PageRank from dense seeds
  5. Random walk with restart

Reports per strategy:
  - graph_unique_relevant_docs
  - graph_final_top10_relevant_docs
  - graph_latency_cost
  - graph_net_gain
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class GraphAblationResult:
    """Result for one graph strategy."""
    strategy_name: str
    ndcg_at_10: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    latency_mean_ms: float = 0.0
    graph_unique_relevant_docs: float = 0.0
    graph_final_top10_relevant_docs: float = 0.0
    graph_latency_cost_ms: float = 0.0
    graph_net_gain: float = 0.0  # nDCG gain from adding graph
    pool_size_mean: float = 0.0


class GraphAblation:
    """Systematic ablation of graph expansion strategies."""

    def __init__(self, corpus_embeddings: np.ndarray, corpus_texts: List[str]):
        self.corpus_embeddings = corpus_embeddings
        self.corpus_texts = corpus_texts
        self.n_docs = len(corpus_texts)
        self._graphs: Dict[str, Dict[int, List[int]]] = {}

    # ── Graph construction strategies ─────────────────────────────────

    def build_knn_graph(self, k: int = 5) -> Dict[int, List[int]]:
        """Strategy 1: kNN semantic graph using FAISS."""
        import faiss
        n, d = self.corpus_embeddings.shape
        idx = faiss.IndexFlatIP(d)
        idx.add(self.corpus_embeddings)
        _, indices = idx.search(self.corpus_embeddings, k + 1)

        graph = {}
        for i in range(n):
            graph[i] = [int(indices[i, j]) for j in range(1, k + 1)
                        if 0 <= indices[i, j] < n and indices[i, j] != i]
        self._graphs["kNN_semantic"] = graph
        print(f"   ✅ kNN semantic graph: {n:,} nodes, k={k}")
        return graph

    def build_lexical_graph(self, min_overlap: int = 3) -> Dict[int, List[int]]:
        """Strategy 2: Lexical-overlap graph based on shared tokens."""
        from collections import defaultdict

        # Build inverted index of tokens -> doc indices
        token_to_docs = defaultdict(set)
        doc_tokens = []
        for i, text in enumerate(self.corpus_texts):
            tokens = set(text.lower().split())
            doc_tokens.append(tokens)
            for token in tokens:
                token_to_docs[token].add(i)

        # Build graph: connect docs with >= min_overlap shared tokens
        graph: Dict[int, List[int]] = {}
        n = len(self.corpus_texts)

        # Only sample for large corpora
        sample_size = min(n, 5000)
        sample_indices = np.random.choice(n, sample_size, replace=False) if n > sample_size else range(n)

        for i in sample_indices:
            neighbors = set()
            for token in doc_tokens[i]:
                for j in token_to_docs[token]:
                    if j != i and j not in neighbors:
                        overlap = len(doc_tokens[i] & doc_tokens[j])
                        if overlap >= min_overlap:
                            neighbors.add(j)
                            if len(neighbors) >= 10:
                                break
                if len(neighbors) >= 10:
                    break
            graph[i] = list(neighbors)[:10]

        self._graphs["lexical_overlap"] = graph
        print(f"   ✅ Lexical overlap graph: {len(graph):,} nodes, min_overlap={min_overlap}")
        return graph

    def build_hybrid_graph(self, k_semantic: int = 3, k_lexical: int = 3) -> Dict[int, List[int]]:
        """Strategy 3: Hybrid semantic + lexical graph."""
        # Build semantic kNN
        import faiss
        n, d = self.corpus_embeddings.shape
        idx = faiss.IndexFlatIP(d)
        idx.add(self.corpus_embeddings)
        _, sem_indices = idx.search(self.corpus_embeddings, k_semantic + 1)

        # Build lexical neighbors (simplified)
        doc_tokens = [set(text.lower().split()) for text in self.corpus_texts]

        graph: Dict[int, List[int]] = {}
        sample_size = min(n, 5000)

        for i in range(min(n, sample_size)):
            neighbors = set()
            # Add semantic neighbors
            for j in range(1, k_semantic + 1):
                nb = int(sem_indices[i, j])
                if 0 <= nb < n and nb != i:
                    neighbors.add(nb)

            # Add top lexical neighbors
            if i < len(doc_tokens):
                best_lex = []
                for j in range(min(n, 500)):
                    if j != i and j < len(doc_tokens):
                        overlap = len(doc_tokens[i] & doc_tokens[j])
                        if overlap >= 2:
                            best_lex.append((overlap, j))
                best_lex.sort(reverse=True)
                for _, j in best_lex[:k_lexical]:
                    neighbors.add(j)

            graph[i] = list(neighbors)

        self._graphs["hybrid_sem_lex"] = graph
        print(f"   ✅ Hybrid graph: {len(graph):,} nodes")
        return graph

    def expand_with_graph(
        self,
        graph: Dict[int, List[int]],
        seed_indices: np.ndarray,
        query_embedding: np.ndarray,
        max_neighbors: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Expand seed indices using a given graph."""
        visited = set(seed_indices.tolist())
        new_candidates = set()

        for node in seed_indices[:10]:
            node = int(node)
            neighbors = graph.get(node, [])
            for nb in neighbors:
                if nb not in visited:
                    new_candidates.add(nb)
                    visited.add(nb)
                if len(new_candidates) >= max_neighbors:
                    break
            if len(new_candidates) >= max_neighbors:
                break

        if not new_candidates:
            return seed_indices, np.ones(len(seed_indices), dtype=np.float32)

        new_list = list(new_candidates)[:max_neighbors]
        new_embs = self.corpus_embeddings[new_list]
        new_scores = np.dot(new_embs, query_embedding)

        all_idx = np.concatenate([seed_indices, np.array(new_list, dtype=np.int64)])
        seed_scores = np.dot(self.corpus_embeddings[seed_indices.astype(int)], query_embedding)
        all_scores = np.concatenate([seed_scores, new_scores])

        sort_order = np.argsort(-all_scores)
        return all_idx[sort_order], all_scores[sort_order].astype(np.float32)

    def personalized_pagerank(
        self,
        graph: Dict[int, List[int]],
        seed_indices: np.ndarray,
        alpha: float = 0.85,
        max_iter: int = 20,
        top_k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Strategy 4: Personalized PageRank from dense seeds."""
        n = self.n_docs
        scores = np.zeros(n, dtype=np.float64)
        seed_set = set(seed_indices.tolist())

        # Initialize with uniform distribution over seeds
        for idx in seed_indices:
            scores[int(idx)] = 1.0 / len(seed_indices)

        for _ in range(max_iter):
            new_scores = np.zeros(n, dtype=np.float64)
            for node, neighbors in graph.items():
                if scores[node] > 0 and len(neighbors) > 0:
                    share = scores[node] / len(neighbors)
                    for nb in neighbors:
                        if 0 <= nb < n:
                            new_scores[nb] += (1 - alpha) * share

            # Teleport back to seeds
            for idx in seed_indices:
                new_scores[int(idx)] += alpha / len(seed_indices)

            scores = new_scores

        # Return top-k by PageRank score
        top_indices = np.argsort(-scores)[:top_k]
        return top_indices, scores[top_indices].astype(np.float32)

    def random_walk_restart(
        self,
        graph: Dict[int, List[int]],
        seed_indices: np.ndarray,
        n_walks: int = 100,
        walk_length: int = 5,
        restart_prob: float = 0.3,
        top_k: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Strategy 5: Random walk with restart from dense seeds."""
        rng = np.random.default_rng(42)
        visit_counts = {}

        for _ in range(n_walks):
            start = int(rng.choice(seed_indices))
            current = start

            for step in range(walk_length):
                visit_counts[current] = visit_counts.get(current, 0) + 1

                if rng.random() < restart_prob:
                    current = int(rng.choice(seed_indices))
                    continue

                neighbors = graph.get(current, [])
                if neighbors:
                    current = int(rng.choice(neighbors))
                else:
                    current = int(rng.choice(seed_indices))

        # Convert to arrays sorted by visit frequency
        sorted_nodes = sorted(visit_counts.items(), key=lambda x: -x[1])[:top_k]
        indices = np.array([n for n, _ in sorted_nodes], dtype=np.int64)
        scores = np.array([c for _, c in sorted_nodes], dtype=np.float32)
        scores = scores / scores.sum() if scores.sum() > 0 else scores

        return indices, scores

    def run_ablation(
        self,
        queries: List[Dict],
        dense_results: List[Dict],
        evaluator,
        corpus_ids: List[str],
        qrels: Dict,
        bm25_results: Optional[List[Dict]] = None,
        reranker=None,
        baseline_ndcg: float = 0.0,
    ) -> Dict[str, GraphAblationResult]:
        """
        Run full graph ablation study.

        Configurations tested:
          - Dense only (baseline)
          - Dense + BM25 (no graph)
          - Dense + kNN graph
          - Dense + BM25 + kNN graph
          - Dense + BM25 + CrossEncoder (no graph)
          - Dense + BM25 + kNN graph + CrossEncoder
        """
        results = {}

        # Build kNN graph (primary strategy)
        print("\n📊 Graph Ablation Study")
        print("   Building graph variants...")
        knn_graph = self.build_knn_graph(k=5)

        strategies = {
            "Dense": None,
            "Dense+kNN_Graph": ("knn_expand", knn_graph),
        }

        # Also build lexical if corpus is small enough
        if self.n_docs <= 10000:
            lex_graph = self.build_lexical_graph(min_overlap=3)
            strategies["Dense+Lexical_Graph"] = ("knn_expand", lex_graph)

        print("   Running ablation configs...")

        for config_name, graph_config in strategies.items():
            print(f"      {config_name}...", end=" ", flush=True)
            metrics_list = []
            latencies = []
            graph_latencies = []
            unique_rel_docs = []

            for qi, (q, dr) in enumerate(zip(queries, dense_results)):
                t0 = time.perf_counter()
                qid = q["id"]
                qrels_q = qrels.get(qid, {})
                dense_idx = dr["indices"]
                dense_scr = dr["scores"]

                if graph_config is not None:
                    strategy, graph = graph_config
                    t_graph = time.perf_counter()
                    expanded_idx, expanded_scr = self.expand_with_graph(
                        graph, dense_idx[:50], q["embedding"], max_neighbors=20
                    )
                    graph_lat = (time.perf_counter() - t_graph) * 1000
                    graph_latencies.append(graph_lat)
                    final_idx = expanded_idx[:10]
                else:
                    final_idx = dense_idx[:10]
                    graph_latencies.append(0.0)

                elapsed = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed)

                qm = evaluator.evaluate_query(
                    final_idx, qrels_q, corpus_ids,
                    qid, elapsed, len(final_idx),
                )
                metrics_list.append(qm)

                # Count unique relevant docs from graph
                if graph_config is not None:
                    dense_set = set(dense_idx[:50].tolist())
                    graph_set = set(expanded_idx.tolist()) - dense_set
                    relevant_set = {corpus_ids[int(i)] for i in graph_set
                                     if 0 <= int(i) < len(corpus_ids)}
                    rel_count = sum(1 for d in relevant_set if qrels_q.get(d, 0) >= 1)
                    unique_rel_docs.append(rel_count)
                else:
                    unique_rel_docs.append(0)

            agg = evaluator.aggregate(metrics_list, config_name)
            ndcg10 = agg.ndcg_at_k.get(10, 0)

            results[config_name] = GraphAblationResult(
                strategy_name=config_name,
                ndcg_at_10=ndcg10,
                recall_at_10=agg.recall_at_k.get(10, 0),
                mrr=agg.mrr,
                latency_mean_ms=float(np.mean(latencies)),
                graph_unique_relevant_docs=float(np.mean(unique_rel_docs)),
                graph_latency_cost_ms=float(np.mean(graph_latencies)),
                graph_net_gain=ndcg10 - baseline_ndcg,
            )

            print(f"nDCG@10={ndcg10:.4f}, gain={ndcg10 - baseline_ndcg:+.4f}, "
                  f"unique_rel={np.mean(unique_rel_docs):.2f}")

        return results

    def format_results(self, results: Dict[str, GraphAblationResult]) -> str:
        """Pretty-print graph ablation results."""
        lines = ["  Graph Ablation Results:"]
        lines.append(f"  {'Config':<30} {'nDCG@10':>8} {'R@10':>7} {'Net Gain':>9} "
                      f"{'Graph Lat':>10} {'Unique Rel':>10}")
        lines.append("  " + "─" * 80)

        for name, r in results.items():
            lines.append(
                f"  {name:<30} {r.ndcg_at_10:>8.4f} {r.recall_at_10:>7.4f} "
                f"{r.graph_net_gain:>+9.4f} {r.graph_latency_cost_ms:>9.1f}ms "
                f"{r.graph_unique_relevant_docs:>10.2f}"
            )

        return "\n".join(lines)
