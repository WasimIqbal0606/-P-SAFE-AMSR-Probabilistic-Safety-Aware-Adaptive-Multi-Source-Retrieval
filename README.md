# Adaptive Hybrid Retrieval Controller (AHRC)

An uncertainty-aware, dynamic retrieval framework that optimizes the cost-quality tradeoff in vector search via multi-signal uncertainty estimation, adaptive candidate sizing, and selective graph expansion.

## 🚀 Overview
Most dense retrieval systems statically retrieve a fixed $k$ number of candidates (e.g., `k=10`) regardless of query difficulty. This rigid approach either wastes compute on easy queries or underperforms on complex, ambiguous queries.

The **Adaptive Hybrid Retrieval Controller (AHRC)** solves this by introducing a dynamic control policy. By estimating the "uncertainty" of a query on-the-fly, AHRC decides whether to halt retrieval early, expand the candidate pool, or traverse an underlying task relationship graph to rescue difficult queries.

### What This System Does
✅ **System-level innovation**: Replaces static retrieval with dynamic inference pipelines.  
✅ **Adaptive inference design**: Allocates compute budget (k-size, reranking depth) dynamically per query.  
✅ **Hybrid retrieval**: Combines dense approximate nearest neighbors (FAISS HNSW) with semantic graph traversals.  
✅ **Cost-quality tradeoffs**: Demonstrates Pareto-optimal performance for Retrieval-Augmented Generation (RAG) and dense search.  

### Domains
- Retrieval Systems (IR / Search)
- Efficient AI (Compute-aware inference)
- Adaptive Decision Policies
- LLM / RAG Infrastructure

---

## 🧠 Core Architecture

The AHRC pipeline consists of 5 modular stages executed sequentially for each query:

1. **Initial Dense Retrieval (Staged)**: Performs a fast, approximate nearest-neighbor search via FAISS HNSW for a small initial candidate set.
2. **Uncertainty Estimation (Quality Guard)**: Computes a multi-signal uncertainty score $U \in [0,1]$ using:
   - **Margin**: Gap between top-1 and top-2 similarities.
   - **Spread**: Gap between top-1 and top-5 similarities.
   - **Graph Ambiguity**: Structural connectivity of the candidate pool.
   - **Historical Context**: Rolling confidence scores for specific domains.
3. **Adaptive Controller**: Maps the uncertainty score $U$ to a dynamic `RetrievalDecision`.
   - *Confident queries* stop early.
   - *Uncertain queries* trigger deeper search ($k \uparrow$).
4. **Selective Graph Expansion**: If uncertainty remains high, the controller expands the top-3 dense seeds by traversing a semantic task relationship graph, rescuing hard queries that dense embeddings alone fail to capture.
5. **Dynamic Reranking**: Executes an exact inner-product rerank (or cross-encoder) exclusively on the expanded pool of high-uncertainty queries.

---

## 🔬 Experimental Framework & Ablation Study

This repository includes a fully synthetic benchmarking suite designed to generate mathematically defensible, rigorous evaluation of the AHRC.

### Benchmark Generation (`dataset_generator.py`)
Generates tasks with localized multi-level ground-truth relevance (0–3 scale) using `all-MiniLM-L6-v2` embeddings, ensuring that relevance isn't purely distance-based but semantic and structural.

### Built-in Baselines (`baselines.py`)
The system evaluates AHRC against controlled baselines to ensure scientific rigor:
- **BM25**: Standard lexical matching using `rank-bm25`.
- **Dense Fixed-k**: Standard static FAISS retrieval.
- **Dense + Graph Fixed**: Always-on, non-adaptive graph expansion.

### Evaluation Metrics (`evaluation.py`)
The framework focuses on metrics relevant to Efficient AI and IR:
- **Quality**: $Recall@k$, $nDCG@k$, $MRR$.
- **Cost / Latency**: Candidates explored, $p50/p95/p99$ latency ($ms$), cost-per-recall ratios.

---

## ⚙️ Running the Research Suite

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install rank-bm25
```

### 2. Run the Full Experiment Suite
Runs dataset generation, index building, all baselines, and multiple AHRC modes (Lite, Balanced, High-Recall) to plot the Pareto frontier.
```bash
python -m ahrc.run_experiments --tasks 50000 --queries 1000 --index hnsw
```

### 3. Run the Ablation Study
Systematically disables components (Uncertainty, Adaptation, Graph Expansion) to measure isolated algorithmic contributions.
```bash
python -m ahrc.ablation_study --tasks 10000 --queries 200
```

### 4. Visualizations
The framework automatically generates publication-quality Matplotlib plots in the `ahrc_results/` directory:
- `accuracy_vs_latency.png`
- `cost_vs_recall.png`
- `ablation_bars.png`
- $nDCG@k$ and $Recall@k$ scaling curves.

---

## 📈 Current Operating Modes

The controller is calibrated for a Pareto frontier of cost-performance tradeoffs:
- **AHRC-lite**: Aggressively shrinks budget for clear queries (min $k=6$, max $k=12$). Best for maximum throughput.
- **AHRC-balanced**: Preserves near-baseline $nDCG$ while reducing candidate exploration.
- **AHRC-high-recall**: Triggers deep graph expansion for edge-case queries, maximizing quality under a strict upper bound.

---

## 📝 License
MIT License. Feel free to use this framework to benchmark adaptive retrieval layers in your own LLM/RAG infrastructures.
