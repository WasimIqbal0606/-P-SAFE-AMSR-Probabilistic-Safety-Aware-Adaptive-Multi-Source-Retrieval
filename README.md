# P-SAFE-AMSR: Probabilistic Safety-Aware Adaptive Multi-Source Retrieval

An open-source research repository demonstrating **P-SAFE**, a dynamic routing architecture for retrieval-augmented systems. P-SAFE treats retrieval escalation as a probabilistic, safety-aware decision problem, dynamically balancing cost, quality, and harm-avoidance.

## Overview

In traditional retrieval systems, developers often face a rigid choice:
- **Dense Retrieval** (e.g., dual-encoders): Very fast and cheap, but sometimes lacks lexical precision and structural depth on hard queries.
- **Deep Hybrid** (e.g., Dense + BM25 + CrossEncoder): Achieves state-of-the-art accuracy, but requires significantly more compute latency, GPU overhead, and token limits.

**P-SAFE (v1)** dynamically routes incoming queries between a fast Dense baseline (A0) and an expensive Deep Hybrid pipeline (A6). By using pre-retrieval features (query lexical specificity) and cheap post-retrieval signals (Dense/BM25 overlap, score distributions, nearest-neighbor graph degree), the router estimates:
1. The **Probability of Gain** (likelihood the Hybrid pipeline will improve nDCG).
2. The **Probability of Harm** (likelihood the Hybrid pipeline will degrade results due to out-of-domain CrossEncoder instability).
3. Expected Latency.

The router uses a safety-constrained utility function to decide whether escalating to Deep Hybrid is worth the cost. 

### Why "Safety-Aware"?
P-SAFE explicitly models and penalizes "harm"—cases where advanced deep learning models (like CrossEncoders) confidently degrade performance on queries perfectly answered by simple Dense retrieval. P-SAFE achieves **Quality Retention** while strictly avoiding unnecessary compute.

## Codebase Architecture

This repository hosts the stable **P-SAFE v1** architecture, consisting of the following key modules:

```text
src/psafe/
├── router.py             # B-P-SAFE-AMSR Bayesian Routing Logic
├── feature_extractor.py  # Canonical signal extraction (BM25/Dense overlap, entropies)
├── evaluation.py         # Leakage-safe, stratified split evaluations
├── latency_tracker.py    # Microsecond precision profiling of the routing overhead
├── metrics.py            # Safety-aware metrics (Quality Retention, Harm Avoidance)
├── statistical_tests.py  # Rigorous permutation and paired t-tests
└── baselines.py          # 9 standard and heuristic routing baselines
```

## Running the Canonical Experiment

The canonical runner reproduces the P-SAFE evaluation across BEIR datasets. 

```bash
# Evaluate P-SAFE on SciFact using seed 42
python experiments/run_psafe_v1_experiments.py --datasets scifact --seeds 42 --modes balanced
```

The runner evaluates 9 different baseline routers alongside P-SAFE, generating:
- `extended_metrics.json` (Performance against the Deep Hybrid ceiling)
- `statistical_tests.json` (Significance bounds)
- `latency_breakdown.json` (Cost tradeoffs)

## Experimental Rigor & Reproducibility

We prioritize honest, reproducible empirical science:
1. **Strict Train/Test Isolation:** Router training and threshold tuning happen on isolated splits, evaluated on a held-out test split.
2. **Comprehensive Baselines:** Evaluated against Random, Margin-based, Entropy-based, Regression-only, and Oracle routers.
3. **Validated Audits:** We publish an automated audit (`docs/result_audit.md`) confirming the integrity of pre-computed runs.
4. **No Free Lunch Disclaimer:** P-SAFE is computationally viable only when `(Routing Overhead) < (Hybrid Cost * (1 - Escalation Rate))`. Our latency tracking enforces this reality.

## Future Work (A0-A16 Action Space)
While P-SAFE v1 focuses on Dense (A0) vs Deep Hybrid (A6), our experimental tracking explores a full 16-action space including Knowledge Graph injections, LLM-rewrites, and recursive chunking. This work is actively archived and developed for future versions.

## License
MIT License.
