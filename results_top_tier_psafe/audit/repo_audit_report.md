# P-SAFE-AMSR / B-P-SAFE-AMSR — Full Repository Audit Report
## Generated: 2026-05-07T23:08:00+05:00

---

## AUDIT DECISION: GO-WITH-FIXES

All 11 GO criteria are satisfiable. No absolute blockers found.
12 critical fixes required before production runs.

---

## PHASE 0 — REPOSITORY STRUCTURE

### Active Modules (psafe/)

| Module | Path | Status | Lines |
|--------|------|--------|-------|
| actions.py | psafe/retrievers/actions.py | ACTIVE — Partial | 41 |
| bpsafe_router.py | psafe/routers/bpsafe_router.py | ACTIVE — Needs fixes | 260 |
| feature_extractor.py | psafe/retrievers/feature_extractor.py | ACTIVE — Needs fixes | 155 |
| bgem3_retriever.py | psafe/retrievers/bgem3_retriever.py | ACTIVE — Needs robustness | 85 |
| evaluator.py | psafe/evaluation/evaluator.py | ACTIVE — Minimal | 68 |
| metrics.py | psafe/statistics/metrics.py | ACTIVE — Needs extension | 57 |
| statistical_tester.py | psafe/statistics/statistical_tester.py | ACTIVE — Stub | 88 |
| plotter.py | psafe/visualization/plotter.py | ACTIVE — Partial | 125 |
| generator.py | psafe/reports/generator.py | ACTIVE — Minimal | 43 |
| graph_contribution.py | psafe/utils/graph_contribution.py | ACTIVE — Minimal | 41 |
| latency_tracker.py | psafe/utils/latency_tracker.py | ACTIVE — Minimal | 32 |

### Archived Modules (archive/ahrc/)

| Module | Lines | Role |
|--------|-------|------|
| safe_router.py | 509 | Old v3/v4 router — BASELINE ONLY |
| psafe_router.py | 267 | Intermediate P-SAFE — BASELINE ONLY |
| psafe_experiment_runner.py | 507 | Full experiment runner — BACKEND |
| evaluation.py | 291 | Full evaluator with nDCG/Recall/MRR — BACKEND |
| feature_extractor.py | 157 | Duplicate of psafe version — ARCHIVE |
| statistical_tests.py | 274 | Rich statistical tester — MERGE SOURCE |
| latency_tracker.py | 193 | Rich latency tracker — MERGE SOURCE |
| multi_dataset_runner.py | 272 | Multi-dataset pipeline — NEEDS FIX |
| visualize_results.py | 315 | 12-plot visualization — ARCHIVE |
| psafe_visualize.py | ~120 | P-SAFE visualization — ARCHIVE |
| graph_expander.py | 159 | Graph expander — BACKEND |
| graph_ablation.py | 380 | Graph ablation study — ARCHIVE |
| hybrid_retriever.py | ~400 | HybridRetriever — FIXED BASELINE |
| reranker.py | ~400 | CrossEncoderReranker — BACKEND |
| dataset_interface.py | ~600 | BEIR loader — BACKEND |
| index_manager.py | ~200 | FAISS index — BACKEND |
| baselines.py | ~200 | Dense/BM25 baselines — BACKEND |
| config.py | ~100 | Configuration — BACKEND |
| leakage_safe_split.py | ~100 | Train/val/test split — BACKEND |
| vram_safe_encoder.py | ~80 | Encoding — BACKEND |
| embedding_cache.py | ~60 | Cache — BACKEND |

### Runner Files

| File | Status |
|------|--------|
| run_top_tier_psafe.py | ACTIVE — Main runner, needs multi-dataset |
| run_psafe_amsr.py | DEPRECATED — Older runner |
| run_safe_amsr_v3.py | DEPRECATED — v3 runner |
| run_safe_amsr_v4.py | DEPRECATED — v4 runner |

---

## PHASE 1 — DUPLICATE MAPPING TABLE

| # | Area | Canonical | Archive | Merge? | Risk | Action |
|---|------|-----------|---------|--------|------|--------|
| 1 | Router | psafe/routers/bpsafe_router.py | archive/ahrc/safe_router.py, archive/ahrc/psafe_router.py | NO | LOW | BPSafe is canonical. Old routers are baselines only. |
| 2 | Action Enum | psafe/retrievers/actions.py | archive/ahrc/safe_router.py:Action, archive/ahrc/psafe_router.py:Action | NO | **HIGH** | 3 conflicting Action enums. Canonical = psafe/retrievers/actions.py (A0–A16). Old enums stay in archive. Runner uses OldAction adapter. |
| 3 | Feature Extractor | psafe/retrievers/feature_extractor.py | archive/ahrc/feature_extractor.py | NO | LOW | Identical copies. Both have fixes. Keep psafe version. |
| 4 | Latency Tracker | psafe/utils/latency_tracker.py (32 lines) | archive/ahrc/latency_tracker.py (193 lines) | **YES** | **MEDIUM** | Archive version is richer. Must merge into psafe version. |
| 5 | Visualization | psafe/visualization/plotter.py | archive/ahrc/visualize_results.py, archive/ahrc/psafe_visualize.py | NO | LOW | plotter.py is canonical. Archive versions are data-driven (good) but tied to old interface. |
| 6 | Statistical Tests | psafe/statistics/statistical_tester.py (88 lines) | archive/ahrc/statistical_tests.py (274 lines) | **YES** | **HIGH** | Archive version has Wilcoxon, bootstrap, permutation, pairwise matrix, Holm-Bonferroni. psafe version is a stub with empty `aggregate_multi_seed`. |
| 7 | Evaluation | psafe/evaluation/evaluator.py | archive/ahrc/evaluation.py | NO | LOW | Different roles. psafe/evaluator handles sensitivity splits. archive/evaluation handles nDCG/Recall computation. Both needed. |
| 8 | Metrics | psafe/statistics/metrics.py | None | NO | **MEDIUM** | Taxonomy order is wrong. Missing extended metrics. |
| 9 | Graph | psafe/utils/graph_contribution.py | archive/ahrc/graph_ablation.py | NO | LOW | Different scope. graph_contribution.py = per-query metrics. graph_ablation.py = full study. |
| 10 | Multi-dataset | None in psafe/ | archive/ahrc/multi_dataset_runner.py | **YES** | **HIGH** | No psafe-native multi-dataset runner. Archive version has bugs (num_docs from candidates_total, hardcoded dense latency). |
| 11 | Experiment Runner | run_top_tier_psafe.py | archive/ahrc/psafe_experiment_runner.py | NO | MEDIUM | run_top_tier_psafe.py delegates to archive for BEIR/encoding/indexing. This is acceptable hybrid architecture. |

---

## PHASE 2 — DETAILED BUG AUDIT (20 dimensions)

### 1. Active modules found
11 active modules in psafe/. See table above.

### 2. Archived modules found
36 files in archive/ahrc/. See table above.

### 3. Duplicate modules found
3 duplicate areas: feature_extractor, latency_tracker, statistical_tester.

### 4. Old vs new module mapping
See Phase 1 table.

### 5–8. File disposition
- **Keep**: All psafe/ modules, run_top_tier_psafe.py
- **Archive**: run_psafe_amsr.py, run_safe_amsr_v3.py, run_safe_amsr_v4.py
- **Merge sources**: archive/ahrc/statistical_tests.py → psafe/statistics/statistical_tester.py, archive/ahrc/latency_tracker.py → psafe/utils/latency_tracker.py
- **Canonical**: psafe/ directory is canonical for all new development

### 9. Import errors
- **CRITICAL**: `run_top_tier_psafe.py:119` imports `from archive.ahrc.psafe_router import Action as OldAction` inside the per-query loop. This is functionally correct but wasteful.
- **CRITICAL**: `bpsafe_router.py:189` uses `self.actions[0]` as fallback instead of `Action.A0_DENSE`.
- **WARNING**: `psafe/retrievers/actions.py` has no `__init__.py` in parent — works because `run_top_tier_psafe.py` runs from project root.

### 10. Missing dependencies
- `FlagEmbedding` — Optional, handled by try/except. ✅
- `faiss` — Required for graph/index. Assumed available.
- `rank_bm25` — Required for BM25. Assumed available.
- `sklearn`, `scipy`, `matplotlib`, `seaborn`, `pandas` — Standard ML stack.

### 11. Inconsistent names
- Action enum names differ between 3 files (safe_router uses 5 actions, psafe_router uses 7, actions.py uses 17).
- `RoutingFeatures` alias in archive vs `QueryFeatures` in psafe — compatible via adapter.

### 12. Broken references
- `bpsafe_router.py:118` — `self.actions[0]` assumes list index = A0_DENSE. Not guaranteed if actions list changes order.
- `bpsafe_router.py:189` — Same issue for route fallback.

### 13. Placeholder/fake visualizations
- `plotter.py:124` — Comment says "Other plots will be implemented similarly based on real data only." Only 4 of 15 required plots implemented.
- **No fake data detected** in any visualization. All plots read from real data files.

### 14. Hardcoded/fake metrics
- `plotter.py:38` — Dense latency hardcoded to 0.5ms in Pareto plot.
- `multi_dataset_runner.py:160` — Same hardcoded 0.5ms.
- No fake nDCG or fake metric values detected.

### 15. Leakage risks
- `evaluator.py:62` — Warns about `full_dataset_descriptive` mode. ✅
- `run_top_tier_psafe.py:129` — Uses `create_stratified_split` for leakage-safe split. ✅
- **No leakage detected in router training path.**

### 16. Graph bugs
- **GraphExpander.adjacency**: The archive `graph_expander.py` uses `self.adjacency` (correct). The runner's `_build_knn_graph_fixed` writes to `graph_exp.adjacency` (correct). No `_adjacency` bug found in active code.
- **Graph contribution**: `graph_contribution.py` assumes same ID space. Missing `corpus_ids` parameter for cross-space validation.

### 17. Router bugs
- **BPSafe fallback**: Uses `self.actions[0]` instead of `Action.A0_DENSE`. If actions list doesn't start with A0, fallback breaks.
- **Soft override threshold**: `delta_pred > 0.05/0.1/0.15` — too high. Should be mode-specific with lower values.
- **lambda_latency in config**: `lite: 0.05`, `balanced: 0.01` — WAY too high. These will penalize latency so heavily that almost nothing gets activated. Should be 0.0002/0.00005/0.00001.

### 18. BGE-M3 implementation risks
- No `encode_corpus()`, `encode_queries()`, `retrieve_dense()`, `retrieve_sparse()` methods. Only raw `encode()` and `compute_scores()`.
- No caching for dense_vecs/lexical_weights.
- ColBERT scores applied only to top-k (correct safety).
- Missing graceful degradation to skip BGE-M3 actions when unavailable.

### 19. Multi-dataset summary risks
- `multi_dataset_runner.py:49` — `num_docs` set from `candidates_total` (wrong — this is sum of all candidates across queries).
- `multi_dataset_runner.py:45` — `hybrid_hard_gain` computed by subtraction rather than from safety_metrics.
- No per-mode summary (only Balanced).
- Missing `quality_retention_vs_best_hybrid`, `oracle_gap_closed`, `p_value_vs_best_hybrid`.

### 20. Report overclaiming risks
- `generator.py` includes dynamic limitations. ✅
- No "state-of-the-art" or "top-tier achieved" claims. ✅
- Final print says "Final claim depends on completed experiments and statistical validation." ✅

---

## PHASE 2 — GO/NO-GO CRITERIA EVALUATION

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | Canonical module plan clear | ✅ GO | psafe/ is canonical. archive/ is backend+baseline. |
| 2 | No fake/placeholder plots in final pipeline | ✅ GO | Plotter reads real data. Missing plots = skip, not fake. |
| 3 | One canonical Action enum | ⚠️ FIX | psafe/retrievers/actions.py is canonical but missing metadata dicts. |
| 4 | BPSafeRouter selected as final router | ✅ GO | bpsafe_router.py is the only active router. |
| 5 | Old routers as baselines only | ✅ GO | safe_router.py and psafe_router.py in archive/. |
| 6 | HybridRetriever as fixed baseline | ✅ GO | In archive/, used by ActionSimulator only. |
| 7 | GraphExpander adjacency bug identified | ✅ GO | No _adjacency bug found. adjacency is used correctly. |
| 8 | Multi-dataset summary bug identified | ⚠️ FIX | num_docs, hybrid_hard_gain, missing columns. |
| 9 | FeatureExtractor compatible | ✅ GO | Already has Jaccard, disagreement, numpy-safe graph. |
| 10 | BGE-M3 optional and graceful | ⚠️ FIX | Needs skip logic + skipped_baselines.json. |
| 11 | Statistical testing merged | ⚠️ FIX | Stub in psafe. Rich version in archive. Must merge. |

**All criteria satisfiable. AUDIT DECISION: GO-WITH-FIXES**

---

## CRITICAL FIXES REQUIRED (12)

1. **FIX 1** — Canonical action system: Add metadata dicts to actions.py
2. **FIX 2** — BPSafeRouter: Fix fallback to A0_DENSE, fix lambda defaults, fix soft override
3. **FIX 3** — FeatureExtractor: Add lexical features, feature_schema.json
4. **FIX 4** — GraphExpander: Add unit test (no code bug found, but test needed)
5. **FIX 5** — Graph contribution: Add corpus_ids parameter, more metrics
6. **FIX 6** — BGE-M3: Add full retriever methods, caching, graceful skip
7. **FIX 7** — Evaluation: Make relevance_threshold configurable
8. **FIX 8** — Statistical tester: Merge archive version into psafe
9. **FIX 9** — Extended metrics: Fix taxonomy order, add missing metrics
10. **FIX 10** — Multi-dataset runner: Fix num_docs, best_hybrid, taxonomy
11. **FIX 11** — Latency tracker: Merge archive version into psafe
12. **FIX 12** — Visualization: Add remaining plots, multi-format output

---
