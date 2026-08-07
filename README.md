# P-SAFE-AMSR: Probabilistic Safety-Aware Adaptive Multi-Source Retrieval

[![Submission Audit](https://img.shields.io/badge/Submission%20Audit-PASS%20(18%2F18)-brightgreen.svg)](docs/INDEPENDENT_REAUDIT.md)
[![Tests](https://img.shields.io/badge/pytest-60%20passed-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/Manuscript-PDF%20Built-blue.svg)](paper/manuscript.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An auditable, reproducible, calibrated per-query retrieval escalation controller over Dense and Deep Hybrid pipelines.

---

## Table of Contents

1. [Key Capabilities & Empirical Findings](#key-capabilities--empirical-findings)
2. [What is "Safety" in P-SAFE?](#what-is-safety-in-p-safe)
3. [Canonical Dataset Scope](#canonical-dataset-scope)
4. [Randomness & Explicit Seed Namespaces](#randomness--explicit-seed-namespaces)
5. [Calibration Diagnostics Suite](#calibration-diagnostics-suite)
6. [Comprehensive 12-Baseline Suite](#comprehensive-12-baseline-suite)
7. [Router Component & Feature Ablations](#router-component--feature-ablations)
8. [Statistical Protocols & Non-Inferiority Testing](#statistical-protocols--non-inferiority-testing)
9. [Automated Submission Auditor CLI](#automated-submission-auditor-cli)
10. [Reproducibility & Execution Commands](#reproducibility--execution-commands)
11. [Failure-Case Analysis & Known Limitations](#failure-case-analysis--known-limitations)

---

## Key Capabilities & Empirical Findings

Modern search and RAG systems commonly execute expensive Cross-Encoder reranking for every query. However, uniform cascade escalation is computationally wasteful and frequently degrades ranking quality on queries where first-stage dense retrieval is already optimal.

**B-P-SAFE-AMSR** dynamically predicts whether a query benefits from compute escalation before running expensive stages.

### Validated Results Summary (Seed 42 Primary Split)

| Benchmark | Mode | Dense nDCG | Deep Hybrid | P-SAFE nDCG | Latency Saving vs Hybrid | Hybrid Activation | Behavioral Regime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SciFact** | High Recall | 0.6466 | 0.7330 | **0.7023** | **31.6%** | 66.4% | Selective Escalation |
| **FiQA** | High Recall | 0.4085 | 0.4327 | **0.4285** | **28.0%** | 64.0% | Selective Escalation |
| **NFCorpus** | High Recall | 0.3339 | 0.3632 | **0.3626** | **20.4%** | 79.6% | Selective Escalation |
| **ArguAna** | Balanced | 0.3946 | 0.3915 | **0.4069** | **66.1%** | 15.0% | Protection / No-Benefit |

* **SciFact, FiQA, NFCorpus:** P-SAFE captures the majority of neural reranking quality while saving **20.4%--31.6%** of measured end-to-end latency.
* **ArguAna (Protection Regime):** Always-on reranking causes net harm (0.3915 vs 0.3946). P-SAFE selectively suppresses escalation on 85% of queries, saving **66.1%** latency and achieving higher nDCG (**0.4069**) than either static endpoint.
* **Matched-Budget Random Baseline (1000 Repetitions):** On ArguAna, P-SAFE ($0.4069$) demonstrates statistically significant query-selection superiority over matched random ($0.3942 \pm 0.0046, \Delta = +0.0128, p=0.003$). On SciFact Balanced, matched random achieves $0.7017 \pm 0.0110$ ($\Delta = -0.0052, p=0.687$), confirming that compute allocation efficiency exists across all sets, while query-selection superiority over random allocation is domain-dependent.
* **Deterministic Training Stability:** 10 independent training seeds on the fixed primary split yield $SD = 0.0000$, confirming closed-form deterministic model convergence. Sample partition sensitivity across split seeds is $SD = 0.0020 - 0.0342$.

---

## What is "Safety" in P-SAFE?

> **Operational Definition:** "Safety" in this repository strictly refers to **risk-aware protection against retrieval quality regression** ($\Delta\text{nDCG}@10 < -0.01$).

* **What it IS:** An inspectable penalty ($\lambda_{\mathrm{harm}} P_{\mathrm{harm}}$) and probability gate ($\pharm \le \tau_{\mathrm{harm}}$) preventing over-treatment on queries where neural rerankers distort already optimal dense lists.
* **What it is NOT:** This work does **not** evaluate content safety, hate speech filtering, prompt-injection defense, adversarial robustness, or clinical decision guarantees.

---

## Canonical Dataset Scope

The primary paper experiment matrix (`configs/paper_experiment.yaml`) freezes:
* **Canonical Primary Datasets (4):** `scifact`, `fiqa`, `nfcorpus`, `arguana`
* **Exploratory Datasets (1):** `trec-covid` (declared exploratory due to small sample size $N=25$ and missing lite mode)
* **Data-Split Seeds (3):** `42` (primary), `123`, `2026`
* **Operating Modes (3):** `lite`, `balanced`, `high_recall`
* **Audited Primary Runs:** $4 \text{ datasets} \times 3 \text{ split seeds} \times 3 \text{ modes} = 36 \text{ audited runs}$

All query splits enforce strict disjointness assertions: $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$.

---

## Randomness & Explicit Seed Namespaces

To prevent conflating split sensitivity with training stochasticity, all randomness is organized into explicit namespaces:

```yaml
# configs/paper_experiment.yaml
seed_namespaces:
  split_seed: [42, 123, 2026]              # Data partition seeds
  router_training_seed: [11, 22, ..., 111]  # 10 repeated fitting seeds
  calibration_seed: 42                      # CalibratedClassifierCV fold seed
  random_baseline_seed: 42                  # Base seed for 100-rep random baseline
  bootstrap_seed: 42                        # 5,000 bootstrap resamples
  permutation_seed: 42                      # 2,000 sign permutations
```

---

## Calibration Diagnostics Suite

Probabilities $\pgain(A_6 \mid x) = P(\Delta > +0.05)$ and $\pharm(A_6 \mid x) = P(\Delta < -0.01)$ are calibrated using Platt sigmoid scaling with 3-fold cross-validation on the validation split.

Diagnostics evaluated on test splits (`results/calibration/calibration_metrics.json`):
* **Brier Score:** Mean squared calibration error across positive and rare events ($0.1042 - 0.1704$ on SciFact).
* **Expected Calibration Error (ECE):** Standard 10-bin uniform ECE ($0.0712 - 0.0886$) and 5-bin adaptive quantile ECE ($0.0654 - 0.0821$).
* **Discriminative Power:** AUROC ($0.763 - 0.819$) and AUPRC ($0.412 - 0.771$).
* **Calibration Slope & Intercept:** Slope $\approx 1.01$ and intercept $\approx -0.04$, confirming well-aligned probabilities.

---

## Comprehensive 12-Baseline Suite

P-SAFE is benchmarked against 12 routing policies on identical candidate pools and test splits:

1. **Dense-only ($A_0$):** Zero-incremental-cost dense retrieval (BGE-M3).
2. **Always-Hybrid ($A_6$):** Full multi-stage pipeline with Cross-Encoder reranking.
3. **Random:** Uniform escalation at validation advantage rate.
4. **Matched-Budget Random:** Matches exact P-SAFE test escalation count $k = \mathrm{round}(f \cdot N)$ across 100 random seeds (with 95% CIs).
5. **Dense-margin:** Escalates when top-score gap (score 1 minus score 2) is below threshold.
6. **Dense-entropy:** Escalates when normalized score entropy exceeds threshold.
7. **BM25-disagreement:** Escalates when lexical-dense Jaccard overlap is low.
8. **Cost-only:** Escalates based on query length and token complexity proxy.
9. **Regression-only:** Thresholds raw predicted $\hat{\delta}$.
10. **Classifier-only:** Thresholds uncalibrated gain probability $\pgain$.
11. **Oracle:** Test-label diagnostic upper bound choosing optimal action per query.
12. **B-P-SAFE:** Calibrated utility- and safety-gated routing.

---

## Router Component & Feature Ablations

Controlled ablations on identical splits isolate the contribution of individual router mechanisms (`results/ablations/ablation_results.json`):

* **Full B-P-SAFE:** Achieves highest macro nDCG ($0.4560$).
* **Minus $P_{\mathrm{harm}}$ Penalty:** Reduces harm avoidance and lowers quality on datasets with reranking risk.
* **Feature Group Ablations:** Disagreement signals (Dense--BM25 overlap, rank correlation) and Dense score distributions provide the strongest routing utility ($\Delta_{\mathrm{full}} = -0.014$ when omitted).

---

## Statistical Protocols & Non-Inferiority Testing

All comparisons are paired by query ID and evaluated using:
* Two-sided paired $t$-tests and Wilcoxon signed-rank tests.
* 5,000-draw paired bootstrap 95% confidence intervals and 2,000-draw sign-permutation tests.
* **Holm-Bonferroni Step-Down Correction:** Controls family-wise error across multiple datasets.
* **Formal Non-Inferiority Testing ($\epsilon = 0.010$):** Tests whether B-P-SAFE is non-inferior to Deep Hybrid within a 1.0% nDCG@10 margin ($H_0: \bar{\Delta} \le -\epsilon$). Non-inferiority is statistically established for ArguAna and NFCorpus.

---

## Automated Submission Auditor CLI

Verify 100% publication readiness and provenance across all 18 criteria:

```powershell
python audit_submission.py
# or
python -m psafe.audit_submission
```

**Verification Checklist (18 Criteria):**
1. Canonical dataset configuration matches manuscript ($N=4$, seeds=[42, 123, 2026]).
2. Exactly 36/36 primary evidence runs present and validated.
3. Strict disjoint split assertion: $\text{train} \cap \text{val} = \emptyset, \text{train} \cap \text{test} = \emptyset, \text{val} \cap \text{test} = \emptyset$.
4. Comprehensive 12-baseline suite verified.
5. Matched-budget random baseline evaluated over 100 repetitions.
6. Calibration diagnostics complete (Brier, ECE, AUROC, AUPRC, slope/intercept).
7. Holm-Bonferroni correction and Non-Inferiority tests ($\epsilon = 0.010$) verified.
8. 10-training-seed fixed-split stability analysis present.
9. Router component and feature-group ablations complete.
10. All LaTeX tables generated from machine-readable JSON/CSV artifacts.
11. Publication figures (PDF & PNG) generated and verified.
12. Machine-readable claim registry (`paper/claim_registry.json`) validated.

---

## Reproducibility & Execution Commands

### 1. Run Complete Evidence Pipeline
```powershell
python experiments/run_comprehensive_evidence.py
```

### 2. Generate Publication Tables
```powershell
python generate_paper_tables.py
```

### 3. Generate Publication Figures (PDF & PNG)
```powershell
python generate_figures.py
```

### 4. Run Automated Submission Audit
```powershell
python audit_submission.py
```

### 5. Run Pytest Test Suite
```powershell
pytest
```

---

## Failure-Case Analysis & Known Limitations

1. **False Positives (Over-Escalation):** On queries where Dense retrieval achieves nDCG = 1.0, high lexical specificity or moderate graph degree can trigger unnecessary Cross-Encoder execution, incurring compute without quality gain.
2. **False Negatives (Under-Escalation):** Hard queries with low dense top-score gap where $P_{\mathrm{gain}}$ was underestimated due to sparse vocabulary overlap, resulting in remaining at Dense.
3. **Binary Action Space:** The current verified release focuses on binary Dense vs Deep Hybrid routing. Intermediate multi-tier action routing (A0--A16) is archived in code and remains future work.
4. **Hardware Specificity:** Latency is measured on an NVIDIA RTX 5070 Ti GPU; CPU-only or distributed environments will exhibit different latency profiles.

---

## Citation

```bibtex
@article{iqbal2026whentorerank,
  title={When to Rerank: Calibrated Risk- and Cost-Aware Routing for Dense--Hybrid Retrieval Cascades},
  author={Iqbal, Wasim},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026}
}
```
