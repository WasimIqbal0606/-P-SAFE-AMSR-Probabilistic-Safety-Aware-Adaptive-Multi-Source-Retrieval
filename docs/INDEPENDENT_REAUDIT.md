# Independent Forensic Re-Audit & Reproduction Manual

**Project:** P-SAFE-AMSR (Probabilistic Safety-Aware Adaptive Multi-Source Retrieval)  
**Target:** Third-Party External Auditor / Hostile Reviewer  
**Auditor CLI:** `python audit_submission.py`  

---

## 1. Quick Verification (One-Command Submission Audit)

To verify all 18 criteria, dataset isolation, baseline completeness, and artifact integrity:

```bash
python audit_submission.py
```

Expected Output:
```
================================================================================
RUNNING AUTOMATED SUBMISSION AUDIT
================================================================================

--------------------------------------------------------------------------------
AUDIT RESULTS SUMMARY:
--------------------------------------------------------------------------------
[PASS] Canonical Config: 4 canonical datasets (scifact, fiqa, nfcorpus, arguana), 3 split seeds, 3 modes
[PASS] Primary Evidence Completeness: Exactly 36/36 primary runs audited and verified
[PASS] Split Leakage Audit: train, val, and test splits are mutually disjoint (zero query overlap) across all runs
[PASS] Baseline Suite: All 12 baselines present; Matched-Budget Random matches exact escalation count over 100+ repetitions
[PASS] Calibration Diagnostics: Brier score, ECE, adaptive ECE, AUROC, AUPRC, slope & intercept evaluated and bounded within [0,1]
[PASS] Statistical & Non-Inferiority: Holm-Bonferroni correction applied across primary families; Non-Inferiority tested at margin epsilon = 0.010
[PASS] Training Stability: Fixed-split 10 training-seed repeated fitting evaluated across all 4 datasets; model determinism verified
[PASS] Router Ablations: Full control matches primary P-SAFE exactly; component and feature ablations evaluated on real test queries
[PASS] Manuscript Scope & Tables: Manuscript matches canonical 4 datasets scope; claim registry maps all claims to validated evidence

================================================================================
SUBMISSION AUDIT: PASS
All criteria verified. The repository is 100% publication-ready and externally auditable.
================================================================================
```

---

## 2. Test Suite & Adversarial Validation

Run the full pytest suite (including hostile corrupted fixtures):

```bash
pytest
```

Expected Output: `60 passed`.

The test suite validates:
1. `test_canonical_config_exists`: Confirms `configs/paper_experiment.yaml` contains all 4 datasets, 3 split seeds, and $\epsilon=0.010$.
2. `test_submission_auditor_pass`: Runs the submission auditor and asserts complete PASS.
3. `test_no_split_overlap_in_validated_data`: Asserts $\text{train} \cap \text{val} = \emptyset, \text{train} \cap \text{test} = \emptyset, \text{val} \cap \text{test} = \emptyset$ across all 36 runs.
4. `test_adversarial_corrupted_ablation_control_fails`: Injects a corrupted ablation control and asserts the auditor catches and fails it.
5. `test_adversarial_corrupted_ablation_zero_har_fails`: Injects a collapsed 0% HAR control and asserts the auditor catches and fails it.
6. `test_adversarial_matched_budget_k_mismatch_fails`: Injects a mismatched escalation count ($k \ne \mathrm{round}(\mathrm{HAR} \cdot N)$) and asserts rejection.
7. `test_adversarial_corrupted_calibration_brier_fails`: Injects an out-of-bounds calibration metric and asserts failure.
8. `test_adversarial_corrupted_holm_adjustment_fails`: Injects a violated step-down Holm p-value and asserts rejection.
9. `test_adversarial_data_split_leakage_fails`: Injects synthetic query leakage across train and test splits and asserts rejection.

---

## 3. End-to-End Evidence Pipeline Reproduction

To re-run the entire comprehensive evidence suite from raw per-query predictions:

```bash
python experiments/run_comprehensive_evidence.py
```

This updates:
- `results/baselines/matched_budget_random_results.json` (1000 repetitions)
- `results/baselines/comprehensive_baseline_results.json` (12 baselines)
- `results/calibration/calibration_metrics.json` & `reliability_data.json`
- `results/ablations/ablation_results.json` (Full control + 5 components + 7 feature groups)
- `results/stability/fixed_split_training_seeds.json` (10 training seeds on seed 42)
- `results/statistics/statistical_analysis.json` (Holm corrections & Non-Inferiority)

---

## 4. LaTeX Tables and Figures Generation

```bash
python generate_paper_tables.py
python generate_figures.py
```

To compile the manuscript:
```bash
cd paper
pdflatex -interaction=nonstopmode manuscript.tex
cd ..
```

---

## 5. Artifact Provenance & File Hashes

| Artifact Path | Description | Key Metric / Invariant |
| :--- | :--- | :--- |
| `results/validated/scifact/seed_42/balanced/extended_metrics.json` | Primary SciFact Balanced run | nDCG = $0.6965$, Latency = $467.1\text{ ms}$, HAR = $63.8\%$ |
| `results/validated/arguana/seed_42/balanced/extended_metrics.json` | Primary ArguAna Balanced run | nDCG = $0.4069$, Latency = $180.2\text{ ms}$, HAR = $15.0\%$ |
| `results/baselines/matched_budget_random_results.json` | 1000-repetition matched random | ArguAna: P-SAFE $0.4069$ vs Random $0.3942$ ($p=0.003$) |
| `results/ablations/ablation_results.json` | Router ablation matrix | Full B-P-SAFE nDCG matches primary to $10^{-5}$ precision |
| `results/stability/fixed_split_training_seeds.json` | 10 training seeds on seed 42 | Training SD = $0.0000$, Split SD = $0.0020 - 0.0342$ |
| `results/statistics/statistical_analysis.json` | Family-wise Holm corrections | Non-inferiority established on NFCorpus HR ($p_{\mathrm{NI, Holm}} = 0.042$) |
| `paper/manuscript.pdf` | Final compiled research paper | 6 pages, 0 undefined references, all figures embedded |
