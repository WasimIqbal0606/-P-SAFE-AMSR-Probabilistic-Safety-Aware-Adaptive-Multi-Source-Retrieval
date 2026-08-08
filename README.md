# B-P-SAFE-AMSR

Evidence-audited implementation of **Binary Probabilistic Safety-Aware Adaptive
Multi-Source Retrieval**, a per-query controller that routes between Dense
retrieval and an always-on Deep Hybrid cascade.

The scientific scope is deliberately narrow. "Safety" means reducing the risk
of retrieval-quality regression, not content safety or security. The retained
paper evidence covers SciFact, FiQA, NFCorpus, and ArguAna with split seeds 42,
123, and 2026 and three routing modes.

## Evidence status

The 36 primary run directories retain per-query nDCG, actions, latency,
statistics, baseline outputs, and reproducibility manifests under
`results/validated/`.

Retained secondary evidence:

- Eight executed baselines with per-query artifacts: Dense-only,
  Always-Hybrid, Random, Dense-margin, Dense-entropy, Regression-only,
  Classification-only, and Oracle.
- Matched-Budget Random with exactly 1000 allocations per condition and the
  exact P-SAFE escalation count.
- Held-out gain/harm diagnostics: Brier, ECE, adaptive ECE, AUROC, AUPRC,
  calibration slope/intercept, and prevalence.
- Paired query-level tests, bootstrap intervals, Holm-adjusted p-values, and
  non-inferiority decisions at the pre-specified margin `epsilon = 0.010`.

Removed secondary evidence:

- The 10-training-seed stability claim was removed because the historical
  records copied primary test metrics and did not preserve genuine fitting
  provenance.
- Feature-group ablations were removed because their feature matrices used
  surrogate retrieval signals and train/validation labels sampled from the
  test delta distribution.
- BM25-disagreement and cost-only baseline claims were removed because valid,
  validation-tuned execution artifacts were not retained.

The removed files remain inspectable under `archive/invalid_evidence/` and are
not publication evidence.

## Defensible results

On split seed 42, High Recall improves over Dense by 0.0557 nDCG@10 on
SciFact, 0.0200 on FiQA, and 0.0287 on NFCorpus, with measured latency savings
of 31.6%, 28.0%, and 20.4% relative to always-on Deep Hybrid. On ArguAna
Balanced, P-SAFE reaches 0.4069 versus 0.3946 for Dense and 0.3915 for Deep
Hybrid, with 15.0% hybrid activation.

At the same escalation count, the 1000-allocation matched-random analysis finds
an ArguAna advantage of +0.0130 (`p = 0.005`). NFCorpus and FiQA have positive
but non-significant margins; SciFact P-SAFE is below the matched-random mean.
Only NFCorpus High Recall establishes Holm-adjusted non-inferiority to Deep
Hybrid at `epsilon = 0.010`.

Gain/harm classifiers use internal cross-validated sigmoid calibration on the
training split; routing thresholds are selected on validation. Held-out
calibration is heterogeneous, so probabilities are treated as operational
policy signals rather than uniformly calibrated estimates.

## Reproducibility limitation

Historical primary manifests preserve split sizes and test query IDs but not
the original train and validation ID lists. Mechanical proof of train/validation/
test disjointness is therefore unavailable for those historical runs. The
repaired canonical runner now passes the requested split seed explicitly and
writes a `split_manifest.json` containing sorted IDs and SHA-256 hashes.

Until the primary runs are regenerated with those manifests, the submission
auditor intentionally reports `SUBMISSION AUDIT: FAIL` for split provenance.

## Commands

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe experiments\run_comprehensive_evidence.py
.\.venv\Scripts\python.exe generate_paper_tables.py
.\.venv\Scripts\python.exe generate_figures.py
.\.venv\Scripts\python.exe audit_submission.py
```

Lightweight Make targets are also provided:

```text
make verify-paper
make audit-paper
```

`verify-paper` regenerates retained secondary evidence and paper artifacts. It
does not rerun neural retrieval. `audit-paper` runs the fail-closed semantic
auditor.

## Traceability

- Claim registry: `paper/claim_registry.json`
- Manuscript: `paper/manuscript.tex`
- Primary evidence: `results/validated/`
- Matched random: `results/baselines/matched_budget_random_results.json`
- Calibration: `results/calibration/calibration_metrics.json`
- Statistics: `results/statistics/statistical_analysis.json`
- Forensic status: `docs/SUBMISSION_AUDIT.md`

This repository does not claim state of the art, universal superiority, zero
degradation, or uniformly reliable calibration.
