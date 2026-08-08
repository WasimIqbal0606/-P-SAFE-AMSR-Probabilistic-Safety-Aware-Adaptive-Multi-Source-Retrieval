# Submission Audit

Status: **NOT SUBMISSION READY**

This document is evidence-derived and does not override the machine-readable
artifacts or the fail-closed auditor.

## Retained evidence

- 36 primary result directories: four datasets x three split seeds x three
  modes.
- Per-query P-SAFE, Dense, Deep Hybrid, latency, prediction, and eight-baseline
  artifacts for every primary run.
- Exactly 1000 matched-budget random allocations per retained condition.
- Held-out calibration diagnostics derived from committed action predictions.
- Paired query-level statistics and Holm-adjusted non-inferiority decisions.

## Removed claims

- Ten independent training-seed stability: removed. The historical records
  copied canonical test nDCG/latency/HAR and lacked genuine split, calibration,
  and prediction provenance.
- Feature-group ablations: removed. The archived generator created surrogate
  score features and sampled train/validation labels from test-side deltas.
- Twelve-baseline completeness: removed. Only eight baseline policies have
  genuine per-run artifacts; matched-budget random is retained separately.
- BM25-disagreement and cost-only results: removed. Their previous construction
  used test-side proxies/tuning and fallback scientific values.
- Mechanically proven zero leakage: removed. Historical primary runs did not
  commit explicit train and validation query IDs.

## Retained claims

- Seed-42 High Recall improves over Dense on SciFact, FiQA, and NFCorpus with
  measured latency savings relative to always-on Deep Hybrid.
- ArguAna Balanced is a protection/no-benefit regime for always-on Deep Hybrid.
- Matched-budget query-selection superiority is established on ArguAna only;
  it is not generalized across datasets.
- Gain/harm calibration and discrimination are heterogeneous.
- Holm-adjusted non-inferiority at epsilon 0.010 is established only for
  NFCorpus High Recall.

## Release blocker

The historical primary artifacts contain held-out test IDs and split sizes but
not explicit train/validation IDs. The repaired runner now writes a complete
`split_manifest.json`, but the old runs have not been rerun. The submission
auditor must therefore fail the split-provenance gate.

## Verification status

- The full repository suite passes in the repaired project environment:
  `63 passed, 1 warning`.
- The dependency-light adversarial auditor suite passes: `13 passed`.
- The clean semantic audit passes every retained-evidence gate and fails the
  split-provenance gate, ending with `SUBMISSION AUDIT: FAIL`.
- The project environment is CPython 3.11.9 with Torch 2.11.0+cu128; CUDA is
  visible on the NVIDIA GeForce RTX 5070 Ti. The missing base interpreter was
  restored without replacing the venv's installed packages.
- All eight retained paper figures were regenerated from committed evidence in
  PDF and PNG form. The LaTeX manuscript PDF remains unbuilt because a LaTeX
  toolchain is not installed.

Terminal status is produced only by:

```text
python src/psafe/audit_submission.py
```
