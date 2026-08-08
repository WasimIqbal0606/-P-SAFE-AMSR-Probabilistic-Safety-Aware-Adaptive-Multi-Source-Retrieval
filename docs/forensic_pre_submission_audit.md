# Forensic Pre-Submission Audit

Frozen baseline commit: `8277a380c7309a12f094d1f9fde0db58b6951ca7`

## VALID

- 36 primary result directories for SciFact, FiQA, NFCorpus, and ArguAna.
- Per-query Dense, Deep Hybrid, and P-SAFE nDCG values and action predictions.
- Per-run latency and eight executed baseline artifacts.
- Matched-budget random analysis after regeneration with exactly 1000
  allocations and exact P-SAFE action counts.
- Held-out calibration diagnostics regenerated from committed predictions.
- Paired statistics, Holm adjustment, and the NFCorpus High Recall
  non-inferiority decision.

## INVALID

- Generated `train_features.npz`, `val_features.npz`, and `test_features.npz`:
  retrieval features were sinusoidal/index-based surrogates and train/validation
  labels were sampled from the test delta distribution.
- Feature-group ablation results derived from those matrices.
- Fixed-split 10-training-seed records that copied primary nDCG, latency, and
  activation rather than evaluating each fitted action vector.
- BM25-disagreement results that used `pred_delta` and test-distribution
  thresholding.
- Cost-only and missing-baseline fallbacks using fixed latency/activation values.

## STALE

- Manuscript and README statements claiming 10 independent training seeds,
  12 verified baselines, 100 matched-random repetitions, and validation-fitted
  calibration.
- Matched-random numbers and baseline/ablation/stability figures generated from
  earlier artifacts.
- Compiled manuscript PDFs containing removed claims.

The stale prose was corrected. Stale or invalid binary artifacts were moved to
`archive/invalid_evidence/` rather than silently retained as paper evidence.

## UNSUPPORTED

- Mechanically proven zero split leakage for the historical primary runs. The
  historical manifests contain test IDs and split sizes but not explicit train
  and validation IDs.
- Fixed-split repeated-fitting variance.
- Causal feature-group importance.
- BM25-disagreement and cost-only baseline performance.
- Universal superiority over random allocation or traditional routers.

## Release Gate

The semantic audit intentionally fails until the canonical primary runs are
rerun with explicit `split_manifest.json` files. The repaired runner passes the
requested seed to the splitter and writes sorted train/validation/test IDs and
SHA-256 hashes for future runs.
