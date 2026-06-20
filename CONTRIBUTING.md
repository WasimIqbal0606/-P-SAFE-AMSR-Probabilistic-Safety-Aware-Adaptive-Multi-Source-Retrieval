# Contributing to P-SAFE-AMSR

Thank you for your interest in contributing to P-SAFE-AMSR (Probabilistic Safety-Aware Adaptive Multi-Source Retrieval).

## Development Setup

1. Create and activate a Python 3.11+ environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run unit tests to verify setup.

```bash
pytest tests/ -v
```

## Code Style

- Follow clear, small, focused changes.
- Keep function and variable names descriptive.
- Preserve evaluation methodology: maintain strict train/validation/test isolation.
- Prefer deterministic scripts and reproducible outputs (always use explicit seeds).
- All routing features must be defined in `src/psafe/feature_extractor.py:FEATURE_NAMES`.
- All actions must be defined in `src/psafe/actions.py`.
- All baselines must be registered in `src/psafe/baselines.py:BASELINE_ROUTERS`.

## Testing

Run the full test suite before opening a pull request:

```bash
pytest tests/ -v
```

For a smoke-test experiment run:

```bash
python experiments/run_psafe_v1_experiments.py \
  --datasets scifact \
  --seeds 42 \
  --modes balanced
```

## Experiment Integrity

- **Never evaluate on training or validation data.** All router evaluations must use held-out test queries.
- **Always save reproducibility artifacts** (`reproducibility_manifest.json`, `split_hash`, query IDs).
- **Do not modify validated results.** Files in `results/validated/` are the ground truth. New runs go into separate directories until validated.
- **Report limitations honestly.** If a new feature does not improve results on some datasets, document that.

## Commit and Pull Request Rules

- Use a concise commit title that explains intent.
- Include a short summary of changed files and why.
- Mention benchmark or reproducibility impact if relevant.
- Do not include large generated datasets, cached embeddings, or temporary outputs in commits.

## Security and Secrets

- Never commit credentials or API keys.
- Use environment variables for secrets.
- Verify logs and output files do not contain sensitive values.
