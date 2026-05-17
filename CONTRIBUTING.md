# Contributing Guide

Thank you for contributing to the Neuromorphic Quantum-Cognitive Task Management System.

## Development Setup

1. Create and activate a Python 3.11 environment.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Start backend API.

```bash
python run_api.py
```

4. Start frontend (optional).

```bash
streamlit run main.py
```

## Code Style

- Follow clear, small, focused changes.
- Keep function and variable names descriptive.
- Preserve benchmark methodology: separate cold start from active-search timing.
- Prefer deterministic scripts and reproducible outputs.

## Testing

Run smoke tests before opening a pull request.

```bash
python -m unittest discover -s tests -p "test_*.py"
```

For benchmark validation:

```bash
python golden_graph_generator.py --max-point 50000
python high_density_simulation.py --tasks 50000 --repeats 5
```

## Commit and Pull Request Rules

- Use a concise commit title that explains intent.
- Include a short summary of changed files and why.
- Mention benchmark or reproducibility impact if relevant.
- Do not include large generated datasets or temporary outputs in commits.

## Security and Secrets

- Never commit credentials or API keys.
- Use environment variables for secrets.
- Verify logs and output files do not contain sensitive values.
