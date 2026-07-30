# Submission Checklist

## Manuscript

- [x] All known typos fixed
- [x] Equations consistent with evaluated router code
- [x] Claims match `results/validated`
- [x] Figures readable and exported as SVG, vector PDF, and PNG
- [x] Statistical tests verified from paired query-level data
- [x] Baselines clearly defined, tuned on validation, and evaluated on test
- [x] Three-seed appendix included
- [x] No unsupported SOTA, guarantee, or universal-superiority claims
- [x] Reproducibility statement included
- [x] Final PDF manually inspected from rendered page images

## Evidence checks

- [x] 36 dataset/seed/mode runs audited
- [x] Query IDs unique and matched within paired comparisons
- [x] Stored means reproduced from per-query CSV files
- [x] Paired t statistics and p-values reproduced
- [x] Wilcoxon W and nonzero-pair counts reproduced
- [x] Bootstrap confidence intervals reproduced
- [x] Paired sign-permutation p-values reproduced
- [x] Seed-42 router baselines use the same test queries
- [x] Random-router unmatched-activation limitation disclosed
- [x] Baseline decision-overhead timing asymmetry disclosed

## Build and QA

- [x] `latexmk -pdf main.tex` attempted in the packaged `submission` directory
- [x] MiKTeX `latexmk` bootstrap failure documented (missing local Perl package and unavailable mirror)
- [x] Deterministic fallback build completed with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`
- [x] No undefined citations or references
- [x] No overfull boxes
- [x] Only non-blocking underfull-box warnings remain
- [x] PDF opens and all 12 pages render
- [x] Figures, captions, tables, equations, appendices, and bibliography visually inspected
- [x] Forbidden typo strings absent from source and extracted PDF text

## Reproduction commands

From the repository root:

```powershell
C:\Users\Waseem\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe paper\tools\build_evidence_tables.py
$env:NODE_PATH='C:\Users\Waseem\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
C:\Users\Waseem\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe paper\tools\generate_submission_figures.js
```

From `paper`:

```powershell
C:\Users\Waseem\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe -interaction=nonstopmode -halt-on-error manuscript.tex
C:\Users\Waseem\AppData\Local\Programs\MiKTeX\miktex\bin\x64\bibtex.exe manuscript
C:\Users\Waseem\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe -interaction=nonstopmode -halt-on-error manuscript.tex
C:\Users\Waseem\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe -interaction=nonstopmode -halt-on-error manuscript.tex
```

## Venue readiness

- [x] arXiv technical-report package is ready
- [x] Workshop-content package is ready
- [ ] Venue-specific anonymization/template/page-limit conversion completed
- [ ] Journal-level matched-activation/disagreement/cost-only/direct-competitor baselines completed
- [ ] Journal-level calibration diagnostics and fixed-test repeated fitting completed
