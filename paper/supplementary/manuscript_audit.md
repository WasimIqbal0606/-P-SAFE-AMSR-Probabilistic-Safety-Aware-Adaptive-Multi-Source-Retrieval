# Reviewer-Style Manuscript Audit

Audit date: 2026-07-30  
Audited sources: `paper/manuscript.tex`, the pre-revision PDF, implementation under
`src/psafe` and `experiments`, and all runs under `results/validated`.

The pre-fix score is **4.47/10** (67/150). The post-fix score is **8.49/10**
(127.3/150). This is an evidence-readiness score, not an estimate of acceptance
probability.

Legend for blocker columns: A = arXiv technical report, W = workshop, J = journal;
“risk” means material reviewer exposure but not an automatic submission blocker.

| # | Category | Before | After | Pre-fix blocker | Post-fix blocker |
|---:|---|---:|---:|---|---|
| 1 | Title and abstract | 6.0 | 9.0 | J | — |
| 2 | Introduction and problem framing | 5.0 | 8.5 | W/J | — |
| 3 | Related work positioning | 4.0 | 8.0 | W/J | J risk |
| 4 | Method clarity | 4.0 | 8.0 | A/W/J | J risk |
| 5 | Equation consistency | 4.0 | 9.0 | A/W/J | — |
| 6 | Notation consistency | 5.0 | 9.0 | W/J | — |
| 7 | Experimental design | 5.0 | 8.0 | W/J | J |
| 8 | Baseline fairness | 3.0 | 7.0 | A/W/J | J |
| 9 | Statistical testing | 5.0 | 8.5 | W/J | J risk |
| 10 | Figure readability | 4.0 | 9.0 | A/W/J | — |
| 11 | Table formatting | 5.0 | 8.5 | W/J | — |
| 12 | Claims versus evidence | 3.0 | 9.0 | A/W/J | — |
| 13 | Limitations | 5.0 | 9.0 | W/J | — |
| 14 | Reproducibility | 5.0 | 8.5 | W/J | J risk |
| 15 | Submission readiness | 4.0 | 8.3 | A/W/J | J |

## 1. Title and abstract

**Exact pre-fix problems.** The title foregrounded “safety” without delimiting
retrieval-quality harm; the abstract mixed stale seed-42 values with broad claims and
did not expose the binary action space, competitive router baselines, or multi-seed
qualification.

**Exact fixes.** The title now states the actual decision and trade-off. The 150-word
abstract names the binary Dense–Deep-Hybrid setting, four datasets, three split seeds,
seed-42 latency savings, ArguAna’s no-benefit regime, baseline scope, and remaining
limitations. It contains no SOTA, guarantee, or universal-superiority wording.

## 2. Introduction and problem framing

**Exact pre-fix problems.** The framing treated uniform reranking too categorically,
blurred retrieval regression with broader AI safety, and implied that all enumerated
actions had been empirically evaluated.

**Exact fixes.** The introduction now says uniform reranking is a common design, not
an inevitable failure; defines harm as an nDCG decrease; states that content safety is
out of scope; and states explicitly: “This work is not a new retriever or reranker. It
is a calibrated routing layer over existing retrieval modules.” Four contributions are
limited to validated work.

## 3. Related work positioning

**Exact pre-fix problems.** One cascade citation was misattributed, one early-exit
title/year was wrong, selective prediction and calibration were thinly connected, and
the closest 2026 adaptive reranking paper was absent.

**Exact fixes.** The bibliography now uses verified primary records for neural
reranking, operational and jointly optimized cascades, early exit, SelectiveNet,
calibration, FrugalGPT, Adaptive-RAG, and Adaptive Re-Ranking. The direct comparison is
conceptual and does not claim a shared benchmark.

**Residual risk.** A journal version should execute the closest adaptive-reranking
competitor in one harness if code and resources permit.

## 4. Method clarity

**Exact pre-fix problems.** The manuscript omitted deployed soft overrides, called
sigmoid calibration isotonic, implied a real utility lower confidence bound, and
claimed HNSW graph construction despite `IndexFlatIP` code.

**Exact fixes.** The method now documents the real Balanced and High-recall overrides,
the positive-utility requirement, sigmoid/Platt-style cross-validation, prior fallback,
the placeholder nature of the LCB flag, exact Dense/BM25/pool/Cross-Encoder depths, and
flat-FAISS cached five-neighbour expansion with no HNSW.

**Residual risk.** Only the binary A0/A6 policy is validated; full multi-action routing
remains future work.

## 5. Equation consistency

**Exact pre-fix problems.** The full utility and deployment rule described different
routers; the candidate and recovery terms were inconsistent; HGR/OGC typography was
malformed; and malformed delta notation appeared in prose.

**Exact fixes.**

- Equation (1) includes expected delta, harm probability, latency, candidate count,
  and recovery probability with consistent `lambda_harm`, `lambda_lat`,
  `lambda_cand`, and `lambda_rec`.
- Equation (2) is the truthful binary deployed rule using positive utility and the
  mode-dependent admissibility gate, including implemented overrides.
- Equation (5) renders HGR with clean mean-quality subscripts.
- Equation (7) renders OGC against the per-query endpoint oracle.
- The candidate term is explicitly constant for deployed A6 and retained for
  generality.

## 6. Notation consistency

**Exact pre-fix problems.** P-SAFE/B-P-SAFE naming drifted; `hyhid`, `barm`, `nec`,
`Ahard`, and “Protection/Nobenefit” variants appeared; Cohen’s effect notation was
inconsistent.

**Exact fixes.** The full system name appears once, followed by B-P-SAFE throughout.
All malformed tokens are gone. Cohen’s paired effect is named \(d_z\). Dense,
Deep Hybrid, LS, HAR, HGR, OGC, RC, and HA use one definition each.

## 7. Experimental design

**Exact pre-fix problems.** The PDF reported stale test sizes and metrics, did not
distinguish split seeds from repeated training on one test set, and left endpoint
depths ambiguous.

**Exact fixes.** The paper now reports seed-42 test sizes 152/325/162/706, 40/10/50
train/validation/test splits, seeds 42/123/2026, BGE-M3 and BGE-reranker-v2-m3,
candidate depths, RTX 5070 Ti hardware, and cache use. Multi-seed evidence is described
as split sensitivity.

**Journal blocker.** Add more datasets/domains and repeated model fitting on fixed test
sets; ideally add a production-like load/throughput evaluation.

## 8. Baseline fairness

**Exact pre-fix problems.** The main comparison was dominated by static endpoints,
additional router artifacts were omitted, and latency definitions were not aligned.

**Exact fixes.** The paper now defines and reports Dense-only, Always-Hybrid, Random,
Dense-margin, Dense-entropy, Regression-only, Classification-only, Oracle, and
B-P-SAFE. It states training/tuning rules, test-only evaluation, activation, latency,
and paired p-values. It explicitly says Random is validation-derived rather than
test-activation matched, the Oracle is unattainable, and baseline router-decision
overhead is uninstrumented.

**Journal blocker.** Add matched-activation random, BM25/dense disagreement, cost-only,
and closest direct competitor baselines with uniform end-to-end timing.

## 9. Statistical testing

**Exact pre-fix problems.** Some prose p-values were stale or malformed, query pairing
was asserted rather than audited, effect-size naming drifted, and multiple comparisons
were not acknowledged clearly.

**Exact fixes.** The independent auditor checks unique/matched query IDs and recomputes
means, paired t statistics/p-values, Wilcoxon W/nonzero counts, 5,000-draw bootstrap
intervals, and 2,000-draw sign-permutation p-values for 36 runs. The paper reports
mixed-test outcomes rather than selecting only significant results and includes the
required multiple-comparison guardrail.

**Verified typo values.**

- SciFact high recall versus Dense: \(p_t=0.005839\), rendered as
  \(5.84\times10^{-3}\); SciFact balanced is \(p_t=0.01136\).
- NFCorpus balanced versus Dense: Wilcoxon \(p=0.0255\), not a guessed exponent.
- ArguAna has \(n=706\) seed-42 test queries.

**Residual risk.** A journal version should predefine a primary family and correction
or hierarchical testing plan.

## 10. Figure readability

**Exact pre-fix problems.** The supplied PDF used dark raster plots, stale or
out-of-scope datasets, ambiguous labels, and references to nonexistent result paths.

**Exact fixes.** Six light, consistent figures were rebuilt as SVG, vector PDF, and
PNG. Captions were checked against scope, seed aggregation, and axes. The architecture
states no HNSW; baseline and latency charts expose caveats; all figures are cited.
The final 12-page PDF was rendered to PNG and visually inspected page by page.

## 11. Table formatting

**Exact pre-fix problems.** Headers contained malformed lambda names, results were
stale, and some tables overflowed or lacked fairness notes.

**Exact fixes.** All result tables are generated from `results/validated`, use booktabs
and consistent abbreviations, and fit without overfull boxes. Baseline captions state
latency and random-router caveats. Appendix tables use a one-column layout.

## 12. Claims versus evidence

**Exact pre-fix problems.** The draft implied a formal calibration guarantee, a broad
safety property, a universal upper-quality Hybrid endpoint, multi-action validation,
and routing superiority not supported by baseline artifacts.

**Exact fixes.** Final wording uses “explicit probabilistic calibration objective,”
“retrieval self-protection,” and “maximum-compute reference and often, but not always,
the higher-quality endpoint.” It limits experiments to A0/A6, reports strong simple
baselines, and describes every multi-seed result as split sensitivity.

## 13. Limitations

**Exact pre-fix problems.** Limitations did not fully expose latency asymmetry,
calibration diagnostics, split-seed interpretation, missing routers, the placeholder
LCB, or content-safety scope.

**Exact fixes.** The final section covers all of those issues plus limited datasets,
hardware specificity, subcomponent timing, multiple tests, absent production/load and
energy evaluation, and absent downstream generation evaluation.

## 14. Reproducibility

**Exact pre-fix problems.** Result paths in the PDF were stale, tables/figures were
manually transcribed, and source-to-claim traceability was weak.

**Exact fixes.** The paper names `results/validated`, model IDs, Python/GPU, seeds,
split hashes/manifests, caches, and generation tools. `build_evidence_tables.py`
validates and regenerates all evidence tables; `generate_submission_figures.js`
generates all figures. CSV/JSON audit artifacts accompany the paper.

**Residual risk.** Full raw datasets/model weights are external, and a fresh
from-scratch rerun was not performed during editorial revision.

## 15. Submission readiness

**Pre-fix.** Not ready for public submission because the manuscript contradicted the
validated artifacts, contained method inaccuracies, and referenced stale figures.

**Post-fix.**

- **arXiv technical report:** ready.
- **Workshop:** ready after applying the venue’s anonymous/template/page-limit rules.
- **Journal:** not yet evidence-complete. The manuscript is clean enough to circulate,
  but the baseline/calibration/repeated-evaluation blockers above should be addressed
  before a strong journal submission.

## Exact claim changes

| Unsupported or risky formulation | Final formulation |
|---|---|
| “formal probabilistic calibration guarantee” | “explicit probabilistic calibration objective”; no ECE/Brier evidence is claimed |
| “major safety property” | “retrieval self-protection”; content safety is explicitly out of scope |
| “Deep Hybrid is the upper-quality reference” | maximum-compute reference that is often, but not always, higher quality |
| “static pipelines always fail” | uniform reranking is a common design that ignores query-level variation |
| validated A0–A16 routing | validated binary A0/A6 routing; multi-action routing is future work |
| “beats all baselines” | strongest practical router varies by dataset |
| guaranteed harm avoidance | observed paired retrieval-quality outcomes only |
| HNSW graph construction | exact flat-inner-product top-neighbour lists cached as an expansion graph |
| stable across seeds | sensitivity across three different train/validation/test split seeds |
| portable latency improvement | within-run measured latency relative to always-on Deep Hybrid |

## Reviewer-risk assessment

The largest remaining risks are empirical rather than editorial: no matched-activation
random baseline, disagreement/cost-only routers, shared-harness direct competitor,
calibration metrics, fixed-test repeated fitting, or production load study. A reviewer
may also challenge “safety-aware” branding despite the explicit definition; the revised
title avoids that term and the paper repeatedly limits harm to retrieval-quality
regression. The evidence trail, equations, figures, tables, references, and PDF are now
internally consistent.
