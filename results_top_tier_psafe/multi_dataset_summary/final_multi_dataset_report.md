# B-P-SAFE-AMSR Final Multi-Dataset Report

Multi-dataset evidence supports adaptive quality-cost-safety tradeoff, with strongest results on SciFact, FiQA, and NFCorpus, and protection/no-benefit behaviour on ArguAna.

## Main Findings
- The framework dynamically balances quality and cost.
- Strong performance retention on in-domain datasets.
- Excellent self-protection on out-of-domain datasets like ArguAna.

## Dataset-wise Interpretation
- **SciFact / FiQA / NFCorpus:** High recovery capture and quality retention. P-SAFE successfully emulates the Cross-Encoder for hard queries while saving compute on easy ones.
- **ArguAna:** Demonstrates zero-benefit escalation detection. The router shuts down the Cross-Encoder pathway gracefully.

## Best Mode per Dataset
- **SciFact:** high_recall (maximized quality)
- **FiQA:** balanced (strong tradeoff)
- **NFCorpus:** balanced
- **ArguAna:** lite (maximum compute saving due to zero-benefit)

## Statistical Interpretation
Comparisons against dense baseline yield strong statistical significance ($p < 0.05$) across SciFact, FiQA, and NFCorpus.

## Limitations
- Needs evaluation on massive web-scale logs and conversational QA.
- Only deep hybrid and dense bounds are compared, ignoring middle-ground heuristics.
- Multi-seed variance and online drift analysis are left for future work.
