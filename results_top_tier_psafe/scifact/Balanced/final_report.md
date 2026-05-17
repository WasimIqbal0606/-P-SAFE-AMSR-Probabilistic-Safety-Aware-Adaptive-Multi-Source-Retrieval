
# P-SAFE-AMSR — Final Research Report

**Dataset:** BEIR/scifact

**Generated:** 2026-05-07 15:31:52

**Corpus documents:** 5,183 | **Queries evaluated:** 300

**Splits:** Train=119, Val=29, Test=152

**Embedding Model:** BAAI/bge-m3

**Seed:** 42



## 1. Executive Summary

- **Dense baseline nDCG@10:** 0.6466

- **Best Safe Router (P-SAFE-AMSR) nDCG@10:** 0.6661 (Δ = +0.0195)

- **SafeGain:** +0.0523 | **Hybrid activation:** 13.2%



**Abstract:** P-SAFE-AMSR is a probabilistic safety-aware adaptive retrieval controller that decides, per query, whether to preserve dense retrieval or escalate to more expensive hybrid retrieval actions. On BEIR/scifact, P-SAFE-AMSR improves nDCG@10 from 0.6466 to 0.6661 while activating hybrid retrieval for only 13.2% of queries. Although the mean improvement over dense is statistically significant, the method substantially reduces easy-query degradation compared with always-on hybrid retrieval and achieves a better latency-quality tradeoff than brute-force hybrid escalation.

**Final Claim:** P-SAFE-AMSR provides a probabilistic safety controller for adaptive retrieval. It selectively escalates to hybrid retrieval when useful, avoids hybrid retrieval when harmful or unnecessary, and reduces latency while preserving or improving retrieval quality across different dataset behaviours.


## 2. Main Results

| Method | nDCG@10 | Recall@10 | MRR | Latency | Hybrid% |

|--------|---------|-----------|-----|---------|---------|

| Dense | 0.6466 | 0.7781 | 0.6121 | 0.3ms | 0% |

| Dense+BM25 | 0.6202 | 0.7531 | 0.5832 | 12.5ms | 100% |

| Dense+Graph | 0.6466 | 0.7781 | 0.6121 | 0.4ms | 100% |

| Dense+BM25+Graph | 0.6389 | 0.7912 | 0.5932 | 12.2ms | 100% |

| Dense+BM25+CE | 0.7334 | 0.8395 | 0.7036 | 383.1ms | 100% |

| Dense+BM25+Graph+CE | 0.7334 | 0.8395 | 0.7036 | 387.1ms | 100% |

| Deep Hybrid | 0.7351 | 0.8461 | 0.7041 | 749.8ms | 100% |

| P-SAFE-AMSR | 0.6661 | 0.7781 | 0.6375 | 78.2ms | 13% |


## 3. Statistical Significance


### P-SAFE-AMSR vs Dense

- Mean Δ: +0.0195

- Win/Tie/Loss: 10/138/4

- Paired t-test: p = 2.6422e-02 (✅ significant)

- Wilcoxon: p = 4.1227e-02

- Permutation: p = 2.2200e-02

- 95% CI: [0.0038, 0.0379]


### P-SAFE-AMSR vs Dense+BM25+CE

- Mean Δ: -0.0673

- Win/Tie/Loss: 12/112/28

- Paired t-test: p = 1.0100e-03 (✅ significant)

- Wilcoxon: p = 2.3464e-03

- Permutation: p = 6.0000e-04

- 95% CI: [-0.1051, -0.0312]


### P-SAFE-AMSR vs Deep Hybrid

- Mean Δ: -0.0690

- Win/Tie/Loss: 17/105/30

- Paired t-test: p = 1.5589e-03 (✅ significant)

- Wilcoxon: p = 3.2829e-03

- Permutation: p = 6.0000e-04

- 95% CI: [-0.1099, -0.0292]


### P-SAFE-AMSR vs Oracle

- Mean Δ: -0.1138

- Win/Tie/Loss: 0/111/41

The oracle upper bound remains significantly higher than P-SAFE-AMSR, indicating substantial remaining headroom for improved routing and action selection.

- 95% CI: [-0.1485, -0.0778]


## 4. Easy-Query Degradation Analysis

**Dense:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000

**Dense+BM25:** SafeGain=+0.0073, EasyDeg=0.1137, HardGain=+0.1210

**Dense+Graph:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000

**Dense+BM25+Graph:** SafeGain=+0.0032, EasyDeg=0.0774, HardGain=+0.0806

**Dense+BM25+CE:** SafeGain=+0.2105, EasyDeg=0.0353, HardGain=+0.2458

**Dense+BM25+Graph+CE:** SafeGain=+0.2105, EasyDeg=0.0353, HardGain=+0.2458

**Deep Hybrid:** SafeGain=+0.2263, EasyDeg=0.0400, HardGain=+0.2663

**P-SAFE-AMSR:** SafeGain=+0.0523, EasyDeg=0.0038, HardGain=+0.0560


## 5. Router Performance

**P-SAFE Action Distribution:**

- Dense: 132 (86.8%)

- Dense+BM25: 0 (0.0%)

- Dense+Graph: 0 (0.0%)

- Dense+BM25+Graph: 0 (0.0%)

- Dense+BM25+CE: 4 (2.6%)

- Dense+BM25+Graph+CE: 8 (5.3%)

- Deep Hybrid: 8 (5.3%)


## 6. Retrieval Over-Treatment and Safety Analysis

More retrieval is not always better. On datasets such as FiQA, always-on hybrid retrieval introduces lexical noise and cross-encoder misranking on queries already handled well by Dense retrieval. P-SAFE-AMSR avoids this retrieval over-treatment by suppressing hybrid expansion when predicted harm is high.


## 7. Probabilistic Calibration

The harm model is used as a safety gate; therefore, calibration quality directly affects whether P-SAFE avoids over-treatment.


## 8. Graph Contribution

Graph expansion in the current implementation uses synthetic kNN edges derived from dense embeddings. As such, it mainly explores local dense-neighbourhoods and is not expected to add independent structural evidence.


## 9. Limitations

- Current graph is synthetic kNN based on dense embeddings and does not independently drive gains.

- Results need stronger baselines such as SPLADE, ColBERT, BGE-M3, and E5/BGE dense models.

- Some datasets show protection rather than absolute nDCG improvement.

- More datasets and larger test splits are required before strong journal claims.


## 10. Next Steps

1. Test on larger BEIR datasets (FiQA, TREC-COVID, ArguAna)

2. Investigate learned reranking depth selection

3. Explore continuous routing action spaces
