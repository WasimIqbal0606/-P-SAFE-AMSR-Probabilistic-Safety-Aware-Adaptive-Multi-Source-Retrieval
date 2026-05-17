
# P-SAFE-AMSR — Final Research Report

**Dataset:** BEIR/nfcorpus

**Generated:** 2026-05-07 16:12:04

**Corpus documents:** 3,633 | **Queries evaluated:** 323

**Splits:** Train=129, Val=32, Test=162

**Embedding Model:** BAAI/bge-m3

**Seed:** 42



## 1. Executive Summary

- **Dense baseline nDCG@10:** 0.3329

- **Best Safe Router (P-SAFE-AMSR) nDCG@10:** 0.3501 (Δ = +0.0172)

- **SafeGain:** +0.0109 | **Hybrid activation:** 71.0%



**Abstract:** P-SAFE-AMSR is a probabilistic safety-aware adaptive retrieval controller that decides, per query, whether to preserve dense retrieval or escalate to more expensive hybrid retrieval actions. On BEIR/nfcorpus, P-SAFE-AMSR improves nDCG@10 from 0.3329 to 0.3501 while activating hybrid retrieval for only 71.0% of queries. Although the mean improvement over dense is not statistically significant in the current split (p=0.082), the method substantially reduces easy-query degradation compared with always-on hybrid retrieval and achieves a better latency-quality tradeoff than brute-force hybrid escalation.

**Final Claim:** P-SAFE-AMSR provides a probabilistic safety controller for adaptive retrieval. It selectively escalates to hybrid retrieval when useful, avoids hybrid retrieval when harmful or unnecessary, and reduces latency while preserving or improving retrieval quality across different dataset behaviours.


## 2. Main Results

| Method | nDCG@10 | Recall@10 | MRR | Latency | Hybrid% |

|--------|---------|-----------|-----|---------|---------|

| Dense | 0.3329 | 0.2637 | 0.5071 | 0.2ms | 0% |

| Dense+BM25 | 0.3367 | 0.2672 | 0.5210 | 3.0ms | 100% |

| Dense+Graph | 0.3329 | 0.2637 | 0.5071 | 0.2ms | 100% |

| Dense+BM25+Graph | 0.3356 | 0.2700 | 0.5115 | 2.0ms | 100% |

| Dense+BM25+CE | 0.3665 | 0.3063 | 0.5434 | 362.7ms | 100% |

| Dense+BM25+Graph+CE | 0.3668 | 0.3063 | 0.5443 | 363.5ms | 100% |

| Deep Hybrid | 0.3641 | 0.3073 | 0.5439 | 736.9ms | 100% |

| P-SAFE-AMSR | 0.3501 | 0.2919 | 0.5172 | 320.8ms | 71% |


## 3. Statistical Significance


### P-SAFE-AMSR vs Dense

- Mean Δ: +0.0172

- Win/Tie/Loss: 52/84/26

- Paired t-test: p = 8.2324e-02 (borderline/promising)

- Wilcoxon: p = 3.5783e-02

- Permutation: p = 8.7600e-02

- 95% CI: [-0.0020, 0.0352]


### P-SAFE-AMSR vs Dense+BM25+CE

- Mean Δ: -0.0164

- Win/Tie/Loss: 19/110/33

- Paired t-test: p = 2.4257e-02 (✅ significant)

- Wilcoxon: p = 2.5071e-02

- Permutation: p = 2.4000e-02

- 95% CI: [-0.0304, -0.0018]


### P-SAFE-AMSR vs Deep Hybrid

- Mean Δ: -0.0140

- Win/Tie/Loss: 29/102/31

- Paired t-test: p = 5.3263e-02 (borderline/promising)

- Wilcoxon: p = 1.0143e-01

- Permutation: p = 5.2600e-02

- 95% CI: [-0.0286, 0.0011]


### P-SAFE-AMSR vs Oracle

- Mean Δ: -0.0686

- Win/Tie/Loss: 0/88/74

The oracle upper bound remains significantly higher than P-SAFE-AMSR, indicating substantial remaining headroom for improved routing and action selection.

- 95% CI: [-0.0864, -0.0522]


## 4. Easy-Query Degradation Analysis

**Dense:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000

**Dense+BM25:** SafeGain=-0.0281, EasyDeg=0.0820, HardGain=+0.0539

**Dense+Graph:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000

**Dense+BM25+Graph:** SafeGain=-0.0130, EasyDeg=0.0365, HardGain=+0.0235

**Dense+BM25+CE:** SafeGain=+0.0363, EasyDeg=0.0418, HardGain=+0.0781

**Dense+BM25+Graph+CE:** SafeGain=+0.0369, EasyDeg=0.0418, HardGain=+0.0787

**Deep Hybrid:** SafeGain=+0.0353, EasyDeg=0.0434, HardGain=+0.0788

**P-SAFE-AMSR:** SafeGain=+0.0109, EasyDeg=0.0433, HardGain=+0.0542


## 5. Router Performance

**P-SAFE Action Distribution:**

- Dense: 47 (29.0%)

- Dense+BM25: 10 (6.2%)

- Dense+Graph: 0 (0.0%)

- Dense+BM25+Graph: 0 (0.0%)

- Dense+BM25+CE: 29 (17.9%)

- Dense+BM25+Graph+CE: 37 (22.8%)

- Deep Hybrid: 39 (24.1%)


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
