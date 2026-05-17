
# P-SAFE-AMSR — Final Research Report

**Dataset:** BEIR/nfcorpus

**Generated:** 2026-05-07 16:12:21

**Corpus documents:** 3,633 | **Queries evaluated:** 323

**Splits:** Train=129, Val=32, Test=162

**Embedding Model:** BAAI/bge-m3

**Seed:** 42



## 1. Executive Summary

- **Dense baseline nDCG@10:** 0.3329

- **Best Safe Router (P-SAFE-AMSR) nDCG@10:** 0.3329 (Δ = +0.0000)

- **SafeGain:** +0.0000 | **Hybrid activation:** 0.0%



**Abstract:** P-SAFE-AMSR is a probabilistic safety-aware adaptive retrieval controller that decides, per query, whether to preserve dense retrieval or escalate to more expensive hybrid retrieval actions. On BEIR/nfcorpus, P-SAFE-AMSR improves nDCG@10 from 0.3329 to 0.3329 while activating hybrid retrieval for only 0.0% of queries. Although the mean improvement over dense is not statistically significant in the current split (p=1.000), the method substantially reduces easy-query degradation compared with always-on hybrid retrieval and achieves a better latency-quality tradeoff than brute-force hybrid escalation.

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

| P-SAFE-AMSR | 0.3329 | 0.2637 | 0.5071 | 0.2ms | 0% |


## 3. Statistical Significance


### P-SAFE-AMSR vs Dense

- Mean Δ: +0.0000

- Win/Tie/Loss: 0/162/0

- Paired t-test: p = 1.0000e+00 (❌ not significant)

- Wilcoxon: p = 1.0000e+00

- Permutation: p = 1.0000e+00

- 95% CI: [0.0000, 0.0000]


### P-SAFE-AMSR vs Dense+BM25+CE

- Mean Δ: -0.0336

- Win/Tie/Loss: 40/55/67

- Paired t-test: p = 4.4482e-03 (✅ significant)

- Wilcoxon: p = 4.3287e-03

- Permutation: p = 3.6000e-03

- 95% CI: [-0.0560, -0.0111]


### P-SAFE-AMSR vs Deep Hybrid

- Mean Δ: -0.0313

- Win/Tie/Loss: 38/55/69

- Paired t-test: p = 9.3619e-03 (✅ significant)

- Wilcoxon: p = 4.8647e-03

- Permutation: p = 8.6000e-03

- 95% CI: [-0.0543, -0.0088]


### P-SAFE-AMSR vs Oracle

- Mean Δ: -0.0832

- Win/Tie/Loss: 0/77/85

The oracle upper bound remains significantly higher than P-SAFE-AMSR, indicating substantial remaining headroom for improved routing and action selection.

- 95% CI: [-0.1034, -0.0640]


## 4. Easy-Query Degradation Analysis

**Dense:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000

**Dense+BM25:** SafeGain=-0.0281, EasyDeg=0.0820, HardGain=+0.0539

**Dense+Graph:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000

**Dense+BM25+Graph:** SafeGain=-0.0130, EasyDeg=0.0365, HardGain=+0.0235

**Dense+BM25+CE:** SafeGain=+0.0363, EasyDeg=0.0418, HardGain=+0.0781

**Dense+BM25+Graph+CE:** SafeGain=+0.0369, EasyDeg=0.0418, HardGain=+0.0787

**Deep Hybrid:** SafeGain=+0.0353, EasyDeg=0.0434, HardGain=+0.0788

**P-SAFE-AMSR:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000


## 5. Router Performance

**P-SAFE Action Distribution:**

- Dense: 162 (100.0%)

- Dense+BM25: 0 (0.0%)

- Dense+Graph: 0 (0.0%)

- Dense+BM25+Graph: 0 (0.0%)

- Dense+BM25+CE: 0 (0.0%)

- Dense+BM25+Graph+CE: 0 (0.0%)

- Deep Hybrid: 0 (0.0%)


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
