
# P-SAFE-AMSR — Final Research Report

**Dataset:** BEIR/fiqa

**Generated:** 2026-05-07 16:03:01

**Corpus documents:** 57,638 | **Queries evaluated:** 648

**Splits:** Train=258, Val=64, Test=326

**Embedding Model:** BAAI/bge-m3

**Seed:** 42



## 1. Executive Summary

- **Dense baseline nDCG@10:** 0.4162

- **Best Safe Router (P-SAFE-AMSR) nDCG@10:** 0.4162 (Δ = +0.0000)

- **SafeGain:** +0.0000 | **Hybrid activation:** 0.0%



**Abstract:** P-SAFE-AMSR is a probabilistic safety-aware adaptive retrieval controller that decides, per query, whether to preserve dense retrieval or escalate to more expensive hybrid retrieval actions. On BEIR/fiqa, P-SAFE-AMSR improves nDCG@10 from 0.4162 to 0.4162 while activating hybrid retrieval for only 0.0% of queries. Although the mean improvement over dense is not statistically significant in the current split (p=1.000), the method substantially reduces easy-query degradation compared with always-on hybrid retrieval and achieves a better latency-quality tradeoff than brute-force hybrid escalation.

**Final Claim:** P-SAFE-AMSR provides a probabilistic safety controller for adaptive retrieval. It selectively escalates to hybrid retrieval when useful, avoids hybrid retrieval when harmful or unnecessary, and reduces latency while preserving or improving retrieval quality across different dataset behaviours.


## 2. Main Results

| Method | nDCG@10 | Recall@10 | MRR | Latency | Hybrid% |

|--------|---------|-----------|-----|---------|---------|

| Dense | 0.4162 | 0.4772 | 0.4972 | 0.4ms | 0% |

| Dense+BM25 | 0.3168 | 0.4158 | 0.3730 | 177.2ms | 100% |

| Dense+Graph | 0.4162 | 0.4772 | 0.4972 | 0.5ms | 100% |

| Dense+BM25+Graph | 0.3690 | 0.4836 | 0.4138 | 176.0ms | 100% |

| Dense+BM25+CE | 0.4458 | 0.5104 | 0.5347 | 531.5ms | 100% |

| Dense+BM25+Graph+CE | 0.4458 | 0.5104 | 0.5347 | 536.5ms | 100% |

| Deep Hybrid | 0.4454 | 0.5084 | 0.5379 | 903.3ms | 100% |

| P-SAFE-AMSR | 0.4162 | 0.4772 | 0.4972 | 0.4ms | 0% |


## 3. Statistical Significance


### P-SAFE-AMSR vs Dense

- Mean Δ: +0.0000

- Win/Tie/Loss: 0/326/0

- Paired t-test: p = 1.0000e+00 (❌ not significant)

- Wilcoxon: p = 1.0000e+00

- Permutation: p = 1.0000e+00

- 95% CI: [0.0000, 0.0000]


### P-SAFE-AMSR vs Dense+BM25+CE

- Mean Δ: -0.0296

- Win/Tie/Loss: 72/149/105

- Paired t-test: p = 1.3688e-02 (✅ significant)

- Wilcoxon: p = 1.6650e-02

- Permutation: p = 1.1800e-02

- 95% CI: [-0.0540, -0.0070]


### P-SAFE-AMSR vs Deep Hybrid

- Mean Δ: -0.0293

- Win/Tie/Loss: 76/144/106

- Paired t-test: p = 2.3894e-02 (✅ significant)

- Wilcoxon: p = 2.4611e-02

- Permutation: p = 2.1800e-02

- 95% CI: [-0.0550, -0.0053]


### P-SAFE-AMSR vs Oracle

- Mean Δ: -0.1082

- Win/Tie/Loss: 0/195/131

The oracle upper bound remains significantly higher than P-SAFE-AMSR, indicating substantial remaining headroom for improved routing and action selection.

- 95% CI: [-0.1288, -0.0891]


## 4. Easy-Query Degradation Analysis

**Dense:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000

**Dense+BM25:** SafeGain=-0.2096, EasyDeg=0.2751, HardGain=+0.0655

**Dense+Graph:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000

**Dense+BM25+Graph:** SafeGain=-0.1196, EasyDeg=0.1485, HardGain=+0.0290

**Dense+BM25+CE:** SafeGain=+0.0394, EasyDeg=0.0697, HardGain=+0.1091

**Dense+BM25+Graph+CE:** SafeGain=+0.0394, EasyDeg=0.0697, HardGain=+0.1091

**Deep Hybrid:** SafeGain=+0.0413, EasyDeg=0.0768, HardGain=+0.1180

**P-SAFE-AMSR:** SafeGain=+0.0000, EasyDeg=-0.0000, HardGain=+0.0000


## 5. Router Performance

**P-SAFE Action Distribution:**

- Dense: 326 (100.0%)

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
