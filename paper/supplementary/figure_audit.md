# Figure Audit

| Figure | Original issue | Fix | Status |
|---|---|---|---|
| 1 | Dark, oversized architecture with disconnected arrows and ambiguous pre/post-routing flow | Rebuilt as a light vector pipeline showing signals, models, utility gate, binary endpoints, and graph truth | PASS |
| 2 | Dark raster Pareto plot mixed five datasets and raw nDCG scales; labels were difficult at column width | Rebuilt as mean quality gain versus measured latency saving with error bars for the four manuscript datasets | PASS |
| 3 | Dark raster activation chart included an out-of-scope fifth dataset and a legend over the plot | Rebuilt as a light grouped chart with direct mean/SD annotations | PASS |
| 4 | Prior action-distribution figure implied autonomous detection without exposing baseline competitiveness | Replaced with selected-router quality–latency small multiples; full values remain in tables | PASS |
| 5 | Previous hard-query annotations were crowded and depended on stale values | Replaced with source-backed log-scale latency components and explicit activation/conditional timing | PASS |
| 6 | Dark multi-seed bars were large and visually heavy | Rebuilt as compact per-seed dot-and-line panels for nDCG and activation | PASS |

All six figures are exported as SVG, vector PDF, and PNG. Captions in the manuscript identify the data scope and metric definition.
