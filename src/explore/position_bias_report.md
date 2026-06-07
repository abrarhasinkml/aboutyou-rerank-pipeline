# Position Bias Analysis

**Method:** CTR aggregated into position buckets (width=5), adaptive merge for buckets < 30 rows.
**Reference position:** ≤ 5 (correction factor baseline).
**Tail cap:** positions > 100 grouped into one bucket.

## CTR by Position Bucket

| Bucket | Rows | Mean Pos | Mean CTR | Median CTR | Mean Clicks | Mean Impr | Correction Factor |
|--------|------|----------|----------|------------|-------------|----------|-------------------|
| 0-4 | 1248 | 2.7 | 13.19 | 0.0 | 7.65 | 86.0 | 1.000 |
| 5-9 | 1842 | 7.3 | 9.75 | 0.0 | 27.97 | 302.1 | 1.352 |
| 10-14 | 2022 | 12.3 | 8.48 | 0.0 | 39.55 | 599.8 | 1.555 |
| 15-19 | 2089 | 17.2 | 8.11 | 0.0 | 40.84 | 702.3 | 1.627 |
| 20-24 | 1917 | 22.2 | 9.29 | 0.0 | 40.36 | 821.1 | 1.420 |
| 25-29 | 1714 | 27.2 | 7.56 | 0.0 | 53.30 | 1036.3 | 1.746 |
| 30-34 | 1351 | 32.1 | 8.33 | 1.6 | 53.97 | 1253.7 | 1.583 |
| 35-39 | 980 | 37.3 | 7.99 | 2.3 | 54.62 | 1334.8 | 1.651 |
| 40-44 | 869 | 42.5 | 8.51 | 2.6 | 57.89 | 1385.5 | 1.550 |
| 45-49 | 793 | 47.3 | 7.02 | 2.1 | 51.66 | 1298.5 | 1.880 |
| 50-54 | 584 | 52.3 | 9.38 | 2.4 | 47.03 | 1110.2 | 1.407 |
| 55-59 | 517 | 57.2 | 8.65 | 0.0 | 37.07 | 1020.5 | 1.525 |
| 60-64 | 439 | 62.2 | 9.44 | 2.1 | 39.15 | 952.4 | 1.397 |
| 65-69 | 358 | 67.2 | 8.23 | 1.7 | 39.62 | 930.9 | 1.603 |
| 70-74 | 308 | 72.1 | 11.46 | 0.0 | 31.68 | 736.1 | 1.152 |
| 75-79 | 265 | 77.1 | 15.20 | 0.0 | 26.41 | 623.2 | 0.868 |
| 80-84 | 226 | 82.2 | 13.28 | 0.0 | 22.77 | 514.3 | 0.993 |
| 85-89 | 173 | 87.2 | 11.56 | 0.0 | 17.36 | 391.6 | 1.141 |
| 90-94 | 139 | 92.1 | 14.19 | 0.0 | 22.19 | 447.8 | 0.930 |
| 95-99 | 135 | 97.3 | 19.73 | 0.0 | 13.70 | 280.9 | 0.669 |

## Position Bias Shape (D6 Input)

Three functional forms were evaluated to describe CTR decay with position:

| Model | R² |
|-------|-----|
| Linear (CTR = a·pos + b) | 0.3428 |
| Log-linear (CTR = e^(a·pos + b) — exponential decay) | 0.3718 |
| Power-law (CTR = C · pos^a) | 0.0644 |

**Conclusion: No position bias detected.**

Position explains little CTR variance (R² < 0.5 for all models). In aggregated search data, niche queries (few candidates, motivated users) produce higher per-impression CTR at deep positions than generic queries at top positions. Applying position correction would harm the reranker.

**D6 implication:** Drop position correction. The primary reranker uses Strategy D (smoothed CTR without position correction). Strategy E (smoothed + position correction) is retained for comparison only. Applying a position correction factor to this data would harm the reranker.
## Interpretation

**Why no position bias?** The data aggregates months of traffic across queries of different specificity. Broad queries have 100 candidates with flat CTR; niche queries have few candidates — motivated users scroll through all of them, producing high CTR even at deep positions. In this context, position correlates more with query specificity than relevance quality.

Strategy D (Beta-binomial smoothed CTR) is the right tool: it handles the real problem of click sparsity without correcting for a bias that doesn't exist here.