# Data Profile Report

**Rows:** 19,129  |  **Search terms:** 305  |  **Products:** 16,697  |  **Memory:** 1.6 MB

## Candidates per Search Term

| Statistic | Value |
|-----------|-------|
| min | 4 |
| max | 100 |
| median | 69.0 |
| mean | 62.7 |
| p5 | 7 |
| p95 | 100 |
| p99 | 100 |

## Click Sparsity

- **Zero clicks:** 11,251 rows (58.8%)
- **Has clicks:** 7,878 rows (41.2%)

### Click Distribution (all rows)

| Statistic | Value |
|-----------|-------|
| count | 19129 |
| mean | 37.95 |
| std | 120.85 |
| min | 0.0 |
| max | 5899.0 |
| p50% | 0.0 |
| p90% | 118.0 |
| p95% | 188.0 |
| p99% | 426.16 |
### Click Distribution (clicked rows only)

| Statistic | Value |
|-----------|-------|
| count | 7878 |
| mean | 92.16 |
| std | 174.55 |
| min | 1.0 |
| max | 5899.0 |
| p50% | 58.0 |
| p90% | 209.0 |
| p95% | 298.0 |
| p99% | 677.46 |

## CTR Distribution

- **CTR = 0%:** 59.3% of rows
- **CTR > 100%:** 51 rows (0.27%)
  - **Interpretation:** Clicks CAN exceed impressions — this is NOT a data error.
    Customers click via bookmarks, push notifications, or email links which
    generate click events without a corresponding SRP impression.
  - **Implication for scoring:** Raw CTR (`clicks/impressions × 100`) is an
    unreliable metric. We decouple clicks and impressions in the scoring formula
    — clicks measure engagement, impressions provide confidence.
  - **Impression distribution for these rows:**
    - 1 impression(s): 44 rows
    - 2 impression(s): 3 rows
    - 3 impression(s): 2 rows
    - 4 impression(s): 2 rows
- **Low impressions (< 3):** 7,339 rows (38.4%)

| Statistic | Value |
|-----------|-------|
| count | 19129 |
| mean | 10.69 |
| std | 26.72 |
| min | 0.0 |
| max | 500.0 |
| p25% | 0.0 |
| p50% | 0.0 |
| p75% | 5.3 |
| p90% | 33.33 |
| p95% | 100.0 |
| p99% | 100.0 |

## Position Columns

### impression_pos_avg
- **Non-null:** 19,027  |  **Null:** 102 (0.5%)

| Statistic | Value |
|-----------|-------|
| min | 1.0 |
| max | 664.0 |
| median | 26.0 |
| mean | 38.0 |

### impression_pos_median
- **Non-null:** 19,027  |  **Null:** 102 (0.5%)

| Statistic | Value |
|-----------|-------|
| min | 1.0 |
| max | 664.0 |
| median | 19.0 |
| mean | 30.8 |

### click_pos_avg
- **Non-null:** 7,878  |  **Null:** 11,251 (58.8%)

| Statistic | Value |
|-----------|-------|
| min | 1.0 |
| max | 658.0 |
| median | 31.0 |
| mean | 44.6 |

### click_pos_median
- **Non-null:** 7,878  |  **Null:** 11,251 (58.8%)

| Statistic | Value |
|-----------|-------|
| min | 1.0 |
| max | 658.0 |
| median | 20.0 |
| mean | 34.9 |

## Position Bias Evidence

- **Correlation (impression_pos_avg ↔ click_pos_avg):** r = 0.959
- **Interpretation:** Positive correlation — products impressed at higher positions also clicked at higher positions.

## Cold-Start Terms (Zero Total Clicks)

- **Terms with zero clicks:** 0 (0.0% of all terms)
- **Implication:** These terms have no behavioural signal — must fall back to baseline ordering (`impression_pos_avg` ascending).

### Clicks per Term Distribution

| Statistic | Value |
|-----------|-------|
| min | 5.0 |
| max | 43594.0 |
| median | 12.0 |
| mean | 2380.4 |
| p95 | 15142.0 |
| p99 | 29177.2 |

## Duplicate Check

- No duplicate (search_term, product_id) pairs found.
