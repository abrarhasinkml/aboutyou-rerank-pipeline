# Architecture: Search Reranking Prototype

## Overview

Build a search reranker for ABOUT YOU's SRP — given a `search_term`, reorder the candidate product set so products users engaged with rank higher, using historical click behaviour. 

**Core constraint:** 5-6 hour timebox. Prototype must work end-to-end. Production vision goes in Task 3/presentation.

---

## Decisions Log

### D1: No dbt in the prototype

**Context:** dbt + dimensional modeling was considered for data ingestion → cleaning → mart pipeline.

**Decision:** dbt is excluded from the prototype build. It requires a SQL warehouse, adds hours of setup, and the dataset is already at mart granularity (one row = one search_term × product, pre-aggregated).

**Where dbt lives:** Task 3 ("Next Steps") in the presentation — as the production architecture vision alongside Airflow for scheduling, lineage, and data quality monitoring.

**Rationale:** The timebox is 5-6 hours. Standing up a database + dbt project eats 2-3 hours before any reranking logic is written. The senior vision is demonstrated through the presentation, not by over-engineering the prototype.

### D2: Reranking strategy — comparative analysis with pluggable approaches

**Context:** Choosing how to compute ranking scores from click/impression/position data.

**Decision:** Build a **pluggable strategy system** — the initial prototype implements a statistical model as the default, but the architecture supports swapping strategies via a `strategy` parameter. This allows the API to serve different approaches for different scenarios and shows the panel we evaluated multiple options rather than picking one naively.

**Strategy comparison:**

| Strategy | Category | How it works | Pros | Cons |
|----------|----------|-------------|------|------|
| **A: Naive clicks** | Heuristic | Rank by `clicks` descending | Dead simple, no parameters | Overfits items with many impressions; position bias unaddressed; sparse terms collapse to random order |
| **B: Raw CTR** | Heuristic | Rank by `ctr` (clicks/impressions × 100) descending | Accounts for exposure differences | Unstable at low impressions (1 click / 1 impression = 100% CTR beats a proven item); no position bias correction |
| **C: Position-debiased CTR** | Heuristic | Weighted CTR corrected by impression position bias | Handles the known position effect (items at rank 1 get more clicks regardless of relevance) | Still unstable at low impression counts; no confidence floor |
| **D: Smoothed CTR (Beta-binomial)** | Statistical | `(clicks + α) / (impressions + β)` — shrinks noisy estimates toward a global prior | Robust at low impressions; handles sparsity well; well-grounded in Bayesian statistics | Requires tuning α/β from data; slightly less interpretable to non-technical stakeholders |
| **E: Smoothed + position-corrected CTR** | Statistical | Strategy D combined with a position bias correction factor | Best of both worlds — handles sparsity and position bias | Most parameters to tune; needs EDA to inform position correction shape |
| **F: Learning-to-rank (e.g., LambdaRank)** | ML | Gradient-boosted pairwise ranking on engineered features | Can learn complex interactions; standard in production search systems | Needs significantly more data, feature engineering, train/val splits; not achievable within 5-6 hour timebox; harder to explain |

**Selected default: Strategy D (smoothed CTR, Beta-binomial)** for the prototype. The EDA revealed no position bias in this dataset (see D6), so position correction was dropped. Strategy E is retained for comparison only.

**Pluggable implementation:** The reranker accepts a `strategy` enum, allowing the API to expose different approaches:

```python
from enum import Enum

class RerankStrategy(str, Enum):
    NAIVE_CLICKS = "naive_clicks"
    RAW_CTR = "raw_ctr"
    POSITION_DEBIASED_CTR = "position_debiased_ctr"
    SMOOTHED_CTR = "smoothed_ctr"
    SMOOTHED_POSITION_CTR = "smoothed_position_ctr"  # default

# API usage: GET /rerank?q=kleid&k=20&strategy=smoothed_position_ctr
```


### D3: Layered Python architecture with dedicated EDA

**Context:** How to organize the prototype code.

**Decision:** Five-package Python module structure, adding a dedicated `explore/` package for exploratory data analysis that feeds directly into model design decisions:

```
src/
├── explore/        # EDA — informs all downstream decisions
│   ├── data_profile.py   # Row counts, sparsity, distributions, null rates
│   ├── position_bias.py  # Click vs. position analysis, bias shape
│   └── visualise.py      # Plots: CTR distributions, position bias curves, term length histograms
├── data/           # Ingestion + validation
│   ├── loader.py         # Read parquet/csv, validate schema
│   └── cleaner.py        # Deduplication, null handling, outlier flags
├── models/         # Reranking logic
│   ├── strategies.py     # Strategy enum + factory — all approaches from D2
│   ├── scorer.py         # CTR scoring, position bias correction, confidence weighting
│   └── reranker.py       # Term-level reordering, tie-breaking, strategy dispatch
├── evaluation/     # Offline evaluation
│   └── metrics.py        # NDCG@K, position-weighted recall, strategy comparison
└── api.py          # FastAPI endpoint for serving
```

**Rationale:** 
- The `explore/` package runs first — its outputs (distribution stats, position bias curves, sparsity metrics) directly inform the parameters in D6 and the `cleaner.py` thresholds
- Separation of concerns mirrors what a production dbt pipeline would do (sources → staging → marts → serving), but without the warehouse dependency
- Each layer is independently testable

### D4: FastAPI serving layer

**Context:** How to expose the reranker for demonstration.

**Decision:** Lightweight FastAPI app with a single endpoint — `GET /rerank?q={search_term}&k={top_n}&strategy={strategy}` returning ordered product IDs with scores, metadata, and product images (via CDN URL construction).

**Rationale:** 
- The `strategy` parameter demonstrates the pluggable approach from D2
- Product images in the response make the presentation demo visually compelling

### D5: Evaluation metric — NDCG@10 with click-based relevance

**Context:** How to measure reranker quality.

**Decision:** NDCG@10 using clicks as relevance labels, computed per search term, then averaged across terms. All strategies from D2 are evaluated against the same baselines.

**Baselines:**
- `impression_pos_avg` ascending (logged display order — the "production today" reference)
- Raw clicks descending (naive heuristic)

**Rationale:**
- NDCG is the standard IR metric for ranked retrieval
- Clicks are the only behavioural signal available; treated as ordinal relevance grades (0, 1, 2, …)
- Per-term evaluation respects that each query has a different candidate set
- Evaluating all strategies against the same baselines makes the comparative analysis from D2 quantitative, not just theoretical

### D6: Scoring parameters — resolved via EDA


| Question | Finding | RerankConfig impact |
|----------|---------|---------------------|
| Sparsity | 58.8% zero-click rows -- moderate | `alpha=1.0`, `beta` auto-computed from global CTR (~2.5%) |
| Position bias | **No bias** -- R2<0.38 all models; deep pos (95-99) CTR 19.7% > top (0-4) 13.2% | `position_correction_type="none"` -- correction would harm reranker |
| Impression floor | 38.4% rows have <3 impressions | `min_impressions_floor=3` |
| CTR > 100% | 51 rows (0.27%) -- clicks from bookmarks/notifications/email exceed SRP impressions | No cap; clicks=engagement, impressions=confidence |
| Cold-start | 0% -- every term has >=5 clicks | Placeholder retained |
| Seasonality | Data spans months -- stale clicks on old trends | **Deferred.** Production: recency decay |


```python
@dataclass
class RerankConfig:
    smoothing_alpha: float = 1.0           # moderate sparsity -> light prior
    smoothing_beta: float | None = None    # auto-computed from global CTR
    position_correction_type: str = "none" # EDA: no bias exists in this data
    position_correction_gamma: float = 0.0
    min_impressions_floor: int = 3         # 38.4% of rows below this
    cold_start_fallback: str = "baseline"  # placeholder for future refreshes
```

---

## Data Schema (from README)

| Column | Type | Role in scoring |
|--------|------|----------------|
| `search_term` | string | Grouping key — one candidate set per term |
| `product_id` | int | Output ID after reordering |
| `clicks` | int | Primary relevance signal |
| `impressions` | int | Exposure — denominator for CTR, threshold for confidence |
| `ctr` | float | Pre-computed click-through rate (may still recalculate with smoothing) |
| `impression_pos_avg` | float | Position bias correction input |
| `impression_pos_median` | float | Robust alternative to avg |
| `click_pos_avg` | float/null | Where clicks happened (null if no clicks) |
| `click_pos_median` | float/null | Robust click position (null if no clicks) |


---

## Prototype Structure

### Module: `src/explore/data_profile.py`
```python
def profile_dataset(df: pd.DataFrame) -> dict:
    """Return dict of summary stats: rows, terms, sparsity, distributions."""

def candidates_per_term_distribution(df: pd.DataFrame) -> pd.Series:
    """Min, max, median, p95 candidates per search term."""

def null_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column null counts and percentages."""
```

### Module: `src/explore/position_bias.py`
```python
def position_vs_ctr(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CTR by impression position bucket — reveals bias shape."""

def click_position_analysis(df: pd.DataFrame) -> dict:
    """Compare impression_pos_avg vs click_pos_avg distributions."""
```

### Module: `src/explore/visualise.py`
```python
def plot_ctr_distribution(df: pd.DataFrame) -> None:
    """Histogram + boxplot of CTR values."""

def plot_position_bias_curve(df: pd.DataFrame) -> None:
    """CTR vs. impression position — the key curve for D6."""

def plot_term_candidates_histogram(df: pd.DataFrame) -> None:
    """Histogram of candidates per search term."""
```

### Module: `src/data/loader.py`
```python
def load_search_data(path: str) -> pd.DataFrame:
    """Read parquet, validate required columns exist, return DataFrame."""
    
def load_product_metadata(path: str) -> pd.DataFrame:
    """Read CSV product metadata."""
```

### Module: `src/data/cleaner.py`
```python
def validate_and_clean(df: pd.DataFrame, config: RerankConfig) -> pd.DataFrame:
    """Deduplicate (search_term, product_id), handle nulls, flag/cap outliers."""

def compute_global_stats(df: pd.DataFrame) -> dict:
    """Global priors for smoothing: mean CTR, mean clicks, etc."""
```

### Module: `src/models/strategies.py`
```python
from enum import Enum

class RerankStrategy(str, Enum):
    NAIVE_CLICKS = "naive_clicks"
    RAW_CTR = "raw_ctr"
    POSITION_DEBIASED_CTR = "position_debiased_ctr"
    SMOOTHED_CTR = "smoothed_ctr"
    SMOOTHED_POSITION_CTR = "smoothed_position_ctr"

def get_scorer(strategy: RerankStrategy) -> callable:
    """Factory: return the scoring function for a given strategy."""
```

### Module: `src/models/scorer.py`
```python
def score_naive_clicks(df: pd.DataFrame) -> pd.Series:
    """Strategy A: raw clicks descending."""

def score_raw_ctr(df: pd.DataFrame) -> pd.Series:
    """Strategy B: clicks / impressions."""

def score_position_debiased_ctr(df: pd.DataFrame) -> pd.Series:
    """Strategy C: CTR with position bias multiplier."""

def score_smoothed_ctr(df: pd.DataFrame, alpha: float, beta: float) -> pd.Series:
    """Strategy D: Beta-binomial smoothed CTR."""

def score_smoothed_position_ctr(df: pd.DataFrame, config: RerankConfig) -> pd.Series:
    """Strategy E (default): smoothed CTR + position bias correction."""

def position_correction_factor(pos: float, config: RerankConfig) -> float:
    """Position bias correction — shape determined by EDA (D6)."""
```

### Module: `src/models/reranker.py`
```python
def rerank(
    term: str,
    df: pd.DataFrame,
    strategy: RerankStrategy = RerankStrategy.SMOOTHED_POSITION_CTR,
    config: RerankConfig | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Filter to term, score with chosen strategy, sort, return ordered list."""

def rerank_all(
    df: pd.DataFrame,
    strategy: RerankStrategy = RerankStrategy.SMOOTHED_POSITION_CTR,
    config: RerankConfig | None = None,
) -> dict[str, list[dict]]:
    """Batch rerank all terms for evaluation."""
```

### Module: `src/evaluation/metrics.py`
```python
def ndcg_at_k(ranked_ids: list, relevance: dict, k: int = 10) -> float:
    """Compute NDCG@k for a single term's ranked list."""

def compare_to_baseline(
    df: pd.DataFrame,
    rerank_fn: callable,
    k: int = 10,
) -> dict:
    """Per-term NDCG@k, summary stats, and delta vs. baseline."""
```

### Module: `src/api.py`
```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/rerank")
def rerank_endpoint(
    q: str = Query(..., description="Search term"),
    k: int = Query(20, description="Top N results"),
    strategy: RerankStrategy = Query(
        RerankStrategy.SMOOTHED_POSITION_CTR,
        description="Scoring strategy"
    ),
):
    """Rerank products for search term and return top-k with scores + images."""
    results = reranker.rerank(q, global_df, strategy=strategy, top_k=k)
    return {"query": q, "strategy": strategy.value, "results": results}
```

---

## QA & Exploration (Task 2 — nice-to-have)

If time permits after the core reranker:

- **Jupyter notebook** leveraging `src/explore/` modules, with inline product images using the CDN URL format (`https://cdn.aboutstatic.com/file/{image_hash}`)
- Per-query side-by-side: baseline ranking (impression_pos_avg) vs. reranker output
- Quick table showing product thumbnail, name, original rank, new rank, click count, score
- 3-5 example queries with annotated observations

---

## Build Order

The architecture imposes a dependency order — each phase unblocks the next:

```
1. src/explore/     →  Produces data profile, position bias curves, sparsity stats
       ↓
3. src/data/        →  Loader and cleaner, using thresholds from EDA
       ↓
4. src/models/      →  All five strategies, with config from D6
       ↓
5. src/evaluation/  →  NDCG@10 comparison across strategies
       ↓
6. src/api.py       →  FastAPI endpoint with strategy parameter
       ↓
7. QA notebook      →  Visual exploration (nice-to-have)
```
