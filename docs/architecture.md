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

**Rationale:** The timebox is 5-6 hours. Standing up a database + dbt project eats 2-3 hours before any reranking logic is written. 

### D2: Statistical/heuristic model (not ML)

**Context:** Approach choice between heuristic, statistical model, and full learning-to-rank.

**Decision:** Position-bias-corrected CTR scoring with confidence weighting. A statistical model grounded in IR best practices.

**Rationale:**
- Achievable within the timebox with room for evaluation, QA tool, and presentation prep
- Explainable to non-technical stakeholders (the presentation audience)
- ML approaches (logistic regression, LambdaRank) add feature engineering, train/val splits, and debugging complexity that risks an incomplete deliverable
- The data is highly sparse (~80%+ rows have zero clicks) — simple heuristics would overfit to noise, but lightweight statistical smoothing handles the long tail well

### D3: Layered Python architecture

**Context:** How to organize the prototype code.

**Decision:** Four-layer Python module structure:

```
src/
├── data/           # Ingestion + validation
│   ├── loader.py       # Read parquet/csv, validate schema
│   └── cleaner.py      # Deduplication, null handling, outlier flags
├── models/         # Reranking logic
│   ├── scorer.py       # CTR scoring, position bias correction, confidence weighting
│   └── reranker.py     # Term-level reordering, tie-breaking
├── evaluation/     # Offline evaluation
│   └── metrics.py      # NDCG@K, position-weighted recall, etc.
└── api.py          # FastAPI endpoint for serving
```

**Rationale:** Separation of concerns mirrors what a production dbt pipeline would do (sources → staging → marts → serving), but without the warehouse dependency. Each layer is independently testable.

### D4: FastAPI serving layer

**Context:** How to expose the reranker for demonstration.

**Decision:** Lightweight FastAPI app with a single endpoint — `GET /rerank?q={search_term}&k={top_n}` returning ordered product IDs with scores and metadata.

**Rationale:** 
- Shows production serving thinking beyond the notebook
- The API seed in the presentation shows how this graduates from prototype to a production service
- Low effort (~30 lines) for high signal to a senior role interview panel

### D5: Evaluation metric — NDCG@10 with click-based relevance

**Context:** How to measure reranker quality.

**Decision:** NDCG@10 using clicks as relevance labels, computed per search term, then averaged across terms.

**Rationale:**
- NDCG is the standard IR metric for ranked retrieval
- @10 reflects the above-the-fold SRP real estate
- Clicks are the only behavioural signal available; treated as ordinal relevance grades (0, 1, 2, …)
- Per-term evaluation respects that each query has a different candidate set
- Baselines: impression_pos_avg (logged display order) and raw clicks descending

### D6: Scoring approach — Smoothed, position-corrected CTR

**Design (pending data confirmation):**

For each search term × product pair:

```
score = (clicks + α) / (impressions + β) × position_correction_factor
```

Where:
- **α (prior clicks)** and **β (prior impressions)** form a Beta-binomial smoothing — shrinks noisy CTR estimates toward the global mean, critical for the long tail of sparse terms
- **position_correction_factor** accounts for the well-known position bias (items at position 1 get more clicks regardless of relevance). Likely a multiplicative factor based on impression_pos_avg vs. a reference position
- **Confidence floor:** If impressions < min_impressions_threshold, score is heavily penalized or assigned a default rank

**Key references grounding this approach:**
- Joachims et al. (2017) — position bias in click data for search
- Beta-binomial smoothing used in Etsy, Airbnb CTR estimation
- Practical examples from LinkedIn's search reranking blog posts

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

### Data unknowns to resolve:

- [ ] Row count and memory size
- [ ] Unique search_terms count and candidates-per-term distribution
- [ ] Click sparsity (% rows with clicks=0)
- [ ] CTR distribution (range, percentiles, extreme values)
- [ ] Position column ranges and null rates (especially click_pos_*)
- [ ] Correlation between impression_pos_avg and clicks (position bias evidence)
- [ ] Terms with zero total clicks (cold-start handling needed)

---

## Prototype Structure

### Module: `src/data/loader.py`
```python
def load_search_data(path: str) -> pd.DataFrame:
    """Read parquet, validate required columns exist, return DataFrame."""
    
def load_product_metadata(path: str) -> pd.DataFrame:
    """Read CSV product metadata."""
```

### Module: `src/data/cleaner.py`
```python
def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate (search_term, product_id), handle nulls, flag outliers."""

def compute_global_stats(df: pd.DataFrame) -> dict:
    """Global priors for smoothing: mean CTR, mean clicks, etc."""
```

### Module: `src/models/scorer.py`
```python
def compute_smoothed_ctr(
    row: pd.Series, 
    global_ctr: float, 
    alpha: float, 
    beta: float
) -> float:
    """Beta-binomial smoothed CTR for a single term-product row."""

def position_correction_factor(impression_pos_avg: float) -> float:
    """Multiplier to offset position bias (lower pos → higher factor)."""

def score_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply full scoring pipeline: clean → smooth → correct → score."""
```

### Module: `src/models/reranker.py`
```python
def rerank(term: str, df: pd.DataFrame, top_k: int | None = None) -> list[dict]:
    """Filter to term, sort by score descending, return ordered product list."""

def rerank_all(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Batch rerank all terms for evaluation."""
```

### Module: `src/evaluation/metrics.py`
```python
def ndcg_at_k(ranked_ids: list, relevance: dict, k: int = 10) -> float:
    """Compute NDCG@k for a single term's ranked list."""

def evaluate_reranker(
    df: pd.DataFrame, 
    rerank_fn: callable, 
    k: int = 10
) -> dict:
    """Per-term NDCG@k, summary stats, and comparison to baselines."""
```

### Module: `src/api.py`
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/rerank")
def rerank_endpoint(q: str, k: int = 20):
    """Rerank products for search term and return top-k with scores."""
    results = reranker.rerank(q, global_df, top_k=k)
    return {"query": q, "results": results}
```

---

## QA & Exploration (Task 2 — nice-to-have)

If time permits after the core reranker:

- **Jupyter notebook** with inline product images using the CDN URL format (`https://cdn.aboutstatic.com/file/{image_hash}`)
- Per-query side-by-side: baseline ranking (impression_pos_avg) vs. reranker output
- Quick table showing product thumbnail, name, original rank, new rank, click count, score
- 3-5 example queries with annotated observations

**Not** building a full Streamlit/Dash dashboard — a notebook with rich table display is sufficient for the timebox.

---

## Presentation Structure (Task 4 — 15-20 min)

### Slide 1: Problem
- Search returns candidates; we need to order them by user preference
- Click behaviour is the signal

### Slide 2: Approach
- Statistical reranking: smoothed CTR + position bias correction
- Why not pure ML (time, explainability, data sparsity)
- Why not simple heuristics (overfit to noise, no confidence handling)

### Slide 3: Results (example queries)
- Before/after ranking for 2-3 queries
- NDCG@10 vs. baselines
- Screenshots or product images to make it visual

### Slide 4: Production Vision (Task 3)
- **Data pipeline:** dbt models for ingestion, cleaning, feature engineering — scheduled via Airflow
- **Serving:** FastAPI microservice with Redis cache for hot queries
- **Experimentation:** Online A/B testing framework, NDCG monitoring
- **Personalization:** User-level features, collaborative filtering for cold-start terms
- **Unseen terms:** Semantic similarity to known terms via embeddings

### Slide 5: Trade-offs & Alternatives
- Simpler: position-debiased clicks only (loses confidence weighting)
- More complex: LambdaRank (requires more data, harder to debug)
- Why the middle ground is right for this phase

---

## Dependencies to Add

```bash
uv add fastapi uvicorn
# Optionally for QA:
uv add jupyter ipykernel tabulate
# For production vision (not needed now, but noted):
# scikit-learn, lightgbm, scipy
```

---

## Open Questions

1. **Click sparsity impact:** If >90% of rows have zero clicks, we need strong smoothing — potentially a floor where below N impressions, default to global rank. Need data to tune N.

2. **Position bias shape:** Is impression_pos_avg linearly correlated with CTR (log-linear, as typical) or does it have plateaus? Determines whether correction is linear factor or log-transform.

3. **Cold-start terms:** How many search_terms have zero total clicks? These terms have no signal — they get the impression_pos_avg baseline as fallback.

4. **Duplicate products per term:** Are there duplicates on (search_term, product_id)? Need to handle in cleaner.

5. **Click bot detection:** Are there any rows with impossibly high CTR (e.g., 1 impression, 1 click = 100%) that might be noise? Should be flagged.

_Answers to these will come from the data exploration script._
