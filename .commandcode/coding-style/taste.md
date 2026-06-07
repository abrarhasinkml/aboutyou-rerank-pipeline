# Coding Standards

This file captures non-negotiable coding standards for the ABOUT YOU search reranking prototype. Every piece of code — from EDA through production serving — is evaluated against these rules.

---

## 1. Tradeoffs — explain every decision

Every non-trivial function, algorithm choice, or data structure decision must carry a brief tradeoff comment. The goal is to make the architecture doc (D1–D6) visible in the code itself.

**Rule:** Functions with algorithmic logic or data structure choices must include a docstring section `### Tradeoffs` listing what was gained and what was sacrificed.

```python
def compute_smoothed_ctr(clicks, impressions, alpha, beta):
    """
    Beta-binomial smoothed CTR.

    ### Tradeoffs
    - Gain: Robust to sparse data — shrinks toward global prior instead of overfitting.
    - Sacrifice: Requires tuning alpha/beta from data; less interpretable than raw CTR.
    - Alternative considered: Wilson score interval (harder to combine with position correction).
    """
```

**Strictness by stage:**
| Stage | Requirement |
|-------|-------------|
| `src/explore/` | Tradeoff notes on analysis design choices and visualization decisions |
| `src/data/` | Tradeoff notes on loader chunking strategy, cleaner threshold choices |
| `src/models/` | Tradeoff notes on **every** scoring and ranking function |
| `src/evaluation/` | Tradeoff notes on metric choice, @k selection, aggregation method |
| `src/api.py` | Tradeoff notes on endpoint design, caching decisions, response shape |

---

## 2. Optimisation — always think in terms of performance

Code must be written with performance awareness, even when working with small data. The habit of thinking about computational cost should be visible.

**Rules:**
- Use **vectorised pandas/numpy** operations — never iterate row-by-row with `.iterrows()` or `for` loops over DataFrames
- Prefer `groupby().transform()` over `groupby().apply()` for per-group operations
- Avoid repeated full-DataFrame scans; compute once and reuse
- Document computational complexity for functions that operate on more than O(n)

```python
def score_dataframe(df, config):
    """
    Scoring pipeline — O(n) single pass over the DataFrame.

    ### Tradeoffs
    - Design: Single vectorised pass vs. chained transformations.
    - Gain: One scan over all rows; memory predictable (no intermediate copies).
    - Sacrifice: Less modular than chaining. For >10M rows, would switch to chunked apply.
    """
```

**Strictness by stage:**
| Stage | Requirement |
|-------|-------------|
| `src/explore/` | Vectorised operations; document complexity of analysis loops |
| `src/data/` | Single-pass cleaning where possible; document scan count |
| `src/models/` | Full vectorisation; document O(n) guarantees |
| `src/evaluation/` | Efficient NDCG computation (no O(n²) per term) |
| `src/api.py` | Response time documentation; call-out any blocking operations |

---

## 3. OOM & distributed computing — design for scale

Every module must be written as if the dataset doesn't fit in memory. This is a **deliberate over-engineering** choice to demonstrate production thinking. The code itself runs on a small dataset, but the **comments and design** must show awareness of what changes at scale.

**Rules:**

### 3.1 Chunked I/O pattern
Even though we can `pd.read_parquet()` the whole file, document the production equivalent:

```python
def load_search_data(path: str) -> pd.DataFrame:
    """
    Load search term data.

    ### Scale notes
    - Prototype: Full read — dataset fits in memory (~10-50MB).
    - Production (100-1000x growth): Switch to `pyarrow.dataset` with 
      `use_threads=True` and row-group filtering. For Spark environments,
      use `spark.read.parquet()` with predicate pushdown on search_term.
    - OOM risk: None at prototype scale. At 10M+ terms: column pruning
      (read only `search_term`, `product_id`, `clicks`, `impressions`,
      `impression_pos_avg`) before loading the full schema.
    """
```

### 3.2 Memory accounting
Functions that allocate significant intermediate objects must document expected memory footprint:

```python
def compute_global_stats(df: pd.DataFrame) -> dict:
    """
    ### Memory
    Allocates ~2x input size (groupby aggregation + result dict).
    At 10M rows with 8 columns: ~640MB input → ~1.3GB peak.
    Production: Use incremental stats (Welford's online algorithm) or Spark's 
    `approx_count_distinct` for cardinality estimation.
    """
```

### 3.3 Partitioning awareness
Group-by operations must document the partitioning key:

```python
def rerank_all(df, strategy, config):
    """
    ### Partitioning
    Natural partition key: `search_term`. Each term's candidates are 
    independent — embarrassingly parallel. Production: `df.groupby('search_term')
    .apply(rerank)` maps cleanly to Spark `groupBy('search_term').applyInPandas()`
    or Dask `map_partitions()`.
    """
```

### 3.4 Streaming patterns
Where iteration is unavoidable, document the streaming alternative:

```python
# Prototype: iterate over all terms (fits in memory)
for term in terms:
    rerank(term, df)

# Production (>1M terms): stream from message queue
# async def consume_kafka():
#     for msg in consumer:
#         rerank(msg.search_term, shared_df)
```

**Strictness by stage:**
| Stage | Requirement |
|-------|-------------|
| `src/explore/` | Chunked read notes; OOM call-outs for any full-Dataset operations |
| `src/data/` | Chunked I/O documentation on loader; partitioning note on cleaner |
| `src/models/` | Memory accounting on all scoring functions; partitioning awareness on reranker; streaming alternatives on batch functions |
| `src/evaluation/` | Memory accounting on evaluation aggregations; note on incremental metric computation |
| `src/api.py` | Request isolation notes; shared-state mutability warnings; caching strategy with memory bounds |

---

## 4. Production-grade design — think beyond the prototype

Every module must show awareness of what changes when this graduates from prototype to production.

**Rules:**

### 4.1 Configuration, not hardcoding
Magic numbers must live in a config dataclass, not inline:

```python
# BAD
if impressions < 10:
    score = 0.0

# GOOD
if impressions < config.min_impressions_floor:
    score = config.cold_start_default_score
```

### 4.2 Observability hooks
Functions in the serving path (`src/models/reranker.py`, `src/api.py`) must include comments showing where metrics/logging/tracing would be added:

```python
def rerank(term, df, strategy, config, top_k=None):
    """
    ### Observability (production add-ons)
    - Timer: `with metrics.timer("rerank.latency"):`
    - Counter: `metrics.increment("rerank.requests", tags={"strategy": strategy})`
    - Log: `logger.info("rerank", term=term, strategy=strategy, candidates=N, top_k=top_k)`
    - Tracing: `span = tracer.start_span("rerank"); span.set_attribute("term", term)`
    """
```

### 4.3 Error handling as comments
For prototype code, document expected failure modes rather than implementing full error handling (which would eat timebox):

```python
def load_search_data(path: str) -> pd.DataFrame:
    """
    ### Failure modes (production hardening)
    - FileNotFound: Raise clear error with path (prototype: let pandas handle it)
    - Schema mismatch: Validate column presence + types (prototype: let pandas fail)
    - Corrupt parquet: Catch PyArrowException, retry once, then alert (prototype: fail fast)
    - Empty file: Return empty DataFrame with schema (prototype: let pandas return empty)
    """
```

### 4.4 Stateless serving
The API and reranker must be designed as stateless functions operating on a shared, read-only DataFrame. Any mutable state must be explicitly documented and justified:

```python
# GOOD — stateless, takes df as input
def rerank(term, df, strategy, config, top_k=None):
    ...

# AVOID — relies on module-level state
# df = load()  # module-level — not safe for multi-worker serving
```

### 4.5 Testability by construction
Functions must have clear input/output contracts that are trivially testable:

```python
def score_smoothed_ctr(df: pd.DataFrame, alpha: float, beta: float) -> pd.Series:
    """
    Pure function — same input always produces same output.
    
    Testable properties:
    - Returns float Series with same index as input
    - All scores in [0, 1] range
    - Higher clicks → higher score (for same impressions)
    - Higher impressions → lower score (for same clicks, < global mean)
    """
```

**Strictness by stage:**
| Stage | Requirement |
|-------|-------------|
| `src/explore/` | Config for thresholds; testable properties on analysis functions |
| `src/data/` | Config-driven cleaning thresholds; failure mode documentation on loader |
| `src/models/` | Full config-driven scoring; observability hooks; stateless design; testable property documentation on every function |
| `src/evaluation/` | Config-driven @k and metric selection; failure mode docs |
| `src/api.py` | Observability hooks on every endpoint; stateless serving; request validation documented |

---

## 5. Commit & review discipline

- **Atomic commits** — one logical change per commit. No "misc fixes" or "update code" commits.
- **Commit messages** — imperative mood, present tense. Describe what the change does and why.
- **Co-author trailer** — every commit must end with `Co-authored-by: CommandCodeBot <noreply@commandcode.ai>`.

---

## Summary Matrix

| Concern | explore/ | data/ | models/ | evaluation/ | api.py |
|---------|----------|-------|---------|-------------|--------|
| Tradeoff docs | Design choices | Threshold choices | Every function | Metric choices | Endpoint design |
| Optimisation | Vectorised + complexity notes | Single-pass | Full vectorised + O(n) | Efficient NDCG | Latency docs |
| OOM / distributed | Chunked read notes | Chunked I/O + partition docs | Memory accounting + streaming | Memory accounting | Shared state warnings |
| Production design | Config-driven | Config + failure modes | Full rigor (config, observability, stateless, testable) | Config + failure modes | Full rigor (observability, stateless, validation) |
