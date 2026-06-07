# Project Status & Next Steps

## Completed

### 1. Explore Phase (`src/explore/`)
- [x] `data_profile.py` — dataset profiling, sparsity, CTR distribution, cold-start
- [x] `position_bias.py` — CTR vs position analysis, model fit (R²<0.38 all models)
- [x] `visualise.py` — 4 plots: CTR distribution, position bias curve, candidates per term, click sparsity
- [x] `eda_findings.md` — merged report with all findings, seasonality bias section, embedded plots
- **Key findings:** 58.8% zero-click rows, no position bias, CTR>100% is legitimate (bookmarks/notifications), 0% cold-start terms

### 2. D6 Resolution (`docs/architecture.md`)
- [x] No position bias → `position_correction_type="none"`
- [x] `smoothing_alpha=1.0`, `smoothing_beta=40.0` (from global CTR ~2.5%)
- [x] `min_impressions_floor=3` (38.4% of rows below this)
- [x] Seasonality bias documented as deferred production concern
- [x] Strategy D (smoothed CTR) is primary; Strategy E retained for comparison only

### 3. Data Layer (`src/data/`) — PR #3 merged
- [x] `src/utils.py` — logging, schema validation, chunked parquet/CSV iterators
- [x] `src/data/loader.py` — load_search_data, load_product_metadata, load_chunked_search (generator)
- [x] `src/data/cleaner.py` — deduplicate, handle_nulls, flag_outliers, compute_global_stats, clean_search_data
- [x] `tests/test_loader.py` — schema validation, actual file loading
- [x] `tests/test_cleaner.py` — dedup, nulls, outliers, global stats, full pipeline
- [x] 36 tests passing

### 4. Models Layer (`src/models/`) — PR #4 open
- [x] `src/models/strategies.py` — RerankStrategy enum + get_scorer factory
- [x] `src/models/scorer.py` — 5 scoring functions (A-E) with uniform (df, params) interface
- [x] `src/models/reranker.py` — rerank() single term, rerank_all() batch
- [x] `tests/test_models.py` — 31 tests passing
- [x] Typed args with Args sections explaining each parameter

---

## Not Started

### 5. Evaluation (`src/evaluation/`)
- [ ] `src/evaluation/metrics.py` — NDCG@K implementation
- [ ] `tests/test_metrics.py` — pytest for NDCG calculation
- [ ] Strategy comparison: run all 5 strategies against 2 baselines (impression_pos_avg ascending, raw clicks descending)
- [ ] Per-term NDCG@10, aggregated mean NDCG, comparison table
- [ ] Document which strategy performs best and why

### 6. API Layer (`src/api.py`)
- [ ] FastAPI endpoint: `GET /rerank?q={search_term}&k={top_k}&strategy={strategy}`
- [ ] **Input sanitisation** (important — not yet implemented):
  - Lowercase query input
  - Strip whitespace
  - Collapse multiple spaces
  - This should live in the API layer, NOT in reranker.py (keep reranker as a pure function)
- [ ] Response: ordered product IDs with scores, metadata, product images (CDN URL)
- [ ] Tests for the API endpoint
- [ ] Error handling: unknown search term, invalid strategy, missing params

### 7. QA Notebook (nice-to-have)
- [ ] Jupyter notebook with inline product images (CDN: `https://cdn.aboutstatic.com/file/{image_hash}`)
- [ ] Side-by-side: baseline ranking vs reranker output for 3-5 example queries
- [ ] Strategy comparison table (NDCG@10 for all strategies)
- [ ] Product thumbnail + name + original rank + new rank + click count + score

---

## Coding Standards (from `.commandcode/coding-style/taste.md`)
- Always discuss tradeoffs for every decision
- Always think optimisation
- Always think OOM, distributed computing, production scale (millions in traffic)
- Chunked loading, not everything in memory
- Logger only, no print statements
- Typed function args with Args sections
- Utils for reusable code (used 3+ times)
- Tests for every module

---

## Branches
| PR | Branch | Status |
|----|--------|--------|
| #1 | architecture-design | Merged |
| #2 | explore-phase | Merged |
| #3 | data-loader-cleaner | Merged |
| #4 | models-strategies | Open |
