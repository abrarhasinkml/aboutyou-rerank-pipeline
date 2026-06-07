# How to Run

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/abrarhasinkml/aboutyou-rerank-pipeline.git
cd aboutyou-rerank-pipeline

# With uv (recommended)
uv sync

```

## Run Tests

```bash
# All tests (103 total)
uv run pytest tests/ -v

# Evaluation module only
uv run pytest tests/test_metrics.py -v

# API tests only
uv run pytest tests/test_api.py -v

# Models tests only
uv run pytest tests/test_models.py -v

# Data layer tests only
uv run pytest tests/test_loader.py tests/test_cleaner.py -v
```

## Run the API Server

```bash
uv run uvicorn src.api:app --reload
```

The server starts on `http://localhost:8000`. OpenAPI docs at `http://localhost:8000/docs`.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rerank?q={term}&k={n}&strategy={strat}` | GET | Rerank products for a search term |
| `/health` | GET | Health check |

### Example requests

```bash
# Rerank "kleid" with default strategy (smoothed_ctr), top 10
curl "http://localhost:8000/rerank?q=kleid&k=10"

# Use a different strategy
curl "http://localhost:8000/rerank?q=kleid&k=10&strategy=naive_clicks"

```

## Run the QA Notebook

```bash
# Install notebook dependencies
uv add jupyter ipykernel tabulate matplotlib seaborn

# Start Jupyter
uv run jupyter notebook notebooks/qa_exploration.ipynb
```

The notebook covers:
1. Data loading and cleaning
2. Strategy comparison (NDCG@10 for all 5 strategies + 2 baselines)
3. Side-by-side baseline vs. reranker output with product images

## Run Evaluation Standalone

```python
from src.data.loader import load_search_data
from src.data.cleaner import clean_search_data
from src.evaluation.metrics import evaluate_all_strategies

raw = load_search_data()
df = clean_search_data(raw)
results = evaluate_all_strategies(df, k=10)
print(results)
```

## Project Structure

```
src/
├── api.py              # FastAPI serving layer
├── utils.py            # Logging, schema validation, helpers
├── data/
│   ├── loader.py       # Parquet/CSV loading with chunked iteration
│   └── cleaner.py      # Deduplication, null handling, outlier flagging
├── models/
│   ├── strategies.py   # RerankStrategy enum + factory
│   ├── scorer.py       # 5 scoring functions (A-E)
│   └── reranker.py     # Term-level reordering
├── evaluation/
│   └── metrics.py      # NDCG@K, strategy comparison
├── explore/            # EDA modules (data_profile, position_bias, visualise)
└── tests/
    ├── test_cleaner.py
    ├── test_loader.py
    ├── test_models.py
    ├── test_metrics.py
    └── test_api.py
```
