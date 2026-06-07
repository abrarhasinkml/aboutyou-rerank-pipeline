"""
Reusable utilities — logging, config, and shared helpers.

### Tradeoffs
- Design: Centralised utilities vs. inline setup in each module.
- Gain: Single place for logging config, path resolution, schema validation.
      Consistent behaviour across all modules.
- Sacrifice: Adds a dependency between modules (all import from utils).
      For this project size it's acceptable; at 50+ modules consider a
      utils package with sub-modules.

### Scale notes
- Logging: Structured JSON logging is the production default. We use a
  text formatter for prototype readability. In production, swap
  JsonFormatter for `python-json-logger` or an OpenTelemetry exporter.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# ── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ── Logging ──────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger with consistent formatting.

    ### Tradeoffs
    - Design: Single shared handler vs. per-module handlers.
    - Gain: One place to control log level, format, and output destination.
      `logging.getLogger(name)` returns the same instance on repeated calls
      — no duplicate handlers.
    - Sacrifice: Global state (the root logger). Acceptable for a prototype.
      Production: use `structlog` or `python-json-logger` for structured
      JSON output that integrates with log aggregators (ELK, Datadog).

    ### Scale notes
    - At high throughput (>10K log lines/sec): replace StreamHandler with
      a buffered handler or emit to stdout as JSON for a log shipper.
    - Production: add correlation IDs (request/trace) via structlog
      contextvars or OpenTelemetry baggage.
    """
    global _configured
    if not _configured:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT))
        root.addHandler(handler)
        _configured = True

    return logging.getLogger(name)


# ── Schema validation ────────────────────────────────────────────────────────

SEARCH_SCHEMA = {
    "search_term": "string",
    "product_id": "int64",
    "clicks": "int64",
    "impressions": "int64",
    "ctr": "float64",
    "impression_pos_avg": "float64",
    "impression_pos_median": "float64",
    "click_pos_avg": "float64",
    "click_pos_median": "float64",
}

PRODUCT_SCHEMA = {
    "product_id": "int64",
    "product_name": "string",
    "image_hash": "string",
    "product_url": "string",
}


def validate_columns(
    df_columns: list[str],
    required: set[str],
    context: str = "",
) -> None:
    """
    Raise ValueError if required columns are missing.

    ### Tradeoffs
    - Fail-fast validation vs. silent column absence.
    - Gain: Clear error messages with the missing column names.
    - Sacrifice: Adds a check at every load boundary. At scale, this is
      a single set intersection — O(n) in column count, negligible.

    Used 3+ times across loader, cleaner, and evaluation — hence in utils.
    """
    missing = required - set(df_columns)
    if missing:
        msg = f"Missing columns{f' ({context})' if context else ''}: {missing}"
        raise ValueError(msg)


# ── Chunked parquet reading ─────────────────────────────────────────────────

def iter_parquet_chunks(
    path: Path | str,
    columns: list[str] | None = None,
    chunk_size: int = 50_000,
):
    """
    Yield row-group chunks from a parquet file without loading everything
    into memory.

    ### Tradeoffs
    - Design: Row-group streaming via pyarrow vs. single pd.read_parquet().
    - Gain: Memory footprint is bounded by chunk_size, not total file size.
      For 100GB production files this is the only viable approach.
    - Sacrifice: Row-groups may not align exactly with chunk_size. We accept
      variable chunk sizes from the underlying parquet structure. For this
      prototype (1.6MB) the chunks are small — the pattern matters more
      than the performance.

    ### Scale notes
    - Prototype: 1 file, ~19K rows → 1 chunk. The API is the point.
    - Production: `pyarrow.dataset` with predicate pushdown + partition
      pruning. For Spark: `spark.read.parquet().repartition("search_term")`.
    - Distributed: Each worker processes one row-group independently.
      `pyarrow.parquet.ParquetFile(path).iter_batches(batch_size=chunk_size)`
      is the low-level API for custom parallelism.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=chunk_size, columns=columns):
        yield batch.to_pandas()


def iter_csv_chunks(
    path: Path | str,
    chunk_size: int = 50_000,
    **kwargs,
):
    """
    Yield chunks from a CSV file.

    ### Tradeoffs
    - Same chunked philosophy as iter_parquet_chunks.
    - CSV has no row-group metadata — chunks are line-count based.
    - Sacrifice: dtype inference per chunk can differ. We force dtypes
      in the caller (cleaner/loader) to ensure consistency.
    """
    import pandas as pd

    yield from pd.read_csv(path, chunksize=chunk_size, **kwargs)
