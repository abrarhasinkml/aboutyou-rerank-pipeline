"""
Reusable utilities — logging, config, and shared helpers.

### Tradeoffs
- Design: Centralised utilities vs. inline setup in each module.
- Gain: Single place for logging config, path resolution, schema validation.
      Consistent behaviour across all modules.

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
    """
    import pandas as pd

    yield from pd.read_csv(path, chunksize=chunk_size, **kwargs)
