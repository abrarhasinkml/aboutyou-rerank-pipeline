"""
Data loader — reads parquet and CSV files with chunked iteration,
schema validation, and structured logging.

### Tradeoffs (module-level)
- Design: Two entry points — load_full() for prototype scale, load_chunked()
  for production-scale streaming. Both validate schema.
- Gain: load_full() is simple and fast for 1.6MB. load_chunked() demonstrates
  the production pattern (bounded memory, chunk-level processing).
- load_chunked() returns a generator — callers must iterate or
  explicitly concatenate. This is intentional: it forces awareness of memory.

### Scale notes
- Prototype (1.6MB): load_full() is fine — 1 chunk, <100ms.
- Production (100GB+): load_chunked() with pyarrow row-group streaming.
  Each chunk is independently processable — maps to Spark partitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pandas as pd

from src.utils import (
    DATA_DIR,
    PRODUCT_SCHEMA,
    SEARCH_SCHEMA,
    get_logger,
    iter_csv_chunks,
    iter_parquet_chunks,
    validate_columns,
)

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SEARCH_FILE = "search_term_products.parquet"
PRODUCTS_FILE = "products.csv"

# Columns to read from parquet — reduces memory by ~40% vs. full schema.
# Click position columns are sparse (58.8% null) and not used in scoring,
# but we keep them for the position_bias analysis and future use.
SEARCH_COLUMNS = list(SEARCH_SCHEMA.keys())

# ── Full load (prototype scale) ──────────────────────────────────────────────


def load_search_data(
    data_dir: Path | str = DATA_DIR,
    filename: str = SEARCH_FILE,
) -> pd.DataFrame:
    """
    Load the full search term–product dataset in one read.

    ### Tradeoffs
    - Design: Single pd.read_parquet() vs. chunked iteration.
    - Gain: Simple, fast for prototype scale (1.6MB, <100ms).
    - Entire file in memory. At 100GB+ this OOMs.
      Use load_chunked_search() for production.

    ### Scale notes
    - Prototype: 1.6MB → ~2MB in memory. Safe.
    - Production: switch to load_chunked_search() or Spark read.
    """
    path = Path(data_dir) / filename
    logger.info("Loading search data: %s", path)

    df = pd.read_parquet(path, columns=SEARCH_COLUMNS)
    _validate_search_schema(df)
    _log_load_stats(df, "search_data")

    return df


def load_product_metadata(
    data_dir: Path | str = DATA_DIR,
    filename: str = PRODUCTS_FILE,
) -> pd.DataFrame:
    """
    Load product metadata CSV.

    ### Tradeoffs
    - Design: Single pd.read_csv() vs. chunked.
    - Gain: Product metadata is small (~16K rows, <2MB). Single read is fine.
    """
    path = Path(data_dir) / filename
    logger.info("Loading product metadata: %s", path)

    df = pd.read_csv(path)
    _validate_product_schema(df)
    _log_load_stats(df, "product_metadata")

    return df


# ── Chunked load (production scale) ─────────────────────────────────────────


def load_chunked_search(
    data_dir: Path | str = DATA_DIR,
    filename: str = SEARCH_FILE,
    chunk_size: int = 50_000,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield validated chunks of search data without loading the full file.

    ### Tradeoffs
    - Design: Generator yielding pd.DataFrames vs. single full read.
    - Gain: Memory bounded by chunk_size (default 50K rows ≈ 4MB).
      Each chunk can be processed independently — maps to distributed
      workers in Spark/Dask.
    - Validation: Schema is validated on the first chunk only. Subsequent
      chunks are assumed consistent (parquet guarantees this; CSV does not
      — for CSV, validate each chunk).

    ### Scale notes
    - Prototype: 1 chunk, 19K rows. Generator is overkill but shows intent.
    - Production: 50K-row chunks from pyarrow row-groups. In Spark, replace
      with `spark.read.parquet(path).repartition(col("search_term"))`.
    - Distributed: Each chunk is a unit of work for a Dask partition or
      Spark task. No cross-chunk dependencies.
    """
    path = Path(data_dir) / filename
    logger.info("Loading search data (chunked, chunk_size=%d): %s", chunk_size, path)

    validated = False
    chunk_idx = 0

    for chunk in iter_parquet_chunks(path, columns=SEARCH_COLUMNS, chunk_size=chunk_size):
        if not validated:
            _validate_search_schema(chunk)
            validated = True

        chunk_idx += 1
        logger.debug(
            "Chunk %d: %d rows, memory=%.2fMB",
            chunk_idx,
            len(chunk),
            chunk.memory_usage(deep=True).sum() / (1024 * 1024),
        )
        yield chunk

    logger.info("Finished loading %d chunks from %s", chunk_idx, path)


def load_chunked_products(
    data_dir: Path | str = DATA_DIR,
    filename: str = PRODUCTS_FILE,
    chunk_size: int = 50000,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield validated chunks of product metadata CSV.

    ### Tradeoffs
    - Design: Same chunked generator pattern as load_chunked_search.
    - CSV has no row-group metadata — chunks are line-count based.
    - Assumption: CSV dtype inference can differ per chunk. We validate
      schema on the first chunk and rely on consistent types thereafter.
    """
    path = Path(data_dir) / filename
    logger.info("Loading product metadata (chunked, chunk_size=%d): %s", chunk_size, path)

    validated = False
    chunk_idx = 0

    for chunk in iter_csv_chunks(path, chunk_size=chunk_size):
        if not validated:
            _validate_product_schema(chunk)
            validated = True

        chunk_idx += 1
        yield chunk

    logger.info("Finished loading %d chunks from %s", chunk_idx, path)


# ── Schema validation helpers ────────────────────────────────────────────────


def _validate_search_schema(df: pd.DataFrame) -> None:
    """Validate search data schema — fail fast with clear message."""
    validate_columns(
        df.columns.tolist(),
        set(SEARCH_SCHEMA.keys()),
        context="search_data",
    )
    logger.debug("Search schema validated: %d columns OK", len(SEARCH_SCHEMA))


def _validate_product_schema(df: pd.DataFrame) -> None:
    """Validate product metadata schema."""
    validate_columns(
        df.columns.tolist(),
        set(PRODUCT_SCHEMA.keys()),
        context="product_metadata",
    )
    logger.debug("Product schema validated: %d columns OK", len(PRODUCT_SCHEMA))


def _log_load_stats(df: pd.DataFrame, name: str) -> None:
    """Log row count, column count, and memory usage after load."""
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(
        "Loaded %s: %d rows, %d columns, %.2fMB",
        name,
        len(df),
        len(df.columns),
        mem_mb,
    )
