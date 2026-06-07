"""
Data cleaner - deduplication, null handling, outlier flagging, text normalisation,
and global stats computation for the search reranking dataset.

### Tradeoffs (module-level)
- Design: Stateless functions operating on DataFrames vs. a Cleaner class.
- Gain: Pure functions are testable, composable, and parallelisable.
      No mutable state between calls.
- Alternative: Config must be threaded through each call. For this project
  (3-4 calls) it's acceptable; at 20+ functions consider a config singleton.

### Scale notes
- All operations are vectorised pandas - O(n) per function.
- For 100GB+ data: each function maps to a Spark `.transform()` or Dask
  `.map_partitions()` call. Stateless design makes this trivial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger, validate_columns

logger = get_logger(__name__)


# -- Text normalisation -------------------------------------------------------


def normalize_text(text: str) -> str:
    """
    Normalise a text string: lowercase, strip whitespace, collapse spaces.

    Used on both search_term (at load time) and API query input (at request time)
    so the two always match regardless of how the user types the query.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    cleaned = str(text).lower().strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


# -- Configuration ------------------------------------------------------------


@dataclass(frozen=True)
class CleanerConfig:
    """
    All cleaning thresholds in one place - nothing hardcoded inline.

    ### Tradeoffs
    - Design: Frozen dataclass vs. plain dict.
    - Gain: Type-safe, IDE-friendly, immutable. Prevents accidental mutation.
    """

    # Deduplication
    dedup_subset: tuple[str, ...] = ("search_term", "product_id")
    dedup_keep: str = "first"  # "first" or "last"

    # Null handling
    impression_pos_fillna_strategy: str = "median"  # "median" | "mean" | "drop"
    click_pos_fillna_strategy: str = "none"

    # Outlier flagging (NOT capping - clicks > impressions is legitimate)
    flag_ctr_above_100: bool = True
    flag_low_impressions: bool = True
    low_impression_threshold: int = 3  # from EDA: 38.4% of rows below this

    # Row filtering
    drop_null_search_term: bool = True
    drop_null_product_id: bool = True


DEFAULT_CONFIG = CleanerConfig()


# -- Core cleaning functions --------------------------------------------------


def normalize_search_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and collapse whitespace on the search_term column."""
    n_unique_before = df["search_term"].nunique()
    df["search_term"] = df["search_term"].astype(str).apply(normalize_text)
    n_unique_after = df["search_term"].nunique()
    if n_unique_before != n_unique_after:
        logger.info(
            "Normalised search_term: unique terms %d -> %d",
            n_unique_before, n_unique_after,
        )
    return df


def normalize_product_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and collapse whitespace on the product_name column."""
    if "product_name" not in df.columns:
        return df
    df["product_name"] = df["product_name"].astype(str).apply(normalize_text)
    logger.debug("Normalised product_name column")
    return df


def deduplicate(
    df: pd.DataFrame,
    config: CleanerConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Remove duplicate (search_term, product_id) pairs."""
    n_before = len(df)
    subset = list(config.dedup_subset)

    df = df.drop_duplicates(subset=subset, keep=config.dedup_keep)
    n_dropped = n_before - len(df)

    if n_dropped > 0:
        logger.info("Deduplication: dropped %d duplicate rows (%.1f%%)", n_dropped, n_dropped / n_before * 100)
    else:
        logger.debug("Deduplication: no duplicates found")

    return df


def handle_nulls(
    df: pd.DataFrame,
    config: CleanerConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Handle null values according to per-column strategy."""
    n_before = len(df)

    # Drop rows with null identifiers
    if config.drop_null_search_term:
        null_terms = df["search_term"].isna().sum()
        if null_terms > 0:
            logger.warning("Dropping %d rows with null search_term", null_terms)
            df = df.dropna(subset=["search_term"])

    if config.drop_null_product_id:
        null_pids = df["product_id"].isna().sum()
        if null_pids > 0:
            logger.warning("Dropping %d rows with null product_id", null_pids)
            df = df.dropna(subset=["product_id"])

    # Fill impression position nulls
    if config.impression_pos_fillna_strategy == "median":
        median_val = df["impression_pos_avg"].median()
        null_count = df["impression_pos_avg"].isna().sum()
        if null_count > 0:
            df["impression_pos_avg"] = df["impression_pos_avg"].fillna(median_val)
            df["impression_pos_median"] = df["impression_pos_median"].fillna(median_val)
            logger.info(
                "Filled %d null impression_pos values with median=%.1f",
                null_count, median_val,
            )
    elif config.impression_pos_fillna_strategy == "drop":
        df = df.dropna(subset=["impression_pos_avg", "impression_pos_median"])

    # Click position nulls: leave as NaN (expected for zero-click rows)
    if config.click_pos_fillna_strategy != "none":
        logger.warning(
            "click_pos_fillna_strategy=%s - this may bias averages for zero-click rows",
            config.click_pos_fillna_strategy,
        )

    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info("Null handling: dropped %d rows total", n_dropped)

    return df


def flag_outliers(
    df: pd.DataFrame,
    config: CleanerConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Add boolean flag columns for outlier rows without modifying the data."""
    flags = pd.DataFrame(index=df.index)

    if config.flag_ctr_above_100:
        flags["flag_ctr_above_100"] = df["ctr"] > 100
        n_flagged = flags["flag_ctr_above_100"].sum()
        if n_flagged > 0:
            logger.info(
                "Flagged %d rows with CTR > 100%% (legitimate: bookmarks/notifications)",
                n_flagged,
            )

    if config.flag_low_impressions:
        flags["flag_low_impressions"] = df["impressions"] < config.low_impression_threshold
        n_flagged = flags["flag_low_impressions"].sum()
        if n_flagged > 0:
            logger.info(
                "Flagged %d rows with impressions < %d (low confidence)",
                n_flagged, config.low_impression_threshold,
            )

    if len(flags.columns) > 0:
        df = pd.concat([df, flags], axis=1)

    return df


# -- Global stats -------------------------------------------------------------


def compute_global_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute global priors for Beta-binomial smoothing and baseline metrics."""
    clicked_mask = df["clicks"] > 0

    stats = {
        "global_mean_ctr": float(df.loc[clicked_mask, "ctr"].mean()) if clicked_mask.any() else 0.0,
        "global_mean_clicks": float(df["clicks"].mean()),
        "global_mean_impressions": float(df["impressions"].mean()),
        "n_rows": len(df),
        "n_unique_terms": int(df["search_term"].nunique()),
        "n_unique_products": int(df["product_id"].nunique()),
        "zero_click_pct": round(float((df["clicks"] == 0).mean() * 100), 1),
        "smoothing_beta": _compute_smoothing_beta(
            alpha=1.0,
            global_mean_ctr=df.loc[clicked_mask, "ctr"].mean() if clicked_mask.any() else 0.0,
        ),
    }

    logger.info(
        "Global stats: mean_ctr=%.2f%%, mean_clicks=%.1f, sparsity=%.1f%%, smoothing_beta=%.1f",
        stats["global_mean_ctr"],
        stats["global_mean_clicks"],
        stats["zero_click_pct"],
        stats["smoothing_beta"],
    )

    return stats


def _compute_smoothing_beta(alpha: float, global_mean_ctr: float) -> float:
    """Derive Beta-binomial smoothing parameter beta from alpha and global CTR."""
    if global_mean_ctr <= 0 or global_mean_ctr > 100:
        logger.warning("global_mean_ctr=%.2f is out of range, using default beta=99", global_mean_ctr)
        return 99.0

    return 100.0 / global_mean_ctr - 1.0


# -- Pipeline -----------------------------------------------------------------


def clean_search_data(
    df: pd.DataFrame,
    config: CleanerConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Full cleaning pipeline: normalise search terms -> deduplicate -> handle nulls -> flag outliers.

    ### Tradeoffs
    - Design: Explicit function chain vs. a pipeline object.
    - Gain: Each step is independently testable and logged. Clear data
      lineage in logs.
    - Future ref: Each step creates a new DataFrame (not in-place). For
      19K rows this is fine; for 100GB+ consider in-place operations or
      a streaming pipeline.
    """
    logger.info("Starting cleaning pipeline: %d rows", len(df))

    df = normalize_search_terms(df)
    df = deduplicate(df, config)
    df = handle_nulls(df, config)
    df = flag_outliers(df, config)

    logger.info("Cleaning pipeline complete: %d rows", len(df))
    return df


def clean_product_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise product_name column in product metadata."""
    logger.info("Cleaning product metadata: %d rows", len(df))
    df = normalize_product_names(df)
    return df
