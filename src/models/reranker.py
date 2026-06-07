"""
Reranker — filters to a single search term, scores candidates with the chosen
strategy, sorts descending, and returns an ordered list.

### Tradeoffs (module-level)
- Design: Stateless functions taking DataFrame + strategy vs. a Reranker class.
- Gain: Pure functions are testable and parallelisable. No mutable state.
- Sacrifice: Config (params dict) must be threaded through each call. At scale,
  consider a RerankerConfig dataclass like CleanerConfig.

### Scale notes
- rerank(): operates on a single term's candidates (~60-100 rows). Trivial.
- rerank_all(): iterates over all 305 terms. Each is independent — maps to
  Spark groupBy('search_term').apply(rerank_fn) or Dask map_partitions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.strategies import RerankStrategy, get_scorer
from src.utils import get_logger

logger = get_logger(__name__)


# ── Default parameters from D6 resolution ────────────────────────────────────

DEFAULT_PARAMS: dict[str, Any] = {
    "alpha": 1.0,               # moderate sparsity -> light prior
    "beta": 40.0,               # derived: 100/2.5 - 1 = 39, rounded to 40
    "reference_position": 5,    # reference for position correction (Strategy E only)
    "gamma": 0.01,              # decay factor for position correction (Strategy E only)
    "min_impressions_floor": 3, # rows below this get penalised score
}


def rerank(
    term: str,
    df: pd.DataFrame,
    strategy: RerankStrategy = RerankStrategy.SMOOTHED_CTR,
    params: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Rerank products for a single search term.

    Args:
        term: The search term to filter on. Must match a value in df['search_term'].
        df: Full search dataset. Will be filtered to rows where
            search_term == term.
        strategy: Which scoring strategy to use. Default: SMOOTHED_CTR (D6 resolved).
        params: Scoring parameters (alpha, beta, gamma, etc.). Merged with
            DEFAULT_PARAMS — caller overrides only what they need.
        top_k: If set, return only the top K results. If None, return all
            candidates for the term.

    Returns:
        List of dicts, each with keys: product_id, score, clicks, impressions,
        impression_pos_avg, ctr, rank. Sorted by score descending.
    """
    merged_params = {**DEFAULT_PARAMS, **(params or {})}

    # Filter to term
    term_df = df[df["search_term"] == term].copy()
    if term_df.empty:
        logger.warning("No candidates found for term=%r", term)
        return []

    # Score
    scorer = get_scorer(strategy)
    term_df["score"] = scorer(term_df, merged_params)

    # Apply confidence floor: penalise rows with very few impressions
    min_impressions: int = merged_params.get("min_impressions_floor", 3)
    low_impression_mask = term_df["impressions"] < min_impressions
    if low_impression_mask.any():
        # Pull low-impression rows toward 0 (conservative: don't trust thin data)
        penalty_factor = term_df.loc[low_impression_mask, "impressions"] / min_impressions
        term_df.loc[low_impression_mask, "score"] *= penalty_factor
        logger.debug(
            "Applied confidence floor to %d/%d candidates (impressions < %d)",
            low_impression_mask.sum(), len(term_df), min_impressions,
        )

    # Sort descending by score
    term_df = term_df.sort_values("score", ascending=False).reset_index(drop=True)

    # Apply top_k
    if top_k is not None:
        term_df = term_df.head(top_k)

    # Build result
    results = []
    for rank, (_, row) in enumerate(term_df.iterrows(), start=1):
        results.append({
            "product_id": int(row["product_id"]),
            "score": round(float(row["score"]), 4),
            "clicks": int(row["clicks"]),
            "impressions": int(row["impressions"]),
            "impression_pos_avg": round(float(row["impression_pos_avg"]), 1),
            "ctr": round(float(row["ctr"]), 2),
            "rank": rank,
        })

    logger.debug(
        "Reranked term=%r with strategy=%s: %d results (top_k=%s)",
        term, strategy.value, len(results), top_k,
    )
    return results


def rerank_all(
    df: pd.DataFrame,
    strategy: RerankStrategy = RerankStrategy.SMOOTHED_CTR,
    params: dict[str, Any] | None = None,
    top_k: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Batch rerank all search terms.

    Args:
        df: Full search dataset with 'search_term' column.
        strategy: Scoring strategy to apply to all terms.
        params: Scoring parameters. Merged with DEFAULT_PARAMS.
        top_k: If set, return only top K results per term.

    Returns:
        Dict mapping search_term -> list of result dicts (same format as rerank()).

    ### Tradeoffs
    - Design: Simple loop over unique terms vs. groupby().apply().
    - Gain: Each term is independent — trivially parallelisable.
      For 305 terms, sequential is fast (<1s). For 1M+ terms, use
      Spark groupBy or Dask map_partitions.
    - Sacrifice: No cross-term scoring or global reordering. Each term
      is reranked independently.
    """
    merged_params = {**DEFAULT_PARAMS, **(params or {})}

    terms = df["search_term"].unique()
    logger.info("Reranking %d terms with strategy=%s", len(terms), strategy.value)

    results: dict[str, list[dict[str, Any]]] = {}
    for term in terms:
        results[term] = rerank(term, df, strategy, merged_params, top_k)

    logger.info("Reranking complete: %d terms processed", len(results))
    return results
