"""
Evaluation metrics — NDCG@K for comparing the smoothed CTR reranker against
the impression position baseline.

### Tradeoffs
- Design: Pure functions for NDCG computation vs. a class-based approach.
- Gain: Functions are testable, composable, and parallelisable per search term.
- Sacrifice: Global stats (e.g., mean NDCG across terms) require aggregating
  results manually. For 305 terms this is trivial.

### Scale notes
- Per-term NDCG: O(n log n) for sorting, O(n) for DCG computation.
- Batch evaluation: 305 terms × 2 approaches = 610 NDCG calls. <1s.
- For 1M+ terms: parallelise with Spark/Dask — each term is independent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.reranker import DEFAULT_PARAMS, rerank
from src.models.strategies import RerankStrategy
from src.utils import get_logger

logger = get_logger(__name__)


def dcg_at_k(
    ranked_ids: list[int],
    relevance: dict[int, float],
    k: int = 10,
) -> float:
    """Discounted Cumulative Gain at position k using log2 discount."""
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k]):
        rel = relevance.get(pid, 0.0)
        dcg += rel / np.log2(i + 2)  # i+2: rank is 1-indexed
    return dcg


def ndcg_at_k(
    ranked_ids: list[int],
    relevance: dict[int, float],
    k: int = 10,
) -> float:
    """Normalised DCG at position k. Returns 0-1 (1 = perfect ranking)."""
    if not ranked_ids or k <= 0:
        return 0.0

    dcg = dcg_at_k(ranked_ids, relevance, k)
    ideal_ids = sorted(relevance.keys(), key=lambda pid: relevance.get(pid, 0.0), reverse=True)
    idcg = dcg_at_k(ideal_ids, relevance, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_relevance(
    term_df: pd.DataFrame,
    relevance_col: str = "clicks",
) -> dict[int, float]:
    """Map product_id -> relevance score (clicks as ordinal grades)."""
    return dict(zip(term_df["product_id"], term_df[relevance_col].astype(float)))


def evaluate_strategy(
    df: pd.DataFrame,
    strategy: RerankStrategy,
    k: int = 10,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Evaluate a single reranking strategy across all search terms.

    Returns:
        DataFrame with columns: [search_term, ndcg_at_k, n_candidates].
    """
    merged_params = {**DEFAULT_PARAMS, **(params or {})}

    terms = df["search_term"].unique()
    results = []

    for term in terms:
        term_df = df[df["search_term"] == term].copy()
        if term_df.empty:
            continue

        relevance = compute_relevance(term_df)
        ranked = rerank(term, df, strategy, merged_params, top_k=k)
        ranked_ids = [r["product_id"] for r in ranked]
        ndcg = ndcg_at_k(ranked_ids, relevance, k=k)

        results.append({
            "search_term": term,
            "ndcg_at_k": ndcg,
            "n_candidates": len(term_df),
        })

    logger.info(
        "Evaluated strategy=%s: %d terms, mean NDCG@%d=%.4f",
        strategy.value,
        len(results),
        k,
        np.mean([r["ndcg_at_k"] for r in results]),
    )

    return pd.DataFrame(results)


def evaluate_baseline(
    df: pd.DataFrame,
    sort_col: str,
    ascending: bool = True,
    k: int = 10,
) -> pd.DataFrame:
    """
    Evaluate a baseline (sort by a single column) across all search terms.

    Returns:
        DataFrame with columns: [search_term, ndcg_at_k, n_candidates].
    """
    terms = df["search_term"].unique()
    results = []

    for term in terms:
        term_df = df[df["search_term"] == term].copy()
        if term_df.empty:
            continue

        relevance = compute_relevance(term_df)
        term_df = term_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        ranked_ids = term_df["product_id"].tolist()
        ndcg = ndcg_at_k(ranked_ids, relevance, k=k)

        results.append({
            "search_term": term,
            "ndcg_at_k": ndcg,
            "n_candidates": len(term_df),
        })

    logger.info(
        "Evaluated baseline (sort by %s, %s): %d terms, mean NDCG@%d=%.4f",
        sort_col,
        "ascending" if ascending else "descending",
        len(results),
        k,
        np.mean([r["ndcg_at_k"] for r in results]),
    )

    return pd.DataFrame(results)


def evaluate_against_baseline(
    df: pd.DataFrame,
    k: int = 10,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Compare smoothed CTR reranker against the impression position baseline.

    Returns:
        DataFrame with columns: [strategy, mean_ndcg, std_ndcg, min_ndcg,
        max_ndcg, n_terms]. Two rows — baseline then reranker.
    """
    all_results = []

    # Baseline: impression_pos_avg ascending (logged display order)
    baseline = evaluate_baseline(df, "impression_pos_avg", ascending=True, k=k)
    all_results.append({
        "strategy": "baseline_impression_pos",
        "mean_ndcg": baseline["ndcg_at_k"].mean(),
        "std_ndcg": baseline["ndcg_at_k"].std(),
        "min_ndcg": baseline["ndcg_at_k"].min(),
        "max_ndcg": baseline["ndcg_at_k"].max(),
        "n_terms": len(baseline),
    })

    # Smoothed CTR (default strategy from D6)
    reranker = evaluate_strategy(df, RerankStrategy.SMOOTHED_CTR, k=k, params=params)
    all_results.append({
        "strategy": RerankStrategy.SMOOTHED_CTR.value,
        "mean_ndcg": reranker["ndcg_at_k"].mean(),
        "std_ndcg": reranker["ndcg_at_k"].std(),
        "min_ndcg": reranker["ndcg_at_k"].min(),
        "max_ndcg": reranker["ndcg_at_k"].max(),
        "n_terms": len(reranker),
    })

    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values("mean_ndcg", ascending=False).reset_index(drop=True)

    logger.info(
        "\n=== Smoothed CTR vs Baseline (NDCG@%d) ===\n%s",
        k, result_df.to_string(index=False),
    )

    return result_df
