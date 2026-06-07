"""
Scoring functions — each strategy produces a score column on the input DataFrame.

All functions share the same signature: (df, params) -> pd.Series.
This uniform interface lets the strategy factory dispatch without knowing
the internals of each scorer.

### Tradeoffs (module-level)
- Design: Separate function per strategy vs. one function with a strategy param.
- Gain: Each function is independently testable, has its own docstring/tradeoff
  notes, and can be optimised independently.
- Sacrifice: Some code duplication across strategies (e.g., the smoothed CTR
  functions both access 'clicks' and 'impressions' columns). For 5 strategies
  this is acceptable; at 20+ consider a base class or composable transforms.

### Scale notes
- All functions are vectorised pandas — O(n) single pass.
- For 100GB+ data: each scorer maps to a Spark UDF or Dask map_partitions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


# ── Heuristic strategies ─────────────────────────────────────────────────────


def score_naive_clicks(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """
    Strategy A: Rank by raw clicks descending.

    Args:
        df: DataFrame with columns ['clicks']. Must not be empty.
        params: Unused. Present for interface consistency.

    Returns:
        pd.Series of scores (raw clicks), same index as df.
    """
    return df["clicks"].astype(float)


def score_raw_ctr(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """
    Strategy B: Rank by raw CTR (clicks / impressions * 100).

    Args:
        df: DataFrame with columns ['clicks', 'impressions'].
        params: Unused.

    Returns:
        pd.Series of CTR scores. Rows with impressions=0 get score 0.
    """
    return (df["clicks"] / df["impressions"].replace(0, np.nan) * 100).fillna(0.0)


def score_position_debiased_ctr(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """
    Strategy C: CTR with a multiplicative position bias correction.

    Note from EDA: This dataset shows NO position bias (R2<0.38). This strategy
    is retained for evaluation comparison only — it will likely underperform D.

    Args:
        df: DataFrame with columns ['clicks', 'impressions', 'impression_pos_avg'].
        params: dict with keys:
            - 'reference_position' (int, default 5): Reference position for correction.
            - 'gamma' (float, default 0.01): Decay factor for position correction.

    Returns:
        pd.Series of position-debiased CTR scores.
    """
    ref_pos: int = params.get("reference_position", 5)
    gamma: float = params.get("gamma", 0.01)

    raw_ctr = (df["clicks"] / df["impressions"].replace(0, np.nan) * 100).fillna(0.0)
    correction = np.exp(gamma * (df["impression_pos_avg"].fillna(ref_pos) - ref_pos))
    return raw_ctr * correction


# ── Statistical strategies ───────────────────────────────────────────────────


def score_smoothed_ctr(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """
    Strategy D (default): Beta-binomial smoothed CTR.

    Formula: (clicks + alpha) / (impressions + beta) * 100

    Shrinks noisy CTR estimates toward a global prior, preventing products
    with 1 click / 1 impression from outranking proven items.

    Args:
        df: DataFrame with columns ['clicks', 'impressions'].
        params: dict with keys:
            - 'alpha' (float, default 1.0): Prior pseudo-clicks. Higher values
              pull estimates more strongly toward the global mean.
            - 'beta' (float, default 40.0): Prior pseudo-impressions. Derived
              from EDA: global mean CTR ~2.5%, so with alpha=1, beta~=100/2.5-1=39.

    Returns:
        pd.Series of smoothed CTR scores, in [0, 100] range.
    """
    alpha: float = params.get("alpha", 1.0)
    beta: float = params.get("beta", 40.0)

    return (df["clicks"] + alpha) / (df["impressions"] + beta) * 100


def score_smoothed_position_ctr(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """
    Strategy E: Smoothed CTR with position bias correction.

    NOTE: EDA found NO position bias in this dataset. This strategy is retained
    for evaluation comparison only — Strategy D should outperform it.

    Args:
        df: DataFrame with columns ['clicks', 'impressions', 'impression_pos_avg'].
        params: dict with keys:
            - 'alpha' (float, default 1.0): Prior pseudo-clicks.
            - 'beta' (float, default 40.0): Prior pseudo-impressions.
            - 'reference_position' (int, default 5): Reference position for correction.
            - 'gamma' (float, default 0.01): Decay factor for position correction.

    Returns:
        pd.Series of smoothed + position-corrected CTR scores.
    """
    smoothed = score_smoothed_ctr(df, params)

    ref_pos: int = params.get("reference_position", 5)
    gamma: float = params.get("gamma", 0.01)

    correction = np.exp(gamma * (df["impression_pos_avg"].fillna(ref_pos) - ref_pos))
    return smoothed * correction
