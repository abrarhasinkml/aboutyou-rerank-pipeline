"""
Reranking strategies — enum of all available scoring approaches and a factory
that returns the appropriate scoring function.

### Tradeoffs (module-level)
- Design: Enum + factory vs. if/elif chain vs. registry dict.
- Gain: Enum gives type safety and IDE autocomplete. Factory decouples
  the strategy name (string/API param) from the implementation (function).
  New strategies are added in one place (the enum + the factory).

### Scale notes
- Strategy dispatch is O(1) — a dict lookup. No performance concern.
- In production: strategies could be loaded as plugins from separate
  modules, registered via entry points (setuptools).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

import pandas as pd

from src.data.cleaner import CleanerConfig
from src.utils import get_logger
from src.models.scorer import (
        score_naive_clicks,
        score_position_debiased_ctr,
        score_raw_ctr,
        score_smoothed_ctr,
        score_smoothed_position_ctr,
    )

logger = get_logger(__name__)


class RerankStrategy(str, Enum):
    """
    Available reranking strategies, ordered by complexity.

    Each strategy produces a 'score' column on the input DataFrame.
    The reranker then sorts by score descending.
    """

    # Heuristic
    NAIVE_CLICKS = "naive_clicks"
    RAW_CTR = "raw_ctr"
    POSITION_DEBIASED_CTR = "position_debiased_ctr"

    # Statistical
    SMOOTHED_CTR = "smoothed_ctr"
    SMOOTHED_POSITION_CTR = "smoothed_position_ctr"


def get_scorer(strategy: RerankStrategy) -> Callable[[pd.DataFrame, dict], pd.Series]:
    """
    Factory: return the scoring function for a given strategy.

    Args:
        strategy: The RerankStrategy enum value identifying which scorer to return.

    Returns:
        A callable that takes (df: pd.DataFrame, params: dict) -> pd.Series of scores.

    Raises:
        ValueError: If the strategy is not implemented.
    """
    
    _registry: dict[RerankStrategy, Callable[[pd.DataFrame, dict], pd.Series]] = {
        RerankStrategy.NAIVE_CLICKS: score_naive_clicks,
        RerankStrategy.RAW_CTR: score_raw_ctr,
        RerankStrategy.POSITION_DEBIASED_CTR: score_position_debiased_ctr,
        RerankStrategy.SMOOTHED_CTR: score_smoothed_ctr,
        RerankStrategy.SMOOTHED_POSITION_CTR: score_smoothed_position_ctr,
    }

    if strategy not in _registry:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Available: {[s.value for s in RerankStrategy]}"
        )

    logger.debug("Resolved strategy %s -> %s", strategy.value, _registry[strategy].__name__)
    return _registry[strategy]
