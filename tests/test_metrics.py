"""Tests for src/evaluation/metrics.py — NDCG@K computation and strategy evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    dcg_at_k,
    ndcg_at_k,
    compute_relevance,
    evaluate_strategy,
    evaluate_baseline,
    evaluate_against_baseline,
)
from src.models.strategies import RerankStrategy


# ── DCG/NDCG unit tests ─────────────────────────────────────────────────────


def test_dcg_perfect_ranking():
    """Perfect ranking (highest relevance first) should maximise DCG."""
    relevance = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0}
    ranked = [1, 2, 3, 4]
    dcg = dcg_at_k(ranked, relevance, k=4)
    # Expected: 3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5)
    expected = 3.0 / np.log2(2) + 2.0 / np.log2(3) + 1.0 / np.log2(4)
    assert abs(dcg - expected) < 1e-6


def test_dcg_worst_ranking():
    """Worst ranking (lowest relevance first) should minimise DCG."""
    relevance = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0}
    ranked = [4, 3, 2, 1]
    dcg_worst = dcg_at_k(ranked, relevance, k=4)
    dcg_best = dcg_at_k([1, 2, 3, 4], relevance, k=4)
    assert dcg_worst < dcg_best


def test_dcg_at_k_truncates():
    """DCG@k should only consider first k elements."""
    relevance = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0}
    ranked = [1, 2, 3, 4]
    dcg_k2 = dcg_at_k(ranked, relevance, k=2)
    dcg_k4 = dcg_at_k(ranked, relevance, k=4)
    assert dcg_k2 < dcg_k4


def test_ndcg_perfect_ranking():
    """NDCG@k should be 1.0 for a perfect ranking."""
    relevance = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0}
    ranked = [1, 2, 3, 4]
    ndcg = ndcg_at_k(ranked, relevance, k=4)
    assert abs(ndcg - 1.0) < 1e-6


def test_ndcg_random_ranking():
    """NDCG@k should be < 1.0 for a non-ideal ranking."""
    relevance = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0}
    ranked = [4, 3, 2, 1]
    ndcg = ndcg_at_k(ranked, relevance, k=4)
    assert ndcg < 1.0


def test_ndcg_empty_ranking():
    """NDCG@k should be 0.0 for an empty ranking."""
    ndcg = ndcg_at_k([], {1: 1.0}, k=10)
    assert ndcg == 0.0


def test_ndcg_zero_relevance():
    """NDCG@k should be 0.0 when all relevance scores are 0."""
    relevance = {1: 0.0, 2: 0.0, 3: 0.0}
    ranked = [1, 2, 3]
    ndcg = ndcg_at_k(ranked, relevance, k=3)
    assert ndcg == 0.0


def test_ndcg_single_item():
    """NDCG@1 for a single item should be 1.0 if ranked first."""
    relevance = {1: 5.0}
    ranked = [1]
    ndcg = ndcg_at_k(ranked, relevance, k=1)
    assert abs(ndcg - 1.0) < 1e-6


def test_dcg_unknown_product():
    """Unknown product IDs should contribute 0 to DCG."""
    relevance = {1: 3.0, 2: 2.0}
    ranked = [1, 999, 2]  # 999 not in relevance
    dcg = dcg_at_k(ranked, relevance, k=3)
    expected = 3.0 / np.log2(2) + 0.0 / np.log2(3) + 2.0 / np.log2(4)
    assert abs(dcg - expected) < 1e-6


# ── compute_relevance tests ─────────────────────────────────────────────────


def test_compute_relevance_basic():
    """Should map product_id to clicks."""
    df = pd.DataFrame({
        "product_id": [10, 20, 30],
        "clicks": [5, 0, 12],
    })
    rel = compute_relevance(df)
    assert rel == {10: 5.0, 20: 0.0, 30: 12.0}


def test_compute_relevance_custom_column():
    """Should support custom relevance column."""
    df = pd.DataFrame({
        "product_id": [10, 20],
        "ctr": [45.0, 12.5],
    })
    rel = compute_relevance(df, relevance_col="ctr")
    assert rel == {10: 45.0, 20: 12.5}


# ── Integration tests with real reranker ─────────────────────────────────────


@pytest.fixture
def sample_df():
    """Minimal search dataset for integration tests."""
    return pd.DataFrame({
        "search_term": ["dress"] * 5 + ["jeans"] * 4,
        "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "clicks": [100, 50, 10, 2, 0, 80, 30, 5, 0],
        "impressions": [500, 400, 300, 200, 100, 300, 250, 100, 50],
        "ctr": [20.0, 12.5, 3.3, 1.0, 0.0, 26.7, 12.0, 5.0, 0.0],
        "impression_pos_avg": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5],
        "impression_pos_median": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5],
        "click_pos_avg": [1.0, 2.0, 3.0, 4.0, np.nan, 1.5, 2.5, 3.5, np.nan],
        "click_pos_median": [1.0, 2.0, 3.0, 4.0, np.nan, 1.5, 2.5, 3.5, np.nan],
    })


def test_evaluate_strategy_returns_per_term(sample_df):
    """Should return one row per search term with NDCG score."""
    result = evaluate_strategy(sample_df, RerankStrategy.SMOOTHED_CTR, k=3)
    assert len(result) == 2  # dress, jeans
    assert "ndcg_at_k" in result.columns
    assert "search_term" in result.columns
    assert all(0.0 <= v <= 1.0 for v in result["ndcg_at_k"])


def test_evaluate_baseline_impression_pos(sample_df):
    """Baseline (impression_pos_avg ascending) should produce NDCG scores."""
    result = evaluate_baseline(sample_df, "impression_pos_avg", ascending=True, k=3)
    assert len(result) == 2
    assert all(0.0 <= v <= 1.0 for v in result["ndcg_at_k"])


def test_evaluate_baseline_raw_clicks(sample_df):
    """Baseline (clicks descending) should produce NDCG scores."""
    result = evaluate_baseline(sample_df, "clicks", ascending=False, k=3)
    assert len(result) == 2


def test_evaluate_against_baseline_returns_table(sample_df):
    """Should return comparison table with baseline + smoothed CTR."""
    result = evaluate_against_baseline(sample_df, k=3)
    assert len(result) == 2
    assert "strategy" in result.columns
    assert "mean_ndcg" in result.columns
    assert result["mean_ndcg"].is_monotonic_decreasing


def test_smoothed_ctr_beats_baseline(sample_df):
    """Smoothed CTR should outperform the impression position baseline."""
    result = evaluate_against_baseline(sample_df, k=3)
    smoothed = result.loc[result["strategy"] == "smoothed_ctr", "mean_ndcg"].iloc[0]
    baseline = result.loc[result["strategy"] == "baseline_impression_pos", "mean_ndcg"].iloc[0]
    assert smoothed >= baseline


def test_evaluate_strategy_with_custom_params(sample_df):
    """Custom params should override defaults."""
    result = evaluate_strategy(
        sample_df,
        RerankStrategy.SMOOTHED_CTR,
        k=3,
        params={"alpha": 2.0, "beta": 80.0},
    )
    assert len(result) == 2
    assert all(0.0 <= v <= 1.0 for v in result["ndcg_at_k"])
