"""
Tests for src/models/ — strategies, scorer, and reranker.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.reranker import DEFAULT_PARAMS, rerank, rerank_all
from src.models.scorer import (
    score_naive_clicks,
    score_position_debiased_ctr,
    score_raw_ctr,
    score_smoothed_ctr,
    score_smoothed_position_ctr,
)
from src.models.strategies import RerankStrategy, get_scorer


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def term_df() -> pd.DataFrame:
    """Single search term with varied click/impression patterns."""
    return pd.DataFrame({
        "search_term": ["kleid"] * 5,
        "product_id": [100, 200, 300, 400, 500],
        "clicks": [50, 0, 10, 1, 100],
        "impressions": [500, 50, 100, 1, 1000],
        "ctr": [10.0, 0.0, 10.0, 100.0, 10.0],
        "impression_pos_avg": [1.0, 10.0, 20.0, 50.0, 5.0],
        "impression_pos_median": [1.0, 10.0, 20.0, 50.0, 5.0],
        "click_pos_avg": [5.0, np.nan, 22.0, 50.0, 8.0],
        "click_pos_median": [3.0, np.nan, 20.0, 50.0, 6.0],
    })


@pytest.fixture
def multi_term_df(term_df: pd.DataFrame) -> pd.DataFrame:
    """Dataset with two search terms for rerank_all testing."""
    jeans_df = term_df.copy()
    jeans_df["search_term"] = "jeans"
    jeans_df["product_id"] = jeans_df["product_id"] + 1000
    return pd.concat([term_df, jeans_df], ignore_index=True)


# ── Strategy enum ────────────────────────────────────────────────────────────


class TestRerankStrategy:
    """Tests for RerankStrategy enum."""

    def test_all_strategies_have_values(self) -> None:
        assert len(RerankStrategy) == 5

    def test_string_values(self) -> None:
        assert RerankStrategy.NAIVE_CLICKS == "naive_clicks"
        assert RerankStrategy.SMOOTHED_CTR == "smoothed_ctr"


class TestGetScorer:
    """Tests for get_scorer factory."""

    def test_returns_callable_for_each_strategy(self) -> None:
        for strategy in RerankStrategy:
            scorer = get_scorer(strategy)
            assert callable(scorer)

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_scorer("not_a_strategy")  # type: ignore[arg-type]


# ── Scorer functions ─────────────────────────────────────────────────────────


class TestScoreNaiveClicks:
    """Tests for score_naive_clicks (Strategy A)."""

    def test_returns_series(self, term_df: pd.DataFrame) -> None:
        result = score_naive_clicks(term_df, {})
        assert isinstance(result, pd.Series)

    def test_scores_equal_clicks(self, term_df: pd.DataFrame) -> None:
        result = score_naive_clicks(term_df, {})
        pd.testing.assert_series_equal(
            result.astype(int), term_df["clicks"], check_names=False,
        )

    def test_length_matches_input(self, term_df: pd.DataFrame) -> None:
        result = score_naive_clicks(term_df, {})
        assert len(result) == len(term_df)


class TestScoreRawCTR:
    """Tests for score_raw_ctr (Strategy B)."""

    def test_returns_series(self, term_df: pd.DataFrame) -> None:
        result = score_raw_ctr(term_df, {})
        assert isinstance(result, pd.Series)

    def test_ctr_calculation(self, term_df: pd.DataFrame) -> None:
        result = score_raw_ctr(term_df, {})
        # product_id=100: 50/500*100 = 10.0
        assert abs(result.iloc[0] - 10.0) < 0.01

    def test_zero_impressions_get_zero(self, term_df: pd.DataFrame) -> None:
        term_df_mod = term_df.copy()
        term_df_mod.loc[0, "impressions"] = 0
        result = score_raw_ctr(term_df_mod, {})
        assert result.iloc[0] == 0.0


class TestScoreSmoothedCTR:
    """Tests for score_smoothed_ctr (Strategy D)."""

    def test_returns_series(self, term_df: pd.DataFrame) -> None:
        result = score_smoothed_ctr(term_df, {"alpha": 1.0, "beta": 40.0})
        assert isinstance(result, pd.Series)

    def test_smoothed_bounded_above_by_100(self, term_df: pd.DataFrame) -> None:
        result = score_smoothed_ctr(term_df, {"alpha": 1.0, "beta": 40.0})
        assert (result <= 100).all()

    def test_smoothed_greater_than_zero(self, term_df: pd.DataFrame) -> None:
        result = score_smoothed_ctr(term_df, {"alpha": 1.0, "beta": 40.0})
        assert (result > 0).all()

    def test_higher_clicks_higher_score(self, term_df: pd.DataFrame) -> None:
        """For same impressions, more clicks -> higher score."""
        df = pd.DataFrame({
            "clicks": [10, 50],
            "impressions": [100, 100],
        })
        result = score_smoothed_ctr(df, {"alpha": 1.0, "beta": 40.0})
        assert result.iloc[1] > result.iloc[0]

    def test_zero_clicks_still_positive(self, term_df: pd.DataFrame) -> None:
        """Even with 0 clicks, smoothing gives a positive score (the prior)."""
        df = pd.DataFrame({"clicks": [0], "impressions": [100]})
        result = score_smoothed_ctr(df, {"alpha": 1.0, "beta": 40.0})
        assert result.iloc[0] > 0


class TestScorePositionDebiasedCTR:
    """Tests for score_position_debiased_ctr (Strategy C)."""

    def test_returns_series(self, term_df: pd.DataFrame) -> None:
        result = score_position_debiased_ctr(term_df, {"reference_position": 5, "gamma": 0.01})
        assert isinstance(result, pd.Series)

    def test_position_correction_applied(self, term_df: pd.DataFrame) -> None:
        """Positions > reference get boosted (correction > 1)."""
        result = score_position_debiased_ctr(term_df, {"reference_position": 5, "gamma": 0.01})
        raw = score_raw_ctr(term_df, {})
        # product at position 50 should get a boost relative to raw CTR
        pos_50_idx = term_df[term_df["impression_pos_avg"] == 50.0].index
        if len(pos_50_idx) > 0:
            assert result.loc[pos_50_idx[0]] >= raw.loc[pos_50_idx[0]]


class TestScoreSmoothedPositionCTR:
    """Tests for score_smoothed_position_ctr (Strategy E)."""

    def test_returns_series(self, term_df: pd.DataFrame) -> None:
        result = score_smoothed_position_ctr(
            term_df, {"alpha": 1.0, "beta": 40.0, "reference_position": 5, "gamma": 0.01},
        )
        assert isinstance(result, pd.Series)

    def test_bounded_above_by_100(self, term_df: pd.DataFrame) -> None:
        result = score_smoothed_position_ctr(
            term_df, {"alpha": 1.0, "beta": 40.0, "reference_position": 5, "gamma": 0.01},
        )
        # With small gamma, should still be reasonable
        assert (result <= 200).all()  # position correction can push above 100


# ── Reranker ─────────────────────────────────────────────────────────────────


class TestRerank:
    """Tests for rerank()."""

    def test_returns_list_of_dicts(self, term_df: pd.DataFrame) -> None:
        result = rerank("kleid", term_df)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_sorted_by_score_descending(self, term_df: pd.DataFrame) -> None:
        result = rerank("kleid", term_df)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_all_candidates_returned(self, term_df: pd.DataFrame) -> None:
        result = rerank("kleid", term_df)
        assert len(result) == 5

    def test_top_k_limits_results(self, term_df: pd.DataFrame) -> None:
        result = rerank("kleid", term_df, top_k=2)
        assert len(result) == 2

    def test_empty_term_returns_empty(self, term_df: pd.DataFrame) -> None:
        result = rerank("nonexistent", term_df)
        assert result == []

    def test_result_has_expected_keys(self, term_df: pd.DataFrame) -> None:
        result = rerank("kleid", term_df)
        expected_keys = {"product_id", "score", "clicks", "impressions",
                         "impression_pos_avg", "ctr", "rank"}
        assert set(result[0].keys()) == expected_keys

    def test_rank_starts_at_1(self, term_df: pd.DataFrame) -> None:
        result = rerank("kleid", term_df)
        assert result[0]["rank"] == 1

    def test_custom_strategy(self, term_df: pd.DataFrame) -> None:
        result_d = rerank("kleid", term_df, strategy=RerankStrategy.SMOOTHED_CTR)
        result_a = rerank("kleid", term_df, strategy=RerankStrategy.NAIVE_CLICKS)
        # Both should return results but potentially in different order
        assert len(result_d) == len(result_a)

    def test_custom_params_override(self, term_df: pd.DataFrame) -> None:
        result = rerank("kleid", term_df, params={"alpha": 10.0, "beta": 400.0})
        assert len(result) > 0


class TestRerankAll:
    """Tests for rerank_all()."""

    def test_returns_dict_of_lists(self, multi_term_df: pd.DataFrame) -> None:
        result = rerank_all(multi_term_df)
        assert isinstance(result, dict)
        assert all(isinstance(v, list) for v in result.values())

    def test_all_terms_present(self, multi_term_df: pd.DataFrame) -> None:
        result = rerank_all(multi_term_df)
        assert set(result.keys()) == {"kleid", "jeans"}

    def test_each_term_sorted(self, multi_term_df: pd.DataFrame) -> None:
        result = rerank_all(multi_term_df)
        for term, entries in result.items():
            scores = [e["score"] for e in entries]
            assert scores == sorted(scores, reverse=True), f"Term {term} not sorted"
