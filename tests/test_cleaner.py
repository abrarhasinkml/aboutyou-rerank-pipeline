"""
Tests for src/data/cleaner.py — deduplication, null handling, outlier flagging,
global stats computation.

### Tradeoffs
- Design: Unit tests with synthetic fixtures vs. integration tests with real data.
- Gain: Synthetic fixtures are fast, deterministic, and test edge cases
  (dupes, nulls, CTR>100%) that may not exist in the real dataset.
- Sacrifice: Doesn't catch data-format issues. Covered by test_loader.py.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import (
    CleanerConfig,
    clean_search_data,
    compute_global_stats,
    deduplicate,
    flag_outliers,
    handle_nulls,
)


# ── Deduplication ────────────────────────────────────────────────────────────


class TestDeduplicate:
    """Tests for deduplicate()."""

    def test_no_duplicates_unchanged(self, sample_search_df: pd.DataFrame) -> None:
        result = deduplicate(sample_search_df)
        assert len(result) == len(sample_search_df)

    def test_duplicates_removed(self, sample_search_df_with_dupes: pd.DataFrame) -> None:
        result = deduplicate(sample_search_df_with_dupes)
        assert len(result) == len(sample_search_df_with_dupes) - 1

    def test_first_row_kept_by_default(self, sample_search_df_with_dupes: pd.DataFrame) -> None:
        result = deduplicate(sample_search_df_with_dupes)
        kleid_100 = result[(result["search_term"] == "kleid") & (result["product_id"] == 100)]
        assert len(kleid_100) == 1
        assert kleid_100.iloc[0]["clicks"] == 10  # original row, not the dupe with clicks=5

    def test_custom_config_keep_last(self, sample_search_df_with_dupes: pd.DataFrame) -> None:
        config = CleanerConfig(dedup_keep="last")
        result = deduplicate(sample_search_df_with_dupes, config)
        kleid_100 = result[(result["search_term"] == "kleid") & (result["product_id"] == 100)]
        assert kleid_100.iloc[0]["clicks"] == 5  # dupe row, kept as "last"

    def test_original_index_reset(self, sample_search_df: pd.DataFrame) -> None:
        result = deduplicate(sample_search_df)
        assert result.index.is_monotonic_increasing


# ── Null handling ────────────────────────────────────────────────────────────


class TestHandleNulls:
    """Tests for handle_nulls()."""

    def test_drops_null_search_term(self, sample_search_df_with_nulls: pd.DataFrame) -> None:
        result = handle_nulls(sample_search_df_with_nulls)
        assert result["search_term"].isna().sum() == 0

    def test_drops_null_product_id(self, sample_search_df_with_nulls: pd.DataFrame) -> None:
        result = handle_nulls(sample_search_df_with_nulls)
        assert result["product_id"].isna().sum() == 0

    def test_fills_impression_pos_with_median(self, sample_search_df_with_nulls: pd.DataFrame) -> None:
        result = handle_nulls(sample_search_df_with_nulls)
        nullpos_row = result[result["search_term"] == "nullpos"]
        assert len(nullpos_row) == 1
        # impression_pos_avg was null, should now be filled with median
        assert not nullpos_row.iloc[0]["impression_pos_avg"] != nullpos_row.iloc[0]["impression_pos_avg"]  # not NaN

    def test_click_pos_nulls_preserved(self, sample_search_df_with_nulls: pd.DataFrame) -> None:
        result = handle_nulls(sample_search_df_with_nulls)
        # Zero-click rows should still have null click positions
        zero_click = result[result["clicks"] == 0]
        if len(zero_click) > 0:
            assert zero_click["click_pos_avg"].isna().any()

    def test_no_nulls_no_change(self, sample_search_df: pd.DataFrame) -> None:
        result = handle_nulls(sample_search_df)
        assert len(result) == len(sample_search_df)

    def test_drop_impression_pos_nulls(self, sample_search_df_with_nulls: pd.DataFrame) -> None:
        config = CleanerConfig(impression_pos_fillna_strategy="drop")
        result = handle_nulls(sample_search_df_with_nulls, config)
        assert result["impression_pos_avg"].isna().sum() == 0


# ── Outlier flagging ─────────────────────────────────────────────────────────


class TestFlagOutliers:
    """Tests for flag_outliers()."""

    def test_flags_ctr_above_100(self, sample_search_df: pd.DataFrame) -> None:
        result = flag_outliers(sample_search_df)
        assert "flag_ctr_above_100" in result.columns
        assert result["flag_ctr_above_100"].sum() == 1  # sneakers row: CTR=200%

    def test_flags_low_impressions(self, sample_search_df: pd.DataFrame) -> None:
        result = flag_outliers(sample_search_df)
        assert "flag_low_impressions" in result.columns
        # jeans product_id=400: impressions=10, threshold=3 -> not flagged
        # But sneakers product_id=600: impressions=1 -> flagged
        low_imp = result[result["impressions"] < 3]
        assert (result.loc[low_imp.index, "flag_low_impressions"]).all()

    def test_no_flags_when_disabled(self, sample_search_df: pd.DataFrame) -> None:
        config = CleanerConfig(flag_ctr_above_100=False, flag_low_impressions=False)
        result = flag_outliers(sample_search_df, config)
        assert "flag_ctr_above_100" not in result.columns
        assert "flag_low_impressions" not in result.columns

    def test_data_unchanged(self, sample_search_df: pd.DataFrame) -> None:
        original = sample_search_df.copy()
        result = flag_outliers(sample_search_df)
        # Original columns should be unchanged
        for col in original.columns:
            pd.testing.assert_series_equal(result[col], original[col], check_names=False)


# ── Global stats ─────────────────────────────────────────────────────────────


class TestComputeGlobalStats:
    """Tests for compute_global_stats()."""

    def test_returns_expected_keys(self, sample_search_df: pd.DataFrame) -> None:
        stats = compute_global_stats(sample_search_df)
        expected_keys = {
            "global_mean_ctr", "global_mean_clicks", "global_mean_impressions",
            "n_rows", "n_unique_terms", "n_unique_products",
            "zero_click_pct", "smoothing_beta",
        }
        assert expected_keys == set(stats.keys())

    def test_row_count(self, sample_search_df: pd.DataFrame) -> None:
        stats = compute_global_stats(sample_search_df)
        assert stats["n_rows"] == len(sample_search_df)

    def test_unique_counts(self, sample_search_df: pd.DataFrame) -> None:
        stats = compute_global_stats(sample_search_df)
        assert stats["n_unique_terms"] == sample_search_df["search_term"].nunique()
        assert stats["n_unique_products"] == sample_search_df["product_id"].nunique()

    def test_smoothing_beta_positive(self, sample_search_df: pd.DataFrame) -> None:
        stats = compute_global_stats(sample_search_df)
        assert stats["smoothing_beta"] > 0

    def test_zero_click_pct(self, sample_search_df: pd.DataFrame) -> None:
        stats = compute_global_stats(sample_search_df)
        expected = (sample_search_df["clicks"] == 0).mean() * 100
        assert abs(stats["zero_click_pct"] - expected) < 0.1

    def test_global_mean_clicks(self, sample_search_df: pd.DataFrame) -> None:
        stats = compute_global_stats(sample_search_df)
        assert abs(stats["global_mean_clicks"] - sample_search_df["clicks"].mean()) < 0.01


# ── Full pipeline ────────────────────────────────────────────────────────────


class TestCleanSearchData:
    """Tests for the full clean_search_data pipeline."""

    def test_pipeline_returns_dataframe(self, sample_search_df: pd.DataFrame) -> None:
        result = clean_search_data(sample_search_df)
        assert isinstance(result, pd.DataFrame)

    def test_pipeline_with_dupes_and_nulls(
        self, sample_search_df_with_nulls: pd.DataFrame
    ) -> None:
        result = clean_search_data(sample_search_df_with_nulls)
        # Should have no null identifiers
        assert result["search_term"].isna().sum() == 0
        assert result["product_id"].isna().sum() == 0
        # Should have flag columns
        assert "flag_ctr_above_100" in result.columns
        assert "flag_low_impressions" in result.columns

    def test_pipeline_preserves_data_integrity(self, sample_search_df: pd.DataFrame) -> None:
        original = sample_search_df.copy()
        result = clean_search_data(sample_search_df)
        # Row count should be same (no dupes, no nulls to drop)
        assert len(result) == len(original)
