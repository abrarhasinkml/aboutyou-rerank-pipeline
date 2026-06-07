"""
Tests for src/data/loader.py — schema validation, chunked loading,
error handling.

### Tradeoffs
- Design: Test with real parquet/csv fixtures vs. mocked IO.
- Gain: Tests validate the actual data pipeline end-to-end.
- Sacrifice: Tests depend on the data/ directory existing. For CI
  without data files, add a pytest marker to skip data-dependent tests.
"""
import pandas as pd
import pytest

from src.data.loader import (
    _validate_product_schema,
    _validate_search_schema,
    load_product_metadata,
    load_search_data,
)


class TestSearchSchemaValidation:
    """Tests for _validate_search_schema."""

    def test_valid_schema_passes(self, sample_search_df: pd.DataFrame) -> None:
        _validate_search_schema(sample_search_df)  # should not raise

    def test_missing_column_raises(self, sample_search_df: pd.DataFrame) -> None:
        df = sample_search_df.drop(columns=["clicks"])
        with pytest.raises(ValueError, match="Missing columns"):
            _validate_search_schema(df)

    def test_extra_columns_allowed(self, sample_search_df: pd.DataFrame) -> None:
        df = sample_search_df.copy()
        df["extra_col"] = 1
        _validate_search_schema(df)  # should not raise


class TestProductSchemaValidation:
    """Tests for _validate_product_schema."""

    def test_valid_schema_passes(self, sample_product_df: pd.DataFrame) -> None:
        _validate_product_schema(sample_product_df)

    def test_missing_column_raises(self, sample_product_df: pd.DataFrame) -> None:
        df = sample_product_df.drop(columns=["image_hash"])
        with pytest.raises(ValueError, match="Missing columns"):
            _validate_product_schema(df)


class TestLoadSearchData:
    """Tests for load_search_data — reads from actual parquet file."""

    def test_loads_with_correct_columns(self) -> None:
        df = load_search_data()
        expected = {"search_term", "product_id", "clicks", "impressions", "ctr",
                    "impression_pos_avg", "impression_pos_median",
                    "click_pos_avg", "click_pos_median"}
        assert set(df.columns) == expected

    def test_returns_dataframe(self) -> None:
        df = load_search_data()
        assert isinstance(df, pd.DataFrame)

    def test_non_empty(self) -> None:
        df = load_search_data()
        assert len(df) > 0

    def test_no_duplicate_term_product_pairs(self) -> None:
        df = load_search_data()
        dupes = df.duplicated(subset=["search_term", "product_id"]).sum()
        assert dupes == 0


class TestLoadProductMetadata:
    """Tests for load_product_metadata — reads from actual CSV."""

    def test_loads_with_correct_columns(self) -> None:
        df = load_product_metadata()
        expected = {"product_id", "product_name", "image_hash", "product_url"}
        assert set(df.columns) == expected

    def test_returns_dataframe(self) -> None:
        df = load_product_metadata()
        assert isinstance(df, pd.DataFrame)

    def test_non_empty(self) -> None:
        df = load_product_metadata()
        assert len(df) > 0
