"""
Shared fixtures for data loader and cleaner tests.
"""
import pandas as pd
import pytest


@pytest.fixture
def sample_search_df() -> pd.DataFrame:
    """Minimal search dataset matching the parquet schema."""
    return pd.DataFrame({
        "search_term": ["kleid", "kleid", "kleid", "jeans", "jeans", "sneakers"],
        "product_id": [100, 200, 300, 400, 500, 600],
        "clicks": [10, 0, 5, 0, 20, 2],
        "impressions": [100, 50, 80, 10, 200, 1],
        "ctr": [10.0, 0.0, 6.25, 0.0, 10.0, 200.0],
        "impression_pos_avg": [1.0, 5.0, 20.0, 10.0, 3.0, 50.0],
        "impression_pos_median": [1.0, 5.0, 18.0, 10.0, 3.0, 50.0],
        "click_pos_avg": [5.0, None, 22.0, None, 8.0, 50.0],
        "click_pos_median": [3.0, None, 20.0, None, 6.0, 50.0],
    })


@pytest.fixture
def sample_search_df_with_dupes(sample_search_df: pd.DataFrame) -> pd.DataFrame:
    """Search dataset with duplicate (search_term, product_id) rows."""
    dupe_row = pd.DataFrame([{
        "search_term": "kleid", "product_id": 100,
        "clicks": 5, "impressions": 50, "ctr": 10.0,
        "impression_pos_avg": 2.0, "impression_pos_median": 2.0,
        "click_pos_avg": 3.0, "click_pos_median": 3.0,
    }])
    return pd.concat([sample_search_df, dupe_row], ignore_index=True)


@pytest.fixture
def sample_search_df_with_nulls(sample_search_df: pd.DataFrame) -> pd.DataFrame:
    """Search dataset with null identifiers and null impression positions."""
    null_rows = pd.DataFrame([
        {
            "search_term": None, "product_id": 999,
            "clicks": 0, "impressions": 10, "ctr": 0.0,
            "impression_pos_avg": 5.0, "impression_pos_median": 5.0,
            "click_pos_avg": None, "click_pos_median": None,
        },
        {
            "search_term": "test", "product_id": None,
            "clicks": 0, "impressions": 10, "ctr": 0.0,
            "impression_pos_avg": 5.0, "impression_pos_median": 5.0,
            "click_pos_avg": None, "click_pos_median": None,
        },
        {
            "search_term": "nullpos", "product_id": 777,
            "clicks": 3, "impressions": 30, "ctr": 10.0,
            "impression_pos_avg": None, "impression_pos_median": None,
            "click_pos_avg": 15.0, "click_pos_median": 10.0,
        },
    ])
    return pd.concat([sample_search_df, null_rows], ignore_index=True)


@pytest.fixture
def sample_product_df() -> pd.DataFrame:
    """Minimal product metadata matching the CSV schema."""
    return pd.DataFrame({
        "product_id": [100, 200, 300, 400],
        "product_name": ["Kleid Blau", "Jeans Slim", "Sneaker White", "Hose Creme"],
        "image_hash": ["abc123", "def456", "ghi789", "jkl012"],
        "product_url": [
            "https://www.aboutyou.de/p/x/x-100",
            "https://www.aboutyou.de/p/x/x-200",
            "https://www.aboutyou.de/p/x/x-300",
            "https://www.aboutyou.de/p/x/x-400",
        ],
    })
