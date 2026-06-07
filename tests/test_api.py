"""Tests for src/api.py — FastAPI endpoint, input sanitisation, error handling."""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app, sanitise_query


@pytest.fixture
def client():
    """Test client with mocked data."""
    mock_search = pd.DataFrame({
        "search_term": ["dress"] * 5 + ["jeans"] * 4,
        "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "clicks": [100, 50, 10, 2, 0, 80, 30, 5, 0],
        "impressions": [500, 400, 300, 200, 100, 300, 250, 100, 50],
        "ctr": [20.0, 12.5, 3.3, 1.0, 0.0, 26.7, 12.0, 5.0, 0.0],
        "impression_pos_avg": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5],
        "impression_pos_median": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5],
        "click_pos_avg": [1.0, 2.0, 3.0, 4.0, None, 1.5, 2.5, 3.5, None],
        "click_pos_median": [1.0, 2.0, 3.0, 4.0, None, 1.5, 2.5, 3.5, None],
    })
    mock_products = pd.DataFrame({
        "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "product_name": [f"Product {i}" for i in range(1, 10)],
        "image_hash": [f"hash_{i}" for i in range(1, 10)],
        "product_url": [f"https://aboutyou.com/p/{i}" for i in range(1, 10)],
    })

    import src.api as api_module
    api_module._search_df = mock_search
    api_module._product_df = mock_products

    client = TestClient(app)
    yield client

    api_module._search_df = None
    api_module._product_df = None


# ── Input sanitisation ───────────────────────────────────────────────────────


def test_sanitise_query_lowercases():
    assert sanitise_query("DRESS") == "dress"


def test_sanitise_query_strips_whitespace():
    assert sanitise_query("  dress  ") == "dress"


def test_sanitise_query_collapses_spaces():
    assert sanitise_query("blue   dress") == "blue dress"


def test_sanitise_query_empty_raises():
    with pytest.raises(Exception):
        sanitise_query("")


def test_sanitise_query_whitespace_only_raises():
    with pytest.raises(Exception):
        sanitise_query("   ")


# ── /rerank endpoint ─────────────────────────────────────────────────────────


def test_rerank_basic(client):
    response = client.get("/rerank?q=dress")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "dress"
    assert data["strategy"] == "smoothed_ctr"
    assert data["n_results"] == 5
    assert len(data["results"]) == 5


def test_rerank_with_k(client):
    response = client.get("/rerank?q=dress&k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["n_results"] == 3
    assert len(data["results"]) == 3


def test_rerank_with_strategy(client):
    response = client.get("/rerank?q=dress&strategy=naive_clicks")
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "naive_clicks"


def test_rerank_unknown_term_404(client):
    response = client.get("/rerank?q=nonexistent")
    assert response.status_code == 404


def test_rerank_sanitises_input(client):
    response = client.get("/rerank?q=DRESS")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "dress"


def test_rerank_result_has_product_metadata(client):
    response = client.get("/rerank?q=dress&k=1")
    data = response.json()
    result = data["results"][0]
    assert "product_name" in result
    assert "image_url" in result
    assert "product_url" in result


def test_rerank_result_has_score_and_rank(client):
    response = client.get("/rerank?q=dress&k=1")
    data = response.json()
    result = data["results"][0]
    assert "score" in result
    assert "rank" in result
    assert result["rank"] == 1


def test_rerank_sorted_by_score_descending(client):
    response = client.get("/rerank?q=dress")
    data = response.json()
    scores = [r["score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_rerank_invalid_strategy_422(client):
    response = client.get("/rerank?q=dress&strategy=invalid")
    assert response.status_code == 422


def test_rerank_k_too_large(client):
    response = client.get("/rerank?q=dress&k=999")
    assert response.status_code == 422


# ── /health endpoint ─────────────────────────────────────────────────────────


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
