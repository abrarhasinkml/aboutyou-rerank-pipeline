"""
FastAPI serving layer — exposes the reranker via HTTP.

### Tradeoffs (module-level)
- Design: Single FastAPI module vs. a multi-file router setup.
- Gain: One file for a single endpoint. FastAPI gives us automatic OpenAPI docs,
  request validation, and async support with ~50 lines of actual code.
- Sacrifice: At scale (auth, rate limiting, middleware), split into a `router/` package.

### Scale notes
- Startup: loads full dataset into memory (~19K rows, ~2MB). For 100GB+ data,
  replace with Redis/cache-backed lookup or a model server (TorchServe, TF Serving).
- Request: per-query reranking is O(n) where n = candidates for that term (~60-100).
  Sub-millisecond per request. For 1M+ queries/sec, add a cache layer (Redis)
  with TTL-based invalidation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.data.cleaner import clean_product_metadata, clean_search_data, normalize_text
from src.data.loader import load_product_metadata, load_search_data
from src.models.reranker import DEFAULT_PARAMS, rerank
from src.models.strategies import RerankStrategy
from src.utils import get_logger

logger = get_logger(__name__)

# ── Global state (loaded once at startup) ────────────────────────────────────

_search_df: pd.DataFrame | None = None
_product_df: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data once at startup, clean it, and hold in memory."""
    global _search_df, _product_df

    logger.info("Loading datasets at startup...")
    raw_search = load_search_data()
    _search_df = clean_search_data(raw_search)
    _product_df = load_product_metadata()
    _product_df = clean_product_metadata(_product_df)
    logger.info(
        "Startup complete: %d search rows, %d products",
        len(_search_df),
        len(_product_df),
    )
    yield

    logger.info("Shutting down — releasing data from memory")
    _search_df = None
    _product_df = None


app = FastAPI(
    title="ABOUT YOU Search Reranker",
    description="Rerank products for a given search term using click-based strategies.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Input sanitisation ───────────────────────────────────────────────────────


def sanitise_query(q: str) -> str:
    """
    Normalise user query input before passing to the reranker.

    Steps: lowercase → strip whitespace → collapse multiple spaces.

    This lives in the API layer (not reranker.py) because the reranker is a pure
    function — it shouldn't modify its input. Sanitisation is a presentation/API concern.

    Args:
        q: Raw user input string.

    Returns:
        Cleaned query string.

    Raises:
        HTTPException 422 if the query is empty after sanitisation.
    """
    cleaned = normalize_text(q)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Search query cannot be empty")
    return cleaned


# ── Response models ──────────────────────────────────────────────────────────


class RerankResultItem(BaseModel):
    product_id: int
    score: float
    clicks: int
    impressions: int
    impression_pos_avg: float
    ctr: float
    rank: int
    product_name: str | None = None
    image_url: str | None = None
    product_url: str | None = None


class RerankResponse(BaseModel):
    query: str
    strategy: str
    n_results: int
    results: list[RerankResultItem]


# ── CDN URL construction ─────────────────────────────────────────────────────

CDN_BASE = "https://cdn.aboutstatic.com/file"


def _build_image_url(image_hash: str | None) -> str | None:
    """Construct CDN image URL from image hash."""
    if not image_hash or pd.isna(image_hash):
        return None
    return f"{CDN_BASE}/{image_hash}?quality=75&height=640&width=480"


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/rerank", response_model=RerankResponse)
def rerank_endpoint(
    q: str = Query(..., description="Search term to rerank", min_length=1),
    k: int = Query(20, description="Top N results to return", ge=1, le=500),
    strategy: RerankStrategy = Query(
        RerankStrategy.SMOOTHED_CTR,
        description="Scoring strategy to use",
    ),
) -> RerankResponse:
    """
    Rerank products for a search term.

    Returns top-k products ordered by score (highest first), with product
    metadata and CDN image URLs for rendering.
    """
    if _search_df is None or _product_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded — server still starting")

    cleaned_query = sanitise_query(q)

    # Check term exists
    if cleaned_query not in _search_df["search_term"].values:
        available = _search_df["search_term"].unique().tolist()
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"No candidates found for query: {cleaned_query!r}",
                "hint": "Check available terms via GET /terms",
            },
        )

    # Rerank
    results = rerank(cleaned_query, _search_df, strategy=strategy, top_k=k)

    # Enrich with product metadata
    product_lookup = _product_df.set_index("product_id")
    enriched: list[RerankResultItem] = []
    for r in results:
        pid = r["product_id"]
        if pid in product_lookup.index:
            row = product_lookup.loc[pid]
            name = row.get("product_name")
            image_hash = row.get("image_hash")
            prod_url = row.get("product_url")
        else:
            name = None
            image_hash = None
            prod_url = None

        enriched.append(RerankResultItem(
            product_id=pid,
            score=r["score"],
            clicks=r["clicks"],
            impressions=r["impressions"],
            impression_pos_avg=r["impression_pos_avg"],
            ctr=r["ctr"],
            rank=r["rank"],
            product_name=name if pd.notna(name) else None,
            image_url=_build_image_url(image_hash),
            product_url=prod_url if pd.notna(prod_url) else None,
        ))

    return RerankResponse(
        query=cleaned_query,
        strategy=strategy.value,
        n_results=len(enriched),
        results=enriched,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    status = "ok" if _search_df is not None else "loading"
    return {"status": status}
