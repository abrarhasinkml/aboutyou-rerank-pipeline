"""
Dataset profile — produces summary statistics that inform smoothing parameters,
confidence thresholds, and position bias correction design (D6).

### Tradeoffs (module-level)
- Design: Single-pass profiling returning a structured dict + markdown report,
  vs. interactive notebook exploration.
- Gain: Reproducible, version-controllable output. The markdown report serves
  as design documentation — the panel can read it without running code.
- Sacrifice: Less interactive than a notebook. For ad-hoc questions, the
  notebook in Task 2 imports these functions and adds inline visuals.

### Scale notes (module-level)
- Prototype: Full read — dataset fits in memory (~10–50 MB).
- Production (100–1000× growth): Switch to `pyarrow.dataset` with column
  pruning and row-group filtering. For Spark, use `spark.read.parquet()` with
  predicate pushdown. Key insight: profiling can sample rather than scan all
  data — reservoir sampling or stratified sampling by search_term frequency
  gives statistically valid summaries without full scans.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
# All thresholds live here — nothing hardcoded inline.
# These are DEFAULTS; the markdown report may recommend tuned values.

PROFILE_CONFIG = {
    "data_dir": "data",
    "search_file": "search_term_products.parquet",
    "products_file": "products.csv",
    # Thresholds for flagging data characteristics
    # CTR > 100% is NOT an error — clicks from bookmarks, push notifications,
    # or email can legitimately exceed SRP impressions (see data_profile_report.md
    # for full analysis). We track it but do not cap.
    "ctr_exceeds_100_threshold": 100.0,   # flag CTR > 100% as signal, not noise
    "low_impression_threshold": 3,        # flag rows with < N impressions
    "output_report": "src/explore/data_profile_report.md",
}


def load_search_data(
    data_dir: str = "data",
    filename: str = "search_term_products.parquet",
) -> pd.DataFrame:
    """
    Read the search term–product parquet file.

    ### Tradeoffs
    - Design: Full read — simplest path for a prototype dataset.
    - Gain: Zero cognitive overhead; all columns available for profiling.
    - Sacrifice: If the dataset were 100× larger, this would OOM. The
      production equivalent (documented below) addresses this.

    ### Scale notes
    - Prototype: `pd.read_parquet()` — single-threaded, full memory load.
    - Production: Use `pyarrow.dataset.dataset(path).to_table(
        columns=["search_term", "product_id", "clicks", "impressions",
                 "impression_pos_avg", "impression_pos_median"]
      )` for column pruning (~40 % memory reduction). For Spark:
      `spark.read.parquet(path).select(cols).cache()`.
    - OOM risk: None at prototype scale. At 100 M+ rows: sample mode
      (`fraction=0.1`) or row-group filtering on date partitions.

    ### Failure modes (production hardening)
    - FileNotFound: Raise with path (prototype: let pandas fail).
    - Schema mismatch: Validate expected columns (prototype: let pandas fail).
    - Corrupt parquet: Catch ArrowException, retry once, alert.
    - Empty file: Return empty DataFrame with schema (prototype: pandas returns
      empty — acceptable for now).
    """
    path = Path(data_dir) / filename
    df = pd.read_parquet(path)

    # Validate schema at prototype level — fail fast with clear message.
    _validate_schema(df)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Raise if required columns are missing. Inline: no tradeoffs needed —
    this is a guard, not algorithmic."""
    required = {
        "search_term", "product_id", "clicks", "impressions",
        "ctr", "impression_pos_avg", "impression_pos_median",
        "click_pos_avg", "click_pos_median",
    }
    missing = required - set(df.columns)
    if missing:
        msg = f"Missing columns in parquet: {missing}"
        raise ValueError(msg)


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Produce a dict of summary statistics covering schema, sparsity,
    distributions, and position information.

    ### Tradeoffs
    - Design: Single function returning a single dict vs. separate functions
      per category.
    - Gain: One scan through `df.describe()` + `df.info()` equivalent; the
      dict is serialisable to markdown/JSON — easy to version and compare
      across data refreshes.

    ### Optimisation
    - O(k) where k = number of stats computed. All operations are pandas
      vectorised built-ins. No row iteration.
    - `df.describe(percentiles=[...])` computes all quantiles in one Cython
      pass — more efficient than separate `.quantile()` calls.

    ### Memory
    - Allocates: describe() result (~10 KB) + groupby results (term-level
      Series, size ≈ n_terms × 8 bytes) + output dict (~5 KB).
    - Production-scale: groupby aggregations are the bottleneck. At 10 M+
      terms, use approximate quantiles (t-digest) and HyperLogLog for
      cardinality estimation.
    """
    stats = {}

    # ── 1. Schema & scale ───────────────────────────────────────────────────
    n_rows = len(df)
    n_terms = df["search_term"].nunique()
    n_products = df["product_id"].nunique()
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    stats["n_rows"] = n_rows
    stats["n_unique_search_terms"] = n_terms
    stats["n_unique_products"] = n_products
    stats["memory_mb"] = round(memory_mb, 1)

    # ── 2. Candidates per search term ───────────────────────────────────────
    # One groupby().size() vectorised call — O(n_rows).
    term_counts = df.groupby("search_term").size()
    stats["candidates_per_term"] = {
        "min": int(term_counts.min()),
        "max": int(term_counts.max()),
        "median": float(term_counts.median()),
        "mean": round(float(term_counts.mean()), 1),
        "p5": int(term_counts.quantile(0.05)),
        "p95": int(term_counts.quantile(0.95)),
        "p99": int(term_counts.quantile(0.99)),
    }

    # ── 3. Click sparsity ───────────────────────────────────────────────────
    zero_clicks = (df["clicks"] == 0).sum()
    has_clicks = n_rows - zero_clicks
    stats["click_sparsity"] = {
        "zero_clicks": int(zero_clicks),
        "has_clicks": int(has_clicks),
        "zero_pct": round(float(zero_clicks / n_rows * 100), 1),
    }

    # Click distribution — all rows and clicked-only.
    stats["clicks_all"] = _describe_series(df["clicks"], pcts=[0.5, 0.9, 0.95, 0.99])
    clicked_mask = df["clicks"] > 0
    stats["clicks_clicked_only"] = _describe_series(
        df.loc[clicked_mask, "clicks"], pcts=[0.5, 0.9, 0.95, 0.99]
    )

    # ── 4. CTR distribution ─────────────────────────────────────────────────
    stats["ctr"] = _describe_series(df["ctr"], pcts=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    stats["ctr_zero_pct"] = round(float((df["ctr"] == 0).mean() * 100), 1)
    stats["ctr_exceeds_100"] = {
        "count": int((df["ctr"] > PROFILE_CONFIG["ctr_exceeds_100_threshold"]).sum()),
        "pct": round(
            float((df["ctr"] > PROFILE_CONFIG["ctr_exceeds_100_threshold"]).mean() * 100), 2
        ),
        # Breakdown of click/impression ratios for these rows
        "impression_dist": {
            int(imp): int(cnt)
            for imp, cnt in df.loc[
                df["ctr"] > PROFILE_CONFIG["ctr_exceeds_100_threshold"],
                "impressions",
            ].value_counts().sort_index().items()
        },
    }
    stats["low_impression_rows"] = int(
        (df["impressions"] < PROFILE_CONFIG["low_impression_threshold"]).sum()
    )

    # ── 5. Position columns ─────────────────────────────────────────────────
    for col in [
        "impression_pos_avg", "impression_pos_median",
        "click_pos_avg", "click_pos_median",
    ]:
        series = df[col].dropna()
        stats[f"{col}_count"] = len(series)
        stats[f"{col}_null"] = int(df[col].isna().sum())
        stats[f"{col}_null_pct"] = round(float(df[col].isna().mean() * 100), 1)
        if len(series) > 0:
            stats[f"{col}"] = {
                "min": round(float(series.min()), 1),
                "max": round(float(series.max()), 1),
                "median": round(float(series.median()), 1),
                "mean": round(float(series.mean()), 1),
            }

    # ── 6. Position–click correlation (position bias evidence) ──────────────
    valid_mask = df["click_pos_avg"].notna() & df["impression_pos_avg"].notna()
    if valid_mask.sum() > 1:
        stats["position_corr"] = round(
            float(df.loc[valid_mask, "impression_pos_avg"].corr(
                df.loc[valid_mask, "click_pos_avg"]
            )),
            3,
        )
    else:
        stats["position_corr"] = None

    # ── 7. Cold-start terms (zero total clicks) ─────────────────────────────
    term_clicks = df.groupby("search_term")["clicks"].sum()
    cold_start = (term_clicks == 0).sum()
    stats["cold_start"] = {
        "terms_with_zero_clicks": int(cold_start),
        "pct_of_all_terms": round(float(cold_start / n_terms * 100), 1),
    }
    total_term_clicks = term_clicks.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    stats["clicks_per_term"] = {
        "min": int(total_term_clicks["min"]),
        "max": int(total_term_clicks["max"]),
        "median": float(total_term_clicks["50%"]),
        "mean": round(float(total_term_clicks["mean"]), 1),
        "p95": float(total_term_clicks["95%"]),
        "p99": float(total_term_clicks["99%"]),
    }
    del term_clicks, total_term_clicks

    # ── 8. Duplicate check — (search_term, product_id) ──────────────────────
    dupes = df.duplicated(subset=["search_term", "product_id"]).sum()
    stats["duplicate_rows"] = int(dupes)

    return stats


def _describe_series(series: pd.Series, pcts: list[float]) -> dict:
    """Compute count/mean/std/min + specified percentiles + max in one pass."""
    desc = series.describe(percentiles=pcts)
    result = {
        "count": int(desc["count"]),
        "mean": round(float(desc["mean"]), 2),
        "std": round(float(desc["std"]), 2),
        "min": round(float(desc["min"]), 2),
        "max": round(float(desc["max"]), 2),
    }
    for p in pcts:
        key = f"{p:.0%}" if p >= 0.01 else str(p)
        # pandas describe uses "50%" etc. keys
        pct_key = f"{int(p * 100)}%"
        result[f"p{pct_key}"] = round(float(desc.get(pct_key, desc.get(f"{p:.2%}", np.nan))), 2)
    return result


def write_markdown_report(stats: dict, output_path: str) -> None:
    """
    Render the profile dict into a markdown report.

    ### Tradeoffs
    - Design: Manual string formatting vs. Jinja template.
    - Gain: Zero dependencies beyond stdlib. The report is append-only
      (each explore module writes its own section).
    - Sacrifice: Verbose formatting code. For >3 modules, switch to
      a template engine.
    """
    lines = []
    w = lines.append  # shorthand

    w("# Data Profile Report")
    w("")
    w(f"**Rows:** {stats['n_rows']:,}  |  "
      f"**Search terms:** {stats['n_unique_search_terms']:,}  |  "
      f"**Products:** {stats['n_unique_products']:,}  |  "
      f"**Memory:** {stats['memory_mb']} MB")
    w("")

    # ── Candidates per term ──────────────────────────────────────────────────
    cpt = stats["candidates_per_term"]
    w("## Candidates per Search Term")
    w("")
    w("| Statistic | Value |")
    w("|-----------|-------|")
    for k in ["min", "max", "median", "mean", "p5", "p95", "p99"]:
        w(f"| {k} | {cpt[k]:,} |" if isinstance(cpt[k], int)
          else f"| {k} | {cpt[k]} |")
    w("")

    # ── Click sparsity ───────────────────────────────────────────────────────
    sp = stats["click_sparsity"]
    w("## Click Sparsity")
    w("")
    w(f"- **Zero clicks:** {sp['zero_clicks']:,} rows ({sp['zero_pct']}%)")
    w(f"- **Has clicks:** {sp['has_clicks']:,} rows "
      f"({round(100 - sp['zero_pct'], 1)}%)")
    w("")

    w("### Click Distribution (all rows)")
    w("")
    _write_desc_table(w, stats["clicks_all"])

    w("### Click Distribution (clicked rows only)")
    w("")
    _write_desc_table(w, stats["clicks_clicked_only"])
    w("")

    # ── CTR ──────────────────────────────────────────────────────────────────
    w("## CTR Distribution")
    w("")
    w(f"- **CTR = 0%:** {stats['ctr_zero_pct']}% of rows")
    ctr_ex = stats["ctr_exceeds_100"]
    w(f"- **CTR > 100%:** {ctr_ex['count']:,} rows ({ctr_ex['pct']}%)")
    w(f"  - **Interpretation:** Clicks CAN exceed impressions — this is NOT a data error.")
    w(f"    Customers click via bookmarks, push notifications, or email links which")
    w(f"    generate click events without a corresponding SRP impression.")
    w(f"  - **Implication for scoring:** Raw CTR (`clicks/impressions × 100`) is an")
    w(f"    unreliable metric. We decouple clicks and impressions in the scoring formula")
    w(f"    — clicks measure engagement, impressions provide confidence.")
    w(f"  - **Impression distribution for these rows:**")
    for imp, cnt in sorted(ctr_ex["impression_dist"].items()):
        w(f"    - {imp} impression(s): {cnt} rows")
    w(f"- **Low impressions (< {PROFILE_CONFIG['low_impression_threshold']}):** "
      f"{stats['low_impression_rows']:,} rows "
      f"({round(stats['low_impression_rows']/stats['n_rows']*100, 1)}%)")
    w("")
    _write_desc_table(w, stats["ctr"])
    w("")

    # ── Position columns ─────────────────────────────────────────────────────
    w("## Position Columns")
    w("")
    for col in [
        "impression_pos_avg", "impression_pos_median",
        "click_pos_avg", "click_pos_median",
    ]:
        w(f"### {col}")
        w(f"- **Non-null:** {stats[f'{col}_count']:,}  |  "
          f"**Null:** {stats[f'{col}_null']:,} "
          f"({stats[f'{col}_null_pct']}%)")
        inner = stats.get(col)
        if inner:
            w("")
            w("| Statistic | Value |")
            w("|-----------|-------|")
            for k, v in inner.items():
                w(f"| {k} | {v} |")
        w("")

    # ── Position bias evidence ───────────────────────────────────────────────
    w("## Position Bias Evidence")
    w("")
    corr = stats["position_corr"]
    if corr is not None:
        w(f"- **Correlation (impression_pos_avg ↔ click_pos_avg):** r = {corr}")
        if abs(corr) < 0.1:
            w("- **Interpretation:** Weak correlation — position bias may not be linear.")
        elif corr > 0.1:
            w("- **Interpretation:** Positive correlation — "
              "products impressed at higher positions also clicked at higher positions.")
        else:
            w("- **Interpretation:** Negative correlation — "
              "position bias may be present (lower-numbered = better positions).")
    else:
        w("- Insufficient data to compute correlation.")
    w("")

    # ── Cold-start terms ─────────────────────────────────────────────────────
    cs = stats["cold_start"]
    w("## Cold-Start Terms (Zero Total Clicks)")
    w("")
    w(f"- **Terms with zero clicks:** {cs['terms_with_zero_clicks']:,} "
      f"({cs['pct_of_all_terms']}% of all terms)")
    w("- **Implication:** These terms have no behavioural signal — must fall "
      "back to baseline ordering (`impression_pos_avg` ascending).")
    w("")

    w("### Clicks per Term Distribution")
    w("")
    cpt_dist = stats["clicks_per_term"]
    w("| Statistic | Value |")
    w("|-----------|-------|")
    for k in ["min", "max", "median", "mean", "p95", "p99"]:
        w(f"| {k} | {cpt_dist[k]:.1f} |")
    w("")

    # ── Duplicates ───────────────────────────────────────────────────────────
    w("## Duplicate Check")
    w("")
    if stats["duplicate_rows"] > 0:
        w(f"- **Duplicate (search_term, product_id) rows:** "
          f"{stats['duplicate_rows']:,} — cleaner must deduplicate.")
    else:
        w("- No duplicate (search_term, product_id) pairs found.")
    w("")

    # ── Write ────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def _write_desc_table(w, desc: dict) -> None:
    """Helper: render a describe-style dict as a markdown table."""
    w("| Statistic | Value |")
    w("|-----------|-------|")
    for k, v in desc.items():
        w(f"| {k} | {v} |")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: uv run python -m src.explore.data_profile
    Or:  uv run python src/explore/data_profile.py

    ### Observability (production add-ons)
    - Timer: `with metrics.timer("profile.duration"):`
    - Log: `logger.info("profile", rows=n_rows, terms=n_terms)`
    - Alert: if zero_clicks_pct > 95% → warn about extreme sparsity.
    """
    df = load_search_data(
        data_dir=PROFILE_CONFIG["data_dir"],
        filename=PROFILE_CONFIG["search_file"],
    )
    stats = profile_dataset(df)
    write_markdown_report(stats, PROFILE_CONFIG["output_report"])

    # Quick terminal summary
    print(f"[OK] Profiled {stats['n_rows']:,} rows "
          f"across {stats['n_unique_search_terms']:,} search terms")
    print(f"  Sparsity: {stats['click_sparsity']['zero_pct']}% zero-click rows")
    print(f"  Cold-start: {stats['cold_start']['terms_with_zero_clicks']:,} "
          f"terms ({stats['cold_start']['pct_of_all_terms']}%)")
    print(f"  Report: {PROFILE_CONFIG['output_report']}")
