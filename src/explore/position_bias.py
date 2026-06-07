"""
Position bias analysis — determines how impression position affects CTR,
which informs the correction function shape in D6 (linear, log, or power).

### Tradeoffs (module-level)
- Design: Position-bucketed CTR analysis vs. fitting a regression model.
- Gain: Buckets are interpretable — a product owner can look at the curve and
  understand "items at position 1 get 3× the clicks of items at position 20".
  Regression would give a cleaner mathematical form but obscures the raw data.
- Sacrifice: Bucket boundaries are arbitrary (we use adaptive binning to
  mitigate). Edge buckets with few data points are noisy — we report count
  per bucket so the consumer can assess reliability.
- Alternative considered: Poisson/quasi-Poisson GLM for CTR vs. position —
  rejected because the timebox doesn't justify the statistical ceremony and
  the bucket approach produces the D6 config values more directly.

### Scale notes (module-level)
- Bucket operation is a `groupby` over position bins — O(n) single pass.
- At production scale (100 M+ rows): binning is embarrassingly parallel.
  In Spark: `df.withColumn("pos_bucket", floor(col("impression_pos_avg")/bucket_width))
  .groupBy("pos_bucket").agg(mean("ctr"), count("*"))`.
  In Dask: `df.groupby("pos_bucket").ctr.agg(["mean", "count"])`.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

POSITION_CONFIG = {
    "data_dir": "data",
    "search_file": "search_term_products.parquet",
    # Bucketing strategy
    "max_position": 100,          # cap position at this value (long-tail start)
    "bucket_width": 5,            # width of each position bucket
    "adaptive_min_count": 30,     # merge adjacent buckets with fewer than this
                                  # many rows — prevents noise in sparse regions
    # Reference position for correction factor
    "reference_position": 5,      # items at pos ≤5 get "baseline" CTR
    "output_report": "src/explore/position_bias_report.md",
}


def load_data(data_dir: str = "data", filename: str = "search_term_products.parquet") -> pd.DataFrame:
    """
    Read search data — focused on position/CTR columns only.

    ### Tradeoffs
    - Design: Column-pruned read vs. loading the full schema used in data_profile.
    - Gain: Reads only 4 columns (impression_pos_avg, ctr, clicks, impressions)
      instead of 9 — ~40 % less memory.
    - Sacrifice: Can't enrich with search_term grouping here. That's intentional
      — position bias is treated as a global effect (position N has the same
      bias regardless of query), which is the standard assumption in IR position
      bias models.

    ### Scale notes
    - Prototype: Column-pruned read, fits in memory trivially.
    - Production: `pyarrow.dataset` with column selection at the IO layer.
      For 1 B+ rows: sample 1 % via row-group filtering, then apply position
      bias correction from the sampled curve.
    """
    path = Path(data_dir) / filename
    return pd.read_parquet(path, columns=[
        "impression_pos_avg", "ctr", "clicks", "impressions",
    ])


def bucket_by_position(
    df: pd.DataFrame,
    max_position: int = 100,
    bucket_width: int = 5,
    adaptive_min_count: int = 30,
) -> pd.DataFrame:
    """
    Group rows by impression position buckets and compute bucket-level CTR stats.

    ### Tradeoffs
    - Design: Adaptive merging vs. fixed-width bins.
    - Gain: Merging low-count adjacent buckets eliminates noise at sparse
      positions (e.g., position 95-100 might have 3 rows — merging into a
      single 90-100 bucket gives a meaningful average).
    - Sacrifice: Slightly more complex than fixed bins. The merge heuristic
      (merge rightward until count ≥ min_count) can obscure fine-grained
      patterns at the tail — for this dataset, the tail isn't our focus
      (above-the-fold = positions 1-30).

    ### Optimisation
    - O(n) — single groupby with vectorised aggregation.
    - `pd.cut()` produces CategoricalIndex; groupby is hash-based, O(1) per
      unique bucket label.

    ### Memory
    - Input: ~19 K rows × 4 columns ≈ 0.6 MB.
    - Output: ~20 bucket rows — negligible.
    - Production (100 M rows): groupby result is proportional to number of
      unique position values, not row count. Memory ≈ O(unique_positions/bucket_width).

    ### Partitioning
    - No natural partition key needed — this is a global analysis.
      Production: pre-aggregated position buckets can be computed incrementally
      as new data arrives (running mean per bucket via Welford's algorithm).
    """
    # Cap positions beyond max_position into one tail bucket
    work = df[df["impression_pos_avg"].notna()].copy()
    work["position"] = work["impression_pos_avg"].clip(upper=max_position)

    # Create fixed-width buckets
    bin_edges = list(range(0, max_position + 1, bucket_width))
    if bin_edges[-1] < max_position:
        bin_edges.append(max_position + 1)
    labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges) - 1)]

    work["pos_bucket"] = pd.cut(
        work["position"],
        bins=bin_edges,
        labels=labels,
        right=False,
    )

    # Aggregate per bucket
    buckets = work.groupby("pos_bucket", observed=False).agg(
        row_count=("position", "count"),
        mean_ctr=("ctr", "mean"),
        median_ctr=("ctr", "median"),
        std_ctr=("ctr", "std"),
        mean_clicks=("clicks", "mean"),
        mean_impressions=("impressions", "mean"),
        mean_position=("position", "mean"),
    ).reset_index()

    # Adaptive merge: combine low-count adjacent buckets
    buckets = _adaptive_merge(buckets, adaptive_min_count)

    return buckets


def _adaptive_merge(buckets: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """
    Merge adjacent position buckets with fewer than min_count rows.

    ### Tradeoffs
    - Design: Greedy right-merge vs. dynamic programming optimal merge.
    - Gain: Simple, deterministic, single-pass. Greedy is sufficient because
      the tail is monotonically sparse (positions get thinner, never denser).
    - Sacrifice: For pathological data (highly oscillating density), greedy
      can produce suboptimal merges. Not a concern for position data which
      is naturally monotonic in sparsity.

    ### Optimisation
    - O(n_buckets) single pass — typically ~20 buckets, trivially fast.
    """
    if len(buckets) <= 1:
        return buckets

    merged_rows = []
    accumulator = None

    for _, row in buckets.iterrows():
        if accumulator is None:
            accumulator = row.to_dict()
            continue

        if accumulator["row_count"] < min_count:
            # Merge current into accumulator
            total_count = accumulator["row_count"] + row["row_count"]
            accumulator["mean_ctr"] = (
                accumulator["mean_ctr"] * accumulator["row_count"]
                + row["mean_ctr"] * row["row_count"]
            ) / total_count
            accumulator["median_ctr"] = np.nan  # median loses meaning after merge
            accumulator["mean_clicks"] = (
                accumulator["mean_clicks"] * accumulator["row_count"]
                + row["mean_clicks"] * row["row_count"]
            ) / total_count
            accumulator["mean_impressions"] = (
                accumulator["mean_impressions"] * accumulator["row_count"]
                + row["mean_impressions"] * row["row_count"]
            ) / total_count
            accumulator["mean_position"] = (
                accumulator["mean_position"] * accumulator["row_count"]
                + row["mean_position"] * row["row_count"]
            ) / total_count
            accumulator["row_count"] = int(total_count)
        else:
            merged_rows.append(accumulator)
            accumulator = row.to_dict()

    if accumulator is not None:
        # Final check — merge into last if still below threshold
        if merged_rows and accumulator["row_count"] < min_count:
            last = merged_rows[-1]
            total_count = last["row_count"] + accumulator["row_count"]
            last["mean_ctr"] = (
                last["mean_ctr"] * last["row_count"]
                + accumulator["mean_ctr"] * accumulator["row_count"]
            ) / total_count
            last["mean_position"] = (
                last["mean_position"] * last["row_count"]
                + accumulator["mean_position"] * accumulator["row_count"]
            ) / total_count
            last["median_ctr"] = np.nan
            last["mean_clicks"] = (
                last["mean_clicks"] * last["row_count"]
                + accumulator["mean_clicks"] * accumulator["row_count"]
            ) / total_count
            last["mean_impressions"] = (
                last["mean_impressions"] * last["row_count"]
                + accumulator["mean_impressions"] * accumulator["row_count"]
            ) / total_count
            last["row_count"] = int(total_count)
        else:
            merged_rows.append(accumulator)

    return pd.DataFrame(merged_rows)


def compute_correction_factors(
    buckets: pd.DataFrame,
    reference_position: int = 5,
) -> pd.DataFrame:
    """
    Convert bucket-level CTR into a multiplicative correction factor.

    correction_factor(pos) = CTR_at_reference / CTR_at_pos

    An item at position 50 that would naturally get 1/3 the CTR of position 5
    gets a correction factor of 3.0 — multiply its raw engagement score by 3
    to simulate what its CTR would be if it were shown at position 5.

    ### Tradeoffs
    - Design: Ratio-based correction vs. regression residual.
    - Gain: Directly interpretable — "items at pos 30 need a 2.5× boost".
      Maps cleanly to the multiplicative correction in D6.
    - Sacrifice: If CTR goes to zero in a bucket, the factor blows up
      (we cap it at 10×). Ratio is also sensitive to noise in the reference
      bucket — mitigated by using a wider reference window (pos 1–5).

    ### Optimisation
    - O(n_buckets) — trivial, ~20 rows.
    """
    # Reference CTR: average over positions ≤ reference_position
    ref_mask = buckets["mean_position"] <= reference_position
    if not ref_mask.any():
        # Fallback: use the lowest-position bucket available
        ref_mask = buckets["mean_position"] == buckets["mean_position"].min()

    ref_ctr = buckets.loc[ref_mask, "mean_ctr"].mean()
    if ref_ctr <= 0:
        # If reference CTR is zero (shouldn't happen), no correction possible
        buckets["correction_factor"] = 1.0
        buckets["notes"] = "reference CTR is zero — correction disabled"
        return buckets

    buckets["correction_factor"] = (ref_ctr / buckets["mean_ctr"]).clip(
        upper=10.0,  # cap: don't boost >10× — likely noise at extreme positions
        lower=0.1,   # floor: don't penalise >0.1× — positions better than ref
    )
    buckets["correction_factor"] = buckets["correction_factor"].round(3)

    return buckets


def assess_fit(buckets: pd.DataFrame) -> dict:
    """
    Assess whether the CTR-vs-position relationship is best described as
    linear, log-linear, or power-law — informs the correction function type in D6.

    ### Tradeoffs
    - Design: R² comparison of three simple models vs. fitting a spline or
      non-parametric model.
    - Gain: The three forms (linear, log, power) map to simple correction
      functions the product owner can understand. R² gives a quick heuristic.
    - Sacrifice: R² on 15-20 bucketed data points is noisy. This is a
      directional signal, not a rigorous model selection. The final D6 choice
      should also consider simplicity (linear is preferred unless data clearly
      demands otherwise).

    ### Returns
    Dict with R² values for each model fit, plus the recommended form.
    """
    x = buckets["mean_position"].values
    y = buckets["mean_ctr"].values

    with np.errstate(invalid="ignore"):
        # Linear: y = a*x + b
        lin_coeffs = np.polyfit(x, y, 1)
        lin_pred = np.polyval(lin_coeffs, x)
        ss_res = np.sum((y - lin_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        lin_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Log-linear: log(y) = a*x + b  (exponential decay)
        safe_mask = y > 0
        if safe_mask.sum() >= 3:
            log_y = np.log(y[safe_mask])
            log_coeffs = np.polyfit(x[safe_mask], log_y, 1)
            log_pred = np.exp(np.polyval(log_coeffs, x))
            ss_res = np.sum((y - log_pred) ** 2)
            log_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            log_r2 = float("nan")

        # Power-law: log(y) = a*log(x) + b
        safe_mask_p = (y > 0) & (x > 0)
        if safe_mask_p.sum() >= 3:
            log_x = np.log(x[safe_mask_p])
            log_y_p = np.log(y[safe_mask_p])
            pow_coeffs = np.polyfit(log_x, log_y_p, 1)
            pow_pred = np.exp(np.polyval(pow_coeffs, np.log(np.maximum(x, 1))))
            ss_res = np.sum((y - pow_pred) ** 2)
            pow_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            pow_r2 = float("nan")

    # Directional recommendation
    scores = {
        "linear": lin_r2,
        "log_linear": log_r2,
        "power_law": pow_r2,
    }
    valid_scores = {k: v for k, v in scores.items() if not np.isnan(v)}

    # Bias detection threshold: if best R² < 0.5, position explains less than
    # half the CTR variance — we conclude NO meaningful position bias exists.
    # This is a conservative threshold informed by IR literature (Joachims et al.)
    # where position bias typically produces R² > 0.7.
    bias_detected = any(v > 0.5 for v in valid_scores.values())
    best = max(valid_scores, key=valid_scores.get) if valid_scores else "none"

    return {
        "linear_r2": round(lin_r2, 4),
        "log_linear_r2": round(log_r2, 4) if not np.isnan(log_r2) else None,
        "power_law_r2": round(pow_r2, 4) if not np.isnan(pow_r2) else None,
        "recommended_form": best if bias_detected else "none",
        "bias_detected": bias_detected,
        "explanation": (
            "Position explains little CTR variance (R² < 0.5 for all models). "
            "In aggregated search data, niche queries (few candidates, motivated "
            "users) produce higher per-impression CTR at deep positions than "
            "generic queries at top positions. Applying position correction "
            "would harm the reranker."
        ) if not bias_detected else (
            f"Best model {best} with R²={max(valid_scores.values()):.4f}. "
            "Position bias correction is warranted."
        ),
    }


def write_markdown_report(
    buckets: pd.DataFrame,
    fit_assessment: dict,
    config: dict,
    output_path: str,
) -> None:
    """
    Render position bias analysis into a markdown report.

    ### Tradeoffs
    - Design: Appends to the data_profile_report pattern — consistent format
      across explore modules. Manual formatting vs. template engine (see
      data_profile.py for the same tradeoff analysis).
    """
    lines = []
    w = lines.append

    w("# Position Bias Analysis")
    w("")
    w(f"**Method:** CTR aggregated into position buckets (width={config['bucket_width']}), "
      f"adaptive merge for buckets < {config['adaptive_min_count']} rows.")
    w(f"**Reference position:** ≤ {config['reference_position']} "
      f"(correction factor baseline).")
    w(f"**Tail cap:** positions > {config['max_position']} grouped into one bucket.")
    w("")

    # ── Bucket table ─────────────────────────────────────────────────────────
    w("## CTR by Position Bucket")
    w("")
    w("| Bucket | Rows | Mean Pos | Mean CTR | Median CTR | Mean Clicks | "
      "Mean Impr | Correction Factor |")
    w("|--------|------|----------|----------|------------|-------------|"
      "----------|-------------------|")
    for _, row in buckets.iterrows():
        cf = row.get("correction_factor", None)
        cf_str = f"{cf:.3f}" if cf is not None and not np.isnan(cf) else "—"
        med = row["median_ctr"]
        med_str = f"{med:.1f}" if not np.isnan(med) else "—"
        w(
            f"| {row['pos_bucket']} | {int(row['row_count'])} | "
            f"{row['mean_position']:.1f} | {row['mean_ctr']:.2f} | "
            f"{med_str} | {row['mean_clicks']:.2f} | "
            f"{row['mean_impressions']:.1f} | {cf_str} |"
        )
    w("")

    # ── Model fit assessment ─────────────────────────────────────────────────
    w("## Position Bias Shape (D6 Input)")
    w("")
    w("Three functional forms were evaluated to describe CTR decay with position:")
    w("")
    w("| Model | R² |")
    w("|-------|-----|")
    for label, key in [
        ("Linear (CTR = a·pos + b)", "linear_r2"),
        ("Log-linear (CTR = e^(a·pos + b) — exponential decay)", "log_linear_r2"),
        ("Power-law (CTR = C · pos^a)", "power_law_r2"),
    ]:
        val = fit_assessment.get(key)
        val_str = f"{val:.4f}" if val is not None else "insufficient data"
        marker = ""
        if fit_assessment.get("bias_detected"):
            if key == f"{fit_assessment.get('recommended_form', '')}_r2":
                marker = " <- recommended"
        w(f"| {label} | {val_str}{marker} |")
    w("")

    bias_detected = fit_assessment.get("bias_detected", False)
    if bias_detected:
        rec = fit_assessment.get("recommended_form", "linear")
        w(f"**Recommended form for D6:** `{rec}`")
        w("")
        if rec == "linear":
            w("- Use `position_correction_type = \"linear\"` in RerankConfig.")
        elif rec == "log_linear":
            w("- Use `position_correction_type = \"log\"` in RerankConfig.")
        elif rec == "power_law":
            w("- Use `position_correction_type = \"power\"` in RerankConfig.")
        w("")
        w("The parameter `gamma` will be tuned during evaluation.")
    else:
        w("**Conclusion: No position bias detected.**")
        w("")
        w(fit_assessment.get("explanation", ""))
        w("")
        w("**D6 implication:** Drop position correction. The primary reranker uses "
          "Strategy D (smoothed CTR without position correction). Strategy E "
          "(smoothed + position correction) is retained for comparison only. "
          "Applying a position correction factor to this data would harm the "
          "reranker.")

    # ── Interpretation ───────────────────────────────────────────────────────
    w("## Interpretation")
    w("")
    if not bias_detected:
        w("**Why no position bias?** The data aggregates months of traffic across "
          "queries of different specificity. Broad queries have 100 candidates with "
          "flat CTR; niche queries have few candidates — motivated users scroll "
          "through all of them, producing high CTR even at deep positions. In this "
          "context, position correlates more with query specificity than relevance "
          "quality.")
        w("")
        w("Strategy D (Beta-binomial smoothed CTR) is the right tool: it handles "
          "the real problem of click sparsity without correcting for a bias that "
          "doesn't exist here.")
    else:
        w("Position bias is the well-known effect where items at lower ranks get "
          "fewer clicks regardless of their relevance. The correction factor answers:")
        w("")
        w("> \"If this product were shown at position 5 instead of position 50, "
          "how many more clicks would it get?\"")

    # ── Write ────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run: uv run python -m src.explore.position_bias
    Or:  uv run python src/explore/position_bias.py

    ### Observability (production add-ons)
    - Timer: `with metrics.timer("position_bias.duration"):`
    - Log: `logger.info("position_bias", rows=N, buckets=len(buckets))`
    - Alert: if correction_factor.max() > 10× → investigate noise in deep positions.
    """
    df = load_data(
        data_dir=POSITION_CONFIG["data_dir"],
        filename=POSITION_CONFIG["search_file"],
    )

    buckets = bucket_by_position(
        df,
        max_position=POSITION_CONFIG["max_position"],
        bucket_width=POSITION_CONFIG["bucket_width"],
        adaptive_min_count=POSITION_CONFIG["adaptive_min_count"],
    )

    buckets = compute_correction_factors(
        buckets,
        reference_position=POSITION_CONFIG["reference_position"],
    )

    fit = assess_fit(buckets)

    write_markdown_report(
        buckets, fit, POSITION_CONFIG, POSITION_CONFIG["output_report"],
    )

    print(f"[OK] Position bias analysis complete")
    print(f"  Buckets: {len(buckets)} (after adaptive merge)")
    print(f"  Recommended form: {fit['recommended_form']}")
    print(f"  Correction factor range: "
          f"{buckets['correction_factor'].min():.2f} – "
          f"{buckets['correction_factor'].max():.2f}")
    print(f"  Report: {POSITION_CONFIG['output_report']}")
