from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from policy_catalog import ReservePolicyCatalog
from support import replay_support_diagnostics


def threshold_radius_sweep(
    panel_dir: Path,
    catalog: ReservePolicyCatalog,
    radii: list[float],
) -> pd.DataFrame:
    rows = []
    for parquet in sorted(panel_dir.glob("*.parquet")):
        frame = pd.read_parquet(parquet, columns=["slot_floor_price", "bid_floor_gap", "bid_price", "filled"])
        for radius in radii:
            part = replay_support_diagnostics(frame, catalog, radius=radius)
            part["panel_file"] = parquet.name
            rows.append(part)
    combined = pd.concat(rows, ignore_index=True)
    summary = (
        combined.groupby(["policy_id", "threshold_band_radius"], as_index=False)
        .agg(
            row_count=("row_count", "sum"),
            filled_count=("filled_count", "sum"),
            floor_changed_count=("floor_changed_count", "sum"),
            retained_impression_count=("retained_impression_count", "sum"),
            threshold_band_count=("threshold_band_count", "sum"),
            changed_threshold_band_count=("changed_threshold_band_count", "sum"),
        )
        .sort_values(["policy_id", "threshold_band_radius"])
    )
    summary["floor_changed_share"] = summary["floor_changed_count"] / summary["row_count"].clip(lower=1)
    summary["retained_impression_share"] = (
        summary["retained_impression_count"] / summary["filled_count"].clip(lower=1)
    )
    summary["threshold_band_share"] = summary["threshold_band_count"] / summary["row_count"].clip(lower=1)
    summary["sqrt_threshold_band_count"] = np.sqrt(summary["threshold_band_count"].clip(lower=0))
    summary["inverse_sqrt_support"] = 1.0 / summary["sqrt_threshold_band_count"].replace(0, np.nan)
    return summary


def topk_shortlist_stability(bounds: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    ordered = bounds.sort_values("lower_bound_lift", ascending=False).reset_index(drop=True)
    rows = []
    for k in k_values:
        if k < 1 or k >= len(ordered):
            continue
        kth = ordered.iloc[k - 1]
        next_policy = ordered.iloc[k]
        rows.append(
            {
                "k": k,
                "kth_policy_id": kth["policy_id"],
                "next_policy_id": next_policy["policy_id"],
                "kth_lower_bound_lift": kth["lower_bound_lift"],
                "next_lower_bound_lift": next_policy["lower_bound_lift"],
                "shortlist_margin": kth["lower_bound_lift"] - next_policy["lower_bound_lift"],
                "max_symmetric_perturbation_preserving_topk": 0.5
                * (kth["lower_bound_lift"] - next_policy["lower_bound_lift"]),
            }
        )
    return pd.DataFrame(rows)


def support_explicit_lower_bound_ranking(
    bounds: pd.DataFrame,
    support_sweep: pd.DataFrame,
    calibration_policy_id: str,
    calibration_radius: float = 10.0,
    penalty_scale: float | None = None,
) -> pd.DataFrame:
    """Rank policies with an explicit c / sqrt(support) penalty.

    The scale ``c`` can be supplied directly. If omitted, it is calibrated so
    that the support-explicit penalty for ``calibration_policy_id`` at
    ``calibration_radius`` equals that policy's simultaneous daily-variation
    radius. This keeps the empirical exercise data anchored while making the
    support penalty from the corollary visible.
    """

    support_cols = [
        "policy_id",
        "threshold_band_radius",
        "threshold_band_count",
        "threshold_band_share",
        "inverse_sqrt_support",
    ]
    bound_cols = [
        "policy_id",
        "daily_mean_lift",
        "simultaneous_radius",
        "lower_bound_lift",
        "upper_bound_lift",
    ]
    missing_support = set(support_cols) - set(support_sweep.columns)
    missing_bounds = set(bound_cols) - set(bounds.columns)
    if missing_support:
        raise ValueError(f"support_sweep is missing columns: {sorted(missing_support)}")
    if missing_bounds:
        raise ValueError(f"bounds is missing columns: {sorted(missing_bounds)}")

    if penalty_scale is None:
        calibration = support_sweep[
            (support_sweep["policy_id"] == calibration_policy_id)
            & (support_sweep["threshold_band_radius"].astype(float) == float(calibration_radius))
        ]
        if calibration.empty:
            raise ValueError("Calibration policy/radius pair was not found in support_sweep.")
        inverse_support = float(calibration["inverse_sqrt_support"].iloc[0])
        if inverse_support <= 0 or pd.isna(inverse_support):
            raise ValueError("Calibration inverse support must be positive.")
        radius = float(
            bounds.loc[bounds["policy_id"] == calibration_policy_id, "simultaneous_radius"].iloc[0]
        )
        penalty_scale = radius / inverse_support

    ranking = support_sweep[support_cols].merge(bounds[bound_cols], on="policy_id", how="left")
    ranking["support_penalty_scale"] = float(penalty_scale)
    ranking["support_explicit_penalty"] = ranking["support_penalty_scale"] * ranking[
        "inverse_sqrt_support"
    ]
    ranking["support_explicit_lower_bound_lift"] = (
        ranking["daily_mean_lift"] - ranking["support_explicit_penalty"]
    )
    ranking["support_explicit_upper_bound_lift"] = (
        ranking["daily_mean_lift"] + ranking["support_explicit_penalty"]
    )
    ranking["certifies_positive_lift"] = ranking["support_explicit_lower_bound_lift"] > 0
    ranking["support_explicit_rank"] = ranking.groupby("threshold_band_radius")[
        "support_explicit_lower_bound_lift"
    ].rank(method="min", ascending=False)
    return ranking.sort_values(["threshold_band_radius", "support_explicit_rank", "policy_id"])
