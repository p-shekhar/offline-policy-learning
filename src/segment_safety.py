from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import ExperimentConfig
from policy_catalog import BASELINE_POLICY_ID, ReservePolicyCatalog
from replay import replay_policy


def segment_replay_lifts(
    panel_dir: Path,
    catalog: ReservePolicyCatalog,
    policy_id: str,
    segment_column: str,
    config: ExperimentConfig,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    base_columns = [
        "event_date",
        "slot_floor_price",
        "bid_floor_gap",
        "bid_price",
        "pay_price",
        "filled",
        "clicked",
        "converted",
    ]
    if segment_column == "inventory_category":
        extra_columns = ["slot_visibility", "slot_format"]
    elif segment_column == "bid_gap_bucket":
        extra_columns = ["bid_floor_gap"]
    else:
        extra_columns = [segment_column]
    required_columns = list(dict.fromkeys([*base_columns, *extra_columns]))
    for parquet in sorted(panel_dir.glob("*.parquet")):
        frame = pd.read_parquet(parquet, columns=required_columns)
        if segment_column == "inventory_category":
            frame[segment_column] = (
                frame["slot_visibility"].astype(str) + "|" + frame["slot_format"].astype(str)
            )
        elif segment_column == "bid_gap_bucket":
            gap = pd.to_numeric(frame["bid_floor_gap"], errors="coerce")
            frame[segment_column] = pd.cut(
                gap,
                bins=[-np.inf, 0, 25, 50, 100, 200, np.inf],
                labels=["<=0", "0-25", "25-50", "50-100", "100-200", ">200"],
                include_lowest=True,
            ).astype(str)
        for segment, group in frame.groupby(segment_column, observed=True, dropna=False):
            if len(group) < config.min_segment_observations:
                continue
            base = replay_policy(group, catalog, BASELINE_POLICY_ID, config.value_proxy_conversion_weight)
            cand = replay_policy(group, catalog, policy_id, config.value_proxy_conversion_weight)
            base_y = base["counterfactual_yield"] / max(base["opportunities"], 1)
            cand_y = cand["counterfactual_yield"] / max(cand["opportunities"], 1)
            lift = (cand_y - base_y) / base_y if base_y else np.nan
            rows.append(
                {
                    "segment_column": segment_column,
                    "segment": str(segment),
                    "event_date": str(group["event_date"].iloc[0]),
                    "opportunities": len(group),
                    "lift": lift,
                }
            )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return daily
    z = norm.ppf(1.0 - alpha / 2.0)
    summary = (
        daily.groupby(["segment_column", "segment"], as_index=False)
        .agg(
            mean_lift=("lift", "mean"),
            sd_lift=("lift", "std"),
            days=("lift", "count"),
            opportunities=("opportunities", "sum"),
        )
        .fillna({"sd_lift": 0.0})
    )
    summary["standard_error"] = summary["sd_lift"] / np.sqrt(summary["days"].clip(lower=1))
    summary["lower_bar"] = summary["mean_lift"] - z * summary["standard_error"]
    summary["upper_bar"] = summary["mean_lift"] + z * summary["standard_error"]
    summary["passes_nonharm_bar"] = summary["lower_bar"] >= 0
    positive_mean_days = np.where(
        summary["mean_lift"].gt(0),
        np.ceil((z * summary["sd_lift"] / summary["mean_lift"]) ** 2),
        np.inf,
    )
    summary["days_required_for_nonharm_bar"] = np.where(
        summary["passes_nonharm_bar"],
        summary["days"],
        positive_mean_days,
    )
    summary["additional_days_required"] = np.maximum(
        summary["days_required_for_nonharm_bar"].replace(np.inf, np.nan) - summary["days"],
        0,
    )
    return summary.sort_values("lower_bar").reset_index(drop=True)


def combine_segment_safety(
    summaries: list[pd.DataFrame],
    alpha: float,
    min_observations: int,
) -> pd.DataFrame:
    result = pd.concat([frame for frame in summaries if not frame.empty], ignore_index=True)
    result["alpha"] = alpha
    result["min_segment_observations"] = min_observations
    return result.sort_values(["lower_bar", "segment_column", "segment"]).reset_index(drop=True)


def segment_radius_sensitivity(
    segment_summary: pd.DataFrame,
    radii: list[float],
    lipschitz_constant: float = 0.02,
) -> pd.DataFrame:
    """Evaluate a simple coverage-radius penalty for segment-safety certification."""

    rows = []
    for radius in radii:
        adjusted = segment_summary.copy()
        adjusted["coverage_radius"] = float(radius)
        adjusted["lipschitz_constant"] = float(lipschitz_constant)
        adjusted["radius_adjusted_lower_bar"] = adjusted["lower_bar"] - lipschitz_constant * radius
        rows.append(
            {
                "coverage_radius": float(radius),
                "lipschitz_constant": float(lipschitz_constant),
                "segment_count": int(len(adjusted)),
                "certified_segment_count": int(adjusted["radius_adjusted_lower_bar"].ge(0).sum()),
                "minimum_radius_adjusted_lower_bar": float(adjusted["radius_adjusted_lower_bar"].min()),
            }
        )
    return pd.DataFrame(rows)


def segment_multiplicity_scaling(
    segment_summary: pd.DataFrame,
    segment_counts: list[int],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Approximate how simultaneous segment penalties scale with segment count."""

    from scipy.stats import norm

    sd = segment_summary["sd_lift"].fillna(0.0)
    days = segment_summary["days"].clip(lower=1)
    mean = segment_summary["mean_lift"]
    rows = []
    for k in segment_counts:
        z = norm.ppf(1.0 - alpha / (2.0 * max(int(k), 1)))
        lower = mean - z * sd / np.sqrt(days)
        rows.append(
            {
                "segment_count": int(k),
                "bonferroni_z": float(z),
                "certified_segment_count": int(lower.ge(0).sum()),
                "minimum_lower_bar": float(lower.min()),
            }
        )
    return pd.DataFrame(rows)
