from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig
from policy_catalog import BASELINE_POLICY_ID, ReservePolicyCatalog
from progress import ProgressLogger


REPLAY_COLUMNS = [
    "bid_id",
    "event_date",
    "hour",
    "region",
    "city",
    "ad_exchange",
    "advertiser_id",
    "slot_visibility",
    "slot_format",
    "support_cluster",
    "slot_floor_price",
    "bid_floor_gap",
    "bid_price",
    "pay_price",
    "filled",
    "clicked",
    "converted",
]


def replay_policy(
    frame: pd.DataFrame,
    catalog: ReservePolicyCatalog,
    policy_id: str,
    conversion_weight: float = 10.0,
) -> dict[str, float | int | str]:
    candidate_floor = catalog.floor(frame, policy_id)
    logged_floor = pd.to_numeric(frame["slot_floor_price"], errors="coerce").fillna(0).to_numpy("float64")
    bid_price = pd.to_numeric(frame["bid_price"], errors="coerce").fillna(-np.inf).to_numpy("float64")
    pay_price = pd.to_numeric(frame["pay_price"], errors="coerce").fillna(0).to_numpy("float64")
    filled = pd.to_numeric(frame["filled"], errors="coerce").fillna(0).astype(bool).to_numpy()
    clicked = pd.to_numeric(frame["clicked"], errors="coerce").fillna(0).to_numpy("int64")
    converted = pd.to_numeric(frame["converted"], errors="coerce").fillna(0).to_numpy("int64")
    retained = filled & (bid_price >= candidate_floor)
    counterfactual_pay = np.where(retained, np.maximum(pay_price, candidate_floor), 0.0)
    return {
        "policy_id": policy_id,
        "opportunities": int(len(frame)),
        "observed_filled_impressions": int(filled.sum()),
        "retained_impressions": int(retained.sum()),
        "counterfactual_yield": float(counterfactual_pay.sum()),
        "retained_clicks": int(np.where(retained, clicked, 0).sum()),
        "retained_conversions": int(np.where(retained, converted, 0).sum()),
        "retained_value_proxy": float(np.where(retained, clicked + conversion_weight * converted, 0).sum()),
        "floor_changed_share": float(np.mean(~np.isclose(candidate_floor, logged_floor))),
        "mean_candidate_floor": float(np.mean(candidate_floor)),
        "p95_candidate_floor": float(np.percentile(candidate_floor, 95)),
    }


def add_replay_rates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fill_rate"] = result["retained_impressions"] / result["opportunities"].replace(0, np.nan)
    result["yield_per_opportunity"] = result["counterfactual_yield"] / result["opportunities"].replace(0, np.nan)
    result["yield_per_retained_impression"] = result["counterfactual_yield"] / result[
        "retained_impressions"
    ].replace(0, np.nan)
    result["value_proxy_per_opportunity"] = result["retained_value_proxy"] / result["opportunities"].replace(
        0, np.nan
    )
    result["retained_impression_share"] = result["retained_impressions"] / result[
        "observed_filled_impressions"
    ].replace(0, np.nan)
    return result


def evaluate_policy_catalog(
    panel_dir: Path,
    catalog: ReservePolicyCatalog,
    config: ExperimentConfig,
    progress: ProgressLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = progress or ProgressLogger(enabled=False)
    registry = catalog.registry()
    daily_rows = []
    for shard in sorted(panel_dir.glob("*.parquet")):
        logger.log(f"Replaying policies on {shard.name}.")
        frame = pd.read_parquet(shard, columns=REPLAY_COLUMNS)
        event_date = str(frame["event_date"].iloc[0])
        for policy_id in registry["policy_id"]:
            row = replay_policy(frame, catalog, policy_id, config.value_proxy_conversion_weight)
            row["event_date"] = event_date
            daily_rows.append(row)
    daily = add_replay_rates(pd.DataFrame(daily_rows))
    aggregate = (
        daily.groupby("policy_id", as_index=False)
        .agg(
            opportunities=("opportunities", "sum"),
            observed_filled_impressions=("observed_filled_impressions", "sum"),
            retained_impressions=("retained_impressions", "sum"),
            counterfactual_yield=("counterfactual_yield", "sum"),
            retained_clicks=("retained_clicks", "sum"),
            retained_conversions=("retained_conversions", "sum"),
            retained_value_proxy=("retained_value_proxy", "sum"),
            floor_changed_share=("floor_changed_share", "mean"),
            mean_candidate_floor=("mean_candidate_floor", "mean"),
            p95_candidate_floor=("p95_candidate_floor", "mean"),
        )
        .pipe(add_replay_rates)
    )
    baseline = aggregate.query("policy_id == @BASELINE_POLICY_ID").iloc[0]
    for metric in ["yield_per_opportunity", "fill_rate", "value_proxy_per_opportunity"]:
        aggregate[f"delta_{metric}_vs_baseline"] = aggregate[metric] - baseline[metric]
        aggregate[f"pct_delta_{metric}_vs_baseline"] = aggregate[f"delta_{metric}_vs_baseline"] / baseline[
            metric
        ]
    aggregate = aggregate.merge(registry, on="policy_id", how="left")
    aggregate = aggregate.sort_values("yield_per_opportunity", ascending=False).reset_index(drop=True)
    return aggregate, daily
