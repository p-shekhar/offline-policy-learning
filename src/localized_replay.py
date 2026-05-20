from __future__ import annotations

import numpy as np
import pandas as pd

from config import ExperimentConfig
from policy_catalog import BASELINE_POLICY_ID, ReservePolicyCatalog
from progress import ProgressLogger


LOCALIZED_COLUMNS = [
    "event_date",
    "slot_floor_price",
    "bid_floor_gap",
    "bid_price",
    "pay_price",
    "filled",
]


def _replay_outcomes(
    frame: pd.DataFrame,
    floor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bid_price = pd.to_numeric(frame["bid_price"], errors="coerce").fillna(-np.inf).to_numpy("float64")
    pay_price = pd.to_numeric(frame["pay_price"], errors="coerce").fillna(0).to_numpy("float64")
    filled = pd.to_numeric(frame["filled"], errors="coerce").fillna(0).astype(bool).to_numpy()
    retained = filled & (bid_price >= floor)
    return np.where(retained, np.maximum(pay_price, floor), 0.0), retained


def q_localized_replay_daily(
    frame: pd.DataFrame,
    catalog: ReservePolicyCatalog,
    q_values: list[float],
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Compute day-level q-localized replay scores for every non-baseline policy.

    The localized score uses only floor-changing observations closest to the
    candidate policy's reserve threshold. For each policy and each q, the code
    finds the smallest empirical radius that includes q of changed opportunities
    on the day, and then computes replay lift inside that boundary window.
    """

    registry = catalog.registry()
    event_date = str(frame["event_date"].iloc[0])
    logged_floor = pd.to_numeric(frame["slot_floor_price"], errors="coerce").fillna(0).to_numpy("float64")
    bid_price = pd.to_numeric(frame["bid_price"], errors="coerce").fillna(-np.inf).to_numpy("float64")
    baseline_yield, baseline_retained = _replay_outcomes(frame, logged_floor)
    baseline_total_yield = float(baseline_yield.sum())
    row_count = int(len(frame))
    baseline_retained_count = int(baseline_retained.sum())
    rows = []

    for policy_id in registry["policy_id"]:
        if policy_id == BASELINE_POLICY_ID:
            continue
        candidate_floor = catalog.floor(frame, policy_id)
        policy_yield, policy_retained = _replay_outcomes(frame, candidate_floor)
        replay_diff = policy_yield - baseline_yield
        full_replay_lift = float(replay_diff.sum() / baseline_total_yield) if baseline_total_yield else np.nan
        changed = ~np.isclose(candidate_floor, logged_floor)
        changed_count = int(changed.sum())
        distance = np.abs(bid_price - candidate_floor)
        changed_distance = distance[changed]

        for q in q_values:
            if changed_count == 0:
                radius = 0.0
                localized = np.zeros(row_count, dtype=bool)
            else:
                radius = float(np.quantile(changed_distance, q))
                localized = changed & (distance <= radius)

            localized_count = int(localized.sum())
            localized_baseline_yield = float(baseline_yield[localized].sum())
            localized_policy_yield = float(policy_yield[localized].sum())
            localized_diff = localized_policy_yield - localized_baseline_yield
            localized_boundary_lift = (
                localized_diff / localized_baseline_yield if localized_baseline_yield else np.nan
            )
            localized_total_lift = (
                localized_diff / baseline_total_yield if baseline_total_yield else np.nan
            )
            localized_retained_count = int(policy_retained[localized].sum())
            rows.append(
                {
                    "event_date": event_date,
                    "q": float(q),
                    "policy_id": policy_id,
                    "opportunities": row_count,
                    "baseline_total_yield": baseline_total_yield,
                    "baseline_retained_impressions": baseline_retained_count,
                    "full_replay_lift": full_replay_lift,
                    "floor_changed_count": changed_count,
                    "floor_changed_share": changed_count / max(row_count, 1),
                    "localized_radius": radius,
                    "localized_count": localized_count,
                    "localized_share_of_opportunities": localized_count / max(row_count, 1),
                    "localized_share_of_changed": localized_count / max(changed_count, 1),
                    "localized_baseline_yield": localized_baseline_yield,
                    "localized_policy_yield": localized_policy_yield,
                    "localized_diff": localized_diff,
                    "localized_boundary_lift": localized_boundary_lift,
                    "localized_total_lift": localized_total_lift,
                    "localized_retained_impressions": localized_retained_count,
                }
            )
    result = pd.DataFrame(rows)
    return result.merge(registry, on="policy_id", how="left")


def aggregate_q_localized_replay(daily: pd.DataFrame) -> pd.DataFrame:
    summary = (
        daily.groupby(["q", "policy_id"], as_index=False)
        .agg(
            opportunities=("opportunities", "sum"),
            baseline_total_yield=("baseline_total_yield", "sum"),
            floor_changed_count=("floor_changed_count", "sum"),
            localized_count=("localized_count", "sum"),
            localized_baseline_yield=("localized_baseline_yield", "sum"),
            localized_policy_yield=("localized_policy_yield", "sum"),
            localized_diff=("localized_diff", "sum"),
            localized_retained_impressions=("localized_retained_impressions", "sum"),
            mean_localized_radius=("localized_radius", "mean"),
            median_localized_radius=("localized_radius", "median"),
            mean_full_replay_lift=("full_replay_lift", "mean"),
        )
    )
    summary["floor_changed_share"] = summary["floor_changed_count"] / summary["opportunities"].clip(lower=1)
    summary["localized_share_of_opportunities"] = summary["localized_count"] / summary["opportunities"].clip(
        lower=1
    )
    summary["localized_share_of_changed"] = summary["localized_count"] / summary[
        "floor_changed_count"
    ].clip(lower=1)
    summary["localized_boundary_lift"] = summary["localized_diff"] / summary[
        "localized_baseline_yield"
    ].replace(0, np.nan)
    summary["localized_total_lift"] = summary["localized_diff"] / summary[
        "baseline_total_yield"
    ].replace(0, np.nan)
    summary["localized_rank"] = summary.groupby("q")["localized_boundary_lift"].rank(
        method="first",
        ascending=False,
    )
    summary["localized_total_rank"] = summary.groupby("q")["localized_total_lift"].rank(
        method="first",
        ascending=False,
    )
    summary["localized_mean_full_replay_rank"] = summary.groupby("q")["mean_full_replay_lift"].rank(
        method="first",
        ascending=False,
    )
    metadata_cols = ["policy_id", "policy_number", "policy_label", "policy_family"]
    metadata = daily[metadata_cols].drop_duplicates("policy_id")
    return summary.merge(metadata, on="policy_id", how="left").sort_values(["q", "localized_rank"])


def bootstrap_q_localized_winners(
    daily: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = np.array(sorted(daily["event_date"].unique()))
    q_values = sorted(daily["q"].unique())
    rows = []
    for q in q_values:
        q_daily = daily[daily["q"].eq(q)].copy()
        for draw in range(iterations):
            sampled_days = rng.choice(days, size=len(days), replace=True)
            sampled = pd.concat(
                [q_daily[q_daily["event_date"].eq(day)] for day in sampled_days],
                ignore_index=True,
            )
            aggregate = aggregate_q_localized_replay(sampled)
            leader = aggregate.sort_values("localized_boundary_lift", ascending=False).iloc[0]
            top5 = aggregate.sort_values("localized_boundary_lift", ascending=False).head(5)
            rows.append(
                {
                    "q": float(q),
                    "bootstrap_draw": draw,
                    "selected_policy_id": leader["policy_id"],
                    "selected_policy_number": leader["policy_number"],
                    "selected_policy_label": leader["policy_label"],
                    "selected_localized_boundary_lift": float(leader["localized_boundary_lift"]),
                    "top5_policy_ids": ",".join(top5["policy_id"].astype(str)),
                    "top5_policy_numbers": ",".join(top5["policy_number"].astype(str)),
                }
            )
    return pd.DataFrame(rows)


def summarize_bootstrap_q_localized_winners(bootstrap: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["q", "selected_policy_id", "selected_policy_number", "selected_policy_label"]
    summary = (
        bootstrap.groupby(group_cols, as_index=False)
        .agg(
            selected_draws=("bootstrap_draw", "count"),
            mean_selected_localized_boundary_lift=("selected_localized_boundary_lift", "mean"),
        )
        .sort_values(["q", "selected_draws"], ascending=[True, False])
    )
    total_draws = bootstrap.groupby("q")["bootstrap_draw"].nunique().rename("total_draws")
    summary = summary.merge(total_draws, on="q", how="left")
    summary["selection_frequency"] = summary["selected_draws"] / summary["total_draws"].clip(lower=1)
    return summary


def q_localized_replay_panel(
    panel_dir,
    catalog: ReservePolicyCatalog,
    q_values: list[float],
    config: ExperimentConfig,
    progress: ProgressLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = progress or ProgressLogger(enabled=False)
    daily_frames = []
    for shard in sorted(panel_dir.glob("*.parquet")):
        logger.log(f"Computing q-localized replay on {shard.name}.")
        frame = pd.read_parquet(shard, columns=LOCALIZED_COLUMNS)
        daily_frames.append(q_localized_replay_daily(frame, catalog, q_values, config))
    daily = pd.concat(daily_frames, ignore_index=True)
    aggregate = aggregate_q_localized_replay(daily)
    return aggregate, daily
