from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def replay_lower_bounds(
    daily_effects: pd.DataFrame,
    aggregate_effects: pd.DataFrame,
    alpha: float = 0.05,
    support_penalty_scale: float = 1.0,
) -> pd.DataFrame:
    """Construct simultaneous finite-policy lower bounds from daily replay variation."""

    policy_count = daily_effects["policy_id"].nunique()
    z = norm.ppf(1.0 - alpha / (2.0 * max(policy_count, 1)))
    daily = daily_effects.copy()
    baseline = daily.query("policy_id == 'logged_floor_status_quo'")[
        ["event_date", "yield_per_opportunity"]
    ].rename(columns={"yield_per_opportunity": "baseline_yield_per_opportunity"})
    daily = daily.merge(baseline, on="event_date", how="left")
    daily["daily_lift"] = (
        daily["yield_per_opportunity"] - daily["baseline_yield_per_opportunity"]
    ) / daily["baseline_yield_per_opportunity"].replace(0, np.nan)
    rows = (
        daily.groupby("policy_id", as_index=False)
        .agg(
            daily_mean_lift=("daily_lift", "mean"),
            daily_sd_lift=("daily_lift", "std"),
            days=("daily_lift", "count"),
        )
        .fillna({"daily_sd_lift": 0.0})
    )
    rows["standard_error"] = rows["daily_sd_lift"] / np.sqrt(rows["days"].clip(lower=1))
    rows["simultaneous_radius"] = z * rows["standard_error"]
    support = aggregate_effects[["policy_id", "retained_impression_share"]].copy()
    rows = rows.merge(support, on="policy_id", how="left")
    rows["support_penalty"] = support_penalty_scale * (1.0 - rows["retained_impression_share"].fillna(0.0))
    rows["lower_bound_lift"] = rows["daily_mean_lift"] - rows["simultaneous_radius"] - rows["support_penalty"]
    rows["upper_bound_lift"] = rows["daily_mean_lift"] + rows["simultaneous_radius"]
    rows["alpha"] = alpha
    rows["bonferroni_z"] = z
    return rows.sort_values("lower_bound_lift", ascending=False).reset_index(drop=True)


def eliminate_dominated_policies(bounds: pd.DataFrame, tolerance: float = 0.0) -> pd.DataFrame:
    """Apply upper-below-best-lower elimination rule."""

    result = bounds.copy()
    best_lower = result["lower_bound_lift"].max()
    leader = result.sort_values("lower_bound_lift", ascending=False).iloc[0]["policy_id"]
    result["best_lower_bound_lift"] = best_lower
    result["lower_bound_leader"] = leader
    result["shortlist_tolerance"] = float(tolerance)
    result["eliminated_by_lower_bound_rule"] = result["upper_bound_lift"] < (best_lower - tolerance)
    result["dominance_gap"] = (best_lower - tolerance) - result["upper_bound_lift"]
    return result.sort_values(
        ["eliminated_by_lower_bound_rule", "lower_bound_lift"],
        ascending=[True, False],
    ).reset_index(drop=True)


def assign_decision_labels(elimination: pd.DataFrame) -> pd.DataFrame:
    """Label policies as certified, dominated, or unresolved.

    The label is intentionally conservative. The lower-bound leader is the
    certified validation target when its lower bound is positive. Policies whose
    upper bound is below the leader's lower bound are dominated. The remaining
    policies are unresolved and stay in the validation shortlist.
    """

    result = elimination.copy()
    leader = str(result["lower_bound_leader"].iloc[0])
    result["decision_label"] = "unresolved"
    result.loc[result["eliminated_by_lower_bound_rule"], "decision_label"] = "dominated"
    leader_mask = result["policy_id"].eq(leader) & result["lower_bound_lift"].gt(0)
    result.loc[leader_mask, "decision_label"] = "certified"
    result["in_validation_shortlist"] = result["decision_label"].isin(["certified", "unresolved"])
    result["certifies_positive_lift"] = result["lower_bound_lift"] > 0
    return result


def shortlist_size_sensitivity(bounds: pd.DataFrame, tolerances: list[float]) -> pd.DataFrame:
    """Evaluate retained shortlist size as the elimination tolerance varies."""

    rows = []
    for tolerance in tolerances:
        labeled = assign_decision_labels(eliminate_dominated_policies(bounds, tolerance=tolerance))
        rows.append(
            {
                "shortlist_tolerance": float(tolerance),
                "shortlist_size": int(labeled["in_validation_shortlist"].sum()),
                "dominated_count": int(labeled["decision_label"].eq("dominated").sum()),
                "certified_count": int(labeled["decision_label"].eq("certified").sum()),
                "unresolved_count": int(labeled["decision_label"].eq("unresolved").sum()),
                "shortlist_policy_ids": ", ".join(
                    labeled.loc[labeled["in_validation_shortlist"], "policy_id"].astype(str)
                ),
            }
        )
    return pd.DataFrame(rows)
