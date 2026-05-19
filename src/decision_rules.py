from __future__ import annotations

import pandas as pd


def ipinyou_decision_rule_ablation(
    replay_effects: pd.DataFrame,
    lower_bounds: pd.DataFrame,
    elimination: pd.DataFrame,
) -> pd.DataFrame:
    """Compare point-estimate and support-aware offline decision rules."""

    replay_winner = replay_effects.sort_values(
        "pct_delta_yield_per_opportunity_vs_baseline", ascending=False
    ).iloc[0]
    lower_winner = lower_bounds.sort_values("lower_bound_lift", ascending=False).iloc[0]
    non_eliminated = elimination.query("not eliminated_by_lower_bound_rule").copy()
    rows = [
        {
            "decision_rule": "Replay-only",
            "evidence_used": "Season 2 replay point estimate",
            "selected_policy": replay_winner["policy_id"],
            "selected_policy_lift": replay_winner["pct_delta_yield_per_opportunity_vs_baseline"],
            "certifies_nonnegative_lift": bool(
                lower_bounds.set_index("policy_id").loc[replay_winner["policy_id"], "lower_bound_lift"] >= 0
            ),
            "shortlist_size": 1,
        },
        {
            "decision_rule": "Lower-bound winner",
            "evidence_used": "Simultaneous lower bound plus support penalty",
            "selected_policy": lower_winner["policy_id"],
            "selected_policy_lift": lower_winner["daily_mean_lift"],
            "certifies_nonnegative_lift": bool(lower_winner["lower_bound_lift"] >= 0),
            "shortlist_size": 1,
        },
        {
            "decision_rule": "Elimination shortlist",
            "evidence_used": "Upper-bound domination by best lower bound",
            "selected_policy": ", ".join(non_eliminated["policy_id"].astype(str).tolist()),
            "selected_policy_lift": float(non_eliminated["lower_bound_lift"].max())
            if not non_eliminated.empty
            else float("nan"),
            "certifies_nonnegative_lift": bool(non_eliminated["lower_bound_lift"].max() >= 0)
            if not non_eliminated.empty
            else False,
            "shortlist_size": int(len(non_eliminated)),
        },
    ]
    return pd.DataFrame(rows)
