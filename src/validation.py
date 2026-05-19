from __future__ import annotations

import pandas as pd

from config import ExperimentConfig, ProjectPaths
from policy_catalog import ReservePolicyCatalog
from progress import ProgressLogger
from replay import evaluate_policy_catalog


def frozen_catalog_season_validation(
    paths: ProjectPaths,
    config: ExperimentConfig,
    progress: ProgressLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply Season 2 policy definitions unchanged to Season 3."""

    season2_dir = paths.panel_dir / "ipinyou_season2"
    season3_dir = paths.panel_dir / "ipinyou_season3"
    catalog = ReservePolicyCatalog.from_panel_directory(season2_dir)
    cached_season2 = paths.table_dir / "02_season2_replay_policy_effects.csv"
    if cached_season2.exists():
        if progress is not None:
            progress.log(f"Using cached Season 2 replay effects: {cached_season2.name}.")
        season2_effects = pd.read_csv(cached_season2)
    else:
        season2_effects, _ = evaluate_policy_catalog(season2_dir, catalog, config, progress)
    season3_effects, season3_daily = evaluate_policy_catalog(season3_dir, catalog, config, progress)

    rank_cols = ["policy_id", "policy_number", "policy_label", "policy_family"]
    s2 = season2_effects[rank_cols + ["pct_delta_yield_per_opportunity_vs_baseline"]].copy()
    s3 = season3_effects[rank_cols + ["pct_delta_yield_per_opportunity_vs_baseline", "retained_impression_share"]].copy()
    s2["season2_rank"] = s2["pct_delta_yield_per_opportunity_vs_baseline"].rank(ascending=False, method="min")
    s3["season3_rank"] = s3["pct_delta_yield_per_opportunity_vs_baseline"].rank(ascending=False, method="min")
    transfer = s2.merge(s3, on=rank_cols, suffixes=("_season2", "_season3"))
    transfer = transfer.rename(
        columns={
            "pct_delta_yield_per_opportunity_vs_baseline_season2": "season2_replay_lift",
            "pct_delta_yield_per_opportunity_vs_baseline_season3": "season3_replay_lift",
        }
    )
    transfer["rank_shift"] = transfer["season3_rank"] - transfer["season2_rank"]
    transfer = transfer.sort_values(["season2_rank", "season3_rank"]).reset_index(drop=True)
    return transfer, season3_effects, season3_daily
