from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig, ProjectPaths
from data_access import IpinYouArchive
from decision_rules import ipinyou_decision_rule_ablation
from figures import (
    save_appendix_decision_robustness_figure,
    save_appendix_replay_diagnostics,
    save_appendix_support_pair_figure,
    save_shortlist_construction_figure,
    save_threshold_resolution_figure,
    save_validation_readiness_figure,
)
from ipinyou_panel import IpinYouPanelBuilder, summarize_panel_directory
from lower_bounds import (
    assign_decision_labels,
    eliminate_dominated_policies,
    replay_lower_bounds,
    shortlist_size_sensitivity,
)
from policy_catalog import BASELINE_POLICY_ID, PRIORITY_POLICY_ID, ReservePolicyCatalog
from progress import ProgressLogger
from replay import evaluate_policy_catalog
from segment_safety import (
    combine_segment_safety,
    segment_multiplicity_scaling,
    segment_radius_sensitivity,
    segment_replay_lifts,
)
from support import pairwise_boundary_support, summarize_pairwise_boundary_support
from tables import write_table
from threshold_resolution import (
    support_explicit_lower_bound_ranking,
    threshold_radius_sweep,
    topk_shortlist_stability,
)
from validation import frozen_catalog_season_validation


SEASON2_PANEL = "ipinyou_season2"
SEASON3_PANEL = "ipinyou_season3"
MAIN_THRESHOLD_RADII = [1, 2, 5, 10, 20, 50, 100, 200]
APPENDIX_THRESHOLD_RADII = [1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 200]


def run_data_audit(paths: ProjectPaths, progress: ProgressLogger | None = None) -> dict[str, pd.DataFrame]:
    """Inventory the local iPinYou archive and write setup tables."""

    paths.ensure()
    logger = progress or ProgressLogger(enabled=False)
    logger.step("Reading the local iPinYou archive inventory.")
    archive = IpinYouArchive(paths.ipinyou_archive)
    inventory = archive.inventory()
    summary = (
        inventory.groupby(["season", "kind"], as_index=False)
        .agg(files=("member", "count"), compressed_size=("compressed_size", "sum"))
        .sort_values(["season", "kind"])
    )
    write_table(inventory, paths.metadata_dir / "00_ipinyou_inventory.csv")
    write_table(summary, paths.table_dir / "00_ipinyou_inventory_summary.csv")
    logger.done("Data audit artifacts are ready.")
    return {"inventory": inventory, "inventory_summary": summary}


def build_ipinyou_panels(
    paths: ProjectPaths,
    config: ExperimentConfig,
    progress: ProgressLogger | None = None,
) -> dict[str, pd.DataFrame]:
    """Build season-two and season-three opportunity panels."""

    paths.ensure()
    logger = progress or ProgressLogger(enabled=False)
    archive = IpinYouArchive(paths.ipinyou_archive)
    builder = IpinYouPanelBuilder(archive, paths, config, logger)
    nrows = None if config.full_run else config.quick_rows_per_day
    season2_manifest = builder.build_season("training2nd", SEASON2_PANEL, nrows_per_day=nrows)
    season3_manifest = builder.build_season("training3rd", SEASON3_PANEL, nrows_per_day=nrows)
    daily_summary = pd.concat(
        [
            summarize_panel_directory(paths.panel_dir / SEASON2_PANEL),
            summarize_panel_directory(paths.panel_dir / SEASON3_PANEL),
        ],
        ignore_index=True,
    )
    write_table(season2_manifest, paths.table_dir / "01_ipinyou_season2_panel_manifest.csv")
    write_table(season3_manifest, paths.table_dir / "01_ipinyou_season3_panel_manifest.csv")
    write_table(daily_summary, paths.table_dir / "01_ipinyou_panel_daily_summary.csv")
    logger.done("Season-two and season-three panels are ready.")
    return {
        "season2_manifest": season2_manifest,
        "season3_manifest": season3_manifest,
        "daily_summary": daily_summary,
    }


def season2_catalog(paths: ProjectPaths) -> ReservePolicyCatalog:
    return ReservePolicyCatalog.from_panel_directory(paths.panel_dir / SEASON2_PANEL)


def _read_or_replay(
    panel_dir: Path,
    catalog: ReservePolicyCatalog,
    config: ExperimentConfig,
    paths: ProjectPaths,
    prefix: str,
    progress: ProgressLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects_path = paths.table_dir / f"{prefix}_replay_policy_effects.csv"
    daily_path = paths.metadata_dir / f"{prefix}_replay_daily_effects.csv"
    if effects_path.exists() and daily_path.exists():
        if progress is not None:
            progress.log(f"Using cached replay artifacts for {prefix}.")
        return pd.read_csv(effects_path), pd.read_csv(daily_path)
    effects, daily = evaluate_policy_catalog(panel_dir, catalog, config, progress)
    write_table(effects, effects_path)
    write_table(daily, daily_path)
    return effects, daily


def _merge_policy_metadata(frame: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = ["policy_id", "policy_number", "policy_label", "policy_family"]
    metadata = effects[metadata_cols].drop_duplicates("policy_id")
    if all(col in frame.columns for col in metadata_cols[1:]):
        return frame
    return frame.merge(metadata, on="policy_id", how="left")


def run_conservative_shortlist(
    paths: ProjectPaths,
    config: ExperimentConfig,
    progress: ProgressLogger | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Run the Section 5.1 conservative shortlist construction experiment."""

    paths.ensure()
    logger = progress or ProgressLogger(enabled=False)
    logger.step("Running Section 5.1 conservative shortlist construction.")
    catalog = season2_catalog(paths)
    registry = catalog.registry()
    effects, daily = _read_or_replay(
        paths.panel_dir / SEASON2_PANEL,
        catalog,
        config,
        paths,
        "02_season2",
        logger,
    )
    bounds = replay_lower_bounds(
        daily,
        effects,
        alpha=config.lower_bound_alpha,
        support_penalty_scale=config.support_penalty_scale,
    )
    bounds = _merge_policy_metadata(bounds, effects)
    decisions = assign_decision_labels(
        eliminate_dominated_policies(bounds, tolerance=config.shortlist_tolerance)
    )
    decisions = decisions.sort_values("lower_bound_lift", ascending=False).reset_index(drop=True)
    shortlist = decisions.query("in_validation_shortlist").copy()

    write_table(registry, paths.table_dir / "02_policy_registry.csv")
    write_table(bounds, paths.table_dir / "02_simultaneous_lower_bound_ranking.csv")
    write_table(decisions, paths.table_dir / "02_policy_decision_labels.csv")
    write_table(shortlist, paths.table_dir / "02_conservative_validation_shortlist.csv")

    figure_path = paths.figure_dir / "02_conservative_shortlist_construction.png"
    save_shortlist_construction_figure(effects, decisions, figure_path)
    logger.done("Section 5.1 artifacts are ready.")
    return {
        "policy_registry": registry,
        "replay_effects": effects,
        "replay_daily": daily,
        "bounds": bounds,
        "decisions": decisions,
        "shortlist": shortlist,
        "figure": figure_path,
    }


def _selected_policy_ids(decisions: pd.DataFrame, max_policies: int = 4) -> list[str]:
    selected = (
        decisions.query("policy_id != @BASELINE_POLICY_ID")
        .sort_values("lower_bound_lift", ascending=False)
        .head(max_policies)["policy_id"]
        .astype(str)
        .tolist()
    )
    if PRIORITY_POLICY_ID not in selected:
        selected = [PRIORITY_POLICY_ID, *selected]
    return list(dict.fromkeys(selected))[:max_policies]


def run_support_localized_threshold_resolution(
    paths: ProjectPaths,
    config: ExperimentConfig,
    progress: ProgressLogger | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Run the Section 5.2 support-localized threshold-resolution experiment."""

    paths.ensure()
    logger = progress or ProgressLogger(enabled=False)
    logger.step("Running Section 5.2 support-localized threshold resolution.")
    shortlist_artifacts = run_conservative_shortlist(paths, config, logger)
    catalog = season2_catalog(paths)
    decisions = shortlist_artifacts["decisions"]
    if not isinstance(decisions, pd.DataFrame):
        raise TypeError("decisions artifact must be a DataFrame")
    selected_policy_ids = _selected_policy_ids(decisions)

    support_sweep = threshold_radius_sweep(paths.panel_dir / SEASON2_PANEL, catalog, MAIN_THRESHOLD_RADII)
    support_sweep = _merge_policy_metadata(support_sweep, shortlist_artifacts["replay_effects"])
    support_bounds = support_explicit_lower_bound_ranking(
        shortlist_artifacts["bounds"],
        support_sweep,
        calibration_policy_id=PRIORITY_POLICY_ID,
        calibration_radius=config.support_radius,
    )
    support_bounds = _merge_policy_metadata(support_bounds, shortlist_artifacts["replay_effects"])
    certification_counts = (
        support_bounds.groupby("threshold_band_radius", as_index=False)
        .agg(
            certified_policy_count=("certifies_positive_lift", "sum"),
            max_support_adjusted_lower_bound=("support_explicit_lower_bound_lift", "max"),
            median_boundary_sample_size=("threshold_band_count", "median"),
        )
        .sort_values("threshold_band_radius")
    )
    topk_stability = topk_shortlist_stability(shortlist_artifacts["bounds"], [1, 2, 3, 5, 10])

    write_table(support_sweep, paths.table_dir / "03_threshold_radius_support_sweep.csv")
    write_table(support_bounds, paths.table_dir / "03_support_explicit_lower_bound_ranking.csv")
    write_table(certification_counts, paths.table_dir / "03_threshold_certification_counts.csv")
    write_table(topk_stability, paths.table_dir / "03_topk_shortlist_stability.csv")

    figure_path = paths.figure_dir / "03_support_localized_threshold_resolution.png"
    save_threshold_resolution_figure(support_sweep, support_bounds, figure_path, selected_policy_ids)
    logger.done("Section 5.2 artifacts are ready.")
    return {
        "support_sweep": support_sweep,
        "support_bounds": support_bounds,
        "certification_counts": certification_counts,
        "topk_stability": topk_stability,
        "selected_policy_ids": pd.DataFrame({"policy_id": selected_policy_ids}),
        "figure": figure_path,
    }


def run_validation_readiness(
    paths: ProjectPaths,
    config: ExperimentConfig,
    progress: ProgressLogger | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Run the Section 5.3 out-of-time transfer and segment-safety experiment."""

    paths.ensure()
    logger = progress or ProgressLogger(enabled=False)
    logger.step("Running Section 5.3 validation-readiness experiment.")
    transfer, season3_effects, season3_daily = frozen_catalog_season_validation(paths, config, logger)
    transfer = transfer.sort_values(["season2_rank", "season3_rank"]).reset_index(drop=True)
    write_table(transfer, paths.table_dir / "04_season2_to_season3_frozen_transfer.csv")
    write_table(season3_effects, paths.table_dir / "04_season3_replay_policy_effects.csv")
    write_table(season3_daily, paths.metadata_dir / "04_season3_replay_daily_effects.csv")

    catalog = season2_catalog(paths)
    segment_columns = ["advertiser_id", "ad_exchange", "region"]
    summaries = []
    for column in segment_columns:
        logger.log(f"Computing segment safety for {column}.")
        summary = segment_replay_lifts(
            paths.panel_dir / SEASON2_PANEL,
            catalog,
            PRIORITY_POLICY_ID,
            column,
            config,
            alpha=config.lower_bound_alpha,
        )
        write_table(summary, paths.table_dir / f"04_segment_safety_{column}.csv")
        summaries.append(summary)
    segment_summary = combine_segment_safety(
        summaries,
        alpha=config.lower_bound_alpha,
        min_observations=config.min_segment_observations,
    )
    write_table(segment_summary, paths.table_dir / "04_segment_safety_all.csv")

    readiness = pd.DataFrame(
        [
            {
                "check": "season_three_rank_of_priority_policy",
                "value": float(
                    transfer.loc[
                        transfer["policy_id"].eq(PRIORITY_POLICY_ID),
                        "season3_rank",
                    ].iloc[0]
                ),
            },
            {
                "check": "covered_segment_count",
                "value": float(len(segment_summary)),
            },
            {
                "check": "segments_with_nonnegative_lower_bar",
                "value": float(segment_summary["passes_nonharm_bar"].sum()),
            },
            {
                "check": "minimum_segment_lower_bar",
                "value": float(segment_summary["lower_bar"].min()),
            },
        ]
    )
    write_table(readiness, paths.table_dir / "04_validation_readiness_summary.csv")

    figure_path = paths.figure_dir / "04_validation_readiness_transfer_segment_safety.png"
    save_validation_readiness_figure(transfer, segment_summary, figure_path)
    logger.done("Section 5.3 artifacts are ready.")
    return {
        "transfer": transfer,
        "season3_effects": season3_effects,
        "season3_daily": season3_daily,
        "segment_summary": segment_summary,
        "readiness": readiness,
        "figure": figure_path,
    }


def _daily_lift_frame(daily: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    baseline = daily.query("policy_id == @BASELINE_POLICY_ID")[
        ["event_date", "yield_per_opportunity"]
    ].rename(columns={"yield_per_opportunity": "baseline_yield_per_opportunity"})
    result = daily.merge(baseline, on="event_date", how="left")
    result["daily_lift"] = (
        result["yield_per_opportunity"] - result["baseline_yield_per_opportunity"]
    ) / result["baseline_yield_per_opportunity"].replace(0, np.nan)
    return _merge_policy_metadata(result, effects)


def _catalog_size_sensitivity(
    daily: pd.DataFrame,
    effects: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    ranked = (
        effects.query("policy_id != @BASELINE_POLICY_ID")
        .sort_values("pct_delta_yield_per_opportunity_vs_baseline", ascending=False)["policy_id"]
        .astype(str)
        .tolist()
    )
    rows = []
    for catalog_size in [3, 5, 10, 15, len(ranked) + 1]:
        policy_ids = [BASELINE_POLICY_ID, *ranked[: catalog_size - 1]]
        sub_daily = daily[daily["policy_id"].isin(policy_ids)].copy()
        sub_effects = effects[effects["policy_id"].isin(policy_ids)].copy()
        bounds = replay_lower_bounds(
            sub_daily,
            sub_effects,
            alpha=config.lower_bound_alpha,
            support_penalty_scale=config.support_penalty_scale,
        )
        leader = bounds.sort_values("lower_bound_lift", ascending=False).iloc[0]
        rows.append(
            {
                "catalog_size": int(catalog_size),
                "leader_policy_id": leader["policy_id"],
                "leader_lower_bound_lift": float(leader["lower_bound_lift"]),
                "bonferroni_z": float(leader["bonferroni_z"]),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_replay_regret(
    daily_lifts: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = np.array(sorted(daily_lifts["event_date"].unique()))
    full_means = daily_lifts.groupby("policy_id")["daily_lift"].mean()
    full_winner = str(full_means.idxmax())
    full_best = float(full_means.max())
    rows = []
    for draw in range(iterations):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        sampled = pd.concat(
            [daily_lifts[daily_lifts["event_date"].eq(day)] for day in sampled_days],
            ignore_index=True,
        )
        means = sampled.groupby("policy_id")["daily_lift"].mean()
        winner = str(means.idxmax())
        rows.append(
            {
                "bootstrap_draw": draw,
                "selected_policy_id": winner,
                "selected_policy_full_sample_lift": float(full_means[winner]),
                "full_sample_winner": full_winner,
                "full_sample_best_lift": full_best,
                "replay_regret": float(full_best - full_means[winner]),
            }
        )
    return pd.DataFrame(rows)


def _rolling_window_rank_sensitivity(daily_lifts: pd.DataFrame) -> pd.DataFrame:
    days = sorted(daily_lifts["event_date"].unique())
    rows = []
    for day in days:
        sample = daily_lifts[daily_lifts["event_date"].ne(day)]
        means = sample.groupby("policy_id")["daily_lift"].mean().sort_values(ascending=False)
        rows.append(
            {
                "window": f"leave_out_{day}",
                "day_count": int(sample["event_date"].nunique()),
                "winner_policy_id": str(means.index[0]),
                "winner_lift": float(means.iloc[0]),
            }
        )
    for end in range(2, len(days) + 1):
        keep = days[:end]
        sample = daily_lifts[daily_lifts["event_date"].isin(keep)]
        means = sample.groupby("policy_id")["daily_lift"].mean().sort_values(ascending=False)
        rows.append(
            {
                "window": f"prefix_{end}_days",
                "day_count": int(end),
                "winner_policy_id": str(means.index[0]),
                "winner_lift": float(means.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _normalization_rank_sensitivity(effects: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "pct_delta_yield_per_opportunity_vs_baseline",
        "delta_yield_per_opportunity_vs_baseline",
        "pct_delta_fill_rate_vs_baseline",
        "pct_delta_value_proxy_per_opportunity_vs_baseline",
        "yield_per_retained_impression",
    ]
    rows = []
    for metric in metrics:
        ranked = effects.sort_values(metric, ascending=False).reset_index(drop=True)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            rows.append(
                {
                    "normalization_metric": metric,
                    "rank": rank,
                    "policy_id": row.policy_id,
                    "policy_number": row.policy_number,
                    "policy_label": row.policy_label,
                    "metric_value": float(getattr(row, metric)),
                }
            )
    return pd.DataFrame(rows)


def _shortlist_bootstrap_stability(
    daily_lifts: pd.DataFrame,
    bounds: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    days = np.array(sorted(daily_lifts["event_date"].unique()))
    policy_ids = sorted(daily_lifts["policy_id"].unique())
    rows = []
    for draw in range(iterations):
        sampled_days = rng.choice(days, size=len(days), replace=True)
        sampled = pd.concat(
            [daily_lifts[daily_lifts["event_date"].eq(day)] for day in sampled_days],
            ignore_index=True,
        )
        means = sampled.groupby("policy_id")["daily_lift"].mean().reindex(policy_ids)
        leader = str(means.idxmax())
        rows.append({"bootstrap_draw": draw, "selected_policy_id": leader})
    frame = pd.DataFrame(rows)
    frequency = (
        frame.groupby("selected_policy_id", as_index=False)
        .agg(selection_frequency=("bootstrap_draw", "count"))
        .sort_values("selection_frequency", ascending=False)
    )
    frequency["selection_frequency"] = frequency["selection_frequency"] / max(iterations, 1)
    return _merge_policy_metadata(frequency.rename(columns={"selected_policy_id": "policy_id"}), bounds)


def _pairwise_boundary_support_from_panels(
    paths: ProjectPaths,
    catalog: ReservePolicyCatalog,
    radius: float,
    effects: pd.DataFrame,
    progress: ProgressLogger | None,
) -> pd.DataFrame:
    frames = []
    for parquet in sorted((paths.panel_dir / SEASON2_PANEL).glob("*.parquet")):
        if progress is not None:
            progress.log(f"Computing pairwise boundary support on {parquet.name}.")
        frame = pd.read_parquet(parquet, columns=["slot_floor_price", "bid_floor_gap", "bid_price", "filled"])
        frames.append(pairwise_boundary_support(frame, catalog, radius=radius))
    pairwise = summarize_pairwise_boundary_support(frames)
    lift_map = effects.set_index("policy_id")["pct_delta_yield_per_opportunity_vs_baseline"].to_dict()
    pairwise["left_replay_lift"] = pairwise["left_policy_id"].map(lift_map)
    pairwise["right_replay_lift"] = pairwise["right_policy_id"].map(lift_map)
    pairwise["absolute_replay_lift_gap"] = (
        pairwise["left_replay_lift"] - pairwise["right_replay_lift"]
    ).abs()
    pairwise["threshold_distance_to_margin_ratio"] = pairwise["mean_floor_distance"] / pairwise[
        "absolute_replay_lift_gap"
    ].replace(0, np.nan)
    return pairwise


def _response_gap_sensitivity(transfer: pd.DataFrame) -> pd.DataFrame:
    ranked = transfer.sort_values("season2_rank").query("policy_id != @BASELINE_POLICY_ID").head(2)
    leader = ranked.iloc[0]
    runner_up = ranked.iloc[1]
    margin = float(leader["season2_replay_lift"] - runner_up["season2_replay_lift"])
    rows = []
    for response_gap in np.linspace(0, max(0.5, margin * 1.5), 21):
        rows.append(
            {
                "leader_policy_id": leader["policy_id"],
                "runner_up_policy_id": runner_up["policy_id"],
                "season2_replay_margin": margin,
                "hypothetical_pairwise_response_gap": float(response_gap),
                "ranking_preserved_under_symmetric_gap": bool(margin > 2.0 * response_gap),
            }
        )
    return pd.DataFrame(rows)


def _stress_test_certification(
    daily: pd.DataFrame,
    effects: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows = []
    for alpha in [0.01, 0.05, 0.10]:
        for support_penalty_scale in [0.0, 0.5, 1.0, 2.0]:
            bounds = replay_lower_bounds(
                daily,
                effects,
                alpha=alpha,
                support_penalty_scale=support_penalty_scale,
            )
            labels = assign_decision_labels(eliminate_dominated_policies(bounds))
            rows.append(
                {
                    "alpha": alpha,
                    "support_penalty_scale": support_penalty_scale,
                    "certified_count": int(labels["decision_label"].eq("certified").sum()),
                    "dominated_count": int(labels["decision_label"].eq("dominated").sum()),
                    "unresolved_count": int(labels["decision_label"].eq("unresolved").sum()),
                    "shortlist_size": int(labels["in_validation_shortlist"].sum()),
                    "leader_policy_id": str(labels.sort_values("lower_bound_lift", ascending=False).iloc[0]["policy_id"]),
                }
            )
    return pd.DataFrame(rows)


def run_appendix_b_diagnostics(
    paths: ProjectPaths,
    config: ExperimentConfig,
    progress: ProgressLogger | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Run the Appendix B supplementary diagnostics and robustness checks."""

    paths.ensure()
    logger = progress or ProgressLogger(enabled=False)
    logger.step("Running Appendix B diagnostics.")
    shortlist = run_conservative_shortlist(paths, config, logger)
    threshold = run_support_localized_threshold_resolution(paths, config, logger)
    validation = run_validation_readiness(paths, config, logger)
    catalog = season2_catalog(paths)
    effects = shortlist["replay_effects"]
    daily = shortlist["replay_daily"]
    bounds = shortlist["bounds"]
    decisions = shortlist["decisions"]
    if not all(isinstance(x, pd.DataFrame) for x in [effects, daily, bounds, decisions]):
        raise TypeError("Main Section 5 artifacts were not produced as DataFrames.")

    daily_lifts = _daily_lift_frame(daily, effects)
    catalog_sensitivity = _catalog_size_sensitivity(daily, effects, config)
    bootstrap_regret = _bootstrap_replay_regret(
        daily_lifts,
        iterations=config.bootstrap_iterations,
        seed=config.random_seed,
    )
    rolling_sensitivity = _rolling_window_rank_sensitivity(daily_lifts)
    normalization_sensitivity = _normalization_rank_sensitivity(effects)

    support_sweep_fine = threshold_radius_sweep(
        paths.panel_dir / SEASON2_PANEL,
        catalog,
        APPENDIX_THRESHOLD_RADII,
    )
    support_sweep_fine = _merge_policy_metadata(support_sweep_fine, effects)
    pairwise = _pairwise_boundary_support_from_panels(
        paths,
        catalog,
        radius=config.support_radius,
        effects=effects,
        progress=logger,
    )

    tolerances = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
    shortlist_sensitivity = shortlist_size_sensitivity(bounds, tolerances)
    ablation = ipinyou_decision_rule_ablation(effects, bounds, decisions)
    dominated = decisions.query("decision_label == 'dominated'").copy()
    bootstrap_selection = _shortlist_bootstrap_stability(
        daily_lifts,
        bounds,
        iterations=config.bootstrap_iterations,
        seed=config.random_seed,
    )

    expanded_segments = []
    for column in ["advertiser_id", "ad_exchange", "region", "inventory_category", "bid_gap_bucket"]:
        logger.log(f"Computing Appendix B expanded segment diagnostics for {column}.")
        expanded_segments.append(
            segment_replay_lifts(
                paths.panel_dir / SEASON2_PANEL,
                catalog,
                PRIORITY_POLICY_ID,
                column,
                config,
                alpha=config.lower_bound_alpha,
            )
        )
    expanded_segment_summary = combine_segment_safety(
        expanded_segments,
        alpha=config.lower_bound_alpha,
        min_observations=config.min_segment_observations,
    )
    radius_sensitivity = segment_radius_sensitivity(expanded_segment_summary, [0.0, 0.25, 0.5, 1.0, 2.0])
    multiplicity = segment_multiplicity_scaling(
        expanded_segment_summary,
        [5, 10, 20, 44, 75, 100, max(len(expanded_segment_summary), 1)],
        alpha=config.lower_bound_alpha,
    )
    sparse_segments = expanded_segment_summary.query("passes_nonharm_bar == False").copy()

    transfer = validation["transfer"]
    if not isinstance(transfer, pd.DataFrame):
        raise TypeError("transfer artifact must be a DataFrame")
    rank_stability = pd.DataFrame(
        [
            {
                "comparison": "season2_vs_season3",
                "spearman_rank_correlation": float(
                    transfer[["season2_rank", "season3_rank"]].corr(method="spearman").iloc[0, 1]
                ),
                "top5_overlap": int(
                    len(
                        set(transfer.nsmallest(5, "season2_rank")["policy_id"])
                        & set(transfer.nsmallest(5, "season3_rank")["policy_id"])
                    )
                ),
            }
        ]
    )
    response_gap = _response_gap_sensitivity(transfer)
    stress_tests = _stress_test_certification(daily, effects, config)
    implementation = pd.DataFrame(
        [
            {
                "artifact": "policy_catalog",
                "description": "Logged baseline, uniform increases, quantile floors, and margin-gated reserve rules.",
                "row_count": int(len(catalog.registry())),
            },
            {
                "artifact": "season2_panel",
                "description": "Development replay panel built from iPinYou training2nd bid and impression logs.",
                "row_count": int(effects["opportunities"].max()),
            },
            {
                "artifact": "season3_panel",
                "description": "Frozen out-of-time replay panel built from iPinYou training3rd logs.",
                "row_count": int(validation["season3_effects"]["opportunities"].max())
                if isinstance(validation["season3_effects"], pd.DataFrame)
                else 0,
            },
        ]
    )

    tables = {
        "05_daily_replay_lifts.csv": daily_lifts,
        "05_catalog_size_sensitivity.csv": catalog_sensitivity,
        "05_bootstrap_replay_regret.csv": bootstrap_regret,
        "05_rolling_window_rank_sensitivity.csv": rolling_sensitivity,
        "05_normalization_rank_sensitivity.csv": normalization_sensitivity,
        "05_fine_threshold_radius_support_sweep.csv": support_sweep_fine,
        "05_pairwise_boundary_support.csv": pairwise,
        "05_shortlist_size_sensitivity.csv": shortlist_sensitivity,
        "05_decision_rule_ablation.csv": ablation,
        "05_dominated_policy_diagnostics.csv": dominated,
        "05_bootstrap_shortlist_selection.csv": bootstrap_selection,
        "05_expanded_segment_safety.csv": expanded_segment_summary,
        "05_segment_radius_sensitivity.csv": radius_sensitivity,
        "05_segment_multiplicity_scaling.csv": multiplicity,
        "05_sparse_segment_behavior.csv": sparse_segments,
        "05_rank_stability.csv": rank_stability,
        "05_response_gap_sensitivity.csv": response_gap,
        "05_validation_readiness_stress_tests.csv": stress_tests,
        "05_implementation_manifest.csv": implementation,
    }
    written = {name: write_table(frame, paths.table_dir / name) for name, frame in tables.items()}

    replay_fig = paths.figure_dir / "05_appendix_replay_diagnostics.png"
    support_fig = paths.figure_dir / "05_appendix_support_pair_diagnostics.png"
    decision_fig = paths.figure_dir / "05_appendix_decision_robustness.png"
    save_appendix_replay_diagnostics(daily_lifts, catalog_sensitivity, replay_fig)
    save_appendix_support_pair_figure(pairwise, support_fig)
    save_appendix_decision_robustness_figure(shortlist_sensitivity, ablation, decision_fig)

    logger.done("Appendix B artifacts are ready.")
    return {
        **written,
        "appendix_replay_figure": replay_fig,
        "appendix_support_figure": support_fig,
        "appendix_decision_figure": decision_fig,
    }
