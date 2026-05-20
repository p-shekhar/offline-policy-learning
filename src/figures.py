from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.titlepad": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
        }
    )


def save_replay_frontier(effects: pd.DataFrame, out: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    plot = effects.query("policy_id != 'logged_floor_status_quo'").copy()
    sns.scatterplot(
        data=plot,
        x="retained_impression_share",
        y="pct_delta_yield_per_opportunity_vs_baseline",
        hue="policy_family",
        ax=ax,
        s=55,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Retained impression share")
    ax.set_ylabel("Replay yield lift vs. logged baseline")
    ax.legend(title="Policy family", frameon=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_shortlist_construction_figure(
    effects: pd.DataFrame,
    decisions: pd.DataFrame,
    out: Path,
    top_n: int = 12,
) -> None:
    """Save the Section 5.1 two-panel replay frontier and decision plot."""

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))

    frontier = effects.query("policy_id != 'logged_floor_status_quo'").copy()
    sns.scatterplot(
        data=frontier,
        x="retained_impression_share",
        y="pct_delta_yield_per_opportunity_vs_baseline",
        hue="policy_family",
        ax=axes[0],
        s=48,
    )
    for row in frontier.sort_values("pct_delta_yield_per_opportunity_vs_baseline", ascending=False).head(6).itertuples(index=False):
        axes[0].text(
            row.retained_impression_share,
            row.pct_delta_yield_per_opportunity_vs_baseline,
            str(row.policy_number),
            fontsize=10,
            ha="left",
            va="bottom",
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("(a) Replay frontier")
    axes[0].set_xlabel("Retained impression share")
    axes[0].set_ylabel("Replay yield lift vs. logged baseline")
    axes[0].legend(title="Policy family", frameon=True, fontsize=9, title_fontsize=10)

    plot = decisions.sort_values("lower_bound_lift", ascending=False).head(top_n).iloc[::-1].copy()
    colors = {
        "certified": "#1b9e77",
        "unresolved": "#7570b3",
        "dominated": "#d95f02",
    }
    for label, group in plot.groupby("decision_label", sort=False):
        axes[1].errorbar(
            group["daily_mean_lift"],
            group["policy_number"] + ": " + group["policy_label"],
            xerr=group["simultaneous_radius"],
            fmt="o",
            capsize=3,
            color=colors.get(label, "#555555"),
            label=label.title(),
        )
    axes[1].scatter(
        plot["lower_bound_lift"],
        plot["policy_number"] + ": " + plot["policy_label"],
        color="black",
        s=18,
        label="Lower bound",
        zorder=3,
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("(b) Simultaneous lower-bound ranking")
    axes[1].set_xlabel("Lift vs. logged baseline")
    axes[1].set_ylabel("")
    axes[1].legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_lower_bound_ranking(bounds: pd.DataFrame, out: Path, top_n: int = 12) -> None:
    set_style()
    plot = bounds.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.errorbar(
        plot["daily_mean_lift"],
        plot["policy_id"],
        xerr=plot["simultaneous_radius"],
        fmt="o",
        capsize=3,
        color="#2f5597",
    )
    ax.scatter(plot["lower_bound_lift"], plot["policy_id"], color="#c00000", label="Support-adjusted lower bound")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lift vs. logged baseline")
    ax.set_ylabel("")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_segment_safety(summary: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    set_style()
    plot = summary.sort_values("lower_bar").head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    xerr = [plot["mean_lift"] - plot["lower_bar"], plot["upper_bar"] - plot["mean_lift"]]
    ax.errorbar(plot["mean_lift"], plot["segment"], xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Segment replay lift with confidence bars")
    ax.set_ylabel("Segment")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_threshold_radius_sweep(sweep: pd.DataFrame, out: Path, policy_ids: list[str]) -> None:
    set_style()
    plot = sweep[sweep["policy_id"].isin(policy_ids)].copy()
    curves = []
    for _, group in plot.groupby("policy_id", sort=False):
        curve = group.sort_values("threshold_band_radius")
        key = tuple(
            zip(
                curve["threshold_band_radius"].tolist(),
                curve["threshold_band_count"].tolist(),
            )
        )
        curves.append((key, curve))

    grouped_curves: dict[tuple, list[pd.DataFrame]] = {}
    for key, curve in curves:
        grouped_curves.setdefault(key, []).append(curve)

    collapsed = []
    for curve_group in grouped_curves.values():
        base = curve_group[0].copy()
        labels = []
        for curve in curve_group:
            policy_number = str(curve["policy_number"].iloc[0])
            policy_label = str(curve["policy_label"].iloc[0])
            labels.append(f"{policy_number}: {policy_label}")
        if len(labels) == 1:
            base["curve_label"] = labels[0]
        else:
            policy_numbers = "/".join(str(curve["policy_number"].iloc[0]) for curve in curve_group)
            base["curve_label"] = f"{policy_numbers}: shared threshold support"
        collapsed.append(base)

    plot = pd.concat(collapsed, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.lineplot(
        data=plot,
        x="threshold_band_radius",
        y="threshold_band_count",
        hue="curve_label",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Threshold radius")
    ax.set_ylabel("Observations in bid-threshold band")
    ax.legend(title="Policy", frameon=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_threshold_resolution_figure(
    support_sweep: pd.DataFrame,
    support_bounds: pd.DataFrame,
    out: Path,
    policy_ids: list[str],
) -> None:
    """Save the Section 5.2 two-panel support-resolution figure."""

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))
    left = support_sweep[support_sweep["policy_id"].isin(policy_ids)].copy()
    right = support_bounds[support_bounds["policy_id"].isin(policy_ids)].copy()
    label_map = left.drop_duplicates("policy_id").set_index("policy_id")["policy_label"].to_dict()
    left["curve_label"] = left["policy_id"].map(label_map)
    right["curve_label"] = right["policy_id"].map(label_map)

    sns.lineplot(
        data=left,
        x="threshold_band_radius",
        y="threshold_band_count",
        hue="curve_label",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Boundary support")
    axes[0].set_xlabel("Threshold radius")
    axes[0].set_ylabel("Effective boundary sample size")
    axes[0].legend(title="Policy", frameon=True, fontsize=9, title_fontsize=10)

    sns.lineplot(
        data=right,
        x="threshold_band_radius",
        y="support_explicit_lower_bound_lift",
        hue="curve_label",
        marker="o",
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_title("(b) Support-adjusted lower bounds")
    axes[1].set_xlabel("Threshold radius")
    axes[1].set_ylabel("Lower-bound lift")
    axes[1].legend(title="Policy", frameon=True, fontsize=9, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_support_explicit_lower_bounds(
    ranking: pd.DataFrame,
    out: Path,
    policy_ids: list[str],
) -> None:
    set_style()
    plot = ranking[ranking["policy_id"].isin(policy_ids)].copy()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.lineplot(
        data=plot,
        x="threshold_band_radius",
        y="support_explicit_lower_bound_lift",
        hue="policy_id",
        marker="o",
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Threshold radius")
    ax.set_ylabel("Support-explicit lower-bound lift")
    ax.legend(title="Policy", frameon=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_season_transfer(transfer: pd.DataFrame, out: Path, top_n: int = 12) -> None:
    set_style()
    plot = transfer.sort_values("season2_rank").head(top_n).copy()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.scatter(plot["season2_replay_lift"], plot["season3_replay_lift"], color="#4472c4")
    for row in plot.itertuples(index=False):
        ax.text(row.season2_replay_lift, row.season3_replay_lift, str(row.policy_number), fontsize=10)
    lim_min = min(plot["season2_replay_lift"].min(), plot["season3_replay_lift"].min())
    lim_max = max(plot["season2_replay_lift"].max(), plot["season3_replay_lift"].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Season 2 replay lift")
    ax.set_ylabel("Season 3 replay lift")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_validation_readiness_figure(
    transfer: pd.DataFrame,
    segment_summary: pd.DataFrame,
    out: Path,
    top_n: int = 10,
    worst_segments: int = 18,
) -> None:
    """Save the Section 5.3 two-panel transfer and segment-safety figure."""

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
    plot = transfer.sort_values("season2_rank").head(top_n).copy()
    axes[0].scatter(plot["season2_replay_lift"], plot["season3_replay_lift"], color="#4472c4")
    for row in plot.itertuples(index=False):
        axes[0].text(
            row.season2_replay_lift,
            row.season3_replay_lift,
            str(row.policy_number),
            fontsize=10,
            ha="left",
            va="bottom",
        )
    lim_min = min(plot["season2_replay_lift"].min(), plot["season3_replay_lift"].min())
    lim_max = max(plot["season2_replay_lift"].max(), plot["season3_replay_lift"].max())
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("(a) Frozen season-three transfer")
    axes[0].set_xlabel("Season 2 replay lift")
    axes[0].set_ylabel("Season 3 replay lift")

    seg = segment_summary.sort_values("lower_bar").head(worst_segments).iloc[::-1].copy()
    seg["display_segment"] = seg["segment_column"].astype(str) + "=" + seg["segment"].astype(str)
    xerr = [seg["mean_lift"] - seg["lower_bar"], seg["upper_bar"] - seg["mean_lift"]]
    axes[1].errorbar(seg["mean_lift"], seg["display_segment"], xerr=xerr, fmt="o", capsize=3)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("(b) Worst covered segment lower bars")
    axes[1].set_xlabel("Segment replay lift")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_appendix_replay_diagnostics(
    daily_lifts: pd.DataFrame,
    catalog_sensitivity: pd.DataFrame,
    out: Path,
) -> None:
    """Save compact Appendix B replay concentration diagnostics."""

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))
    top = (
        daily_lifts.groupby("policy_id", as_index=False)["daily_lift"].mean()
        .sort_values("daily_lift", ascending=False)
        .head(8)["policy_id"]
    )
    plot = daily_lifts[daily_lifts["policy_id"].isin(top)].copy()
    sns.boxplot(data=plot, x="daily_lift", y="policy_label", ax=axes[0], color="#9ecae1")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("(a) Daily replay lift dispersion")
    axes[0].set_xlabel("Daily lift")
    axes[0].set_ylabel("")

    sns.lineplot(
        data=catalog_sensitivity,
        x="catalog_size",
        y="bonferroni_z",
        marker="o",
        ax=axes[1],
        label="Bonferroni z",
    )
    ax2 = axes[1].twinx()
    sns.lineplot(
        data=catalog_sensitivity,
        x="catalog_size",
        y="leader_lower_bound_lift",
        marker="s",
        color="#d95f02",
        ax=ax2,
        label="Leader lower bound",
    )
    axes[1].set_title("(b) Catalog-size sensitivity")
    axes[1].set_xlabel("Catalog size")
    axes[1].set_ylabel("Bonferroni critical value")
    ax2.set_ylabel("Leader lower-bound lift")
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, loc="best", frameon=True)
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_appendix_support_pair_figure(pairwise: pd.DataFrame, out: Path) -> None:
    """Save Appendix B pairwise boundary-support and margin diagnostics."""

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))
    sns.histplot(data=pairwise, x="boundary_share", bins=30, ax=axes[0], color="#4472c4")
    axes[0].set_title("(a) Pairwise boundary-support distribution")
    axes[0].set_xlabel("Boundary support share")
    axes[0].set_ylabel("Policy pairs")
    axes[0].legend(["Policy-pair count"], frameon=True, loc="best")

    sns.scatterplot(
        data=pairwise,
        x="mean_floor_distance",
        y="absolute_replay_lift_gap",
        size="boundary_share",
        sizes=(15, 120),
        ax=axes[1],
        color="#70ad47",
        legend="brief",
    )
    axes[1].set_title("(b) Resolution margin and threshold distance")
    axes[1].set_xlabel("Mean candidate-floor distance")
    axes[1].set_ylabel("Absolute replay-lift gap")
    axes[1].legend(title="Boundary support share", frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_appendix_decision_robustness_figure(
    shortlist_sensitivity: pd.DataFrame,
    ablation: pd.DataFrame,
    out: Path,
) -> None:
    """Save Appendix B shortlist and decision-rule robustness diagnostics."""

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))
    sns.lineplot(
        data=shortlist_sensitivity,
        x="shortlist_tolerance",
        y="shortlist_size",
        marker="o",
        ax=axes[0],
        label="Retained shortlist size",
    )
    axes[0].set_title("(a) Shortlist-size sensitivity")
    axes[0].set_xlabel("Elimination tolerance")
    axes[0].set_ylabel("Retained shortlist size")
    axes[0].legend(frameon=True, loc="best")

    plot = ablation.copy()
    plot["rule_type"] = plot["shortlist_size"].map(lambda value: "Single selected policy" if value == 1 else "Validation shortlist")
    sns.barplot(
        data=plot,
        x="shortlist_size",
        y="decision_rule",
        hue="rule_type",
        dodge=False,
        ax=axes[1],
        palette={
            "Single selected policy": "#9ecae1",
            "Validation shortlist": "#70ad47",
        },
    )
    axes[1].set_title("(b) Alternative decision rules")
    axes[1].set_xlabel("Policies selected or retained")
    axes[1].set_ylabel("")
    axes[1].legend(title="Rule output", frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_q_localized_selection_figure(
    ranking: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    out: Path,
    policy_ids: list[str],
) -> None:
    """Save q-localized ranking and resampling stability diagnostics."""

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))
    plot = ranking[ranking["policy_id"].isin(policy_ids)].copy()
    plot["q_percent"] = 100.0 * plot["q"]
    sns.lineplot(
        data=plot,
        x="q_percent",
        y="localized_boundary_lift",
        hue="policy_number",
        style="policy_number",
        markers=True,
        dashes=False,
        ax=axes[0],
    )
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("(a) q-localized replay score")
    axes[0].set_xlabel("Boundary evidence fraction q (%)")
    axes[0].set_ylabel("Localized boundary lift")
    axes[0].legend(title="Policy", frameon=True, fontsize=9, title_fontsize=10)

    boot = bootstrap_summary[bootstrap_summary["selected_policy_id"].isin(policy_ids)].copy()
    boot["q_percent"] = 100.0 * boot["q"]
    sns.barplot(
        data=boot,
        x="q_percent",
        y="selection_frequency",
        hue="selected_policy_number",
        ax=axes[1],
    )
    axes[1].set_title("(b) Day-bootstrap winner frequency")
    axes[1].set_xlabel("Boundary evidence fraction q (%)")
    axes[1].set_ylabel("Selection frequency")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(title="Selected policy", frameon=True, fontsize=9, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_q_localized_transfer_figure(
    transfer: pd.DataFrame,
    out: Path,
) -> None:
    """Save season-three transfer for q-localized season-two selections."""

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))
    plot = transfer.copy()
    plot["q_percent"] = 100.0 * plot["q"]
    sns.scatterplot(
        data=plot,
        x="season2_localized_boundary_lift",
        y="season3_full_replay_lift",
        hue="policy_number",
        style="selection_source",
        s=90,
        ax=axes[0],
    )
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("(a) Season-three lift of localized selections")
    axes[0].set_xlabel("Season 2 q-localized boundary lift")
    axes[0].set_ylabel("Season 3 full replay lift")
    axes[0].legend(title="Policy / source", frameon=True, fontsize=9, title_fontsize=10)

    sns.lineplot(
        data=plot,
        x="q_percent",
        y="season3_full_replay_lift",
        hue="policy_number",
        marker="o",
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("(b) Out-of-time lift by q")
    axes[1].set_xlabel("Boundary evidence fraction q (%)")
    axes[1].set_ylabel("Season 3 full replay lift")
    axes[1].legend(title="Policy", frameon=True, fontsize=9, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_decision_ablation(ablation: pd.DataFrame, out: Path) -> None:
    set_style()
    plot = ablation.copy()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = plot["certifies_nonnegative_lift"].map({True: "#70ad47", False: "#c00000"})
    ax.barh(plot["decision_rule"], plot["shortlist_size"], color=colors)
    ax.set_xlabel("Selected/retained policies")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
