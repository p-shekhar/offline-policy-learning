from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import ExperimentConfig, ProjectPaths
from data_access import IPINYOU_BID_COLUMNS, IPINYOU_EVENT_COLUMNS, IpinYouArchive
from progress import ProgressLogger


class IpinYouPanelBuilder:
    """Builds auction-opportunity panels from iPinYou bid/impression/click/conversion logs."""

    numeric_columns = [
        "region",
        "city",
        "ad_exchange",
        "slot_width",
        "slot_height",
        "slot_floor_price",
        "bid_price",
        "advertiser_id",
        "pay_price",
    ]

    categorical_columns = [
        "region",
        "city",
        "ad_exchange",
        "advertiser_id",
        "slot_visibility",
        "slot_format",
    ]

    def __init__(
        self,
        archive: IpinYouArchive,
        paths: ProjectPaths,
        config: ExperimentConfig,
        progress: ProgressLogger | None = None,
    ) -> None:
        self.archive = archive
        self.paths = paths
        self.config = config
        self.progress = progress or ProgressLogger(enabled=False)

    def _members_for_day(self, season: str, date_label: str) -> dict[str, str]:
        matrix = self.archive.season_matrix(season)
        row = matrix.loc[matrix["date_label"].astype(str).eq(str(date_label))]
        if row.empty:
            raise FileNotFoundError(f"No archive members for {season} date {date_label}.")
        return {
            kind: str(row.iloc[0][kind])
            for kind in ["bid", "imp", "clk", "conv"]
            if kind in row.columns and pd.notna(row.iloc[0][kind])
        }

    @staticmethod
    def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
        stamp = frame["timestamp"].astype(str).str.slice(0, 14)
        frame["event_time"] = pd.to_datetime(stamp, format="%Y%m%d%H%M%S", errors="coerce")
        frame["event_date"] = frame["event_time"].dt.date.astype(str)
        frame["hour"] = frame["event_time"].dt.hour.fillna(-1).astype(int)
        frame["day_of_week"] = frame["event_time"].dt.dayofweek.fillna(-1).astype(int)
        return frame

    def engineer(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in self.numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = self._add_time_features(frame)
        frame["slot_floor_price"] = frame["slot_floor_price"].fillna(0.0).clip(lower=0.0)
        frame["bid_price"] = frame["bid_price"].fillna(0.0).clip(lower=0.0)
        frame["pay_price"] = frame["pay_price"].fillna(0.0).clip(lower=0.0)
        frame["slot_area"] = frame["slot_width"].fillna(0.0) * frame["slot_height"].fillna(0.0)
        frame["bid_floor_gap"] = frame["bid_price"] - frame["slot_floor_price"]
        frame["floor_to_bid_ratio"] = np.where(
            frame["bid_price"].gt(0), frame["slot_floor_price"] / frame["bid_price"], 0.0
        )
        clean_tags = frame["user_tags"].fillna("").astype(str)
        clean_tags = clean_tags.where(~clean_tags.isin(["null", "nan", "None", "0"]), "")
        frame["user_tag_count"] = clean_tags.map(lambda value: 0 if value == "" else len(value.split(",")))
        frame["has_user_tags"] = frame["user_tag_count"].gt(0).astype(int)
        for column in self.categorical_columns:
            frame[column] = frame[column].astype("string").fillna("missing")
        frame["support_cluster"] = (
            frame["ad_exchange"].astype(str)
            + "|"
            + frame["region"].astype(str)
            + "|"
            + frame["slot_visibility"].astype(str)
            + "|"
            + frame["slot_format"].astype(str)
        )
        return frame

    def build_day(self, season: str, date_label: str, nrows: int | None = None) -> tuple[pd.DataFrame, dict]:
        start = time.perf_counter()
        mode = "full day" if nrows is None else f"first {nrows:,} bid rows"
        self.progress.log(f"Building iPinYou {season} {date_label} panel ({mode}).")
        members = self._members_for_day(season, date_label)
        bid = self.archive.read_tsv_bz2(members["bid"], IPINYOU_BID_COLUMNS, nrows=nrows)
        bid = bid.drop_duplicates("bid_id", keep="first")
        imp = self.archive.read_tsv_bz2(
            members["imp"], IPINYOU_EVENT_COLUMNS, usecols=["bid_id", "pay_price"]
        ).drop_duplicates("bid_id", keep="last")
        click_ids = (
            set(
                self.archive.read_tsv_bz2(members["clk"], IPINYOU_EVENT_COLUMNS, usecols=["bid_id"])[
                    "bid_id"
                ].dropna()
            )
            if "clk" in members
            else set()
        )
        conv_ids = (
            set(
                self.archive.read_tsv_bz2(members["conv"], IPINYOU_EVENT_COLUMNS, usecols=["bid_id"])[
                    "bid_id"
                ].dropna()
            )
            if "conv" in members
            else set()
        )
        frame = bid.merge(imp, on="bid_id", how="left", validate="one_to_one")
        frame["season"] = season
        frame["date_label"] = str(date_label)
        frame["filled"] = frame["pay_price"].notna().astype(int)
        frame["clicked"] = frame["bid_id"].isin(click_ids).astype(int) * frame["filled"]
        frame["converted"] = frame["bid_id"].isin(conv_ids).astype(int) * frame["filled"]
        frame["value_proxy"] = (
            frame["clicked"] + self.config.value_proxy_conversion_weight * frame["converted"]
        )
        frame = self.engineer(frame)
        manifest = {
            "dataset": "ipinyou",
            "season": season,
            "date_label": str(date_label),
            "run_mode": "full" if nrows is None else "quick",
            "bid_rows": int(len(frame)),
            "filled": int(frame["filled"].sum()),
            "clicks": int(frame["clicked"].sum()),
            "conversions": int(frame["converted"].sum()),
            "fill_rate": float(frame["filled"].mean()),
            "elapsed_seconds": round(time.perf_counter() - start, 2),
        }
        return frame, manifest

    def build_season(
        self,
        season: str,
        output_name: str,
        nrows_per_day: int | None = None,
    ) -> pd.DataFrame:
        self.paths.ensure()
        matrix = self.archive.season_matrix(season)
        matrix.to_csv(self.paths.metadata_dir / f"{output_name}_file_matrix.csv", index=False)
        dates = matrix.dropna(subset=["bid", "imp"])["date_label"].astype(str).tolist()
        output_dir = self.paths.panel_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        manifests = []
        for date_label in dates:
            frame, manifest = self.build_day(season, date_label, nrows=nrows_per_day)
            out = output_dir / f"{output_name}_{date_label}.parquet"
            frame.to_parquet(out, index=False)
            manifest["panel_path"] = str(out.relative_to(self.paths.project_root))
            manifests.append(manifest)
            self.progress.log(
                f"Wrote {out.name}: {manifest['bid_rows']:,} rows, "
                f"{manifest['filled']:,} fills."
            )
        manifest_df = pd.DataFrame(manifests)
        manifest_df.to_csv(self.paths.metadata_dir / f"{output_name}_panel_manifest.csv", index=False)
        return manifest_df


def summarize_panel_directory(panel_dir: Path) -> pd.DataFrame:
    rows = []
    for parquet in sorted(panel_dir.glob("*.parquet")):
        frame = pd.read_parquet(
            parquet,
            columns=["season", "date_label", "event_date", "bid_id", "filled", "clicked", "converted"],
        )
        rows.append(
            {
                "panel": panel_dir.name,
                "file": parquet.name,
                "season": frame["season"].iloc[0],
                "date_label": frame["date_label"].iloc[0],
                "event_date": frame["event_date"].iloc[0],
                "opportunities": len(frame),
                "filled": int(frame["filled"].sum()),
                "clicks": int(frame["clicked"].sum()),
                "conversions": int(frame["converted"].sum()),
                "fill_rate": float(frame["filled"].mean()),
            }
        )
    return pd.DataFrame(rows)
