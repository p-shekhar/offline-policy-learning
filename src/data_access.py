from __future__ import annotations

import bz2
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


IPINYOU_BID_COLUMNS = [
    "bid_id",
    "timestamp",
    "user_id",
    "user_agent",
    "ip",
    "region",
    "city",
    "ad_exchange",
    "domain",
    "url",
    "url_id",
    "slot_id",
    "slot_width",
    "slot_height",
    "slot_visibility",
    "slot_format",
    "slot_floor_price",
    "creative_id",
    "bid_price",
    "advertiser_id",
    "user_tags",
]

IPINYOU_EVENT_COLUMNS = [
    "bid_id",
    "timestamp",
    "log_type",
    "user_id",
    "user_agent",
    "ip",
    "region",
    "city",
    "ad_exchange",
    "domain",
    "url",
    "url_id",
    "slot_id",
    "slot_width",
    "slot_height",
    "slot_visibility",
    "slot_format",
    "slot_floor_price",
    "creative_id",
    "bid_price",
    "pay_price",
    "key_page",
    "advertiser_id",
    "user_tags",
]


@dataclass(frozen=True)
class IpinYouArchive:
    """Reader for the original iPinYou contest archive."""

    archive_path: Path

    member_pattern = re.compile(
        r"ipinyou\.contest\.dataset/(training(?:1st|2nd|3rd))/(bid|imp|clk|conv)\.(\d{8})\.txt\.bz2$"
    )

    def __post_init__(self) -> None:
        if not self.archive_path.exists():
            raise FileNotFoundError(f"Missing iPinYou archive: {self.archive_path}")

    def inventory(self) -> pd.DataFrame:
        rows = []
        with zipfile.ZipFile(self.archive_path) as zf:
            for member in zf.namelist():
                match = self.member_pattern.search(member)
                if match is None:
                    continue
                season, kind, date_label = match.groups()
                info = zf.getinfo(member)
                rows.append(
                    {
                        "dataset": "ipinyou",
                        "season": season,
                        "kind": kind,
                        "date_label": date_label,
                        "member": member,
                        "compressed_size": info.compress_size,
                        "uncompressed_size": info.file_size,
                    }
                )
        return pd.DataFrame(rows).sort_values(["season", "date_label", "kind"]).reset_index(drop=True)

    def season_matrix(self, season: str) -> pd.DataFrame:
        inv = self.inventory().query("season == @season").copy()
        return (
            inv.pivot_table(index="date_label", columns="kind", values="member", aggfunc="first")
            .reset_index()
            .sort_values("date_label")
            .reset_index(drop=True)
        )

    def read_tsv_bz2(
        self,
        member: str,
        columns: list[str],
        usecols: list[str] | None = None,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        with zipfile.ZipFile(self.archive_path) as zf:
            with zf.open(member) as zipped:
                with bz2.open(zipped, "rt", encoding="utf-8", errors="replace") as handle:
                    return pd.read_csv(
                        handle,
                        sep="\t",
                        names=columns,
                        usecols=usecols,
                        nrows=nrows,
                        dtype=str,
                        low_memory=False,
                    )

