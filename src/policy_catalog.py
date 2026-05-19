from __future__ import annotations

import numpy as np
import pandas as pd


BASELINE_POLICY_ID = "logged_floor_status_quo"
PRIORITY_POLICY_ID = "hybrid_q75_if_gap_100"

POLICY_LABELS = {
    BASELINE_POLICY_ID: "Logged Status Quo",
    "uniform_raise_05pct": "Uniform +5%",
    "uniform_raise_10pct": "Uniform +10%",
    "uniform_raise_15pct": "Uniform +15%",
    "uniform_raise_20pct": "Uniform +20%",
    "uniform_raise_30pct": "Uniform +30%",
    "add_5_all_floors": "Add 5 To All Floors",
    "add_10_all_floors": "Add 10 To All Floors",
    "add_20_all_floors": "Add 20 To All Floors",
    "min_positive_floor_q25": "Positive Floors To Q25",
    "min_positive_floor_q50": "Positive Floors To Q50",
    "min_positive_floor_q75": "Positive Floors To Q75",
    "zero_and_low_floor_to_q25": "All Low Floors To Q25",
    "zero_and_low_floor_to_q50": "All Low Floors To Q50",
    "margin_gap_25_add_5": "Gap 25 Add 5",
    "margin_gap_50_add_10": "Gap 50 Add 10",
    "margin_gap_100_add_20": "Gap 100 Add 20",
    "hybrid_q50_if_gap_50": "Q50 Margin-Gated Floor",
    PRIORITY_POLICY_ID: "Q75 Margin-Gated Floor",
}


def reader_policy_label(policy_id: str) -> str:
    return POLICY_LABELS.get(policy_id, policy_id.replace("_", " ").title())


class ReservePolicyCatalog:
    """Finite catalog of monotone floor policies for offline screening."""

    def __init__(self, q25: float, q50: float, q75: float) -> None:
        self.q25 = float(q25)
        self.q50 = float(q50)
        self.q75 = float(q75)

    @classmethod
    def from_panel_directory(cls, panel_dir) -> "ReservePolicyCatalog":
        values = []
        for parquet in sorted(panel_dir.glob("*.parquet")):
            floors = pd.read_parquet(parquet, columns=["slot_floor_price"])["slot_floor_price"]
            floors = pd.to_numeric(floors, errors="coerce").dropna()
            values.append(floors[floors.gt(0)])
        if not values:
            raise FileNotFoundError(f"No panel shards found in {panel_dir}")
        all_floors = pd.concat(values, ignore_index=True)
        if all_floors.empty:
            raise ValueError(f"No positive floor prices found in {panel_dir}")
        q25, q50, q75 = all_floors.quantile([0.25, 0.50, 0.75]).to_numpy()
        return cls(q25=q25, q50=q50, q75=q75)

    def registry(self) -> pd.DataFrame:
        rows = [
            (BASELINE_POLICY_ID, "baseline", "Use the logged floor."),
            ("uniform_raise_05pct", "uniform_percent", "Raise every floor by 5 percent."),
            ("uniform_raise_10pct", "uniform_percent", "Raise every floor by 10 percent."),
            ("uniform_raise_15pct", "uniform_percent", "Raise every floor by 15 percent."),
            ("uniform_raise_20pct", "uniform_percent", "Raise every floor by 20 percent."),
            ("uniform_raise_30pct", "uniform_percent", "Raise every floor by 30 percent."),
            ("add_5_all_floors", "absolute_increment", "Add 5 to every floor."),
            ("add_10_all_floors", "absolute_increment", "Add 10 to every floor."),
            ("add_20_all_floors", "absolute_increment", "Add 20 to every floor."),
            ("min_positive_floor_q25", "positive_floor_quantile", "Lift positive floors below q25 to q25."),
            ("min_positive_floor_q50", "positive_floor_quantile", "Lift positive floors below q50 to q50."),
            ("min_positive_floor_q75", "positive_floor_quantile", "Lift positive floors below q75 to q75."),
            ("zero_and_low_floor_to_q25", "all_floor_quantile", "Lift all floors below q25 to q25."),
            ("zero_and_low_floor_to_q50", "all_floor_quantile", "Lift all floors below q50 to q50."),
            ("margin_gap_25_add_5", "margin_increment", "Add 5 when bid-floor gap is at least 25."),
            ("margin_gap_50_add_10", "margin_increment", "Add 10 when bid-floor gap is at least 50."),
            ("margin_gap_100_add_20", "margin_increment", "Add 20 when bid-floor gap is at least 100."),
            ("hybrid_q50_if_gap_50", "hybrid_quantile_margin", "Lift to q50 when gap is at least 50."),
            (PRIORITY_POLICY_ID, "hybrid_quantile_margin", "Lift to q75 when gap is at least 100."),
        ]
        frame = pd.DataFrame(rows, columns=["policy_id", "policy_family", "description"])
        frame["policy_number"] = [f"P{i}" for i in range(len(frame))]
        frame["policy_label"] = frame["policy_id"].map(reader_policy_label)
        frame["q25"] = self.q25
        frame["q50"] = self.q50
        frame["q75"] = self.q75
        return frame

    def floor(self, frame: pd.DataFrame, policy_id: str) -> np.ndarray:
        logged = pd.to_numeric(frame["slot_floor_price"], errors="coerce").fillna(0).clip(lower=0)
        gap = pd.to_numeric(frame["bid_floor_gap"], errors="coerce").fillna(-np.inf)
        logged_array = logged.to_numpy("float64")
        gap_array = gap.to_numpy("float64")
        if policy_id == BASELINE_POLICY_ID:
            return logged_array
        if policy_id.startswith("uniform_raise_"):
            pct = float(policy_id.split("_")[-1].replace("pct", "")) / 100.0
            return logged_array * (1.0 + pct)
        if policy_id == "add_5_all_floors":
            return logged_array + 5.0
        if policy_id == "add_10_all_floors":
            return logged_array + 10.0
        if policy_id == "add_20_all_floors":
            return logged_array + 20.0
        if policy_id == "min_positive_floor_q25":
            return np.where(logged_array > 0, np.maximum(logged_array, self.q25), logged_array)
        if policy_id == "min_positive_floor_q50":
            return np.where(logged_array > 0, np.maximum(logged_array, self.q50), logged_array)
        if policy_id == "min_positive_floor_q75":
            return np.where(logged_array > 0, np.maximum(logged_array, self.q75), logged_array)
        if policy_id == "zero_and_low_floor_to_q25":
            return np.maximum(logged_array, self.q25)
        if policy_id == "zero_and_low_floor_to_q50":
            return np.maximum(logged_array, self.q50)
        if policy_id == "margin_gap_25_add_5":
            return np.where(gap_array >= 25.0, logged_array + 5.0, logged_array)
        if policy_id == "margin_gap_50_add_10":
            return np.where(gap_array >= 50.0, logged_array + 10.0, logged_array)
        if policy_id == "margin_gap_100_add_20":
            return np.where(gap_array >= 100.0, logged_array + 20.0, logged_array)
        if policy_id == "hybrid_q50_if_gap_50":
            return np.where(gap_array >= 50.0, np.maximum(logged_array, self.q50), logged_array)
        if policy_id == PRIORITY_POLICY_ID:
            return np.where(gap_array >= 100.0, np.maximum(logged_array, self.q75), logged_array)
        raise ValueError(f"Unknown policy_id: {policy_id}")
