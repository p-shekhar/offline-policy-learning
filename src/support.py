from __future__ import annotations

import numpy as np
import pandas as pd

from policy_catalog import ReservePolicyCatalog


def effective_sample_size(weights: np.ndarray | pd.Series) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w >= 0)]
    denom = np.square(w).sum()
    if denom == 0:
        return 0.0
    return float(np.square(w.sum()) / denom)


def replay_support_diagnostics(
    frame: pd.DataFrame,
    catalog: ReservePolicyCatalog,
    radius: float = 10.0,
) -> pd.DataFrame:
    """Policy support diagnostics for reserve-policy replay.

    ``threshold_band_count`` counts observations close to the candidate clearing
    threshold. Thin threshold bands are the empirical version of the paper's
    support-limited threshold-resolution problem.
    """

    rows = []
    bid = pd.to_numeric(frame["bid_price"], errors="coerce").fillna(-np.inf).to_numpy("float64")
    logged = pd.to_numeric(frame["slot_floor_price"], errors="coerce").fillna(0).to_numpy("float64")
    filled = pd.to_numeric(frame["filled"], errors="coerce").fillna(0).to_numpy(bool)
    row_count = len(frame)
    filled_count = int(filled.sum())
    for policy_id in catalog.registry()["policy_id"]:
        floor = catalog.floor(frame, policy_id)
        changed = ~np.isclose(floor, logged)
        retained = filled & (bid >= floor)
        threshold_band = np.abs(bid - floor) <= radius
        floor_changed_count = int(changed.sum())
        retained_impression_count = int(retained.sum())
        threshold_band_count = int(threshold_band.sum())
        changed_threshold_band_count = int((changed & threshold_band).sum())
        rows.append(
            {
                "policy_id": policy_id,
                "row_count": row_count,
                "filled_count": filled_count,
                "floor_changed_count": floor_changed_count,
                "floor_changed_share": float(floor_changed_count / max(row_count, 1)),
                "retained_impression_count": retained_impression_count,
                "retained_impression_share": float(retained_impression_count / max(filled_count, 1)),
                "threshold_band_radius": radius,
                "threshold_band_count": threshold_band_count,
                "threshold_band_share": float(threshold_band_count / max(row_count, 1)),
                "changed_threshold_band_count": changed_threshold_band_count,
            }
        )
    return pd.DataFrame(rows)


def combine_support_diagnostics(diagnostics: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(diagnostics, ignore_index=True)
    summary = (
        frame.groupby("policy_id", as_index=False)
        .agg(
            row_count=("row_count", "sum"),
            filled_count=("filled_count", "sum"),
            floor_changed_count=("floor_changed_count", "sum"),
            retained_impression_count=("retained_impression_count", "sum"),
            threshold_band_radius=("threshold_band_radius", "first"),
            threshold_band_count=("threshold_band_count", "sum"),
            changed_threshold_band_count=("changed_threshold_band_count", "sum"),
        )
    )
    summary["floor_changed_share"] = summary["floor_changed_count"] / summary["row_count"].clip(lower=1)
    summary["retained_impression_share"] = (
        summary["retained_impression_count"] / summary["filled_count"].clip(lower=1)
    )
    summary["threshold_band_share"] = summary["threshold_band_count"] / summary["row_count"].clip(lower=1)
    return summary.sort_values("threshold_band_count", ascending=False)


def pairwise_boundary_support(
    frame: pd.DataFrame,
    catalog: ReservePolicyCatalog,
    radius: float,
) -> pd.DataFrame:
    """Compute pairwise local support around candidate reserve boundaries.

    A pair is locally supported when logged bids fall close to either candidate
    floor, or when the two candidate floors induce different retained/not-retained
    replay decisions. This is the empirical counterpart of the
    ``m_{tau,tau'}`` term in the threshold-resolution discussion.
    """

    registry = catalog.registry()
    policy_ids = registry["policy_id"].astype(str).tolist()
    label = registry.set_index("policy_id")["policy_label"].to_dict()
    number = registry.set_index("policy_id")["policy_number"].to_dict()
    bid = pd.to_numeric(frame["bid_price"], errors="coerce").fillna(-np.inf).to_numpy("float64")
    floors = {policy_id: catalog.floor(frame, policy_id) for policy_id in policy_ids}
    row_count = len(frame)
    rows = []
    for left_idx, left_policy in enumerate(policy_ids):
        left_floor = floors[left_policy]
        left_decision = bid >= left_floor
        for right_policy in policy_ids[left_idx + 1 :]:
            right_floor = floors[right_policy]
            right_decision = bid >= right_floor
            close_to_left = np.abs(bid - left_floor) <= radius
            close_to_right = np.abs(bid - right_floor) <= radius
            disagreement = left_decision != right_decision
            boundary = close_to_left | close_to_right | disagreement
            floor_distance = np.abs(left_floor - right_floor)
            rows.append(
                {
                    "left_policy_id": left_policy,
                    "right_policy_id": right_policy,
                    "left_policy_number": number[left_policy],
                    "right_policy_number": number[right_policy],
                    "left_policy_label": label[left_policy],
                    "right_policy_label": label[right_policy],
                    "threshold_band_radius": float(radius),
                    "row_count": row_count,
                    "decision_disagreement_count": int(disagreement.sum()),
                    "boundary_count": int(boundary.sum()),
                    "boundary_share": float(boundary.sum() / max(row_count, 1)),
                    "mean_floor_distance": float(np.mean(floor_distance)),
                    "p95_floor_distance": float(np.percentile(floor_distance, 95)),
                }
            )
    return pd.DataFrame(rows)


def summarize_pairwise_boundary_support(pairwise_frames: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(pairwise_frames, ignore_index=True)
    key_cols = [
        "left_policy_id",
        "right_policy_id",
        "left_policy_number",
        "right_policy_number",
        "left_policy_label",
        "right_policy_label",
        "threshold_band_radius",
    ]
    summary = (
        frame.groupby(key_cols, as_index=False)
        .agg(
            row_count=("row_count", "sum"),
            decision_disagreement_count=("decision_disagreement_count", "sum"),
            boundary_count=("boundary_count", "sum"),
            mean_floor_distance=("mean_floor_distance", "mean"),
            p95_floor_distance=("p95_floor_distance", "mean"),
        )
        .sort_values("boundary_count")
    )
    summary["boundary_share"] = summary["boundary_count"] / summary["row_count"].clip(lower=1)
    summary["effective_boundary_n"] = summary["boundary_count"]
    return summary.reset_index(drop=True)
