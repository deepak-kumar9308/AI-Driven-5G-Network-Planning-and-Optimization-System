"""
Module 1 - RF Planning.

Computes a coverage map (best-server RSRP on a grid), identifies weak-signal areas
and greedily recommends locations for additional towers that maximise coverage gain.
Signal model: simplified log-distance path loss, no shadowing (smooth map).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.network_simulator import AREA_KM, distance_matrix, rsrp_dbm

DEFAULT_WEAK_DBM = -100.0
DEFAULT_NEW_TX_DBM = 41.0


def make_grid(area_km: float = AREA_KM, resolution: int = 60):
    xs = np.linspace(0, area_km, resolution)
    ys = np.linspace(0, area_km, resolution)
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, gx.ravel(), gy.ravel()


def signal_grid(towers: pd.DataFrame, area_km: float = AREA_KM, resolution: int = 60, active=None):
    """Best-server RSRP on a grid. `active` is an optional boolean mask of towers in use."""
    xs, ys, fx, fy = make_grid(area_km, resolution)
    t = towers if active is None else towers[np.asarray(active, dtype=bool)]
    if t.empty:
        return xs, ys, np.full((resolution, resolution), -130.0), np.full((resolution, resolution), -1)
    sig = rsrp_dbm(t["Tx_Power_dBm"].to_numpy()[None, :], distance_matrix(fx, fy, t))
    best = sig.max(axis=1).reshape(resolution, resolution)
    best_idx = sig.argmax(axis=1).reshape(resolution, resolution)
    return xs, ys, best, best_idx


def coverage_stats(best_signal: np.ndarray, weak_threshold: float = DEFAULT_WEAK_DBM) -> dict:
    """Coverage / weak-area percentages and mean best-server signal."""
    covered = best_signal >= weak_threshold
    return {
        "coverage_pct": float(covered.mean() * 100),
        "weak_pct": float((~covered).mean() * 100),
        "mean_signal_dbm": float(best_signal.mean()),
        "min_signal_dbm": float(best_signal.min()),
    }


def tower_coverage_table(towers: pd.DataFrame, resolution: int = 60,
                         weak_threshold: float = DEFAULT_WEAK_DBM) -> pd.DataFrame:
    """Per-tower share of the area for which it is the best server and is above threshold."""
    _, _, best, idx = signal_grid(towers, resolution=resolution)
    rows = []
    for i, tw in towers.reset_index(drop=True).iterrows():
        mask = (idx == i) & (best >= weak_threshold)
        rows.append({"Tower_ID": tw["Tower_ID"], "X_km": tw["X_km"], "Y_km": tw["Y_km"],
                     "Tx_Power_dBm": tw["Tx_Power_dBm"],
                     "Area_Served_pct": round(float(mask.mean() * 100), 2),
                     "Mean_Signal_dBm": round(float(best[idx == i].mean()), 2) if (idx == i).any() else np.nan})
    return pd.DataFrame(rows)


def recommend_new_towers(towers: pd.DataFrame, n_new: int = 3, weak_threshold: float = DEFAULT_WEAK_DBM,
                         tx_power_dbm: float = DEFAULT_NEW_TX_DBM, resolution: int = 48):
    """
    Greedy placement: repeatedly choose the candidate location (a weak grid cell) that
    increases covered area the most. Returns (recommendations_df, towers_with_new_df).
    """
    _, _, fx, fy = make_grid(AREA_KM, resolution)
    cur = rsrp_dbm(towers["Tx_Power_dBm"].to_numpy()[None, :], distance_matrix(fx, fy, towers)).max(axis=1)
    working = towers.copy()
    recs, base_cov = [], float((cur >= weak_threshold).mean() * 100)
    for k in range(n_new):
        weak_idx = np.where(cur < weak_threshold)[0]
        if len(weak_idx) == 0:
            break
        # subsample candidates so the search stays fast
        cand = weak_idx[:: max(1, len(weak_idx) // 250)]
        cx, cy = fx[cand], fy[cand]
        d = np.sqrt((cx[:, None] - fx[None, :]) ** 2 + (cy[:, None] - fy[None, :]) ** 2)
        new_sig = rsrp_dbm(tx_power_dbm, d)                       # (cand, grid)
        merged = np.maximum(new_sig, cur[None, :])
        gains = (merged >= weak_threshold).mean(axis=1) * 100
        best = int(np.argmax(gains))
        if gains[best] <= (cur >= weak_threshold).mean() * 100 + 1e-9:
            break
        cur = merged[best]
        cov_now = float((cur >= weak_threshold).mean() * 100)
        tid = f"N{k + 1:02d}"
        recs.append({"Recommended_ID": tid, "X_km": round(float(cx[best]), 2), "Y_km": round(float(cy[best]), 2),
                     "Tx_Power_dBm": tx_power_dbm, "Coverage_After_pct": round(cov_now, 2),
                     "Coverage_Gain_pct_points": round(cov_now - base_cov, 2)})
        base_cov = cov_now
        working = pd.concat([working, pd.DataFrame([{
            "Tower_ID": tid, "X_km": float(cx[best]), "Y_km": float(cy[best]), "Tx_Power_dBm": tx_power_dbm,
            "Bandwidth_MHz": 100.0, "Capacity_Mbps": 800.0, "Base_Load_Pct": 30.0,
            "P_Idle_kW": 2.0, "P_Max_kW": 3.8}])], ignore_index=True)
    return pd.DataFrame(recs, columns=["Recommended_ID", "X_km", "Y_km", "Tx_Power_dBm",
                                       "Coverage_After_pct", "Coverage_Gain_pct_points"]), working
