"""
Module 5 - Energy Efficiency Optimisation.

Strategy (NOT "switch off anything with low traffic"): a lightly loaded tower may enter sleep
mode only if ALL of these hold:
  1. its load is below `load_threshold` (candidate),
  2. area coverage (>= weak threshold) stays within `coverage_tol_pts` of the original,
  3. neighbouring active towers that can reach its location have enough spare capacity to absorb
     its traffic while staying under `max_target_load`.
Otherwise the tower stays on, and the reason is recorded in the decision log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.network_simulator import distance_matrix, rsrp_dbm, tower_power_kw
from modules.rf_planning import coverage_stats, signal_grid

# Minimum usable signal for the energy check (more lenient than the -100 dBm RF-planning 'weak' flag)
ENERGY_MIN_SIGNAL_DBM = -105.0


def power_table(towers: pd.DataFrame, loads: np.ndarray, sleeping: np.ndarray) -> np.ndarray:
    out = np.zeros(len(towers))
    for i, t in enumerate(towers.itertuples()):
        out[i] = float(tower_power_kw(loads[i], t.P_Idle_kW, t.P_Max_kW, sleeping=bool(sleeping[i])))
    return out


def optimize_energy(towers: pd.DataFrame, load_threshold: float = 30.0, max_target_load: float = 80.0,
                    coverage_tol_pts: float = 1.0, weak_threshold: float = ENERGY_MIN_SIGNAL_DBM,
                    resolution: int = 48, max_helpers: int = 4) -> dict:
    """Greedy sleep-mode optimisation. Returns before/after tables, decisions and a summary."""
    towers = towers.reset_index(drop=True)
    n = len(towers)
    loads0 = towers["Base_Load_Pct"].to_numpy(dtype=float)
    loads = loads0.copy()
    sleeping = np.zeros(n, dtype=bool)
    cap = towers["Capacity_Mbps"].to_numpy(dtype=float)
    base_cov = coverage_stats(signal_grid(towers, resolution=resolution)[2], weak_threshold)["coverage_pct"]
    decisions = []

    for i in np.argsort(loads0):                       # lightest-loaded first
        tid = towers.loc[i, "Tower_ID"]
        if loads0[i] >= load_threshold:
            continue
        trial = ~sleeping.copy()
        trial[i] = False
        cov = coverage_stats(signal_grid(towers, resolution=resolution, active=trial)[2], weak_threshold)["coverage_pct"]
        if base_cov - cov > coverage_tol_pts:
            decisions.append({"Tower_ID": tid, "Load_pct": round(loads0[i], 1), "Action": "Keep ON",
                              "Reason": f"Coverage would drop {base_cov - cov:.2f} pts (limit {coverage_tol_pts})"})
            continue
        # helpers: active towers that reach this tower's location above the threshold
        sig_here = rsrp_dbm(towers["Tx_Power_dBm"].to_numpy(),
                            distance_matrix([towers.loc[i, "X_km"]], [towers.loc[i, "Y_km"]], towers)[0])
        dist_here = distance_matrix([towers.loc[i, "X_km"]], [towers.loc[i, "Y_km"]], towers)[0]
        helpers = [j for j in np.argsort(dist_here) if j != i and not sleeping[j] and sig_here[j] >= weak_threshold]
        helpers = helpers[:max_helpers]
        headroom = np.array([max(0.0, (max_target_load - loads[j]) / 100 * cap[j]) for j in helpers])
        helpers = [j for j, h in zip(helpers, headroom) if h > 0]   # only towers with spare capacity
        headroom = headroom[headroom > 0]
        traffic = loads[i] / 100 * cap[i]
        if not helpers or headroom.sum() < traffic:
            decisions.append({"Tower_ID": tid, "Load_pct": round(loads0[i], 1), "Action": "Keep ON",
                              "Reason": f"Neighbours lack capacity ({headroom.sum():.0f} < {traffic:.0f} Mbps needed)"})
            continue
        for j, h in zip(helpers, headroom):            # offload proportionally to spare capacity
            loads[j] += traffic * h / headroom.sum() / cap[j] * 100
        loads[i], sleeping[i] = 0.0, True
        decisions.append({"Tower_ID": tid, "Load_pct": round(loads0[i], 1), "Action": "Sleep mode",
                          "Reason": f"Coverage kept ({cov:.2f}%), traffic moved to {', '.join(towers.loc[helpers, 'Tower_ID'])}"})

    after_cov = coverage_stats(signal_grid(towers, resolution=resolution, active=~sleeping)[2], weak_threshold)["coverage_pct"]
    p_before = power_table(towers, loads0, np.zeros(n, dtype=bool))
    p_after = power_table(towers, loads, sleeping)
    before = pd.DataFrame({"Tower_ID": towers["Tower_ID"], "Load_pct": loads0.round(1), "Power_kW": p_before.round(3)})
    after = pd.DataFrame({"Tower_ID": towers["Tower_ID"], "Load_pct": loads.round(1), "Power_kW": p_after.round(3),
                          "State": np.where(sleeping, "Sleep", "Active")})
    return {
        "before": before, "after": after, "decisions": pd.DataFrame(decisions),
        "sleeping_ids": towers.loc[sleeping, "Tower_ID"].tolist(),
        "summary": {
            "power_before_kw": float(p_before.sum()), "power_after_kw": float(p_after.sum()),
            "savings_pct": float((p_before.sum() - p_after.sum()) / p_before.sum() * 100),
            "coverage_before_pct": float(base_cov), "coverage_after_pct": float(after_cov),
            "towers_sleeping": int(sleeping.sum()),
            "max_load_after_pct": float(loads[~sleeping].max()),
            "mean_active_load_before_pct": float(loads0.mean()),
            "mean_active_load_after_pct": float(loads[~sleeping].mean()),
        },
    }
