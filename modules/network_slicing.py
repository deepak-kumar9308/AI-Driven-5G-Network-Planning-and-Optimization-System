"""
Module 4 - Network Slicing Optimisation (simulation).

Slices: eMBB, URLLC, mMTC. Compared policies over a simulated 24-hour demand profile:
  * Static equal split (baseline)
  * Dynamic priority-aware allocation (guaranteed minimum + priority-weighted water-filling)
  * Q-learning agent (reinforcement-learning-INSPIRED demo, genuinely trained on simulated days)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLICES = ["eMBB", "URLLC", "mMTC"]
SLICE_CONFIG = {
    #         guaranteed min share, priority weight, headroom factor, SLA = min fraction of demand served, latency target
    "eMBB":  {"min_share": 0.30, "priority": 1.0, "headroom": 0.10, "sla": 0.90, "latency_ms": 50},
    "URLLC": {"min_share": 0.15, "priority": 3.0, "headroom": 0.30, "sla": 0.99, "latency_ms": 5},
    "mMTC":  {"min_share": 0.05, "priority": 0.7, "headroom": 0.10, "sla": 0.90, "latency_ms": 200},
}


def hourly_demand(capacity_mbps: float = 1000.0, seed: int = 42, noise: float = 0.08) -> pd.DataFrame:
    """Simulated demand (Mbps) per slice for 24 hours. eMBB peaks in the evening, mMTC at night."""
    rng = np.random.default_rng(seed)
    h = np.arange(24)
    embb = 0.40 + 0.45 * np.exp(-((h - 20) ** 2) / 12.0) + 0.15 * np.exp(-((h - 12) ** 2) / 10.0)
    urllc = 0.13 + 0.04 * np.sin((h - 6) / 24 * 2 * np.pi)
    mmtc = 0.09 + 0.04 * np.exp(-((h - 2) ** 2) / 10.0)
    frac = np.vstack([embb, urllc, mmtc]).T * (1 + rng.normal(0, noise, (24, 3)))
    return pd.DataFrame(np.clip(frac, 0.01, None) * capacity_mbps, columns=SLICES).rename_axis("Hour").reset_index()


def allocate_static(demand: dict, capacity: float) -> dict:
    """Baseline: split capacity equally between slices."""
    return {s: capacity / len(SLICES) for s in SLICES}


def allocate_dynamic(demand: dict, capacity: float) -> dict:
    """
    Guaranteed minimum (capped at the slice's own need), then remaining capacity is handed out
    by priority-weighted water-filling to slices that still need more. Never exceeds capacity.
    """
    need = {s: demand[s] * (1 + SLICE_CONFIG[s]["headroom"]) for s in SLICES}
    alloc = {s: min(need[s], SLICE_CONFIG[s]["min_share"] * capacity) for s in SLICES}
    remaining = capacity - sum(alloc.values())
    for _ in range(20):
        hungry = [s for s in SLICES if need[s] - alloc[s] > 1e-9]
        if remaining <= 1e-9 or not hungry:
            break
        wsum = sum(SLICE_CONFIG[s]["priority"] for s in hungry)
        given = 0.0
        for s in hungry:
            g = min(need[s] - alloc[s], remaining * SLICE_CONFIG[s]["priority"] / wsum)
            alloc[s] += g
            given += g
        remaining -= given
    return alloc


def evaluate_allocation(demand: dict, alloc: dict, capacity: float) -> dict:
    """Served traffic, unmet demand, utilisation and SLA flags for one allocation."""
    served = {s: min(demand[s], alloc[s]) for s in SLICES}
    return {
        "served": served,
        "unmet": {s: demand[s] - served[s] for s in SLICES},
        "slice_utilization_pct": {s: (served[s] / alloc[s] * 100 if alloc[s] > 0 else 0.0) for s in SLICES},
        "capacity_utilization_pct": sum(served.values()) / capacity * 100,
        "sla_met": {s: served[s] >= SLICE_CONFIG[s]["sla"] * demand[s] - 1e-9 for s in SLICES},
    }


def simulate_day(demand_df: pd.DataFrame, capacity: float, policy: str = "dynamic", alloc_fn=None) -> pd.DataFrame:
    """Run a policy over every hour -> tidy table with allocation and KPIs per hour/slice."""
    fn = alloc_fn or {"static": allocate_static, "dynamic": allocate_dynamic}[policy]
    rows = []
    for _, r in demand_df.iterrows():
        demand = {s: float(r[s]) for s in SLICES}
        alloc = fn(demand, capacity)
        ev = evaluate_allocation(demand, alloc, capacity)
        for s in SLICES:
            rows.append({"Hour": int(r["Hour"]), "Slice": s, "Demand_Mbps": demand[s], "Allocated_Mbps": alloc[s],
                         "Served_Mbps": ev["served"][s], "Unmet_Mbps": ev["unmet"][s],
                         "Slice_Utilization_pct": ev["slice_utilization_pct"][s],
                         "Capacity_Utilization_pct": ev["capacity_utilization_pct"], "SLA_Met": ev["sla_met"][s]})
    return pd.DataFrame(rows)


def policy_summary(table: pd.DataFrame, capacity: float) -> dict:
    hours = table["Hour"].nunique()
    return {
        "served_total_mbps": float(table["Served_Mbps"].sum() / hours),
        "unmet_total_mbps": float(table["Unmet_Mbps"].sum() / hours),
        "capacity_utilization_pct": float(table.groupby("Hour")["Capacity_Utilization_pct"].first().mean()),
        "sla_satisfaction_pct": float(table["SLA_Met"].mean() * 100),
        "urllc_sla_pct": float(table.loc[table["Slice"] == "URLLC", "SLA_Met"].mean() * 100),
    }


# ----------------------------- Q-learning (RL-inspired) --------------------------------
ACTIONS = [  # capacity fractions (eMBB, URLLC, mMTC)
    (1 / 3, 1 / 3, 1 / 3), (0.60, 0.25, 0.15), (0.70, 0.20, 0.10), (0.50, 0.30, 0.20),
    (0.80, 0.12, 0.08), (0.65, 0.20, 0.15), (0.75, 0.15, 0.10), (0.55, 0.15, 0.30),
]


def _state(demand: dict, capacity: float) -> int:
    """Discretise each slice's demand into low/medium/high (27 states)."""
    cuts = {"eMBB": (0.45, 0.70), "URLLC": (0.12, 0.16), "mMTC": (0.09, 0.12)}
    idx = 0
    for s in SLICES:
        f = demand[s] / capacity
        level = 0 if f < cuts[s][0] else (1 if f < cuts[s][1] else 2)
        idx = idx * 3 + level
    return idx


def _reward(demand: dict, frac, capacity: float) -> float:
    alloc = {s: frac[i] * capacity for i, s in enumerate(SLICES)}
    ev = evaluate_allocation(demand, alloc, capacity)
    return ev["capacity_utilization_pct"] / 100 + 0.3 * sum(ev["sla_met"].values()) + (0.5 if ev["sla_met"]["URLLC"] else 0)


def train_q_agent(capacity: float = 1000.0, episodes: int = 300, seed: int = 42,
                  alpha: float = 0.2, gamma: float = 0.5, eps_start: float = 1.0, eps_end: float = 0.05):
    """Tabular Q-learning over simulated days (state=demand level, action=capacity split preset)."""
    rng = np.random.default_rng(seed)
    q = np.zeros((27, len(ACTIONS)))
    history = []
    for ep in range(episodes):
        eps = max(eps_end, eps_start * (1 - ep / max(1, episodes * 0.8)))
        day = hourly_demand(capacity, seed=int(rng.integers(1e9)), noise=0.15)
        total = 0.0
        for h in range(24):
            d = {s: float(day.loc[h, s]) for s in SLICES}
            st = _state(d, capacity)
            a = int(rng.integers(len(ACTIONS))) if rng.random() < eps else int(q[st].argmax())
            r = _reward(d, ACTIONS[a], capacity)
            nxt = {s: float(day.loc[(h + 1) % 24, s]) for s in SLICES}
            q[st, a] += alpha * (r + gamma * q[_state(nxt, capacity)].max() - q[st, a])
            total += r
        history.append(total / 24)
    return q, history


def q_allocator(q: np.ndarray, capacity: float):
    """Turn a trained Q-table into an allocation function (greedy policy)."""
    def fn(demand: dict, cap: float) -> dict:
        frac = ACTIONS[int(q[_state(demand, cap)].argmax())]
        return {s: frac[i] * cap for i, s in enumerate(SLICES)}
    return fn
