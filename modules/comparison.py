"""
Before-vs-after comparison. Every value is calculated from the simulation outputs - nothing is
hard-coded and no metric is assumed to improve. Rows can come out as "Worse" or "Neutral".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.beamforming import simulate_beamforming, summarize_beamforming
from modules.energy_optimization import optimize_energy
from modules.handover_prediction import a3_rule_predict, predict_dataframe
from modules.network_simulator import estimate_latency_ms
from modules.network_slicing import hourly_demand, policy_summary, simulate_day
from modules.rf_planning import coverage_stats, recommend_new_towers, signal_grid
from utils.data_preprocessing import split_dataset
from utils.metrics import classification_metrics, comparison_row


def handover_effect(df: pd.DataFrame, bundle: dict, seed: int = 42) -> dict:
    """Apply ML-predicted handovers to the held-out test users and re-estimate signal and latency."""
    _, X_test, _, y_test = split_dataset(df, seed=seed)
    test = df.loc[X_test.index]
    pred = predict_dataframe(bundle, test)["Predicted_Handover"].to_numpy()
    rule = a3_rule_predict(test)
    y = y_test.to_numpy()

    def apply(flags):
        sig = np.where(flags == 1, test["Neighbor_Signal_dBm"], test["Signal_Strength_dBm"])
        load = np.where(flags == 1, test["Neighbor_Traffic_Percentage"], test["Network_Traffic_Percentage"])
        return sig.mean(), estimate_latency_ms(sig, load, test["User_Speed_kmh"]).mean()

    sig0, lat0 = apply(np.zeros(len(test), dtype=int))
    sig1, lat1 = apply(pred)
    return {"signal_before": sig0, "signal_after": sig1, "latency_before": lat0, "latency_after": lat1,
            "rule_metrics": classification_metrics(y, rule), "ml_metrics": classification_metrics(y, pred),
            "rule_false_handovers": int(((rule == 1) & (y == 0)).sum()),
            "ml_false_handovers": int(((pred == 1) & (y == 0)).sum()), "n_test": int(len(test))}


def build_comparison(df: pd.DataFrame, towers: pd.DataFrame, bundle: dict, n_new_towers: int = 3,
                     n_elements: int = 16, capacity_mbps: float = 1000.0, seed: int = 42):
    """Return (comparison_df, details_dict)."""
    rows = []
    # 1) RF planning: coverage before/after adding recommended towers
    cov0 = coverage_stats(signal_grid(towers)[2])
    recs, with_new = recommend_new_towers(towers, n_new=n_new_towers)
    cov1 = coverage_stats(signal_grid(with_new)[2])
    rows.append(comparison_row("Area coverage (>= -100 dBm)", cov0["coverage_pct"], cov1["coverage_pct"], True, "%", "RF Planning"))
    rows.append(comparison_row("Mean best-server signal", cov0["mean_signal_dbm"], cov1["mean_signal_dbm"], True, "dBm", "RF Planning"))
    # 2) Beamforming
    bf = summarize_beamforming(simulate_beamforming(towers, n_elements=n_elements, seed=seed))
    rows.append(comparison_row("Mean SINR (omni vs beamformed)", bf["sinr_before_db"], bf["sinr_after_db"], True, "dB", "Beamforming (simplified sim)"))
    # 3) Handover
    ho = handover_effect(df, bundle, seed)
    rows.append(comparison_row("Mean serving signal (test users)", ho["signal_before"], ho["signal_after"], True, "dBm", "Handover (ML decisions)"))
    rows.append(comparison_row("Mean latency (test users, modelled)", ho["latency_before"], ho["latency_after"], False, "ms", "Handover (ML decisions)"))
    rows.append(comparison_row("Handover decision F1 (A3 rule vs ML)", ho["rule_metrics"]["f1"], ho["ml_metrics"]["f1"], True, "score", "Handover model"))
    rows.append(comparison_row("Unnecessary handovers (A3 rule vs ML)", ho["rule_false_handovers"], ho["ml_false_handovers"], False, "count", "Handover model"))
    # 4) Slicing
    demand = hourly_demand(capacity_mbps, seed)
    s0 = policy_summary(simulate_day(demand, capacity_mbps, "static"), capacity_mbps)
    s1 = policy_summary(simulate_day(demand, capacity_mbps, "dynamic"), capacity_mbps)
    rows.append(comparison_row("Capacity utilisation (slicing)", s0["capacity_utilization_pct"], s1["capacity_utilization_pct"], True, "%", "Network Slicing"))
    rows.append(comparison_row("Slice SLA satisfaction", s0["sla_satisfaction_pct"], s1["sla_satisfaction_pct"], True, "%", "Network Slicing"))
    rows.append(comparison_row("Unmet demand", s0["unmet_total_mbps"], s1["unmet_total_mbps"], False, "Mbps", "Network Slicing"))
    # 5) Energy
    en = optimize_energy(towers)["summary"]
    rows.append(comparison_row("Total power consumption", en["power_before_kw"], en["power_after_kw"], False, "kW", "Energy Optimisation"))
    rows.append(comparison_row("Coverage after sleep mode", en["coverage_before_pct"], en["coverage_after_pct"], True, "%", "Energy Optimisation"))
    rows.append(comparison_row("Mean load of active towers", en["mean_active_load_before_pct"], en["mean_active_load_after_pct"], None, "%", "Energy Optimisation"))
    return pd.DataFrame(rows), {"handover": ho, "recommendations": recs, "beamforming": bf, "energy": en}
