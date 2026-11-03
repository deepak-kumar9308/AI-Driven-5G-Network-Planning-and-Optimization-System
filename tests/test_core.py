"""Unit tests for the simulator, preprocessing, ML, slicing, energy, beamforming and comparison logic."""
import numpy as np
import pandas as pd
import pytest

from modules import beamforming as bf
from modules import network_slicing as ns
from modules.comparison import build_comparison
from modules.energy_optimization import optimize_energy
from modules.handover_prediction import (a3_rule_predict, load_bundle, predict_dataframe, predict_handover,
                                         save_bundle, train_and_evaluate)
from modules.network_simulator import generate_network_dataset, generate_towers, snapshot_for_location
from modules.rf_planning import coverage_stats, recommend_new_towers, signal_grid
from modules.traffic_clustering import cluster_traffic
from utils.data_preprocessing import FEATURES, TARGET, preprocess, validate_dataset


@pytest.fixture(scope="module")
def towers():
    return generate_towers(seed=42)


@pytest.fixture(scope="module")
def df(towers):
    return generate_network_dataset(n_users=2000, towers=towers, seed=42)


@pytest.fixture(scope="module")
def bundle(df):
    return train_and_evaluate(df)


# ---- dataset / preprocessing
def test_dataset_is_reproducible(towers):
    a = generate_network_dataset(500, towers, seed=1)
    b = generate_network_dataset(500, towers, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_dataset_valid_and_balanced(df):
    validate_dataset(df)
    assert 0.25 < df[TARGET].mean() < 0.35


def test_validation_rejects_bad_data(df):
    bad = df.copy()
    bad.loc[0, "Latency_ms"] = 9999
    with pytest.raises(ValueError):
        validate_dataset(bad)
    with pytest.raises(ValueError):
        validate_dataset(df.drop(columns=["Latency_ms"]))


def test_no_target_leakage(df):
    X, y = preprocess(df)
    assert TARGET not in X.columns and "User_ID" not in X.columns and "Tower_ID" not in X.columns
    assert list(X.columns) == FEATURES and len(X) == len(y)


# ---- ML
def test_model_trains_and_beats_chance(bundle):
    for m in bundle["metrics"].values():
        assert m["accuracy"] > 0.8
    assert bundle["metrics"]["Random Forest"]["f1"] > 0.75


def test_model_save_load_predict(bundle, tmp_path):
    save_bundle(bundle, tmp_path / "m.pkl")
    loaded = load_bundle(tmp_path / "m.pkl")
    strong_need = {"Signal_Strength_dBm": -112, "Neighbor_Signal_dBm": -80, "User_Speed_kmh": 90, "Tower_Distance_km": 5,
                   "Network_Traffic_Percentage": 90, "Neighbor_Traffic_Percentage": 30, "Latency_ms": 40, "Network_Speed_Mbps": 2}
    stay = {**strong_need, "Signal_Strength_dBm": -70, "Neighbor_Signal_dBm": -100, "Tower_Distance_km": 0.5,
            "User_Speed_kmh": 3, "Network_Traffic_Percentage": 30, "Neighbor_Traffic_Percentage": 80}
    assert predict_handover(loaded, strong_need)["handover_required"] is True
    assert predict_handover(loaded, stay)["handover_required"] is False
    with pytest.raises(ValueError):
        predict_handover(loaded, {"Latency_ms": 1})
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.pkl")


def test_predict_dataframe_and_rule(bundle, df):
    out = predict_dataframe(bundle, df.head(50))
    assert set(out["Predicted_Handover"]) <= {0, 1} and out["Handover_Probability"].between(0, 1).all()
    assert set(np.unique(a3_rule_predict(df))) <= {0, 1}


def test_clustering(df):
    out, profile, sil = cluster_traffic(df, 3)
    assert out["Cluster"].nunique() == 3 and -1 <= sil <= 1 and len(profile) == 3
    with pytest.raises(ValueError):
        cluster_traffic(df, 1)


# ---- RF / beamforming
def test_coverage_and_recommendations(towers):
    _, _, grid, _ = signal_grid(towers, resolution=40)
    before = coverage_stats(grid)["coverage_pct"]
    recs, with_new = recommend_new_towers(towers, 2, resolution=40)
    after = coverage_stats(signal_grid(with_new, resolution=40)[2])["coverage_pct"]
    assert after >= before and len(with_new) == len(towers) + len(recs)


def test_snapshot_for_location(towers):
    s = snapshot_for_location(towers, 5, 5, towers.loc[0, "Tower_ID"], 50)
    assert s["neighbor_id"] != s["serving_id"]
    with pytest.raises(ValueError):
        snapshot_for_location(towers, 5, 5, "NOPE", 50)


def test_beam_pattern_properties():
    assert bf.beam_gain_db(0, 16) == pytest.approx(10 * np.log10(16))
    assert bf.beam_gain_db(30, 16) < bf.beam_gain_db(0, 16)
    assert bf.beam_gain_db(0, 32) > bf.beam_gain_db(0, 8)


def test_beamforming_improves_sinr(towers):
    res = bf.simulate_beamforming(towers, n_elements=16, n_slots=15, seed=3)
    s = bf.summarize_beamforming(res)
    assert s["sinr_gain_db"] > 0 and np.isfinite(res.select_dtypes("number")).all().all()


# ---- slicing
def test_slicing_allocation_constraints():
    cap = 1000.0
    for demand in ({"eMBB": 900, "URLLC": 150, "mMTC": 100}, {"eMBB": 200, "URLLC": 50, "mMTC": 30}):
        a = ns.allocate_dynamic(demand, cap)
        assert sum(a.values()) <= cap + 1e-6 and all(v >= 0 for v in a.values())
    low = ns.allocate_dynamic({"eMBB": 200, "URLLC": 50, "mMTC": 30}, cap)
    assert low["URLLC"] >= 50                      # URLLC demand always fully covered when capacity is ample


def test_dynamic_not_worse_than_static_on_unmet():
    cap = 1000.0
    d = ns.hourly_demand(cap)
    s0 = ns.policy_summary(ns.simulate_day(d, cap, "static"), cap)
    s1 = ns.policy_summary(ns.simulate_day(d, cap, "dynamic"), cap)
    assert s1["unmet_total_mbps"] <= s0["unmet_total_mbps"]


def test_q_learning_trains():
    q, hist = ns.train_q_agent(1000.0, episodes=30)
    assert q.shape == (27, len(ns.ACTIONS)) and np.abs(q).sum() > 0 and len(hist) == 30
    alloc = ns.q_allocator(q, 1000.0)({"eMBB": 500, "URLLC": 130, "mMTC": 90}, 1000.0)
    assert sum(alloc.values()) == pytest.approx(1000.0)


# ---- energy
def test_energy_preserves_coverage_and_capacity(towers):
    r = optimize_energy(towers, coverage_tol_pts=1.0)
    s = r["summary"]
    assert s["savings_pct"] >= 0 and s["coverage_before_pct"] - s["coverage_after_pct"] <= 1.0 + 1e-6
    if r["sleeping_ids"]:
        assert s["savings_pct"] > 0
        # total traffic is conserved (offloaded, not dropped)
        cap = towers.set_index("Tower_ID")["Capacity_Mbps"]
        before = (r["before"].set_index("Tower_ID")["Load_pct"] / 100 * cap).sum()
        after = (r["after"].set_index("Tower_ID")["Load_pct"] / 100 * cap).sum()
        assert after == pytest.approx(before, rel=0.02)


def test_energy_strict_tolerance_blocks_sleep(towers):
    r = optimize_energy(towers, coverage_tol_pts=0.0, weak_threshold=-95)
    assert r["summary"]["towers_sleeping"] == 0 and r["summary"]["savings_pct"] == 0


# ---- comparison
def test_comparison_table_is_computed(df, towers, bundle):
    comp, details = build_comparison(df, towers, bundle)
    assert {"Metric", "Baseline", "Optimized", "Result"} <= set(comp.columns)
    assert comp["Result"].isin(["Improved", "Worse", "No change", "Neutral / context"]).all()
    assert len(comp) >= 10 and "handover" in details
