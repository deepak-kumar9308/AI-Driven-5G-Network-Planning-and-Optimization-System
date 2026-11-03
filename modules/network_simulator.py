"""
Core synthetic network simulator.

Everything in this project is built on top of this module. It creates:
  * a set of simulated 5G base stations (towers),
  * a synthetic per-user measurement dataset (signal, latency, speed, load, ...),
  * the simplified radio / power models shared by every other module.

IMPORTANT: all values are SIMULATED. The radio model is a simplified log-distance
path-loss model. Nothing here represents a real telecom operator's network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Global simulation constants (documented in docs/methodology.md)
# ----------------------------------------------------------------------------
AREA_KM = 12.0            # simulated service area is AREA_KM x AREA_KM
ANT_GAIN_DB = 15.0        # transmit antenna gain (dBi)
RE_OFFSET_DB = 35.0       # total power -> per-resource-element power (~10*log10(3276))
PL0_DB = 105.0            # path loss at 1 km
PL_EXPONENT = 3.8         # path-loss exponent (urban-like)
SHADOW_SIGMA_DB = 4.0     # log-normal shadowing std-dev for per-user samples
RSRP_MIN, RSRP_MAX = -130.0, -40.0
SLEEP_POWER_FRACTION = 0.10   # a tower in sleep mode draws 10% of its idle power
SERVICE_TYPES = ["eMBB", "URLLC", "mMTC"]

DATASET_COLUMNS = [
    "User_ID", "Tower_ID", "Service_Type", "X_km", "Y_km",
    "Signal_Strength_dBm", "Neighbor_Tower_ID", "Neighbor_Signal_dBm",
    "Network_Speed_Mbps", "Latency_ms", "User_Speed_kmh", "Tower_Distance_km",
    "Network_Traffic_Percentage", "Neighbor_Traffic_Percentage",
    "Bandwidth_MHz", "Power_Consumption_kWh", "Handover_Required",
]


# ----------------------------------------------------------------------------
# Radio / power models
# ----------------------------------------------------------------------------
def path_loss_db(distance_km):
    """Simplified log-distance path loss (dB). Distance is clamped to 50 m."""
    d = np.maximum(np.asarray(distance_km, dtype=float), 0.05)
    return PL0_DB + 10.0 * PL_EXPONENT * np.log10(d)


def rsrp_dbm(tx_power_dbm, distance_km, shadowing_db=0.0):
    """Received signal power per resource element (RSRP-like), in dBm."""
    value = tx_power_dbm + ANT_GAIN_DB - RE_OFFSET_DB - path_loss_db(distance_km) + shadowing_db
    return np.clip(value, RSRP_MIN, RSRP_MAX)


def estimate_sinr_db(signal_dbm, load_pct=0.0):
    """Very simple SINR proxy (dB) from signal strength and cell load."""
    sinr = np.asarray(signal_dbm, dtype=float) + 100.0 - 0.08 * np.asarray(load_pct, dtype=float)
    return np.clip(sinr, -10.0, 35.0)


def estimate_speed_mbps(signal_dbm, load_pct, bandwidth_mhz):
    """Per-user throughput estimate (noise-free): capped Shannon-like efficiency x BW x load factor."""
    sinr = estimate_sinr_db(signal_dbm, load_pct)
    spectral_eff = np.minimum(np.log2(1.0 + 10.0 ** (sinr / 10.0)), 7.4) * 0.75
    share = 0.25 * (1.0 - 0.7 * np.asarray(load_pct, dtype=float) / 100.0)
    return np.maximum(bandwidth_mhz * spectral_eff * share, 0.1)


def estimate_latency_ms(signal_dbm, load_pct, user_speed_kmh=0.0):
    """Latency estimate (noise-free): grows with weak signal, high load and mobility."""
    weak = np.maximum(0.0, -80.0 - np.asarray(signal_dbm, dtype=float))
    lat = 6.0 + 0.25 * weak + 25.0 * (np.asarray(load_pct, dtype=float) / 100.0) ** 3 \
        + 0.03 * np.asarray(user_speed_kmh, dtype=float)
    return np.clip(lat, 1.0, 150.0)


def tower_power_kw(load_pct, p_idle_kw, p_max_kw, sleeping=False):
    """Linear load-dependent power model. Sleeping towers draw a small fraction of idle power."""
    if sleeping:
        return np.asarray(p_idle_kw, dtype=float) * SLEEP_POWER_FRACTION
    load = np.clip(np.asarray(load_pct, dtype=float), 0.0, 100.0) / 100.0
    return p_idle_kw + (p_max_kw - p_idle_kw) * load


# ----------------------------------------------------------------------------
# Towers
# ----------------------------------------------------------------------------
def generate_towers(n_towers: int = 12, area_km: float = AREA_KM, seed: int = 42) -> pd.DataFrame:
    """Place towers on a jittered grid (avoids unrealistic clumping) with random attributes."""
    if n_towers < 2:
        raise ValueError("n_towers must be at least 2")
    rng = np.random.default_rng(seed)
    cols = int(np.ceil(np.sqrt(n_towers)))
    rows = int(np.ceil(n_towers / cols))
    cw, ch = area_km / cols, area_km / rows
    slots = [(c, r) for r in range(rows) for c in range(cols)]
    chosen = rng.choice(len(slots), size=n_towers, replace=False)
    chosen.sort()
    records = []
    for i, idx in enumerate(chosen):
        c, r = slots[idx]
        x = np.clip((c + 0.5) * cw + rng.uniform(-0.3, 0.3) * cw, 0.2, area_km - 0.2)
        y = np.clip((r + 0.5) * ch + rng.uniform(-0.3, 0.3) * ch, 0.2, area_km - 0.2)
        profile = rng.choice(["low", "mid", "high"], p=[0.30, 0.40, 0.30])
        load = {"low": rng.uniform(8, 28), "mid": rng.uniform(35, 65), "high": rng.uniform(70, 92)}[profile]
        bw = float(rng.choice([60, 80, 100]))
        p_idle = rng.uniform(1.8, 2.2)
        records.append({
            "Tower_ID": f"T{i + 1:02d}", "X_km": round(float(x), 3), "Y_km": round(float(y), 3),
            "Tx_Power_dBm": round(float(rng.uniform(38, 44)), 1),
            "Bandwidth_MHz": bw,
            "Capacity_Mbps": bw * 8.0,
            "Base_Load_Pct": round(float(load), 1),
            "P_Idle_kW": round(float(p_idle), 3),
            "P_Max_kW": round(float(p_idle + rng.uniform(1.6, 2.2)), 3),
        })
    return pd.DataFrame(records)


def distance_matrix(xs, ys, towers: pd.DataFrame) -> np.ndarray:
    """Euclidean distances (km): shape (n_points, n_towers)."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    dx = xs[:, None] - towers["X_km"].to_numpy()[None, :]
    dy = ys[:, None] - towers["Y_km"].to_numpy()[None, :]
    return np.sqrt(dx ** 2 + dy ** 2)


def signal_matrix(xs, ys, towers: pd.DataFrame, shadowing: np.ndarray | None = None) -> np.ndarray:
    """RSRP (dBm) from every tower at every point: shape (n_points, n_towers)."""
    d = distance_matrix(xs, ys, towers)
    sh = 0.0 if shadowing is None else shadowing
    return rsrp_dbm(towers["Tx_Power_dBm"].to_numpy()[None, :], d, sh)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
def generate_network_dataset(n_users: int = 6000, towers: pd.DataFrame | None = None, seed: int = 42,
                             target_handover_rate: float = 0.30) -> pd.DataFrame:
    """
    Generate the synthetic per-user dataset.

    Label design (Handover_Required): a latent "handover need" score is built from the
    signal gap to the best neighbour, weak serving signal, user speed, distance and load
    imbalance, plus random noise that represents unobserved factors. The label is the
    thresholded score, so the ML model can learn a strong - but not perfect - relationship.
    The label is never used to create any feature (no target leakage).
    """
    rng = np.random.default_rng(seed)
    if towers is None:
        towers = generate_towers(seed=seed)
    n_t = len(towers)

    # --- user positions: 70% uniform, 30% clustered around random towers (hotspots)
    n_hot = int(0.3 * n_users)
    xs = rng.uniform(0, AREA_KM, n_users)
    ys = rng.uniform(0, AREA_KM, n_users)
    hot = rng.integers(0, n_t, n_hot)
    xs[:n_hot] = np.clip(towers["X_km"].to_numpy()[hot] + rng.normal(0, 1.2, n_hot), 0, AREA_KM)
    ys[:n_hot] = np.clip(towers["Y_km"].to_numpy()[hot] + rng.normal(0, 1.2, n_hot), 0, AREA_KM)

    dist = distance_matrix(xs, ys, towers)
    sig = signal_matrix(xs, ys, towers, rng.normal(0, SHADOW_SIGMA_DB, size=(n_users, n_t)))

    # --- serving tower: usually one of the 3 nearest ("sticky" connections), not always the best
    order = np.argsort(dist, axis=1)
    rank = rng.choice([0, 1, 2], size=n_users, p=[0.60, 0.25, 0.15])
    rank = np.minimum(rank, n_t - 1)
    ar = np.arange(n_users)
    serving = order[ar, rank]
    serv_sig = sig[ar, serving]
    serv_dist = dist[ar, serving]

    # --- best neighbour = strongest other tower
    masked = sig.copy()
    masked[ar, serving] = -np.inf
    neigh = masked.argmax(axis=1)
    neigh_sig = sig[ar, neigh]

    # --- loads
    base_load = towers["Base_Load_Pct"].to_numpy()
    serv_load = np.clip(base_load[serving] + rng.normal(0, 4, n_users), 1, 100)
    neigh_load = np.clip(base_load[neigh] + rng.normal(0, 4, n_users), 1, 100)

    # --- user mobility (pedestrian / urban / vehicular)
    cat = rng.choice(3, size=n_users, p=[0.35, 0.20, 0.45])
    user_speed = np.where(cat == 0, rng.uniform(0, 6, n_users),
                  np.where(cat == 1, rng.uniform(6, 30, n_users), rng.uniform(30, 120, n_users)))

    bw = towers["Bandwidth_MHz"].to_numpy()[serving]
    speed = estimate_speed_mbps(serv_sig, serv_load, bw) * rng.lognormal(0, 0.10, n_users)
    latency = np.clip(estimate_latency_ms(serv_sig, serv_load, user_speed) + rng.exponential(1.5, n_users), 1, 150)
    power = tower_power_kw(serv_load, towers["P_Idle_kW"].to_numpy()[serving], towers["P_Max_kW"].to_numpy()[serving])

    # --- latent handover-need score -> label
    diff = neigh_sig - serv_sig
    z = (0.50 * (diff - 4.0)
         + 0.15 * np.maximum(0.0, -98.0 - serv_sig)
         + 0.012 * (user_speed - 50.0)
         + 0.30 * (serv_dist - 2.5)
         + 0.04 * (serv_load - neigh_load)
         + rng.normal(0, 1.5, n_users))
    threshold = np.quantile(z, 1.0 - target_handover_rate)
    label = (z > threshold).astype(int)

    ids = towers["Tower_ID"].to_numpy()
    df = pd.DataFrame({
        "User_ID": [f"U{i + 1:05d}" for i in range(n_users)],
        "Tower_ID": ids[serving],
        "Service_Type": rng.choice(SERVICE_TYPES, size=n_users, p=[0.60, 0.15, 0.25]),
        "X_km": xs.round(3), "Y_km": ys.round(3),
        "Signal_Strength_dBm": serv_sig.round(2),
        "Neighbor_Tower_ID": ids[neigh],
        "Neighbor_Signal_dBm": neigh_sig.round(2),
        "Network_Speed_Mbps": speed.round(2),
        "Latency_ms": latency.round(2),
        "User_Speed_kmh": user_speed.round(1),
        "Tower_Distance_km": serv_dist.round(3),
        "Network_Traffic_Percentage": serv_load.round(1),
        "Neighbor_Traffic_Percentage": neigh_load.round(1),
        "Bandwidth_MHz": bw,
        "Power_Consumption_kWh": np.asarray(power).round(3),
        "Handover_Required": label,
    })
    return df[DATASET_COLUMNS]


def snapshot_for_location(towers: pd.DataFrame, x_km: float, y_km: float, serving_id: str,
                          user_speed_kmh: float) -> dict:
    """Noise-free network conditions for a user at (x, y) attached to `serving_id`."""
    sig = signal_matrix([x_km], [y_km], towers)[0]
    dist = distance_matrix([x_km], [y_km], towers)[0]
    ids = towers["Tower_ID"].tolist()
    if serving_id not in ids:
        raise ValueError(f"Unknown tower {serving_id}")
    s = ids.index(serving_id)
    others = [i for i in range(len(ids)) if i != s]
    n = others[int(np.argmax(sig[others]))]
    load_s, load_n = float(towers.loc[s, "Base_Load_Pct"]), float(towers.loc[n, "Base_Load_Pct"])
    return {
        "serving_id": serving_id, "neighbor_id": ids[n],
        "Signal_Strength_dBm": float(sig[s]), "Neighbor_Signal_dBm": float(sig[n]),
        "User_Speed_kmh": float(user_speed_kmh), "Tower_Distance_km": float(dist[s]),
        "Network_Traffic_Percentage": load_s, "Neighbor_Traffic_Percentage": load_n,
        "Latency_ms": float(estimate_latency_ms(sig[s], load_s, user_speed_kmh)),
        "Network_Speed_Mbps": float(estimate_speed_mbps(sig[s], load_s, towers.loc[s, "Bandwidth_MHz"])),
    }
