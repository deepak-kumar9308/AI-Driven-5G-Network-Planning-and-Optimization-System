"""
Module 2 - Beamforming (SIMPLIFIED SIMULATION).

This is NOT real antenna control. It models a uniform linear array (ULA, half-wavelength
spacing) and applies its normalised array-factor pattern to the desired link and to
interfering towers, comparing an omni-directional baseline against beamformed transmission.

Assumptions: one scheduled user per tower per time slot, azimuth-only pattern, small random
beam-steering error, fixed -25 dB side/back-lobe floor, interference scaled by neighbour load.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.network_simulator import AREA_KM, distance_matrix, rsrp_dbm

NOISE_DBM = -122.0        # thermal noise per resource element (30 kHz, 7 dB noise figure)
SIDELOBE_FLOOR = 10 ** (-25 / 10)


def array_factor_power(offset_deg, n_elements: int):
    """Normalised ULA array-factor power (0..1) at an angular offset from the beam centre."""
    psi = np.pi * np.sin(np.radians(np.asarray(offset_deg, dtype=float)))
    num = np.sin(n_elements * psi / 2.0)
    den = n_elements * np.sin(psi / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        af = np.where(np.abs(den) < 1e-9, 1.0, (num / den) ** 2)
    return np.maximum(af, SIDELOBE_FLOOR)


def beam_gain_db(offset_deg, n_elements: int):
    """Directional gain in dB relative to an omni-directional (0 dB) transmitter."""
    return 10 * np.log10(n_elements * array_factor_power(offset_deg, n_elements))


def beam_pattern(n_elements: int, steer_deg: float = 0.0, n_points: int = 361) -> pd.DataFrame:
    """Gain vs absolute azimuth for plotting (polar). 0 deg = +Y axis (north), clockwise."""
    az = np.linspace(-180, 180, n_points)
    offset = (az - steer_deg + 180) % 360 - 180
    return pd.DataFrame({"Azimuth_deg": az, "Gain_dB": beam_gain_db(offset, n_elements)})


def _bearing(dx, dy):
    return np.degrees(np.arctan2(dx, dy))  # 0 = north, clockwise


def simulate_beamforming(towers: pd.DataFrame, n_users: int = 400, n_elements: int = 16, n_slots: int = 40,
                         steering_error_deg: float = 2.0, seed: int = 42) -> pd.DataFrame:
    """
    Returns one row per (slot, scheduled user) with SINR / signal / interference before and
    after beamforming.
    """
    if n_elements < 2:
        raise ValueError("n_elements must be >= 2")
    rng = np.random.default_rng(seed)
    xs, ys = rng.uniform(0, AREA_KM, n_users), rng.uniform(0, AREA_KM, n_users)
    dist = distance_matrix(xs, ys, towers)
    sig = rsrp_dbm(towers["Tx_Power_dBm"].to_numpy()[None, :], dist, rng.normal(0, 4, dist.shape))
    serving = sig.argmax(axis=1)
    load = towers["Base_Load_Pct"].to_numpy() / 100.0
    tx, ty = towers["X_km"].to_numpy(), towers["Y_km"].to_numpy()
    noise_mw = 10 ** (NOISE_DBM / 10)
    served_by = {t: np.where(serving == t)[0] for t in range(len(towers))}

    rows = []
    for slot in range(n_slots):
        sched = {t: int(rng.choice(u)) for t, u in served_by.items() if len(u)}
        for s, u in sched.items():
            sig_mw = 10 ** (sig[u, s] / 10)
            i_base, i_beam = 0.0, 0.0
            for j, uj in sched.items():
                if j == s:
                    continue
                p_j = 10 ** (sig[u, j] / 10) * load[j]          # omni interference from tower j
                beam_dir = _bearing(xs[uj] - tx[j], ys[uj] - ty[j])
                to_victim = _bearing(xs[u] - tx[j], ys[u] - ty[j])
                off = (to_victim - beam_dir + 180) % 360 - 180
                i_base += p_j
                i_beam += p_j * n_elements * float(array_factor_power(off, n_elements))
            err = rng.normal(0, steering_error_deg)
            g_desired = n_elements * float(array_factor_power(err, n_elements))
            rows.append({
                "Slot": slot, "Tower_ID": towers.loc[s, "Tower_ID"], "User_Index": u,
                "Signal_Before_dBm": 10 * np.log10(sig_mw),
                "Signal_After_dBm": 10 * np.log10(sig_mw * g_desired),
                "Interference_Before_dBm": 10 * np.log10(i_base + 1e-15),
                "Interference_After_dBm": 10 * np.log10(i_beam + 1e-15),
                "SINR_Before_dB": 10 * np.log10(sig_mw / (noise_mw + i_base)),
                "SINR_After_dB": 10 * np.log10(sig_mw * g_desired / (noise_mw + i_beam)),
            })
    return pd.DataFrame(rows)


def summarize_beamforming(df: pd.DataFrame) -> dict:
    """Aggregate statistics from `simulate_beamforming`."""
    return {
        "sinr_before_db": float(df["SINR_Before_dB"].mean()),
        "sinr_after_db": float(df["SINR_After_dB"].mean()),
        "sinr_gain_db": float((df["SINR_After_dB"] - df["SINR_Before_dB"]).mean()),
        "interference_change_db": float((df["Interference_After_dBm"] - df["Interference_Before_dBm"]).median()),
        "signal_gain_db": float((df["Signal_After_dBm"] - df["Signal_Before_dBm"]).mean()),
        "pct_users_improved": float((df["SINR_After_dB"] > df["SINR_Before_dB"]).mean() * 100),
    }
