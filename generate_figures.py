"""Create static report figures (Matplotlib/Seaborn) from the real simulation outputs -> assets/project_images/.
These are generated charts, NOT dashboard screenshots."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from modules.energy_optimization import optimize_energy
from modules.handover_prediction import load_bundle
from modules.rf_planning import signal_grid
from modules.beamforming import simulate_beamforming

ROOT = Path(__file__).parent
OUT = ROOT / "assets" / "project_images"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    towers = pd.read_csv(ROOT / "dataset" / "towers.csv")
    sns.set_theme(style="whitegrid")

    xs, ys, grid, _ = signal_grid(towers, resolution=80)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(xs, ys, grid, cmap="RdYlGn", vmin=-110, vmax=-60, shading="auto")
    ax.scatter(towers["X_km"], towers["Y_km"], marker="^", c="black", s=60)
    ax.set(title="Simulated best-server signal (dBm)", xlabel="X (km)", ylabel="Y (km)")
    fig.colorbar(im, ax=ax, label="dBm")
    fig.tight_layout(); fig.savefig(OUT / "rf_coverage_map.png", dpi=130); plt.close(fig)

    bundle = load_bundle(ROOT / "models" / "handover_model.pkl")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(np.array(bundle["metrics"]["Random Forest"]["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No HO", "HO"], yticklabels=["No HO", "HO"])
    ax.set(title="Random Forest confusion matrix (test set)", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=130); plt.close(fig)

    r = optimize_energy(towers)
    both = pd.concat([r["before"].assign(Scenario="Before"), r["after"].assign(Scenario="After")])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=both, x="Tower_ID", y="Power_kW", hue="Scenario", ax=ax)
    ax.set(title=f"Power per tower - estimated saving {r['summary']['savings_pct']:.1f}%")
    fig.tight_layout(); fig.savefig(OUT / "energy_before_after.png", dpi=130); plt.close(fig)

    res = simulate_beamforming(towers)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(res["SINR_Before_dB"], color="grey", label="Before (omni)", ax=ax, alpha=0.6)
    sns.histplot(res["SINR_After_dB"], color="#1f5fbf", label="After (beamformed)", ax=ax, alpha=0.6)
    ax.set(title="SINR distribution (simplified simulation)", xlabel="SINR (dB)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "beamforming_sinr.png", dpi=130); plt.close(fig)
    print("Saved figures to", OUT)


if __name__ == "__main__":
    main()
