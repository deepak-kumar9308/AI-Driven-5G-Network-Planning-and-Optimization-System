"""Generate the synthetic network dataset and tower table (reproducible via --seed)."""
import argparse
from pathlib import Path

from modules.network_simulator import generate_network_dataset, generate_towers
from utils.data_preprocessing import validate_dataset

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"


def main(n_users: int = 6000, n_towers: int = 12, seed: int = 42):
    DATASET_DIR.mkdir(exist_ok=True)
    towers = generate_towers(n_towers=n_towers, seed=seed)
    df = generate_network_dataset(n_users=n_users, towers=towers, seed=seed)
    validate_dataset(df)  # raises ValueError if anything is out of range
    towers.to_csv(DATASET_DIR / "towers.csv", index=False)
    df.to_csv(DATASET_DIR / "network_data.csv", index=False)
    print(f"Saved {len(df)} synthetic user records and {len(towers)} towers to {DATASET_DIR}")
    print(f"Handover_Required rate: {df['Handover_Required'].mean():.1%}  (SYNTHETIC DATA)")
    return df, towers


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--users", type=int, default=6000)
    p.add_argument("--towers", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.users, a.towers, a.seed)
