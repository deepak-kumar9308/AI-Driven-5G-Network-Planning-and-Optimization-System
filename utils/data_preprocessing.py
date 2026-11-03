"""Data validation, feature engineering and train/test splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Sensible physical ranges used for validation: column -> (min, max)
VALID_RANGES = {
    "Signal_Strength_dBm": (-130, -40),
    "Neighbor_Signal_dBm": (-130, -40),
    "Network_Speed_Mbps": (0, 3000),
    "Latency_ms": (0, 500),
    "User_Speed_kmh": (0, 350),
    "Tower_Distance_km": (0, 50),
    "Network_Traffic_Percentage": (0, 100),
    "Neighbor_Traffic_Percentage": (0, 100),
    "Bandwidth_MHz": (1, 400),
    "Power_Consumption_kWh": (0, 50),
    "Handover_Required": (0, 1),
}
REQUIRED_COLUMNS = ["User_ID", "Tower_ID"] + list(VALID_RANGES)

# Model inputs. The label (Handover_Required) and identifiers are NOT included -> no leakage.
BASE_FEATURES = [
    "Signal_Strength_dBm", "Neighbor_Signal_dBm", "User_Speed_kmh", "Tower_Distance_km",
    "Network_Traffic_Percentage", "Neighbor_Traffic_Percentage", "Latency_ms", "Network_Speed_Mbps",
]
ENGINEERED_FEATURES = ["Signal_Diff_dB", "Load_Diff_Pct"]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
TARGET = "Handover_Required"


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError when the dataset is malformed; return df unchanged otherwise."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df[REQUIRED_COLUMNS].isna().any().any():
        bad = df[REQUIRED_COLUMNS].columns[df[REQUIRED_COLUMNS].isna().any()].tolist()
        raise ValueError(f"Dataset contains missing values in: {bad}")
    if df["User_ID"].duplicated().any():
        raise ValueError("Dataset contains duplicate User_ID values")
    for col, (lo, hi) in VALID_RANGES.items():
        if not df[col].between(lo, hi).all():
            raise ValueError(f"Column '{col}' has values outside the valid range [{lo}, {hi}]")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features computed ONLY from measurable inputs (never from the label)."""
    out = df.copy()
    out["Signal_Diff_dB"] = out["Neighbor_Signal_dBm"] - out["Signal_Strength_dBm"]
    out["Load_Diff_Pct"] = out["Network_Traffic_Percentage"] - out["Neighbor_Traffic_Percentage"]
    return out


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate, engineer features, return (X, y)."""
    validate_dataset(df)
    full = add_engineered_features(df)
    return full[FEATURES], full[TARGET].astype(int)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified, reproducible train/test split -> X_train, X_test, y_train, y_test."""
    X, y = preprocess(df)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
