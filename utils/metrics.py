"""Metric helpers: classification metrics, KPI calculation and comparison tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict:
    """Accuracy, precision, recall, F1 and confusion matrix (as nested list)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def pct_change(before: float, after: float) -> float:
    """Relative change in percent; returns 0 when `before` is 0."""
    if before == 0:
        return 0.0
    return (after - before) / abs(before) * 100.0


def compute_kpis(df: pd.DataFrame, towers: pd.DataFrame) -> dict:
    """Dashboard KPIs computed from the dataset - nothing hard-coded."""
    return {
        "towers": int(len(towers)),
        "users": int(df["User_ID"].nunique()),
        "avg_signal_dbm": float(df["Signal_Strength_dBm"].mean()),
        "avg_latency_ms": float(df["Latency_ms"].mean()),
        "avg_speed_mbps": float(df["Network_Speed_Mbps"].mean()),
        "utilization_pct": float(towers["Base_Load_Pct"].mean()),
        "handover_rate_pct": float(df["Handover_Required"].mean() * 100),
    }


def comparison_row(metric: str, before: float, after: float, higher_is_better: bool | None,
                   unit: str, source: str) -> dict:
    """One row of the before/after table. `higher_is_better=None` marks context-only metrics."""
    change = after - before
    if higher_is_better is None or abs(change) < 1e-9:
        verdict = "Neutral / context" if higher_is_better is None else "No change"
    else:
        improved = (change > 0) == higher_is_better
        verdict = "Improved" if improved else "Worse"
    return {"Metric": metric, "Unit": unit, "Baseline": round(float(before), 3),
            "Optimized": round(float(after), 3), "Change": round(float(change), 3),
            "Change_%": round(pct_change(before, after), 2), "Result": verdict, "Source": source}
