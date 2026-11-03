"""
Module 3 - Intelligent Handover Management (primary ML module).

Implemented and actually trained here:
  * Random Forest classifier  -> handover prediction (main model, saved with joblib)
  * K-Nearest Neighbours, SVM -> comparison classifiers (trained and evaluated on the same split)
  * A3-style rule baseline    -> "handover if neighbour is > hysteresis dB stronger"
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.data_preprocessing import FEATURES, add_engineered_features, split_dataset
from utils.metrics import classification_metrics

SEED = 42


def build_models(seed: int = SEED) -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                                class_weight="balanced", random_state=seed, n_jobs=-1),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, class_weight="balanced",
                                                          random_state=seed)),
    }


def train_and_evaluate(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Train all models on the training split, evaluate on the held-out test split."""
    X_train, X_test, y_train, y_test = split_dataset(df, seed=seed)
    models, results = build_models(seed), {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = classification_metrics(y_test, model.predict(X_test))
    rf = models["Random Forest"]
    cv = cross_val_score(RandomForestClassifier(**rf.get_params()), X_train, y_train, cv=5, scoring="f1")
    return {
        "model": rf, "features": FEATURES, "metrics": results,
        "cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std()),
        "feature_importance": dict(zip(FEATURES, map(float, rf.feature_importances_))),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)), "seed": seed,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "comparison_models": {k: v for k, v in models.items() if k != "Random Forest"},
    }


def save_bundle(bundle: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}. Run `python train_model.py` first.")
    return joblib.load(p)


def _row_frame(conditions: dict) -> pd.DataFrame:
    base = ["Signal_Strength_dBm", "Neighbor_Signal_dBm", "User_Speed_kmh", "Tower_Distance_km",
            "Network_Traffic_Percentage", "Neighbor_Traffic_Percentage", "Latency_ms", "Network_Speed_Mbps"]
    missing = [k for k in base if k not in conditions]
    if missing:
        raise ValueError(f"Missing input conditions: {missing}")
    return add_engineered_features(pd.DataFrame([{k: float(conditions[k]) for k in base}]))[FEATURES]


def predict_handover(bundle: dict, conditions: dict, threshold: float = 0.5) -> dict:
    """Predict handover need for one set of conditions using the trained Random Forest."""
    proba = float(bundle["model"].predict_proba(_row_frame(conditions))[0, 1])
    return {"handover_probability": proba, "handover_required": proba >= threshold}


def predict_dataframe(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add Predicted_Handover and Handover_Probability columns to a dataset copy."""
    X = add_engineered_features(df)[FEATURES]
    out = df.copy()
    out["Handover_Probability"] = bundle["model"].predict_proba(X)[:, 1].round(4)
    out["Predicted_Handover"] = (out["Handover_Probability"] >= 0.5).astype(int)
    return out


def a3_rule_predict(df: pd.DataFrame, hysteresis_db: float = 3.0) -> np.ndarray:
    """Classic rule baseline: hand over when the neighbour is more than `hysteresis_db` stronger."""
    return ((df["Neighbor_Signal_dBm"] - df["Signal_Strength_dBm"]) > hysteresis_db).astype(int).to_numpy()
