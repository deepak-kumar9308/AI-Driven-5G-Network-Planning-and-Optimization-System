"""K-Means clustering of user records into traffic/quality groups (unsupervised, actually trained)."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["Network_Traffic_Percentage", "Latency_ms", "Network_Speed_Mbps", "Signal_Strength_dBm"]


def cluster_traffic(df: pd.DataFrame, k: int = 3, seed: int = 42):
    """Return (df with Cluster + PCA columns, profile table, silhouette score)."""
    if not 2 <= k <= 8:
        raise ValueError("k must be between 2 and 8")
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    out = df.copy()
    out["_raw"] = km.labels_
    # number clusters by ascending mean traffic (1 = lowest average load). Clusters also differ in
    # signal/latency, so the profile table - not the number - describes what each group is.
    order = out.groupby("_raw")["Network_Traffic_Percentage"].mean().sort_values().index.tolist()
    names = [f"Cluster {i + 1}" for i in range(k)]
    mapping = {raw: names[i] for i, raw in enumerate(order)}
    out["Cluster"] = out["_raw"].map(mapping)
    pcs = PCA(n_components=2, random_state=seed).fit_transform(X)
    out["PC1"], out["PC2"] = pcs[:, 0], pcs[:, 1]
    out = out.drop(columns="_raw")
    sil = float(silhouette_score(X, km.labels_, sample_size=min(2000, len(df)), random_state=seed))
    profile = out.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2)
    profile["Users"] = out["Cluster"].value_counts()
    return out, profile.reset_index(), sil
