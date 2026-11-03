"""Plotly figure builders used by the dashboard (plus small Matplotlib/Seaborn helpers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = {"primary": "#1f5fbf", "accent": "#e07b00", "good": "#2e8b57", "bad": "#c0392b", "grey": "#7f8c8d"}
LAYOUT = dict(margin=dict(l=10, r=10, t=45, b=10), font=dict(family="Inter, Segoe UI, sans-serif", size=13),
              legend=dict(orientation="h", y=-0.2))


def _style(fig, title=None, height=420):
    fig.update_layout(title=title, height=height, **LAYOUT)
    return fig


def signal_heatmap(xs, ys, grid, towers, title="Best-server signal (dBm)", extra=None, sleeping=None, threshold=None):
    fig = go.Figure(go.Heatmap(x=xs, y=ys, z=grid, colorscale="RdYlGn", zmin=-110, zmax=-60,
                               colorbar=dict(title="dBm")))
    if threshold is not None:
        fig.add_trace(go.Contour(x=xs, y=ys, z=grid, showscale=False, contours=dict(start=threshold, end=threshold, size=1,
                       coloring="none"), line=dict(color="black", width=2, dash="dash"), name=f"{threshold:.0f} dBm", showlegend=True))
    sleeping = sleeping or []
    on = towers[~towers["Tower_ID"].isin(sleeping)]
    fig.add_trace(go.Scatter(x=on["X_km"], y=on["Y_km"], mode="markers+text", text=on["Tower_ID"], textposition="top center",
                             marker=dict(symbol="triangle-up", size=13, color="black"), name="Tower"))
    if sleeping:
        sl = towers[towers["Tower_ID"].isin(sleeping)]
        fig.add_trace(go.Scatter(x=sl["X_km"], y=sl["Y_km"], mode="markers+text", text=sl["Tower_ID"], textposition="top center",
                                 marker=dict(symbol="x", size=13, color=PALETTE["bad"]), name="Sleeping tower"))
    if extra is not None and len(extra):
        fig.add_trace(go.Scatter(x=extra["X_km"], y=extra["Y_km"], mode="markers+text", text=extra["Recommended_ID"],
                                 textposition="top center", marker=dict(symbol="star", size=15, color=PALETTE["accent"],
                                 line=dict(color="black", width=1)), name="Recommended"))
    fig.update_xaxes(title="X (km)")
    fig.update_yaxes(title="Y (km)", scaleanchor="x")
    return _style(fig, title, 520)


def histogram(df, col, title, color=None, nbins=40):
    return _style(px.histogram(df, x=col, color=color, nbins=nbins, barmode="overlay", opacity=0.75), title, 340)


def before_after_hist(before, after, label, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=before, name="Before (omni)", opacity=0.65, marker_color=PALETTE["grey"]))
    fig.add_trace(go.Histogram(x=after, name="After (beamformed)", opacity=0.65, marker_color=PALETTE["primary"]))
    fig.update_layout(barmode="overlay", xaxis_title=label)
    return _style(fig, title, 360)


def polar_beam(patterns: dict):
    fig = go.Figure()
    for name, p in patterns.items():
        r = p["Gain_dB"].clip(lower=-25) + 25
        fig.add_trace(go.Scatterpolar(theta=p["Azimuth_deg"], r=r, mode="lines", name=name))
    fig.update_layout(polar=dict(angularaxis=dict(rotation=90, direction="clockwise"),
                                 radialaxis=dict(title="Gain (dB, +25 offset)")))
    return _style(fig, "Beam pattern (relative to omni = 0 dB; radius = gain + 25 dB)", 440)


def confusion_heatmap(cm, title="Confusion matrix"):
    fig = px.imshow(np.array(cm), text_auto=True, color_continuous_scale="Blues", x=["No handover", "Handover"],
                    y=["No handover", "Handover"], labels=dict(x="Predicted", y="Actual", color="Count"))
    return _style(fig, title, 380)


def bar(df, x, y, color=None, title="", barmode="group", height=380):
    return _style(px.bar(df, x=x, y=y, color=color, barmode=barmode), title, height)


def feature_importance_bar(importance: dict):
    d = pd.Series(importance).sort_values()
    return _style(px.bar(x=d.values, y=d.index, orientation="h", labels=dict(x="Importance", y="")), "Random Forest feature importance", 380)


def line(df, x, y, color=None, title=""):
    return _style(px.line(df, x=x, y=y, color=color, markers=True), title, 360)


def tower_map(towers, users=None, color_col=None, title="Simulated towers and users"):
    fig = go.Figure()
    if users is not None:
        s = users.sample(min(len(users), 1500), random_state=1)
        fig.add_trace(go.Scatter(x=s["X_km"], y=s["Y_km"], mode="markers", name="Users",
                                 marker=dict(size=4, color=s[color_col], colorscale="RdYlGn", showscale=True,
                                             colorbar=dict(title=color_col.replace("_", " ")), opacity=0.7)))
    fig.add_trace(go.Scatter(x=towers["X_km"], y=towers["Y_km"], mode="markers+text", text=towers["Tower_ID"], textposition="top center",
                             marker=dict(symbol="triangle-up", size=14, color="black"), name="Tower"))
    fig.update_xaxes(title="X (km)")
    fig.update_yaxes(title="Y (km)", scaleanchor="x")
    return _style(fig, title, 500)


def cluster_scatter(df):
    return _style(px.scatter(df.sample(min(2500, len(df)), random_state=1), x="PC1", y="PC2", color="Cluster", opacity=0.6),
                  "K-Means clusters (PCA projection)", 420)
