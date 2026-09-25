"""Matplotlib figures for the Graph Explorer app.

Two figures:
  * plot_landmark_timeline  - alternating-side timeline of landmark articles
    * plot_cluster_timeline   - CiteSpace-style "timeline view": one horizontal
                                                            lane per breakthrough cluster, study-cluster
                                                            points coloured by bin, and citation arcs.
"""
from __future__ import annotations

import re
import textwrap
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import FancyArrowPatch, Patch, PathPatch
from matplotlib.path import Path

YEAR_TICK_COLOR = "#c026d3"

_STOPWORDS = set(
    """
    a about after all also among an and are as at based between but by can
    during effect effects for from has have in into is it its of on or over
    study studies patients patient veterans veteran using use used versus
    that the their these this through to under with within after
    trial trials clinical human animal analysis review randomized
    """.split()
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _fractional_year(dates: pd.Series) -> pd.Series:
    """Datetime-like series -> float year (e.g. 2010.5)."""
    d = pd.to_datetime(dates, errors="coerce")
    return d.dt.year + (d.dt.dayofyear - 1) / 365.25


def cluster_label(titles, n_words: int = 2) -> str:
    """Cheap cluster name: the most frequent non-stopwords in the titles."""
    words = re.findall(r"[a-z][a-z\-]{3,}", " ".join(map(str, titles)).lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    return " ".join(w for w, _ in counts.most_common(n_words)) or "misc"


def _arc(ax, p0, p1, color, alpha, lw, bulge):
    """Quadratic-bezier arc between two points, bowed toward earlier years."""
    ctrl = ((p0[0] + p1[0]) / 2 - bulge, (p0[1] + p1[1]) / 2)
    path = Path([p0, ctrl, p1], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(
        PathPatch(path, fc="none", ec=color, alpha=alpha, lw=lw, zorder=1)
    )


# ── cluster timeline (CiteSpace-style) ───────────────────────────────────────
def plot_cluster_timeline(
    df_studies: pd.DataFrame,
    df_edges: pd.DataFrame | None = None,
    *,
    max_clusters: int = 10,
    max_links: int = 40,
):
    """Plot one lane per breakthrough cluster and points for study clusters.

    Expects df_studies columns: BreakthruCluster, Cluster, Bin, and Date.
    Title is optional and used for lane labels. df_edges should contain
    Source_cluster, Target_cluster, and count.
    Returns a Figure, or None if there is nothing to draw.
    """
    df = df_studies.copy()
    required = {"BreakthruCluster", "Cluster", "Bin", "Date"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing timeline columns: {sorted(missing)}")
    df = df.dropna(subset=["BreakthruCluster", "Cluster", "Bin"])
    df["BreakthruCluster"] = df["BreakthruCluster"].astype(str)
    df["Cluster"] = df["Cluster"].astype(str)
    df["Bin"] = df["Bin"].astype(str).str.lower()
    df["t"] = _fractional_year(df["Date"])
    df = df.dropna(subset=["t"])
    if df.empty:
        return None

    # Largest breakthrough clusters first (top lane = biggest lane).
    breakthrough_clusters = list(
        df["BreakthruCluster"].value_counts().index[:max_clusters]
    )
    df = df[df["BreakthruCluster"].isin(breakthrough_clusters)]
    lane_gap = 2.0
    lane = {
        cluster: (len(breakthrough_clusters) - 1 - i) * lane_gap
        for i, cluster in enumerate(breakthrough_clusters)
    }
    df["year"] = np.floor(df["t"]).astype(int)

    point_clusters = (
        df.groupby(["BreakthruCluster", "Cluster"], sort=False)["Bin"]
        .first()
        .reset_index()
    )
    point_clusters["y"] = point_clusters["BreakthruCluster"].map(lane)
    cluster_y = point_clusters.set_index("Cluster")["y"].to_dict()

    per = (
        df.groupby(["BreakthruCluster", "Cluster", "Bin"], sort=False)
        .agg(t=("t", "median"), n=("t", "size"))
        .reset_index()
    )
    per["year"] = np.floor(per["t"]).astype(int)
    per["y"] = per["Cluster"].map(cluster_y)

    y_min = int(np.floor(per["t"].min()))
    y_max = int(np.ceil(per["t"].max()))
    span = max(y_max - y_min, 1)
    bin_colors = {
        "clinical": "#177a5e",
        "human": "#1d6fa4",
        "animal": "#7a4fb5",
    }
    default_color = "#64748b"

    fig = go.Figure()

    # Citation edges between study clusters, anchored at their median dates.
    if df_edges is not None and len(df_edges):
        anchors = df.groupby("Cluster")["t"].median()
        e = df_edges.copy()
        e["Source_cluster"] = e["Source_cluster"].astype(str)
        e["Target_cluster"] = e["Target_cluster"].astype(str)
        e = e[
            e["Source_cluster"].isin(cluster_y)
            & e["Target_cluster"].isin(cluster_y)
            & (e["Source_cluster"] != e["Target_cluster"])
        ].nlargest(max_links, "count")
        cmax = e["count"].max() if len(e) else 1
        for edge_index, (_, r) in enumerate(e.iterrows()):
            s, t = r["Source_cluster"], r["Target_cluster"]
            start_x, end_x = anchors[s], anchors[t]
            start_y, end_y = cluster_y[s], cluster_y[t]
            curve_t = np.linspace(0, 1, 25)
            control_x = (start_x + end_x) / 2
            control_y = (
                (start_y + end_y) / 2
                + (0.35 * lane_gap if edge_index % 2 == 0 else -0.35 * lane_gap)
            )
            curve_x = (
                (1 - curve_t) ** 2 * start_x
                + 2 * (1 - curve_t) * curve_t * control_x
                + curve_t ** 2 * end_x
            )
            curve_y = (
                (1 - curve_t) ** 2 * start_y
                + 2 * (1 - curve_t) * curve_t * control_y
                + curve_t ** 2 * end_y
            )
            fig.add_trace(
                go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode="lines",
                    line={
                        "color": "rgba(71, 85, 105, 0.45)",
                        "width": 1 + 4 * r["count"] / cmax,
                    },
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Breakthrough-cluster lanes sit behind the point-level study clusters.
    for breakthrough_cluster in breakthrough_clusters:
        sub = df[df["BreakthruCluster"] == breakthrough_cluster]
        fig.add_trace(
            go.Scatter(
                x=[sub["t"].min() - 0.4, sub["t"].max() + 0.4],
                y=[lane[breakthrough_cluster]] * 2,
                mode="lines",
                line={"color": "#cbd5e1", "width": 1.6},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # One hoverable point per study cluster, coloured by study bin.
    for bin_name in ("clinical", "human", "animal"):
        points = per[per["Bin"] == bin_name]
        if points.empty:
            continue
        customdata = points[["Cluster", "BreakthruCluster", "Bin", "n"]].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=points["t"],
                y=points["y"],
                mode="markers",
                name=bin_name.capitalize(),
                marker={
                    "size": 12 + 22 * (points["n"] / per["n"].max()) ** 0.9,
                    "color": bin_colors.get(bin_name, default_color),
                    "line": {"color": "white", "width": 1},
                    "opacity": 0.85,
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>Cluster %{customdata[0]}</b><br>"
                    "Breakthrough cluster: %{customdata[1]}<br>"
                    "Study type: %{customdata[2]}<br>"
                    "Studies: %{customdata[3]}<br>"
                    "Median publication year: %{x:.0f}<extra></extra>"
                ),
            )
        )

    # Years run along the top; breakthrough-cluster lanes run horizontally.
    step = max(1, int(round(span / 8)))
    tick_years = np.arange(y_min, y_max + 1, step)
    fig.update_layout(
        height=max(420, 90 * len(breakthrough_clusters) + 150),
        margin={"l": 20, "r": 240, "t": 70, "b": 35},
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"title": {"text": "Study type"}, "x": 1.02, "y": 1},
        xaxis={
            "side": "top",
            "tickmode": "array",
            "tickvals": tick_years,
            "ticktext": [str(year) for year in tick_years],
            "showgrid": False,
            "zeroline": False,
            "range": [y_min - 0.5, y_max + 0.5],
        },
        yaxis={
            "tickmode": "array",
            "tickvals": [lane[c] for c in breakthrough_clusters],
            "ticktext": [f"# {c}" for c in breakthrough_clusters],
            "range": [-1, (len(breakthrough_clusters) - 1) * lane_gap + 1],
            "showgrid": False,
            "zeroline": False,
            "side": "right",
        },
        showlegend=True,
    )
    return fig


# ── landmark-article timeline ────────────────────────────────────────────────
def plot_landmark_timeline(
    df: pd.DataFrame,
    *,
    wrap: int = 70,
    log_scale: bool = True,
):
    """Alternating left/right timeline. df needs Date, Years, Title.

    Articles that share a value in `Years` are merged into one marker placed
    at their median date.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    if df.empty:
        return None

    dates, labels = [], []
    for _, grp in df.groupby("Years", sort=False):
        dates.append(grp["Date"].median())
        labels.append(
            "\n".join(
                f"{textwrap.fill(r.Title, width=wrap, break_long_words=False)}\n{r.Date:%b %d, %Y}"
                for r in grp.itertuples()
            )
        )

    levels = np.tile([-1, 1], int(np.ceil(len(dates) / 2)))[: len(dates)]

    fig, ax = plt.subplots(figsize=(14, max(10, 1.4 * len(dates) + 2)))
    ax.axvline(0, color="black", lw=1.5)
    ax.scatter(np.zeros(len(dates)), dates, s=50, zorder=3, color="red")
    for date, label, side in zip(dates, labels, levels):
        ax.add_patch(
            FancyArrowPatch(
                (0, date), (side, date),
                arrowstyle="-",
                connectionstyle=f"arc3,rad={0.18 * side}",
                transform=ax.transData,
                linestyle="--",
                color="#6b7280",
                linewidth=1,
                zorder=1,
            )
        )
        ax.text(
            side * 1.1, date, label,
            ha="left" if side > 0 else "right", va="center", fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gainsboro", lw=1),
        )
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for side in ("left", "top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlim(-1.5, 1.5)
    ax.margins(y=0.08)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig
