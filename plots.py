
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


def cluster_label(titles, n_words: int = 3) -> str:
    """Return the most frequent non-stopword trigram in the titles."""
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", " ".join(map(str, titles)).lower())
    filtered = [w for w in words if w not in _STOPWORDS and len(w) > 1]

    if len(filtered) >= 3:
        trigrams = [" ".join(filtered[i : i + 3]) for i in range(len(filtered) - 2)]
        counts = Counter(trigrams)
        return counts.most_common(1)[0][0] if counts else "misc"

    counts = Counter(filtered)
    return " ".join(w for w, _ in counts.most_common(max(1, n_words))) or "misc"


# ── cluster timeline (CiteSpace-style) ───────────────────────────────────────
def plot_cluster_timeline(
    df_studies: pd.DataFrame,
    df_edges: pd.DataFrame,
    selected_cluster: str | None = None,
):
    """Plot one lane per breakthrough cluster and points for study clusters.
    Returns a Figure, or None if there is nothing to draw.
    """
    bin_colors = {
        "clinical": "#177a5e",
        "human": "#1d6fa4",
        "animal": "#7a4fb5",
    }
    default_color = "#64748b"


    # Building the dataset
    df = df_studies.copy()
    df["BreakthruCluster"] = df["BreakthruCluster"].astype(str).str.replace(r"(?is)<citations>.*?(?:<citations/>|</citations>)", "", regex=True)
    df["Cluster"] = df["Cluster"].astype(str)
    df["Bin"] = df["Bin"].astype(str).str.lower()
    df["t"] = _fractional_year(df["Date"])

    # Largest breakthrough clusters first (top lane = biggest lane).
    breakthrough_clusters = df["BreakthruCluster"].value_counts().index.to_list()
    lane = {cluster: (len(breakthrough_clusters) - 1 - i) * 2 for i, cluster in enumerate(breakthrough_clusters)}
    
    point_clusters = (df.groupby(["BreakthruCluster", "Cluster", "Bin"], sort=False)
                    .agg(
                            t=("t", "median"),
                            n=("t", "size"),
                            first_year=("Years", "min"),
                            last_year=("Years", "max"),
                        ) 
                      .reset_index())   
    point_clusters["y"] = point_clusters["BreakthruCluster"].map(lane)
    cluster_y = point_clusters.set_index("Cluster")["y"].to_dict()

    
    fig = go.Figure()

   # Citation edges between study clusters, anchored at their median dates.
    anchors = df.groupby("Cluster")["t"].median()

    e = df_edges.nlargest(int(len(df_edges) * .3), "count")

    for edge_index, r in e.iterrows():
        s, t = r["Source_cluster"], r["Target_cluster"]
        start_x, end_x = anchors[s], anchors[t]
        start_y, end_y = cluster_y[s], cluster_y[t]
        curve_t = np.linspace(0, 1, 25)
        control_x = (start_x + end_x) / 2
        control_y = (
            (start_y + end_y) / 2
            + (0.35 * 2 if edge_index % 2 == 0 else -0.35 * 2)
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
                    "width": 1 + 4 * r["count"] /  e["count"].max(),
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
    for bin_name in ("clinical", "human", "animal"):
        points = point_clusters[point_clusters["Bin"] == bin_name]
        customdata = points[["Cluster", "BreakthruCluster", "Bin", "n", "first_year", "last_year"]].to_numpy()
        fig.add_trace(
                go.Scatter(
                    x=points["t"],
                    y=points["y"],
                    mode="markers",
                    name=bin_name.capitalize(),
                    marker={
                        "size": 12 + 22 * (points["n"] / point_clusters["n"].max()) ** 0.9,
                        "color": bin_colors.get(bin_name, default_color),
                        "line": {"color": "white", "width": 1},
                        "opacity": 0.85,
                    },
                    customdata=customdata,
                    hoverlabel={
                        "align": "left",
                        "showarrow": True,
                    },
                    hovertemplate=(
                        "<b>Cluster %{customdata[0]}</b><br>"
                        "Breakthrough: %{customdata[1]}<br>"
                        "Study type: %{customdata[2]}<br>"
                        "Studies: %{customdata[3]}<br>"
                        "Publication year range: %{customdata[4]}-%{customdata[5]}<br>"
                        "Median publication year: %{x:.0f}<extra></extra>"
                    ),
                )
            )   

    if selected_cluster is not None:
        selected_points = point_clusters[point_clusters["Cluster"] == str(selected_cluster)]
        fig.add_trace(
            go.Scatter(
                x=selected_points["t"],
                y=selected_points["y"],
                mode="markers",
                marker={
                    "symbol": "circle-open",
                    "size": 42,
                    "color": "#facc15",
                    "line": {"color": "#facc15", "width": 3},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        ) 

 
    y_min = int(np.floor(point_clusters["t"].min()))
    y_max = int(np.ceil(point_clusters["t"].max()))
    span = max(y_max - y_min, 1)

    # Years run along the top; breakthrough-cluster lanes run horizontally.
    step = max(1, int(round(span / 8)))
    tick_years = np.arange(y_min, y_max + 1, step)
    fig.update_layout(
        template="plotly_dark",
        height=max(420, 90 * len(breakthrough_clusters) + 150),
        margin={"l": 20, "r": 20, "t": 70, "b": 90},
        hovermode="closest",
        legend={
            "title": {"text": "Study type"},
            "x": 0,
            "y": -0.15,
            "xanchor": "left",
            "yanchor": "top",
            "orientation": "h",
        },
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
            "ticktext": [
                f"# {textwrap.fill(c, width=44, subsequent_indent='  ').replace(chr(10), '<br>')}"
                for c in breakthrough_clusters
            ],
            "range": [-1, (len(breakthrough_clusters) - 1) * 2 + 1],
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
