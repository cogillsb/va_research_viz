
from __future__ import annotations
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import FancyArrowPatch


YEAR_TICK_COLOR = "#c026d3"


# ── helpers ──────────────────────────────────────────────────────────────────
def _fractional_year(dates: pd.Series) -> pd.Series:
    """Datetime-like series -> float year (e.g. 2010.5)."""
    d = pd.to_datetime(dates, errors="coerce")
    return d.dt.year + (d.dt.dayofyear - 1) / 365.25



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
        "clinical": "rgba(23, 122, 94, 0.5)",
        "human": "rgba(29, 111, 184, 0.5)",
        "animal": "rgba(122, 79, 181, 0.5)",
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

    
    fig = go.FigureWidget()

   # Citation edges between study clusters, anchored at their median dates.
    anchors = df.groupby("Cluster")["t"].median()

    e = df_edges.nlargest(int(len(df_edges) * .3), "count")
    neighbors = []
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
        e_opacity = 0.1
        if selected_cluster is not None:
            if s == selected_cluster:
                e_opacity = 1
                neighbors.append(t)
            
        fig.add_trace(
            go.Scatter(
                x=curve_x,
                y=curve_y,
                mode="lines",
                line={
                    "color": f"rgba(71, 85, 105, {e_opacity})",
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
        customdata = points[
            ["Cluster", "BreakthruCluster", "Bin", "n", "first_year", "last_year"]
        ].astype(object).to_numpy()
        colors = []
        for c in list(points['Cluster'].values):
            add_c = bin_colors[bin_name]
            if selected_cluster is not None:
                if c == selected_cluster or c in neighbors:
                    add_c = add_c[:-4] + "1)"
            colors.append(add_c)

        for column in (0, 1):
            customdata[:, column] = [
                textwrap.fill(str(label), width=30).replace("\n", "<br>")
                for label in customdata[:, column]
            ]

        fig.add_trace(
                go.Scatter(
                    x=points["t"],
                    y=points["y"],
                    mode="markers",
                    name=bin_name.capitalize(),
                    marker={
                        "size": 12 + 22 * (points["n"] / point_clusters["n"].max()) ** 0.9,
                        "color": colors,
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
