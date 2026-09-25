"""Node/edge construction for the tripartite cluster graph (streamlit-agraph)."""
from __future__ import annotations

import pandas as pd
from streamlit_agraph import Config, Edge, Node

GAP = 45            # px between neighbouring nodes
ROW_OFFSET = 220    # vertical gap between the human row and the other layers

# Bin -> label noun, legend text, fill, border
BIN_STYLE = {
    "human":    dict(noun="human studies",    legend="Non-Clinical Human VA Study Clusters",  fill="#1d6fa4", border="#4a90d9"),
    "clinical": dict(noun="clinical studies", legend="Clinical VA Study Clusters",            fill="#177a5e", border="#1fad80"),
    "animal":   dict(noun="animal studies",   legend="Non-Clinical Animal VA Study Clusters", fill="#7a4fb5", border="#a47dd6"),
}


def _node_size(row: pd.Series) -> float:
    """Scale counts into readable circle radii without losing relative size."""
    return max(32, 14 + 7 * float(row["count"]) ** 0.5)


def _node(row, bin_name: str, x: float, y: float) -> Node:
    style = BIN_STYLE[bin_name]
    return Node(
        id=row["Cluster"],
        label=row["Cluster"],
        size=_node_size(row),
        x=x,
        y=y,
        color={
            "background": style["fill"],
            "border": style["border"],
            "highlight": {"background": "#4a90d9", "border": "#82b8f0"},
        },
        group=bin_name,
    )


def _place_nodes(df_nodes: pd.DataFrame) -> list[Node]:
    nodes: list[Node] = []

    # Human: single row, left -> right
    x_cursor = 0
    for _, row in df_nodes[df_nodes["Bin"] == "human"].iterrows():
        sz = _node_size(row)
        x = x_cursor + sz
        nodes.append(_node(row, "human", x, 0))
        x_cursor = x + sz + GAP
    row_end = x_cursor

    # Clinical / animal: diagonal cascades below the human row.
    # Clinical starts at the left and steps right, animal starts at the
    # right end of the human row and steps left.
    for bin_name, x_start, direction in (("clinical", 0, 1), ("animal", row_end, -1)):
        x_cursor, y_cursor = x_start, ROW_OFFSET
        for _, row in df_nodes[df_nodes["Bin"] == bin_name].iterrows():
            sz = _node_size(row)
            x = x_cursor + direction * sz
            y = y_cursor + sz
            nodes.append(_node(row, bin_name, x, y))
            x_cursor = x + direction * (sz + GAP)
            y_cursor = y + sz + GAP
    return nodes


def build_graph(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, top_edge_frac: float = 0.3):
    """Return (nodes, edges, config) for agraph.

    Only the strongest `top_edge_frac` of edges (by citation count) are drawn.
    """
    nodes = _place_nodes(df_nodes)

    keep = df_edges.nlargest(int(len(df_edges) * top_edge_frac), "count")
    edges = [
        Edge(
            source=r["Source_cluster"],
            target=r["Target_cluster"],
            type="CURVE_SMOOTH",
            width=r["count"],
            color={"color": "#252d3d", "highlight": "#4a90d9", "opacity": 0.8},
        )
        for _, r in keep.iterrows()
    ]

    config = Config(
        directed=False,
        physics=False,
        staticGraphWithDragAndDrop=False,
        nodeHighlightBehavior=True,
        highlightColor="#4a90d9",
        node={
            "shape": "dot",
            "labelProperty": "label",
            "fontColor": "#f8fafc",
            "fontSize": 12,
            "renderLabel": True,
        },
        link={"highlightColor": "#4a90d9"},
        height=700,
        width=1400,
        background="#13181f",
        hierarchical=True,
    )
    return nodes, edges, config
