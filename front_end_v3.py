"""Graph Explorer - Streamlit front end."""
from __future__ import annotations

from copy import copy
import html
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_agraph import agraph
from wordcloud import WordCloud
#from summary_gen import summarize
from graph_layout import BIN_STYLE, build_graph
from network_build import build_network
from plots import plot_cluster_timeline, plot_landmark_timeline

DEBUG_DUMP_CSV = False  # write the dataframes returned by build_network to disk
DATA_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="VA Study Explorer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #252d3d; }
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

.graph-header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 1.25rem; }
.graph-title  { font-size: 1.15rem; font-weight: 600; color: #e2e8f0; letter-spacing: -0.01em; margin: 0; }
.graph-subtitle { font-size: 0.78rem; color: #556070; font-family: 'SF Mono', 'Fira Code', monospace; }

[data-testid="stExpander"] { background: #161b27; border: 1px solid #252d3d; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p { color: #ffffff !important; }
[data-testid="stExpander"] summary svg { color: #ffffff !important; }

.stMarkdown ul li, .stMarkdown ul li::marker { color: white !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Small UI helpers ─────────────────────────────────────────────────────────
def heading(text: str, level: int = 1) -> None:
    st.markdown(
        f"<h{level} style='text-align:center; color:white;'>{text}</h{level}>",
        unsafe_allow_html=True,
    )


def white_text(text: str) -> None:
    st.markdown(f'<span style="color:white">{text}</span>', unsafe_allow_html=True)


def show_figure(fig) -> None:
    """Render Plotly or Matplotlib figures and release Matplotlib resources."""
    if fig is None:
        st.info("Not enough dated studies to draw this figure.")
        return
    if isinstance(fig, go.Figure):
        st.plotly_chart(fig, width="stretch")
        return
    st.pyplot(fig)
    plt.close(fig)


# ── State ────────────────────────────────────────────────────────────────────
STATE_DEFAULTS = {
    "studies": None,
    "df_edges": None,
    "nodes": [],
    "edges": [],
    "graph_config": None,
    "authors": [],
    "term": "",
}
for key, default in STATE_DEFAULTS.items():
    st.session_state.setdefault(key, default)


@st.cache_data(ttl="1h", max_entries=32, show_spinner=False)
def _load_search_data(query: str, _progress_callback=None):
    if query.strip().casefold() == "test":
        df_studies = pd.read_csv(DATA_DIR / "test_df_studies.csv")
        df_network = pd.read_csv(DATA_DIR / "test_df_network.csv")
        df_edges = pd.read_csv(DATA_DIR / "test_df_edges.csv")
        df_nodes = pd.read_csv(DATA_DIR / "test_df_nodes.csv")
        authors_path = DATA_DIR / "test_df_authors.csv"
        authors = (
            pd.read_csv(authors_path)["Author"].dropna().astype(str).tolist()
            if authors_path.exists()
            else []
        )
        return df_studies, df_network, df_edges, df_nodes, authors

    return build_network(query, progress_callback=_progress_callback)


def run_search(query: str, progress_callback=None) -> None:
    df_studies, df_network, df_edges, df_nodes, authors = _load_search_data(
        query, _progress_callback=progress_callback
    )
    
    if DEBUG_DUMP_CSV:
        
        for name, df in [("studies", df_studies), ("network", df_network),
                         ("edges", df_edges), ("nodes", df_nodes)]:
            df.to_csv(DATA_DIR / f"test_df_{name}.csv", index=False)
           
        pd.Series(authors, name="Author").to_csv(
            DATA_DIR / "test_df_authors.csv", index=False
        )

    if progress_callback is not None:
        progress_callback("Step 7 of 7: Assembling the interactive cluster graph.")
    nodes, edges, config = build_graph(df_nodes, df_edges)
    st.session_state.update(
        studies=df_studies,
        df_edges=df_edges,
        nodes=nodes,
        edges=edges,
        graph_config=config,
        authors=authors,
        term=query,
    )


# ── Header + search ──────────────────────────────────────────────────────────
has_results = st.session_state.studies is not None
if has_results:
    n_clusters = st.session_state.studies["Cluster"].nunique()
    subtitle = f"{n_clusters} clusters · {len(st.session_state.nodes)} nodes · {len(st.session_state.edges)} edges"
else:
    subtitle = "search to begin"

st.markdown(
    f"""
<div class="graph-header">
  <p class="graph-title">⬡ Graph Explorer</p>
  <span class="graph-subtitle">{subtitle}</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.form("search_form", border=False):  # form => pressing Enter also submits
    search_query = st.text_input("Search", placeholder="Search for a treatment or drug.")
    submitted = st.form_submit_button("Submit")

if submitted and search_query.strip():
    with st.status(
        "Step 1 of 7: Searching PubMed for VA studies across three study types.",
        expanded=True,
    ) as search_status:
        def update_search_status(message: str) -> None:
            search_status.update(label=message)

        run_search(search_query.strip(), progress_callback=update_search_status)
        search_status.update(
            label="Search complete. The study network is ready.",
            state="complete",
            expanded=False,
        )
    st.rerun()  # so the header stats above reflect the new results


# ── Sections ─────────────────────────────────────────────────────────────────
def render_landmark_timelines(studies) -> None:
    with st.expander("Landmark article timelines", expanded=True):
        for bin_name in studies["Bin"].unique():
            df_bin = studies[(studies["Bin"] == bin_name) & (studies["Degree"] == 1)]
            df_top = df_bin.head(10).sort_values("Date")

            heading(f"{bin_name.capitalize()} Landmark VA Articles Timeline")
            show_figure(plot_landmark_timeline(df_top))

            heading("Key Findings", 2)
            findings = df_top.sort_values(by ="Strength", ascending=False)["Abstract"].dropna()
            findings = [x.strip() for x in findings]
            st.markdown("\n".join(f"* {f}" for f in findings))


def render_cluster_timelines(studies, df_edges, clicked_node=None) -> None:
    with st.expander("VA research timeline by cluster", expanded=True):
        heading("VA Research Timeline by Cluster")
        white_text(
            "Each lane is a breakthrough on the development cycle. Points show individual study clusters, "
            "colours identify study types, and arcs show citations between clusters."
        )
        show_figure(
            plot_cluster_timeline(
                studies, df_edges, selected_cluster=clicked_node
            )
        )


def render_graph() -> str | None:
    with st.expander("VA research graph", expanded=True):
        heading("VA Research Graph")
        white_text(
            "Below is a tripartite graph of VA studies for clinical trials, human studies, "
            "and animal studies. The studies have been clustered based on similarity. The "
            "connections between the clusters are citations, where the thickness of the line "
            "indicates the number of times either cluster cited the other."
        )
        legend = " &nbsp;&nbsp;&nbsp; ".join(
            f"<span style='font-size:32px; color:{s['fill']};'>■</span> "
            f"<span style='color:white;'>{s['legend']}</span>"
            for s in BIN_STYLE.values()
        )
        st.markdown(
            f"<span style='color:white;'>**Legend:**</span> &nbsp;&nbsp;&nbsp; {legend}",
            unsafe_allow_html=True,
        )
        with st.container(border=True, height=760, horizontal_alignment="center"):
            graph_config = copy(st.session_state.graph_config)
            graph_config.width = "100%"
            graph_config.height = "740px"

            return agraph(
                nodes=st.session_state.nodes,
                edges=st.session_state.edges,
                config=graph_config,
            )


def render_authors() -> None:
    with st.expander("Contributing VA researchers", expanded=True):
        authors = sorted(
            set(st.session_state.authors),
            key=lambda name: name.rsplit(maxsplit=1)[-1].casefold(),
        )
        heading("Contributing VA Researchers")
        heading(f"{len(authors)} VA researchers have worked toward advancing {st.session_state.term}", 3)
        cols = st.columns(3)
        for i, name in enumerate(authors):
            cols[i % 3].markdown(f'<span style="color:white">{name}</span>', unsafe_allow_html=True)


def render_sidebar(clicked_node) -> None:
    with st.sidebar:
        with st.expander("Cluster info", expanded=True):
            if not clicked_node:
                st.caption("Click a node in the graph to see its studies.")
                return

            st.write(f"**Selected Node:** {clicked_node}")
            studies = st.session_state.studies
            selected_studies = studies.loc[
                studies["Cluster"].astype(str) == str(clicked_node), ["ID", "Title", "Years"]
            ].dropna(subset=["ID", "Title", "Years"])
            if selected_studies.empty:
                return
            selected_studies.sort_values(by="Years", inplace=True)
            titles = selected_studies["Title"].astype(str).tolist()
            cloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(titles))
            st.image(cloud.to_array())
            for i, (study_id, title, year) in enumerate(
                selected_studies[["ID", "Title", "Years"]].itertuples(index=False, name=None),
                start=1,
            ):
                study_id = str(study_id).removesuffix(".0")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{quote(study_id, safe='')}/"
                st.markdown(
                    f'<p>{i}. <a href="{url}" target="_blank" '
                    f'rel="noopener noreferrer">{html.escape(str(title) + "  (" + str(year)) + ")"}</a></p>',
                    unsafe_allow_html=True,
                )


# ── Page body (only after a search has been run) ─────────────────────────────
clicked = None
if has_results:
    studies = st.session_state.studies   
    cluster_timeline_slot = st.empty()
    clicked = render_graph()
    with cluster_timeline_slot.container():
        render_cluster_timelines(studies, st.session_state.df_edges, clicked)
    render_landmark_timelines(studies)
    render_authors()
render_sidebar(clicked)
