"""Graph Explorer - Streamlit front end."""
from __future__ import annotations

import copy
import html
import re
from pathlib import Path
from urllib.parse import quote
from network_build import assign_super_clusters
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

DEBUG_DUMP_CSV = True  # write the dataframes returned by build_network to disk
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
[data-testid="stExpander"] summary p { color: #000000 !important; }
[data-testid="stExpander"] summary svg { color: #000000 !important; }
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:hover p,
[data-testid="stExpander"] summary:hover svg { color: #ffffff !important; }

.stMarkdown ul li, .stMarkdown ul li::marker { color: white !important; }
button:hover, button:hover * { color: #ffffff !important; }
.st-key-cluster_timeline_years [data-testid="stSlider"] * { color: #ffffff !important; }
.st-key-cluster_timeline_years [data-testid="stSlider"] [role="slider"] {
    background-color: #ffffff !important;
}
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
    if isinstance(fig, go.FigureWidget):
        st.plotly_chart(fig)
    else:
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
    "breakthrough_descriptions": [],
    "processing": False,
    "clicked_node_id": None,
    "masked_nodes": {}
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
        breakthrough_descriptions_path = DATA_DIR / "test_df_breakthrough_descriptions.csv"
        breakthrough_descriptions = (
            pd.read_csv(breakthrough_descriptions_path)["Breakthrough_Description"].dropna().astype(str).tolist()
            if breakthrough_descriptions_path.exists()
            else []
        )

        return df_studies, df_network, df_edges, df_nodes, authors, breakthrough_descriptions

    return build_network(query, progress_callback=_progress_callback)


def run_search(query: str, progress_callback=None) -> None:
    df_studies, df_network, df_edges, df_nodes, authors, breakthrough_descriptions = _load_search_data(
        query, _progress_callback=progress_callback
    )
    
    if DEBUG_DUMP_CSV:
        
        for name, df in [("studies", df_studies), ("network", df_network),
                         ("edges", df_edges), ("nodes", df_nodes)]:
            df.to_csv(DATA_DIR / f"test_df_{name}.csv", index=False)
           
        pd.Series(authors, name="Author").to_csv(
            DATA_DIR / "test_df_authors.csv", index=False
        )

        pd.Series(breakthrough_descriptions, name="Breakthrough_Description").to_csv(
            DATA_DIR / "test_df_breakthrough_descriptions.csv", index=False
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
        breakthrough_descriptions=breakthrough_descriptions
    )


# ── Header + search ──────────────────────────────────────────────────────────
has_results = st.session_state.studies is not None


def start_process():
    term = st.session_state.term.strip()
    if not term:
        return

    st.session_state.update(
        studies=None,
        df_edges=None,
        nodes=[],
        edges=[],
        graph_config=None,
        authors=[],
        breakthrough_descriptions=[],
        masked_nodes={},
        processing=True,
    )
        

with st.form("search_form", border=False):  # form => pressing Enter also submits
    search_query = st.text_input("Search", placeholder="Search for a treatment or drug.", key="term")
    submitted = st.form_submit_button("Submit",
                                       disabled=st.session_state.processing,
                                         on_click=start_process
    )


if st.session_state.processing:  

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
    st.session_state.processing = False  

    st.rerun()
    

# ── Sections ─────────────────────────────────────────────────────────────────
def render_landmark_timelines(studies) -> None:
    with st.expander("Landmark article timelines", expanded=True):
        for bin_name in studies["Bin"].unique():
            df_bin = studies[(studies["Bin"] == bin_name) & (studies["Degree"] == 1)].copy()
            df_bin = df_bin.drop_duplicates(subset=["Abstract"]).copy()
            df_top = df_bin.head(10).sort_values("Date")

            heading(f"{bin_name.capitalize()} Landmark VA Articles Timeline")
            show_figure(plot_landmark_timeline(df_top))

            heading("Key Findings", 2)
            for btc in df_top.BreakthruCluster.unique():

                sub_head = re.sub(r"(?is)<citations>.*?(?:<citations/>|</citations>)", "", btc)
                sub_head = re.sub(r"\s+", " ", sub_head)
                sub_head = sub_head.strip()

                heading(sub_head, 3)
                cluster_df = df_top[df_top.BreakthruCluster == btc]
                findings = (
                    cluster_df.sort_values(by="Strength", ascending=False)
                    .dropna(subset=["Title", "Abstract", "ID", "Years"])
                    .head(5)
                    .copy()
                )

                for _, row in findings.iterrows():
                    text = str(row.get("Abstract", "")).strip()
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
                    excerpt = " ".join(sentences[-3:]) if len(sentences) >= 3 else " ".join(sentences)
                    if not excerpt:
                        continue

                    study_id = str(row.get("ID", "")).strip().removesuffix(".0")
                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{quote(study_id, safe='')}/"
                    title = html.escape(str(row.get("Title", "")).strip())
                    year = str(row.get("Years", "")).strip()

                    st.markdown(
                        f'<p style="color:white; margin-bottom:0.5rem;">'
                        f'{html.escape(excerpt)}</p>'
                        f'<a href="{pubmed_url}" target="_blank" rel="noopener noreferrer">'
                        f'{title} ({year})</a><br>',
                        unsafe_allow_html=True,
                    )


def render_cluster_timelines(studies, df_edges) -> None:
    clicked_node = st.session_state.clicked_node_id
    with st.expander("VA research timeline by cluster", expanded=True):        
        heading("VA Research Timeline by Cluster")
        white_text(
            "Each lane is a breakthrough on the development cycle. Points show individual study clusters, "
            "colours identify study types, and arcs show citations between clusters."
        )
        studies['Years'] = studies['Years'].astype(int) 

        # Select a range of float values
        values = st.slider(
            label="Select a range of Years:",
            min_value=int(studies.Years.min())-1,
             max_value= int(studies.Years.max()) + 1 ,
             value= (int(studies.Years.min())-1, int(studies.Years.max()) + 1),
             step= 1,
             key="cluster_timeline_years",
        )

        #Filter for selected timeline range
        rng_studies = studies[studies.Years.between(values[0], values[1])] 

        

        #Reassign the clusters
        rng_studies['BreakthruCluster'] = assign_super_clusters(rng_studies, st.session_state['breakthrough_descriptions'])

        rng_clusters = list(rng_studies.Cluster.unique())
        rng_edges = df_edges[(df_edges.Source_cluster.isin(rng_clusters))&(df_edges.Target_cluster.isin(rng_clusters))]
        show_figure(
            plot_cluster_timeline(
                rng_studies,
                rng_edges,
                selected_cluster=clicked_node
            )
        )


def _connected_node_mask(clicked_node: str | None) -> set[str]:
    #connected = []
    e = st.session_state.df_edges.copy()
    e = e.nlargest(int(len(e) * .3), "count")


    def nav(node, connected=None):
        if connected is None:
            connected = {}
        connected[node] = list(e[e.Source_cluster==node]['Target_cluster'].values)
        cons = list(e[e.Source_cluster==node]['Target_cluster'].values)
        for c in cons:
            if c not in list(connected.keys()):
                nav(c, connected)
        #if len(cons)>0:
            #for c in cons:
             #   connected.extend(nav(c, connected))
        #else:
            #connected.append(node)

        return connected
            
        
    connected = nav(clicked_node)

    #connected = list(set(connected))


    st.session_state.masked_nodes = connected




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
        neighbors =[]
        #modify nodes list if clicked
        if st.session_state.clicked_node_id:
            #Track down the associated edges
            df_edges = st.session_state.df_edges.copy()
            e = df_edges.nlargest(int(len(df_edges) * .3), "count")
            neighbors = list(e[e["Source_cluster"] == str(st.session_state.clicked_node_id)]['Target_cluster'].values)
            
        



        with st.container(border=True, height=760, horizontal_alignment="center"):
            graph_config = st.session_state.graph_config
            graph_config.width = "100%"
            graph_config.height = "740px"

            nodes = list(copy.deepcopy(st.session_state.nodes))
            edges = list(copy.deepcopy(st.session_state.edges))
        

            if len(st.session_state.masked_nodes) > 0:
                masks = []
                for v in list(st.session_state.masked_nodes.values()):
                    masks.extend(v)
                masks = list(set(masks))

                for node in nodes:
                    #print(f"{node.id} {node.color['background']}")
                    if node.id in masks:
                        #node.color['background'] = "#8c1108"
                        node.color['border'] = "#01060c"
                        node.color['background'] = "rgba(0, 0, 0, 0)" 
                        node.opacity=1

                    if node.id==st.session_state.clicked_node_id:
                        #node.color['background'] = "#4a90d9"
                        node.color['border'] = "#000509"
                        node.color['background'] = "rgba(211, 211, 211, 1)" 
                        node.opacity=1  
                for edge in edges:
                    opac = .5                    
                    if edge.source in st.session_state.masked_nodes.keys():
                        #print(vars(edge))
                        if edge.to in st.session_state.masked_nodes[edge.source]:
                             opac = .8              
                    edge.color['opacity']=opac

                    


            elif len(neighbors)>0:
                for node in nodes:
                    #print(f"{node.id} {node.color['background']}")
                    if node.id in neighbors:
                        #node.color['background'] = "#8c1108"
                        node.color['border'] = "#01060c"
                        node.opacity=1
                    if node.id==st.session_state.clicked_node_id:
                        #node.color['background'] = "#4a90d9"
                        node.color['border'] = "#000509"
                        node.opacity=1  
                for edge in edges:  
                    if edge.source == st.session_state.clicked_node_id:                    
                        edge.color['opacity']=.8                    
           

            
        
   

            return agraph(
                nodes=nodes,
                edges=edges,
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
        heading(f"{len(st.session_state.studies)} studies were generated", 3)
                
        cols = st.columns(3)
        for i, name in enumerate(authors):
            cols[i % 3].markdown(f'<span style="color:white">{name}</span>', unsafe_allow_html=True)


def render_sidebar() -> None:
    clicked_node = st.session_state.clicked_node_id
    with st.sidebar:
        with st.expander("Cluster info", expanded=True):
            if not clicked_node:
                st.caption("Click a node in the graph to see its studies.")
                return

            st.write(f"**Selected Node:** {clicked_node}")

            if len(st.session_state.masked_nodes)>0:
                unmask_clicked = st.button("Unmask")
                if unmask_clicked:
                    st.session_state.masked_nodes = {}
                    st.rerun()
            else:
                mask_clicked = st.button("Mask")
                if mask_clicked:
                    _connected_node_mask(clicked_node)
                    st.rerun()



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
if has_results and not st.session_state.processing:
    studies = st.session_state.studies
    st.download_button(
        label="Download studies CSV",
        data=studies.to_csv(index=False).encode("utf-8"),
        file_name="df_studies.csv",
        mime="text/csv",
    )
    cluster_timeline_slot = st.empty()
    clicked = render_graph()
    if clicked:
        if str(clicked) != st.session_state.clicked_node_id:
            st.session_state.clicked_node_id = str(clicked)
            st.rerun()
    with cluster_timeline_slot.container():
        render_cluster_timelines(studies, st.session_state.df_edges)
    render_landmark_timelines(studies)
    render_authors()

render_sidebar()
