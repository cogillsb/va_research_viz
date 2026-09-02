import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import pandas as pd
from Bio import Entrez
from pathlib import Path
from itertools import batched
from tqdm import tqdm

from network_build import build_network



# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graph Explorer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #252d3d;
}
[data-testid="stSidebar"] * {
    color: #c8d0e0 !important;
}
.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
}

/* ── Header ── */
.graph-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 1.25rem;
}
.graph-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: -0.01em;
    margin: 0;
}
.graph-subtitle {
    font-size: 0.78rem;
    color: #556070;
    font-family: 'SF Mono', 'Fira Code', monospace;
}

/* ── Graph container ── */
.graph-wrap {
    background: #13181f;
    border: 1px solid #1e2a3a;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Sidebar: empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2.5rem 1rem;
    text-align: center;
    gap: 0.6rem;
}
.empty-icon {
    font-size: 1.8rem;
    opacity: 0.35;
}
.empty-label {
    font-size: 0.8rem;
    color: #4a5568;
    line-height: 1.5;
}

/* ── Sidebar: node detail ── */
.node-card {
    background: #1a2235;
    border: 1px solid #252d3d;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
}
.node-id-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4a90d9 !important;
    font-family: 'SF Mono', 'Fira Code', monospace;
    margin: 0 0 4px;
}
.node-id-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0 !important;
    font-family: 'SF Mono', 'Fira Code', monospace;
    margin: 0;
}

/* ── Connection list ── */
.conn-header {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #556070 !important;
    margin: 1rem 0 0.5rem;
}
.conn-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a2235;
    border: 1px solid #252d3d;
    border-radius: 5px;
    padding: 5px 10px;
    margin: 3px 0;
    font-size: 0.76rem;
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: #8ca0b8 !important;
    width: 100%;
}
.conn-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #2d6a9f;
    flex-shrink: 0;
}
.conn-count {
    font-size: 0.7rem;
    color: #4a90d9 !important;
    margin: 0.75rem 0 0;
    font-family: 'SF Mono', 'Fira Code', monospace;
}

/* ── Stats bar ── */
.stats-row {
    display: flex;
    gap: 1px;
    margin-bottom: 1rem;
    background: #1e2a3a;
    border-radius: 7px;
    overflow: hidden;
    border: 1px solid #1e2a3a;
}
.stat-cell {
    flex: 1;
    padding: 0.55rem 0.9rem;
    background: #13181f;
}
.stat-cell:not(:last-child) {
    border-right: 1px solid #1e2a3a;
}
.stat-val {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.stat-lbl {
    font-size: 0.65rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)


# ── Helper ───────────────────────────────────────────────────────────────────
def get_connected_nodes(clicked_id, nodes_list, edges_list):
    adj_list = {}
    for edge in edges_list:
        s, t = edge.source, edge.to
        adj_list.setdefault(s, set()).add(t)
        adj_list.setdefault(t, set()).add(s)

    node_labels = {n.id: n.label for n in nodes_list}
    visited, results = set(), []

    def dfs(node_id):
        if node_id not in visited:
            visited.add(node_id)
            results.append({"id": node_id, "label": node_labels.get(node_id, node_id)})
            for neighbor in adj_list.get(node_id, []):
                dfs(neighbor)

    dfs(clicked_id)
    return results

def build_graph(df_nodes, df_edges):
    config = Config(
        directed=False,
        physics=False,
        staticGraphWithDragAndDrop=False,
        nodeHighlightBehavior=True,
        highlightColor="#4a90d9",
        node={"labelProperty": "label", "fontColor": "#c8d0e0", "fontSize": 10, "renderLabel": True},
        link={"highlightColor": "#4a90d9"},
        height=400,
        width=1200,
        background="#13181f",
        hierarchical=True,
    )
    nodes = []
    edges = []
    
    LAYER_X      = [0, 500, 1020, 2030]
    Y_SPACING    = 85
    # Node color palette per layer
    LAYER_COLORS = ["#1d6fa4", "#177a5e", "#7a4fb5", "#b54f4f"]
    LAYER_BORDER = ["#4a90d9", "#1fad80", "#a47dd6", "#d67a7a"]

    #Human Studies
    df_nodes_sub = df_nodes[df_nodes['Bin']=='human']
    h = 0   
    x_offset= 0  
    for j, rw in df_nodes_sub.iterrows():
        h+=1
        sz = 1*rw['count']
        x_position = 1*rw['count'] + x_offset
        nodes.append(Node(
            id=rw['Cluster'],
            label=f'{rw['count']} human studies',
            size=1*rw['count'],
            #size =25,
            x=x_position,
            y=0,
            color={"background": LAYER_COLORS[0], "border": LAYER_BORDER[0],
                "highlight": {"background": "#4a90d9", "border": "#82b8f0"}},
            group='human'
        ))
        x_offset = x_position + sz + 5 
    right_offset = x_offset
    #Clinical Studies
    df_nodes_sub = df_nodes[df_nodes['Bin']=='clinical']
    h = 0   
    x_offset= 0 
    y_offset = 100 
    for j, rw in df_nodes_sub.iterrows():
        h+=1
        sz = 1*rw['count']
        x_position = sz + x_offset
        y_position = sz + y_offset
        nodes.append(Node(
            id=rw['Cluster'],
            label=f'{rw['count']} clinical studies',
            size=sz,
            #size =25,
            x=x_position,
            y=y_position,
            color={"background": LAYER_COLORS[1], "border": LAYER_BORDER[1],
                "highlight": {"background": "#4a90d9", "border": "#82b8f0"}},
            group='clinical'
        ))
        x_offset = x_position + sz + 5 
        y_offset = y_position + sz + 5 

    #Animal Studies
    df_nodes_sub = df_nodes[df_nodes['Bin']=='animal']
    h = 0   
    x_offset= right_offset
    y_offset = 100 
    for j, rw in df_nodes_sub.iterrows():
        h+=1
        sz = 1*rw['count']
        x_position = x_offset - sz 
        y_position = sz + y_offset
        nodes.append(Node(
            id=rw['Cluster'],
            label=f'{rw['count']} animal studies',
            size=sz,
            #size =25,
            x=x_position,
            y=y_position,
            color={"background": LAYER_COLORS[2], "border": LAYER_BORDER[2],
                "highlight": {"background": "#4a90d9", "border": "#82b8f0"}},
            group='clinical'
        ))
        x_offset = x_position - sz - 5 
        y_offset = y_position + sz + 5 


    df_edges.sort_values(by='count', inplace=True)
    for i, rw in df_edges.head(int(len(df_edges)*.1)).iterrows():
        edges.append(Edge(
            source=rw['Source_cluster'],
            target=rw['Target_cluster'],
            type="CURVE_SMOOTH",
            width=1*rw['count'],
            color={"color": "#252d3d", "highlight": "#4a90d9", "opacity": 0.8,},
        ))
    print('edges added')
    
   
    return nodes, edges, config
    
  

    



config = Config(
        directed=True,
        physics=False,
        staticGraphWithDragAndDrop=False,
        nodeHighlightBehavior=True,
        highlightColor="#4a90d9",
        node={"labelProperty": "label", "fontColor": "#c8d0e0", "fontSize": 10, "renderLabel": True},
        link={"highlightColor": "#4a90d9"},
        height=820,
        width="100%",
        background="#13181f",
    )
if "nodes" not in st.session_state:
    st.session_state.nodes = [
        Node(id="A", label="Node A", size=25),
        Node(id="B", label="Node B", size=25),
    ]

if "edges" not in st.session_state:
    st.session_state.edges = [
        Edge(source="A", target="B", )    ]

if "studies" not in st.session_state:
    st.session_state.studies = None

if config not in st.session_state:
    st.session_state.config = config

if 'authors' not in st.session_state:
    st.session_state.authors = []

# ── Layout ───────────────────────────────────────────────────────────────────
# Main header
st.markdown("""
<div class="graph-header">
  <p class="graph-title">⬡ Graph Explorer</p>
  <span class="graph-subtitle">4 layers · 40 nodes · 30 edges</span>
</div>
""", unsafe_allow_html=True)



# Create the search input box
search_query = st.text_input("Search", placeholder="Search for a treatment or drug.")
submit_button = st.button(label="Submit")
# Execute function when submit is clicked
if submit_button:
    df_studies, df_network, df_edges, df_nodes, authors = build_network(search_query)
    
    #df_nodes = pd.read_csv('testing_nodes.csv')
    #df_edges = pd.read_csv('testing_edges.csv')
    st.session_state.studies =  df_studies
    #studies = pd.read_csv('studies.csv')
    #clicked_node = build_graph(studies, network)
    nodes, edges, config = build_graph( df_nodes, df_edges)
    st.session_state.nodes = nodes
    st.session_state.edges = edges
    st.session_state.config = config
    st.session_state.authors = authors

st.markdown(
    "<h1 style='text-align: center; color: white;'>VA Research Graph</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<span style="color:white;">**Legend:**</span> &nbsp;&nbsp;&nbsp;
<span style='font-size: 32px; color:#1d6fa4;'>■</span> <span style="color:white;">Non-Clinical Human VA Study Clusters</span> &nbsp;&nbsp;&nbsp;
<span style='font-size: 32px; color:#177a5e;'>■</span> <span style="color:white;">Clinical VA Study Clusters</span> &nbsp;&nbsp;&nbsp;
<span style='font-size: 32px; color:#7a4fb5;'>■</span> <span style="color:white;">Non-Clinical Animal VA Study Clusters</span> &nbsp;&nbsp;&nbsp;

""",
    unsafe_allow_html=True,
)
# Graph
#st.markdown('<div class="graph-wrap">', unsafe_allow_html=True)
with st.container(border=True, height=400):
#with st.container(border=True):
    clicked_node = agraph(
        nodes=st.session_state.nodes, 
        edges=st.session_state.edges, 
        config=st.session_state.config
    )
#st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    "<h1 style='text-align: center; color: white;'>Contributing VA Researchers</h1>",
    unsafe_allow_html=True,
)
# Create 3 columns
cols = st.columns(3)

# Distribute items across the 3 columns

auths = st.session_state.authors
auths.sort()
for index, item in enumerate(st.session_state.authors):
  cols[index % 3].write(f'<span style="color:white">{item}</span>', unsafe_allow_html=True)
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Cluster Info")
    st.markdown("---")
    if clicked_node:
        st.write(f"**Selected Node:** {clicked_node}")

        if st.session_state.studies is not None:
          
            df_titles = st.session_state.studies
            titles = df_titles[df_titles.Cluster==str(clicked_node)].Title.values
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white'
            ).generate(';'.join(titles))

            # 3. Display the generated image using Matplotlib
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Hide the pixel grid axes
            st.pyplot(plt)
            for i, title in enumerate(titles):
                st.write(f'{i+1} {title}')


    