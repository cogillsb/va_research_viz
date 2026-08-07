import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

import requests
import pandas as pd
from Bio import Entrez
from pathlib import Path
from itertools import batched
from tqdm import tqdm

def fetch_clinical_trials(term):
    """
    Takes in a term and gets all the studies associated with a term
    """
    ids = []
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.intr": term,
        "filter.overallStatus": "COMPLETED",
        "pageSize": 1000
    }
    # Loop while 'nextPageToken' exists
    while True:
        response = requests.get(base_url, params=params) 
        
        #export the data
        studies = response.json().get("studies", [])
        ids.extend([study["protocolSection"]["identificationModule"]["nctId"] for study in studies])
  
        data = response.json()  
        if not data.get("nextPageToken"): 
            break
        #set pagination
        params["pageToken"] = data["nextPageToken"]
        
    return ids

def fetch_pmids(ids, query_filters, id_tag='[UID]'):
    """
    Takes in a list of the clinical trial ids, forms a query for pubmed
    and returns a list of ids for all va studies
    """  
    filtered_ids = []
    # Loop in chunks of 1000
    for chunk in batched(ids, 1000):
        ####Finding published non_clinical studies####
        # Format query"
        id_query = f'({" OR ".join([f"{pmid}{id_tag}" for pmid in chunk])})'        
        query = ' AND '.join([id_query] + query_filters)
       
        # Execute search on PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=10000)
        record = Entrez.read(handle)
        handle.close()
        
        # Retrieved PubMed IDs (PMIDs)
        filtered_ids.extend(record["IdList"])
        
    return filtered_ids

def fetch_citations(cite_article):
    #Check the citations
    cites = []
    if cite_article['PubmedData']['ReferenceList']:
        for cite in cite_article['PubmedData']['ReferenceList'][0]['Reference']:
            if 'ArticleIdList' in cite:                
                for a in cite['ArticleIdList']:
                    if a.attributes['IdType'] == 'pubmed':
                        cites.append(a)
    
    return cites

def fetch_authors(auth_article):
    """
    Going through the tags and finding all the authors.
    """
    authors = []
    try:
        #Drill down to author list
        for author in auth_article['MedlineCitation']['Article']['AuthorList']:
            for affil in author['AffiliationInfo']:
                # Look for VA-related strings
                if (any(sub in affil['Affiliation'] for sub in ['Veterans Affairs', 'VA ', ', VA'])):     
                    authors.append(f"{author['ForeName']} {author['LastName']}")
    except:
        pass
    return authors

def build_layer(pmids, lyr_bin, lyr_organism = None, network=True, ):
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml", retmode="text")
    records = Entrez.read(handle)
    handle.close()
    records = records['PubmedArticle']
    
    lyr_authors = []
    sources = []
    targets = []
    tbl_titles = []
    organisms = [] 
    #Run through the records
    for i, pmid  in enumerate(pmids):
        article = records[i]
        
        #Add the authors
        authors = fetch_authors(article)
        lyr_authors.extend(authors)
        if network:
            #Get all the sources
            for cite in fetch_citations(article):
                if cite not in pmids:
                    sources.append(cite)
                    targets.append(pmid)      
        if lyr_organism:
            organisms.append(lyr_organism)
        else:
            organisms.append(", ".join([mesh['DescriptorName'] for mesh in article['MedlineCitation']['MeshHeadingList']]))
        
        #Add it to our table
        tbl_titles.append(article ['MedlineCitation']['Article']['ArticleTitle'])    

    df_lyr_studies = pd.DataFrame({'ID':pmids,
                                   'Title':tbl_titles,
                                   'Bin': lyr_bin,
                                   'Cluster':None,
                                   'Organism': organisms,
                                   'VA_Afil': 1,      
                                  })
    if network:
        df_lyr_network = pd.DataFrame({'Source':sources, 'Target':targets})    
        return df_lyr_studies, df_lyr_network, lyr_authors
    else:
        return df_lyr_studies, lyr_authors
    
def build_research_net(term):

    # Required by NCBI Entrez API
    Entrez.email = "steven.cogill@va.gov"
    df_term_node = pd.DataFrame({'ID': [term],
                                    'Title': [term],
                                    'Bin': ['Search Term'],
                                    'Cluster': [None],
                                    'Organism': ['Human'],
                                    'VA_Afil': [1],      
                                    })


    #Query filters for binning
    human_filter_query = '"Humans"[MeSH Terms]  NOT "Clinical Trial"[Publication Type]'
    animal_filter_query = '"animals"[Mesh] NOT "humans"[Mesh]  NOT "Clinical Trial"[Publication Type]'
    va_filter_query = '("va funded"[Filter] OR "Veterans Affairs"[ad] OR "VA"[ad] OR "Department of Veterans Affairs"[ad])'  

    ####Building first and second layers. The term to clinical studies####
    #Get clinical studies from clin trials.gov
    clin_studies_ids = fetch_clinical_trials(term)

    print(f"There are {len(clin_studies_ids)} completed clinical trials as of now.")

    #Use pubmed and ncbi to get all the va clinical trial research articles
    clin_pmids = fetch_pmids(clin_studies_ids, [], id_tag='[SI]')
    va_pmids = fetch_pmids(clin_studies_ids, [va_filter_query], id_tag='[SI]')

    print(f"We found {len(clin_pmids)} clinical trial publications and {len(va_pmids)} va specific clinical trials")

    print("For the clinical studies, we are pulling in their titles, authors, and citations.")

    #Simple network layer
    df_L1_network = pd.DataFrame({'Source': clin_pmids, 'Target': term})


    ####Building second and third layers. The clinical studies to human and animal studies####

    #Building the nodes for the clinical studies and connections back to animal and human studies
    print("Linking the clinical studies to downstream VA studies on humans and animals.")
    df_L2_studies, df_L2_network, L2_authors = build_layer(clin_pmids, 'Clinical Studies', lyr_organism='Human' )

    #Need to fix the L1 studies for va affiliation
    df_L2_studies['VA_Afil'] = [1 if x in va_pmids else 0 for x in df_L2_studies['ID'].values]

    #Filter out to relevant studies
    hum_pmids = fetch_pmids(df_L2_network.Source.unique(), [va_filter_query, human_filter_query])
    ani_pmids = fetch_pmids(df_L2_network.Source.unique(), [va_filter_query, animal_filter_query])

    print(f"We found {len(hum_pmids)} human studies and {len(ani_pmids)} animal studies")

    #Parse down the network for studies of interest
    df_L2_network = df_L2_network[df_L2_network.Source.isin(hum_pmids + ani_pmids)]

    ####Building third and fourth layers. The human to animal studies####

    #Building nodes for human studies and connections back to animal studies
    print("For the human studies, we are pulling in their titles, authors, and citations.")
    df_L3_studies, df_L3_network, L3_authors = build_layer(hum_pmids, 'Non-clinical Human Studies', lyr_organism='Human')


    #Get the animal pmids and add in those with direct connections to the clinical studies
    ani_h_pmids = fetch_pmids(df_L3_network.Source.unique(),  [va_filter_query, animal_filter_query]) 
    print(f"We found an additional {len(ani_h_pmids)}")
    #Parse down the network for studies of interest
    df_L3_network = df_L3_network[df_L3_network.Source.isin(ani_h_pmids)]

    ani_pmids = list(set(ani_pmids + ani_h_pmids))

    #Building nodes for animal studies. I don't really neeed conncetions to sub animal studies
    print("For the animal studies, we are pulling in their titles, authors.")
    df_L4_studies, L4_authors = build_layer(ani_pmids, 'Non_clinical Animal Studies', network=False)

    #Finding all authors involved in this research
    va_authors = L2_authors + L3_authors + L4_authors
    va_authors = list(set(va_authors))

    studies = pd.concat([df_term_node, df_L2_studies, df_L3_studies, df_L4_studies], ignore_index = True)
    studies.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)

    network = pd.concat([df_L1_network, df_L2_network, df_L3_network], ignore_index = True)
    network.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)

    clin_pmids = studies[studies['Bin'] == 'Clinical Studies']['ID'].values 
    drops = [pid for pid in clin_pmids if pid not in list(network['Target'].values)]

    studies = studies[~studies['ID'].isin(drops)]
    network = network[~network['Source'].isin(drops)]
   

    return network, studies, va_authors




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

def build_graph(studies, network):

    nodes = []
    edges = []
    
    LAYER_X      = [0, 10000, 22000, 33000]
    Y_SPACING    = 85
    # Node color palette per layer
    LAYER_COLORS = ["#1d6fa4", "#177a5e", "#7a4fb5", "#b54f4f"]
    LAYER_BORDER = ["#4a90d9", "#1fad80", "#a47dd6", "#d67a7a"]

    for i, bin in enumerate(studies['Bin'].unique()):
        print(bin)
        df_nodes = studies[studies['Bin']==bin] 
        print(df_nodes.shape)  
        h = 0     
        for j, rw in df_nodes.iterrows():
            h+=1
            nodes.append(Node(
                id=rw['ID'],
                label=rw['Title'],
                size=22,
                x=LAYER_X[i],
                y=h * Y_SPACING,
                color={"background": LAYER_COLORS[i], "border": LAYER_BORDER[i],
                    "highlight": {"background": "#4a90d9", "border": "#82b8f0"}},
            ))
    print('now running edges')
    for i, rw in network.iterrows():
        edges.append(Edge(
            source=rw['Source'],
            target=rw['Target'],
            type="CURVE_SMOOTH",
            color={"color": "#252d3d", "highlight": "#4a90d9", "opacity": 0.8},
        ))
    print('edges added')

   
    return nodes, edges
  

    


def build_df_graph(): 

    # ── Graph data ───────────────────────────────────────────────────────────────
    LAYERS       = 4
    NODES_LAYER  = 10
    LAYER_X      = [0, 220, 440, 660]
    Y_SPACING    = 85

    # Node color palette per layer
    LAYER_COLORS = ["#1d6fa4", "#177a5e", "#7a4fb5", "#b54f4f"]
    LAYER_BORDER = ["#4a90d9", "#1fad80", "#a47dd6", "#d67a7a"]

    nodes, edges = [], []

    for li, x in enumerate(LAYER_X):
        for ni in range(NODES_LAYER):
            nid = f"L{li+1}_N{ni+1}"
            nodes.append(Node(
                id=nid,
                label=nid,
                size=22,
                x=x,
                y=(ni + 1) * Y_SPACING,
                color={"background": LAYER_COLORS[li], "border": LAYER_BORDER[li],
                    "highlight": {"background": "#4a90d9", "border": "#82b8f0"}},
            ))

    for li in range(LAYERS - 1):
        for ni in range(NODES_LAYER):
            edges.append(Edge(
                source=f"L{li+1}_N{ni+1}",
                target=f"L{li+2}_N{ni+1}",
                type="CURVE_SMOOTH",
                color={"color": "#252d3d", "highlight": "#4a90d9", "opacity": 0.8},
            ))

    config = Config(
        directed=True,
        physics=False,
        staticGraphWithDragAndDrop=False,
        nodeHighlightBehavior=True,
        highlightColor="#4a90d9",
        node={"labelProperty": "label", "fontColor": "#c8d0e0", "fontSize": 10},
        link={"highlightColor": "#4a90d9"},
        height=820,
        width="100%",
        background="#13181f",
    )

    return nodes, edges

config = Config(
        directed=True,
        physics=False,
        staticGraphWithDragAndDrop=False,
        nodeHighlightBehavior=True,
        highlightColor="#4a90d9",
        node={"labelProperty": "label", "fontColor": "#c8d0e0", "fontSize": 10},
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
        Edge(source="A", target="B", label="Connected to")
    ]
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
    network, studies, authors = build_research_net(search_query)
    #network = pd.read_csv('network.csv')
    #studies = pd.read_csv('studies.csv')
    #clicked_node = build_graph(studies, network)
    nodes, edges = build_graph(studies, network)
    st.session_state.nodes = nodes
    st.session_state.edges = edges
    st.session_state.authors = authors



# Graph
st.markdown('<div class="graph-wrap">', unsafe_allow_html=True)
clicked_node = agraph(
    nodes=st.session_state.nodes, 
    edges=st.session_state.edges, 
    config=config
)
st.markdown('</div>', unsafe_allow_html=True)

# Create 3 columns
cols = st.columns(3)

# Distribute items across the 3 columns
for index, item in enumerate(st.session_state.authors):
  cols[index % 3].write(item)
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Node inspector")
    st.markdown("---")

    