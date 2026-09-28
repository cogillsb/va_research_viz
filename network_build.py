import requests
import pandas as pd
import re
from Bio import Entrez
from pathlib import Path
from itertools import batched
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import math
from elicit_interface import elicit_search

def elicit_prompt(term):
    """
    Creating the elicit prompt.
    """
    prompt = f"""
    Give me just 10 breakthroughs in research, development, and clinical trials
    for the bringing of {term} to market and continued research as bullet points
    and not as a timeline. Use only high impact review articles.
    Give a list of citations.
    """

    return prompt

def cluster_label(titles):
    """Return the two most frequent non-stopwords from cluster titles."""
    words = re.findall(
        r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", " ".join(map(str, titles)).lower()
    )
    frequent_words = Counter(
        word for word in words if word not in ENGLISH_STOP_WORDS
    ).most_common(2)
    return " ".join(word for word, _ in frequent_words) or "misc"


def _study_text(df_studies):
    return (
        df_studies["Title"].fillna("").astype(str)
        + " "
        + df_studies["Abstract"].fillna("").astype(str)
    )


def assign_super_clusters(df_studies, descriptions):
    """Assign each generated cluster to the closest Elicit description."""
    descriptions = [
        description.strip()
        for description in descriptions
        if description and description.strip()
    ]
    if not descriptions:
        raise ValueError("Elicit returned no super-cluster descriptions")

    study_text = _study_text(df_studies)
    cluster_text = study_text.groupby(df_studies["Cluster"], sort=False).agg(
        " ".join
    )
    categories = []
    for index, description in enumerate(descriptions, start=1):
        base_label = description
        label = base_label
        suffix = 2
        while label in categories:
            label = f"{base_label} ({suffix})"
            suffix += 1
        categories.append(label)

    corpus = list(cluster_text) + descriptions
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(corpus)
    cluster_vectors = vectors[:len(cluster_text)]
    category_vectors = vectors[len(cluster_text):]
    closest_categories = cosine_similarity(cluster_vectors, category_vectors).argmax(axis=1)
    cluster_to_super_cluster = dict(zip(cluster_text.index, [categories[i] for i in closest_categories]))
    return df_studies["Cluster"].map(cluster_to_super_cluster)

def fetch_pmids_on_search_term(quer_comps):
    # Format query"        
    query = ' AND '.join(quer_comps)
    # Execute search on PubMed
    handle = Entrez.esearch(db="pubmed", term=query, retmax=50000)
    record = Entrez.read(handle)
    handle.close()
    
    # Retrieved PubMed IDs (PMIDs)
    return record["IdList"]

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
                if (any(sub in affil['Affiliation'] for sub in ['Veterans Affairs'])):     
                    authors.append(f"{author['ForeName']} {author['LastName']}")
    except:
        pass
    return authors

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



def build_layer(pmids, lyr_bin, lyr_organism, degree):

    #Fetch all the records
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml", retmode="text")
    records = Entrez.read(handle)
    handle.close()
    records = records['PubmedArticle']

    ids = []
    lyr_authors = []
    sources = []
    targets = []
    tbl_titles = []
    dates = []
    years = []
    abstracts = []
    #Run through the records
    for i, article  in enumerate(records):
        aid = str(article['MedlineCitation']['PMID'])
        ids.append(aid)
        #Add the authors
        authors = fetch_authors(article)
        lyr_authors.extend(authors)

        #Add the dates of publication
        date = article['MedlineCitation']['DateCompleted']
        dates.append(f'{date['Year']}-{date['Month']}-{date['Day']}')
        years.append(date['Year'])
        
        for cite in fetch_citations(article):
            if cite not in pmids:
                sources.append(cite)
                targets.append(aid)      

        #Add it to our table
        tbl_titles.append(article ['MedlineCitation']['Article']['ArticleTitle']) 

        if 'Abstract' in article['MedlineCitation']['Article'].keys():
            abstract_texts = (article['MedlineCitation']['Article']['Abstract']['AbstractText'])
            conclusion = " ".join(abstract_texts)
            for section in abstract_texts:
                if "conclusion" in section.attributes.get("Label", "").lower():
                    conclusion = str(section)
                    break
            abstracts.append(conclusion)        

            
        else:
            abstracts.append("")
            

    df_lyr_studies = pd.DataFrame({'ID': ids,
                                   'Title':tbl_titles,
                                   'Abstract': abstracts,
                                   'Date':dates,
                                   'Years': years,
                                   'Bin': lyr_bin,
                                   'Organism': lyr_organism,
                                   'VA_Afil': 1,  
                                   'Degree': degree
                                  })
    
    df_lyr_network = pd.DataFrame({'Source':sources, 'Target':targets})    
    
    
    return df_lyr_studies, df_lyr_network, lyr_authors

def filter_pmids(ids, query_filters):
    """
    Takes in a list of the clinical trial ids, forms a query for pubmed
    and returns a list of ids for all va studies
    """  
    filtered_ids = []
    # Loop in chunks of 1000
    for chunk in batched(ids, 1000):
        ####Finding published non_clinical studies####
        # Format query"
        id_query = f'({" OR ".join([f"{pmid}[UID]" for pmid in chunk])})'        
        query = ' AND '.join([id_query] + query_filters)
       
        # Execute search on PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=10000)
        record = Entrez.read(handle)
        handle.close()
        
        # Retrieved PubMed IDs (PMIDs)
        filtered_ids.extend(record["IdList"])
        
    return filtered_ids

def build_network(term, progress_callback=None):
    #Term to be searched on
    #term = "glp-1"

    def update_progress(message):
        if progress_callback is not None:
            progress_callback(message)

    # Required by NCBI Entrez API
    Entrez.email = "steven.cogill@va.gov"



    #Query filters for binning
    clinical_filter_query = '(Clinical Trial[Publication Type])'
    human_filter_query = '(Humans[MeSH])  NOT (Clinical Trial[Publication Type])'
    animal_filter_query = '(Animals[Mesh]) NOT (Humans[Mesh])  NOT (Clinical Trial[Publication Type])'
    va_filter_query = '((va funded[Filter]) OR (Veterans Affairs[ad]) OR (Department of Veterans Affairs[ad]))'

    update_progress("Step 1 of 7: Searching PubMed for VA studies across three study types.")
    clinical_studies = fetch_pmids_on_search_term([term, clinical_filter_query, va_filter_query])
    human_studies = fetch_pmids_on_search_term([term, human_filter_query, va_filter_query])
    animal_studies = fetch_pmids_on_search_term([term, animal_filter_query, va_filter_query])


    update_progress("Step 2 of 7: Retrieving primary article details, abstracts, authors, and citations.")
    #Clinical
    df_clinical_studies, df_clinical_network, clinical_authors = build_layer(clinical_studies, 'clinical', 'Human', 1)
    hum_pmids = filter_pmids(df_clinical_network.Source.unique(), [va_filter_query, human_filter_query])
    ani_pmids = filter_pmids(df_clinical_network.Source.unique(), [va_filter_query, animal_filter_query])
    df_clinical_network = df_clinical_network[df_clinical_network.Source.isin(hum_pmids + ani_pmids)]




    #Human
    df_human_studies, df_human_network, human_authors = build_layer(human_studies, 'human', 'Human', 1)
    clin_pmids = filter_pmids(df_human_network.Source.unique(), [va_filter_query, clinical_filter_query])
    ani_pmids += filter_pmids(df_human_network.Source.unique(), [va_filter_query, animal_filter_query])
    df_human_network = df_human_network[df_human_network.Source.isin(clin_pmids + ani_pmids)]

    #Animal
    df_animal_studies, df_animal_network, animal_authors = build_layer(animal_studies, 'animal', 'Animal', 1)
    hum_pmids += filter_pmids(df_animal_network.Source.unique(), [va_filter_query, human_filter_query])
    df_animal_network = df_animal_network[df_animal_network.Source.isin(clin_pmids + hum_pmids)]

    update_progress("Step 3 of 7: Finding and retrieving studies cited by the primary articles.")
    #Buld out secondary tables
    df_clinical_sec_studies, df_clinical_sec_network, clinical_sec_authors = build_layer(clin_pmids, 'clinical', 'Human', 2)
    df_clinical_sec_network = df_clinical_sec_network[df_clinical_sec_network.Source.isin(hum_pmids + list(df_human_studies.ID.values) +
                                                                                ani_pmids + list(df_animal_studies.ID.values))]



    df_human_sec_studies, df_human_sec_network, human_sec_authors = build_layer(hum_pmids, 'human', 'Human', 2)
    df_human_sec_network = df_human_sec_network[df_human_sec_network.Source.isin(clin_pmids + list(df_clinical_studies.ID.values) +
                                                                                ani_pmids + list(df_animal_studies.ID.values))]




    df_animal_sec_studies, df_animal_sec_network, animal_sec_authors = build_layer(ani_pmids, 'animal', 'Animal', 2)
    df_animal_sec_network = df_animal_sec_network[df_animal_sec_network.Source.isin(clin_pmids + list(df_clinical_studies.ID.values) +
                                                                                hum_pmids + list(df_human_studies.ID.values))]




    update_progress("Step 4 of 7: Combining study records and clustering titles with abstracts.")
    df_studies = pd.concat([df_clinical_studies, df_human_studies, df_animal_studies,
                        df_clinical_sec_studies, df_human_sec_studies, df_animal_sec_studies])

    df_network = pd.concat([df_clinical_network, df_human_network, df_animal_network,
                        df_clinical_sec_network, df_human_sec_network, df_animal_sec_network])

    df_studies.drop_duplicates(subset=['ID'], keep='first', inplace=True, ignore_index=True)
    df_network.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)

    authors = list(set(clinical_authors + human_authors + animal_authors + clinical_sec_authors + human_sec_authors + animal_sec_authors))

    #Cluster
    cluster_labels = []
    base_ids = []
    used_labels = set()
    for b in df_studies.Bin.unique():    
        studs = df_studies[df_studies.Bin==b]
        
        clusters = int(math.sqrt((len(studs)/2)))
    
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(_study_text(studs).values)
        #Fit a model
        model = KMeans(n_clusters=clusters, init='k-means++', max_iter=200, n_init=10, random_state=42)
        model.fit(X)
        labels = []
        label_by_cluster = {
            cluster_number: cluster_label(
                studs.loc[model.labels_ == cluster_number, "Title"]
            )
            for cluster_number in range(clusters)
        }
        for cluster_number in range(clusters):
            base_label = label_by_cluster[cluster_number]
            label = base_label
            suffix = 2
            while label in used_labels:
                label = f"{base_label} ({suffix})"
                suffix += 1
            used_labels.add(label)
            label_by_cluster[cluster_number] = label
        labels = [label_by_cluster[cluster_number] for cluster_number in model.labels_]
        
        cluster_labels += labels
        base_ids += list(studs.ID.values)

    df_studies = pd.merge(df_studies, pd.DataFrame({'CID':base_ids, 'Cluster':cluster_labels}), left_on='ID', right_on='CID', how='left')
    df_studies.drop(columns=['CID'], inplace=True)
    update_progress("Step 5 of 7: Generating breakthrough descriptions with Elicit.")
    breakthrough_descriptions = elicit_search(term)
    df_studies['BreakthruCluster'] = assign_super_clusters(
        df_studies, breakthrough_descriptions
    )

    update_progress("Step 6 of 7: Linking clusters and calculating citation strength.")
    #Recalc edges
    df_network_test = df_network.copy(deep=True)
    df_network_test = pd.merge(df_network_test, df_studies[['ID', 'Cluster']], left_on='Source', right_on='ID', how='left')
    df_network_test = df_network_test.rename(columns={"Cluster": "Source_cluster"})
    df_network_test.drop(columns=['ID'], inplace=True)
    df_network_test = pd.merge(df_network_test, df_studies[['ID', 'Cluster']], left_on='Target', right_on='ID', how='left')
    df_network_test = df_network_test.rename(columns={"Cluster": "Target_cluster"})
    df_network_test.drop(columns=['ID'], inplace=True)
    df_edges = df_network_test.groupby(['Source_cluster', 'Target_cluster']).size().reset_index(name='count')
    df_nodes = df_studies.groupby(['Bin', 'Cluster', 'BreakthruCluster']).size().reset_index(name='count')

    pmid = list(df_studies.ID.values)
    
    handle = Entrez.elink(dbfrom="pubmed", LinkName="pubmed_pubmed_citedin", id=pmid)
    record = Entrez.read(handle)
    handle.close()
    
    cites = []
    for r in record:
        if r['LinkSetDb']:
            cites.append(len(r['LinkSetDb'][0]['Link']))
        else:
            cites.append(0)
    df_studies['Strength'] = cites
    df_studies.sort_values(by='Strength', ascending=False, inplace=True)
    
    return df_studies, df_network, df_edges, df_nodes, authors

