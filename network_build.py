
import pandas as pd
import re
import os
from Bio import Entrez
from itertools import batched
from pathlib import Path
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from elicit_interface import elicit_search


def _study_texts(df_studies):
    """Combine the article fields used for TF-IDF matching and clustering."""
    text_fields = (
        "MeSH Terms",
        "Major MeSH Terms",
        "Title",
    )
    return pd.concat(
        [
            df_studies.get(field, pd.Series("", index=df_studies.index))
            .fillna("")
            .astype(str)
            for field in text_fields
        ],
        axis=1,
    ).agg(" ".join, axis=1)


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




def assign_super_clusters(df_studies, descriptions):
    """Assign each article to the closest Elicit description using article text."""
    descriptions = [
        description.strip()
        for description in descriptions
        if description and description.strip()
    ]
    if not descriptions:
        raise ValueError("Elicit returned no super-cluster descriptions")

    categories = []
    for description in descriptions:
        base_label = description
        label = base_label
        suffix = 2
        while label in categories:
            label = f"{base_label} ({suffix})"
            suffix += 1
        categories.append(label)

    study_texts = _study_texts(df_studies)
    has_study_texts = study_texts.str.strip().ne("")
    assignments = pd.Series("", index=df_studies.index, dtype="object")
    if not has_study_texts.any():
        return assignments

    corpus = study_texts[has_study_texts].tolist() + descriptions
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    vectors = vectorizer.fit_transform(corpus)
    article_vectors = vectors[:has_study_texts.sum()]
    category_vectors = vectors[has_study_texts.sum():]
    closest_categories = cosine_similarity(
        article_vectors, category_vectors
    ).argmax(axis=1)
    assignments.loc[has_study_texts] = [
        categories[i] for i in closest_categories
    ]
    return assignments


def _cluster_studies(df_studies):
    """Cluster article text within each bin and breakthrough group."""
    df_studies = df_studies.copy()
    cluster_labels = pd.Series(index=df_studies.index, dtype="object")
    next_cluster_id = 1
    junk_words = {
        "cells", "cell", "patient", "patients", "participant", "participants",
        "years", "placebo", "group", "chief", "outcomes", "human", "humans",
        "animals", "animal", "sup sup", "periodicals topic", "disease models"
    }

    for _, group in df_studies.groupby(
        ["Bin", "BreakthruCluster"], sort=False, dropna=False
    ):
        study_texts = _study_texts(group)
        has_study_texts = study_texts.str.strip().ne("")
        labeled_indices = group.index[has_study_texts]

        if not has_study_texts.any():
            cluster_labels.loc[group.index] = (
                f"No MeSH terms (Cluster {next_cluster_id})"
            )
            next_cluster_id += 1
            continue

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
        vectors = vectorizer.fit_transform(study_texts.loc[labeled_indices])
        distinct_terms = study_texts.loc[labeled_indices].nunique()
        cluster_count = min(
            max(1, int(math.sqrt(len(labeled_indices) / 2))),
            distinct_terms,
        )
        model = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            max_iter=200,
            n_init=10,
            random_state=42,
        )
        model.fit(vectors)

        terms = vectorizer.get_feature_names_out()
        major_mesh_terms = group.get(
            "Major MeSH Terms", pd.Series("", index=group.index)
        ).fillna("").astype(str)
        analyze = vectorizer.build_analyzer()
        labels_by_cluster = {}
        for cluster_number, center in enumerate(model.cluster_centers_):
            cluster_id = next_cluster_id
            next_cluster_id += 1
            cluster_indices = labeled_indices[model.labels_ == cluster_number]
            major_features = {
                feature
                for major_term in major_mesh_terms.loc[cluster_indices]
                for feature in analyze(major_term)
            }
            ranked_terms = [
                terms[index]
                for index in center.argsort()[::-1]
                if len(terms[index].split()) == len(set(terms[index].split()))
                and not any(
                    token in junk_words or token.isdigit()
                    for token in terms[index].split()
                )
                and len(terms[index].split()) >= 2
                and not re.search(r"\d", terms[index])
            ]
            ranked_terms.sort(key=lambda term: term not in major_features)

            label_terms = []
            for term in ranked_terms:
                term_tokens = set(term.split())
                redundant = False
                replaced = False
                for selected_index, selected_term in enumerate(label_terms):
                    selected_term_tokens = set(selected_term.split())
                    if term_tokens.issubset(selected_term_tokens):
                        redundant = True
                        break
                    if selected_term_tokens.issubset(term_tokens):
                        if (
                            selected_term in major_features
                            and term not in major_features
                        ):
                            redundant = True
                            break
                        label_terms[selected_index] = term
                        replaced = True
                        break
                    if term_tokens & selected_term_tokens:
                        redundant = True
                        break
                if redundant:
                    continue
                if not replaced:
                    label_terms.append(term)
                if len(label_terms) == 4:
                    break
            labels_by_cluster[cluster_number] = (
                f"{'; '.join(label_terms) or 'No MeSH terms'} "
                f"(Cluster {cluster_id})"
            )

        cluster_labels.loc[labeled_indices] = [
            labels_by_cluster[cluster_number]
            for cluster_number in model.labels_
        ]
        missing_indices = group.index[~has_study_texts]
        if len(missing_indices):
            cluster_labels.loc[missing_indices] = (
                f"No MeSH terms (Cluster {next_cluster_id})"
            )
            next_cluster_id += 1

    df_studies["Cluster"] = cluster_labels
    return df_studies


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
        # Drill down to the author list and keep authors whose affiliation matches
        # Veterans Affairs or VA-based healthcare system naming patterns.
        va_re = re.compile(
            r"(?:\bVeterans Affairs\b|\bVA\b).*?(?:\bHealthcare System\b|\bHealth Care System\b)",
            flags=re.IGNORECASE,
        )
        for author in auth_article['MedlineCitation']['Article']['AuthorList']:
            for affil in author.get('AffiliationInfo', []):
                affiliation = str(affil.get('Affiliation', '')).strip()
                lower_affiliation = affiliation.lower()
                if (
                    'veterans affairs' in lower_affiliation
                    or 'va' in lower_affiliation
                    or va_re.search(affiliation)
                ):
                    authors.append(f"{author['ForeName']} {author['LastName']}")
    except Exception:
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
    mesh_terms = []
    major_mesh_terms = []
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
        mesh_headings = article["MedlineCitation"].get("MeshHeadingList", [])
        article_major_mesh_terms = [
            str(mesh["DescriptorName"])
            for mesh in mesh_headings
            if mesh["DescriptorName"].attributes.get("MajorTopicYN") == "Y"
        ]
        major_mesh_terms.append(", ".join(article_major_mesh_terms))
        mesh_terms.append(
            ", ".join(str(mesh["DescriptorName"]) for mesh in mesh_headings)
        )

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
                                   'MeSH Terms': mesh_terms,
                                   'Major MeSH Terms': major_mesh_terms,
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
    Entrez.email = "cogillsb@kesho-ai.com"
    load_dotenv(Path(__file__).with_name(".env"))
    Entrez.api_key = os.getenv("NCBI_API_KEY")


    #Query filters for binning
    clinical_filter_query = '(Clinical Trial[Publication Type])'
    human_filter_query = '(Humans[MeSH])  NOT (Clinical Trial[Publication Type])'
    animal_filter_query = '(Animals[Mesh]) NOT (Humans[Mesh])  NOT (Clinical Trial[Publication Type])'
    va_filter_query = '((va funded[Filter]) OR (Veterans Affairs[ad])  OR (Department of Veterans Affairs[ad]) OR (VA[ad] AND Healthcare System[ad]))'

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

    print(len(df_animal_studies))

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




    update_progress("Step 4 of 7: Combining studies and generating breakthrough descriptions.")
    df_studies = pd.concat([df_clinical_studies, df_human_studies, df_animal_studies,
                        df_clinical_sec_studies, df_human_sec_studies, df_animal_sec_studies])

    df_network = pd.concat([df_clinical_network, df_human_network, df_animal_network,
                        df_clinical_sec_network, df_human_sec_network, df_animal_sec_network])

    df_studies.drop_duplicates(subset=['ID'], keep='first', inplace=True, ignore_index=True)
    df_network.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)

    authors = list(set(clinical_authors + human_authors + animal_authors + clinical_sec_authors + human_sec_authors + animal_sec_authors))

    breakthrough_descriptions = elicit_search(term)
    df_studies['BreakthruCluster'] = assign_super_clusters(
        df_studies, breakthrough_descriptions
    )

    update_progress(
        "Step 5 of 7: Clustering studies by MeSH terms within breakthrough groups."
    )
    df_studies = _cluster_studies(df_studies)

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
    
    handle = Entrez.elink(
        dbfrom="pubmed", LinkName="pubmed_pubmed_citedin", id=pmid
    )
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
    
    return df_studies, df_network, df_edges, df_nodes, authors, breakthrough_descriptions
