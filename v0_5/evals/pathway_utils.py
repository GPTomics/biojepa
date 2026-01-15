import numpy as np
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

PATHWAY_CACHE = {}


def load_pathway_annotations(libraries=None):
    '''Load gene set libraries via gseapy. Returns dict: {library_name: {pathway: [genes]}}.'''
    import gseapy as gp

    if libraries is None:
        libraries = ['KEGG_2021_Human', 'Reactome_Pathways_2024', 'GO_Biological_Process_2023']

    result = {}
    for lib_name in libraries:
        if lib_name in PATHWAY_CACHE:
            result[lib_name] = PATHWAY_CACHE[lib_name]
        else:
            print(f'Loading {lib_name}...')
            gene_sets = gp.get_library(lib_name)
            PATHWAY_CACHE[lib_name] = gene_sets
            result[lib_name] = gene_sets
            print(f'  {len(gene_sets)} pathways loaded')

    return result


def build_gene_to_pathways(pathway_dict, min_pathway_size=10, max_pathway_size=500):
    '''Invert pathway dict to get gene -> [pathways] mapping. Filters by pathway size.'''
    gene_to_pathways = defaultdict(list)

    for pathway, genes in pathway_dict.items():
        if min_pathway_size <= len(genes) <= max_pathway_size:
            for gene in genes:
                gene_to_pathways[gene.upper()].append(pathway)

    return dict(gene_to_pathways)


def map_genes_to_pathways(gene_names, pathway_dict, min_pathway_size=10, max_pathway_size=500):
    '''Map a list of gene names to their pathway memberships.

    Returns:
        gene_pathway_labels: dict {gene_name: primary_pathway} for genes with pathway membership
        pathway_to_genes: dict {pathway: [gene_indices]} mapping pathways to gene indices
    '''
    gene_to_pathways = build_gene_to_pathways(pathway_dict, min_pathway_size, max_pathway_size)

    gene_names_upper = [g.upper() for g in gene_names]
    name_to_idx = {name: idx for idx, name in enumerate(gene_names_upper)}

    gene_pathway_labels = {}
    pathway_to_genes = defaultdict(list)

    for idx, gene in enumerate(gene_names_upper):
        if gene in gene_to_pathways:
            primary_pathway = gene_to_pathways[gene][0]
            gene_pathway_labels[gene] = primary_pathway
            pathway_to_genes[primary_pathway].append(idx)

    return gene_pathway_labels, dict(pathway_to_genes)


def compute_pathway_clustering_metrics(embeddings, labels, min_samples_per_class=5):
    '''Compute clustering quality metrics for embeddings grouped by pathway labels.

    Args:
        embeddings: np.array of shape [N, D]
        labels: list/array of pathway labels (same length as embeddings)
        min_samples_per_class: minimum samples needed per pathway to include

    Returns:
        dict with silhouette_score, knn_accuracy, n_classes, n_samples
    '''
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    unique, counts = np.unique(encoded_labels, return_counts=True)
    valid_classes = unique[counts >= min_samples_per_class]

    if len(valid_classes) < 2:
        return {'silhouette_score': None, 'knn_accuracy': None, 'n_classes': 0, 'n_samples': 0, 'error': 'Not enough valid classes'}

    mask = np.isin(encoded_labels, valid_classes)
    filtered_embeddings = embeddings[mask]
    filtered_labels = encoded_labels[mask]

    filtered_labels = LabelEncoder().fit_transform(filtered_labels)

    sil_score = silhouette_score(filtered_embeddings, filtered_labels)

    knn = KNeighborsClassifier(n_neighbors=min(5, min_samples_per_class - 1), metric='cosine')
    knn.fit(filtered_embeddings, filtered_labels)
    knn_preds = knn.predict(filtered_embeddings)
    knn_acc = np.mean(knn_preds == filtered_labels)

    return {
        'silhouette_score': float(sil_score),
        'knn_accuracy': float(knn_acc),
        'n_classes': len(valid_classes),
        'n_samples': len(filtered_embeddings)
    }


def get_pathway_summary(pathway_to_genes, top_n=20):
    '''Get summary statistics about pathway coverage.'''
    sizes = [(p, len(genes)) for p, genes in pathway_to_genes.items()]
    sizes.sort(key=lambda x: -x[1])

    return {
        'total_pathways': len(pathway_to_genes),
        'total_genes_covered': sum(len(g) for g in pathway_to_genes.values()),
        'top_pathways': sizes[:top_n],
        'mean_pathway_size': np.mean([s[1] for s in sizes]) if sizes else 0,
        'median_pathway_size': np.median([s[1] for s in sizes]) if sizes else 0
    }
