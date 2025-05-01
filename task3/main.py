from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import re
from typing import List, Dict, Tuple

# Part (a): Sentence splitting and embeddings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from networkx.algorithms import community

# Ensure punkt tokenizer is available
nltk.download('punkt')


def split_sentences(text: str) -> List[str]:
    """
    Zerlegt einen langen Text in eine Liste von Sätzen.
    """
    sentences = nltk.tokenize.sent_tokenize(text, language='german')
    return sentences


def compute_tfidf_embeddings(sentences: List[str]) -> np.ndarray:
    """
    Erzeugt TF-IDF Vektoren für eine Liste von Sätzen.
    Rückgabe: (n_sentences, n_features)
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    # In dense Form zurückgeben
    return tfidf_matrix.toarray()


def compute_sentence_transformer_embeddings(sentences: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Erzeugt Embeddings mittels Sentence-Transformers.
    Rückgabe: (n_sentences, dim)
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


# Part (b): Graph-Bau und TextRank

def build_similarity_graph(embeddings: np.ndarray, threshold: float = None) -> nx.Graph:
    n = embeddings.shape[0]
    sim_matrix = np.dot(embeddings, embeddings.T)

    # Normierte Embeddings: Skalarprodukt = Kosinus
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j]
            if threshold is None or score >= threshold:
                G.add_edge(i, j, weight=score)
    return G


def text_rank(graph: nx.Graph, top_n: int = 5) -> List[Tuple[int, float]]:
    # PageRank mit Kantengewichten
    scores = nx.pagerank(graph, weight='weight')
    # Sortieren nach Score
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


def analyze_graph_structure(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)  # 2m / (n*(n-1))
    avg_deg = sum(dict(G.degree()).values()) / n
    num_components = nx.number_connected_components(G)
    avg_clustering = nx.average_clustering(G, weight='weight')

    print(f"Nodes: {n}")
    print(f"Edges: {m}")
    print(f"Density: {density:.4f}")
    print(f"Avg. degree: {avg_deg:.2f}")
    print(f"Connected components: {num_components}")
    print(f"Avg. clustering coeff.: {avg_clustering:.4f}")

def plot_degree_histogram(G: nx.Graph):
    degs = [d for _, d in G.degree()]
    plt.figure()
    plt.hist(degs, bins=20)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()

def central_nodes(G):
    pagerank   = nx.pagerank(G, weight='weight')
    betweeness = nx.betweenness_centrality(G, weight='weight')
    closeness  = nx.closeness_centrality(G)

    print("Top 5 nodes by PageRank:")
    for idx, score in sorted(pagerank.items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"Node {idx}: {score:.4f}")
    
    print("\nTop 5 nodes by Betweenness Centrality:")
    for idx, score in sorted(betweeness.items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"Node {idx}: {score:.4f}")
    print("\nTop 5 nodes by Closeness Centrality:")
    for idx, score in sorted(closeness.items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"Node {idx}: {score:.4f}")

def communities(G):
    communities = list(community.greedy_modularity_communities(G, weight='weight'))
    print(f"Found {len(communities)} communities, sizes: {[len(c) for c in communities]}")

def plot_graph(G: nx.Graph):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Graph Visualization")
    plt.show()


if __name__ == '__main__':
    with open("./1984.txt") as f:
        text = f.read()

    sentences = split_sentences(text)
    tfidf_emb = compute_tfidf_embeddings(sentences)

    model_name = 'all-MiniLM-L6-v2' # "sentence-transformers/distiluse-base-multilingual-cased-v1"
    st_emb = compute_sentence_transformer_embeddings(sentences, model_name=model_name)

    # b) Graph für TF-IDF
    G_tfidf = build_similarity_graph(tfidf_emb, threshold=0.1)
    top_tfidf = text_rank(G_tfidf, top_n=5)

    # b) Graph für Sentence-Transformer
    G_st = build_similarity_graph(st_emb, threshold=0.2)
    top_st = text_rank(G_st, top_n=5)

    print("Top Sätze (TF-IDF):")
    for idx, score in top_tfidf:
        print(f"[{idx}] {score:.4f} - {sentences[idx]}")

    print(f"\nTop Sätze {model_name}:")
    for idx, score in top_st:
        print(f"[{idx}] {score:.4f} - {sentences[idx]}")



    # Graph-Analyse

    # print("\nGraph-Analyse (TF-IDF):")
    # analyze_graph_structure(G_tfidf)
    # plot_degree_histogram(G_tfidf)
    # central_nodes(G_tfidf)
    # communities(G_tfidf)
    # plot_graph(G_tfidf)
    # print("\nGraph-Analyse (Sentence-Transformer):")
    # analyze_graph_structure(G_st)
    # plot_degree_histogram(G_st)
    # central_nodes(G_st)
    # communities(G_st)
    # plot_graph(G_st)

