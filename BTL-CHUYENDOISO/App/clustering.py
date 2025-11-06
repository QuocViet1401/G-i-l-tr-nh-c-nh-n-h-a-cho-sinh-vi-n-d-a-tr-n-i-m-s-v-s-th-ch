import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def evaluate_clustering(features: np.ndarray, method: str = 'kmeans', n_clusters: int = 8) -> tuple:
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=2)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError("Invalid clustering method")

    labels = model.fit_predict(features)
    if len(set(labels)) > 1:
        score = silhouette_score(features, labels)
    else:
        score = -1
    return model, labels, score

def find_best_clustering(features: np.ndarray) -> tuple:
    methods = ['kmeans', 'dbscan', 'hierarchical']
    best_score = -1
    best_model = None
    best_labels = None
    best_method = None

    for method in methods:
        try:
            model, labels, score = evaluate_clustering(features, method=method, n_clusters=8)
            if score > best_score:
                best_score = score
                best_model = model
                best_labels = labels
                best_method = method
        except:
            continue

    return best_model, best_labels, best_method