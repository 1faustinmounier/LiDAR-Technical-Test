import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
# Removed skimage dependency - using alternative implementation
import networkx as nx

def run_ransac(points, residual_threshold=0.05, min_samples=40, max_iter=10):
    labels = np.full(len(points), -1)
    remaining = points.copy()
    orig_idx = np.arange(len(points))
    cable_id = 0
    for _ in range(max_iter):
        if len(remaining) < min_samples:
            break
        X = remaining[:, 1].reshape(-1, 1)
        y = remaining[:, 2]
        model = make_pipeline(
            PolynomialFeatures(2),
            RANSACRegressor(
                residual_threshold=residual_threshold,
                min_samples=min_samples,
                random_state=42
            )
        )
        try:
            model.fit(X, y)
            inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
            if inlier_mask.sum() < min_samples:
                break
            labels[orig_idx[inlier_mask]] = cable_id
            remaining = remaining[~inlier_mask]
            orig_idx = orig_idx[~inlier_mask]
            cable_id += 1
        except Exception:
            break
    return labels

def run_direction_clustering(points, k=10, eps=0.2, min_samples=5):
    directions = []
    knn = NearestNeighbors(n_neighbors=k+1).fit(points)
    dists, idxs = knn.kneighbors(points)
    for i in range(len(points)):
        neighbors = points[idxs[i, 1:]]
        if len(neighbors) < 2:
            directions.append([0, 0, 0])
            continue
        pca = PCA(n_components=1)
        try:
            pca.fit(neighbors)
            directions.append(pca.components_[0])
        except Exception:
            directions.append([0, 0, 0])
    directions = np.array(directions)
    features = np.hstack([points, directions])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    return clustering.labels_

def run_agglomerative(points, n_clusters=3, linkage='ward'):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return clustering.fit_predict(points)

def run_hough2d(points, res=0.05, num_peaks=4):
    """
    Alternative implementation of Hough line detection without skimage dependency.
    Uses a simplified approach based on angle-based clustering.
    """
    proj = points[:, [1, 2]]
    
    # Calculate angles between consecutive points
    angles = []
    for i in range(len(proj) - 1):
        dy = proj[i+1, 0] - proj[i, 0]
        dz = proj[i+1, 1] - proj[i, 1]
        if dy != 0 or dz != 0:
            angle = np.arctan2(dz, dy)
            angles.append(angle)
    
    if len(angles) == 0:
        return np.full(len(points), -1)
    
    angles = np.array(angles)
    
    # Cluster points based on angle similarity
    try:
        kmeans = KMeans(n_clusters=min(num_peaks, len(angles)), random_state=42)
        angle_labels = kmeans.fit_predict(angles.reshape(-1, 1))
        
        # Assign labels to points based on their position in the sequence
        labels = np.full(len(points), -1)
        for i in range(len(angles)):
            labels[i] = angle_labels[i]
            labels[i+1] = angle_labels[i]  # Assign same label to consecutive points
        
        return labels
    except:
        # Fallback to simple clustering
        return run_agglomerative(points, n_clusters=num_peaks)

def run_graph_knn(points, k=6):
    A = kneighbors_graph(points, k, mode='connectivity', include_self=False)
    G = nx.from_numpy_array(A.toarray())
    components = list(nx.connected_components(G))
    labels = np.full(len(points), -1)
    for i, comp in enumerate(components):
        for idx in comp:
            labels[idx] = i
    return labels

def detect_cables(points, method="RANSAC", **kwargs):
    if method == "RANSAC":
        return run_ransac(points, **kwargs)
    elif method == "Directional":
        return run_direction_clustering(points, **kwargs)
    elif method == "Agglomerative":
        return run_agglomerative(points, **kwargs)
    elif method == "Hough2D":
        return run_hough2d(points, **kwargs)
    elif method == "GraphKNN":
        return run_graph_knn(points, **kwargs)
    else:
        raise ValueError(f"Méthode inconnue: {method}")