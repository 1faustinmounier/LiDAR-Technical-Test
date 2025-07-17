import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from skimage.transform import hough_line, hough_line_peaks
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
    proj = points[:, [1, 2]]
    min_y, min_z = proj.min(axis=0)
    max_y, max_z = proj.max(axis=0)
    grid_y = np.arange(min_y, max_y, res)
    grid_z = np.arange(min_z, max_z, res)
    img = np.zeros((len(grid_y), len(grid_z)), dtype=np.uint8)
    for y, z in proj:
        iy = int((y - min_y) / res)
        iz = int((z - min_z) / res)
        if 0 <= iy < img.shape[0] and 0 <= iz < img.shape[1]:
            img[iy, iz] = 1
    h, theta, d = hough_line(img)
    accums, angles, dists = hough_line_peaks(h, theta, d, num_peaks=num_peaks)
    labels = np.full(len(points), -1)
    for i, (angle, dist) in enumerate(zip(angles, dists)):
        y0 = min_y
        z0 = (dist - y0 * np.sin(angle)) / np.cos(angle)
        y1 = max_y
        z1 = (dist - y1 * np.sin(angle)) / np.cos(angle)
        for j, (y, z) in enumerate(proj):
            num = abs((z1 - z0) * y - (y1 - y0) * z + z1 * y0 - y1 * z0)
            den = np.sqrt((z1 - z0) ** 2 + (y1 - y0) ** 2)
            if den > 0 and num / den < res * 2:
                labels[j] = i
    return labels

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