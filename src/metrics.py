import numpy as np
from sklearn.decomposition import PCA
from . import clustering


def courbure_moyenne(points):
    """Calcule la courbure moyenne basée sur l'angle entre vecteurs successifs."""
    if len(points) < 3:
        return 0.0
    # Trier les points selon l'axe principal
    pca = PCA(n_components=1)
    proj = pca.fit_transform(points)
    sorted_indices = np.argsort(proj.flatten())
    sorted_points = points[sorted_indices]

    # Calculer les vecteurs entre points successifs
    vectors = np.diff(sorted_points, axis=0)
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    if len(vectors) < 2:
        return 0.0

    # Calculer les angles entre vecteurs successifs
    angles = []
    for i in range(len(vectors) - 1):
        dot_product = np.clip(np.dot(vectors[i], vectors[i+1]), -1, 1)
        angle = np.arccos(dot_product)
        angles.append(angle)

    return np.mean(angles) if angles else 0.0


def compute_metrics(points, labels):
    """Calcul les métriques de qualité de segmentation."""
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    n_total = len(points)
    n_noise = np.sum(labels == -1)

    metrics = {
        'n_clusters': n_clusters,
        'n_points_total': n_total,
        'n_noise': n_noise,
        'pct_bruit': n_noise / n_total if n_total > 0 else 0
    }

    if n_clusters == 0:
        metrics['cv_taille'] = 1.0  # valeur par défaut pour éviter KeyError
        metrics['continuite'] = 0.0
        metrics['linearite'] = 0.0
        metrics['compacite_laterale'] = 1.0
        metrics['courbure'] = 1.0
        metrics['taille_moy'] = 0.0
        metrics['taille_std'] = 0.0
        return metrics

    # Taille des clusters
    sizes = [np.sum(labels == l) for l in unique_labels]
    metrics['taille_moy'] = np.mean(sizes)
    metrics['taille_std'] = np.std(sizes)
    metrics['cv_taille'] = metrics['taille_std'] / metrics['taille_moy'] if metrics['taille_moy'] > 0 else 1.0

    # Continuité et linéarité
    continuites = []
    linearites = []
    compacites = []
    courbures = []

    for label in unique_labels:
        cluster_points = points[labels == label]
        if len(cluster_points) < 3:
            continuites.append(0)
            linearites.append(0)
            compacites.append(1)
            courbures.append(1.0)
            continue

        # Continuité (longueur axiale / longueur totale)
        pca = PCA(n_components=1)
        proj = pca.fit_transform(cluster_points)
        length_axial = proj.max() - proj.min()
        length_total = np.linalg.norm(cluster_points.max(axis=0) - cluster_points.min(axis=0))
        continuite = length_axial / (length_total + 1e-8)
        continuites.append(continuite)

        # Linéarité (R² de la régression PCA)
        recon = pca.inverse_transform(proj)
        ss_res = np.sum((cluster_points - recon) **2)
        ss_tot = np.sum((cluster_points - cluster_points.mean(axis=0)) **2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        linearites.append(r2)

        # Compacité latérale (moyenne des distances perpendiculaires)
        dists = np.linalg.norm(cluster_points - recon, axis=1)
        compacites.append(np.mean(dists))

        # Courbure moyenne
        courbure = courbure_moyenne(cluster_points)
        courbures.append(courbure)

    metrics['continuite'] = np.mean(continuites)
    metrics['linearite'] = np.mean(linearites)
    metrics['compacite_laterale'] = np.mean(compacites)
    metrics['courbure'] = np.mean(courbures)

    return metrics


def evaluate_segmentation(points, labels, target_min=3, target_max=6):
    """Évalue la qualité de segmentation avec des critères cibles."""
    metrics = compute_metrics(points, labels)

    # Score global
    score = 0

    # Nombre de clusters dans la cible
    if target_min <= metrics['n_clusters'] <= target_max:
        score += 2
    else:
        score -= abs(metrics['n_clusters'] - (target_min + target_max) / 2)

    # Bruit faible
    score += max(0, 1 - metrics['pct_bruit'] * 10)
    # Équilibre des tailles
    score += max(0, 1 - metrics.get('cv_taille', 1.0) * 5)

    # Continuité
    score += metrics.get('continuite', 0.0)

    # Linéarité
    score += metrics.get('linearite', 0.0)

    # Compacité (plus bas = mieux)
    score += max(0, 1 - metrics.get('compacite_laterale', 1.0) * 5)

    # Courbure (pénalise les courbures élevées)
    score += max(0, 1 - metrics.get('courbure', 1.0))

    metrics['score_global'] = score
    return metrics


def compare_all_methods(points, target_min=3, target_max=6):
    """Compare toutes les méthodes de clustering sur les mêmes données."""
    methods = {
        'RANSAC': clustering.run_ransac,
        'Directional': clustering.run_direction_clustering,
        'Agglomerative': clustering.run_agglomerative,
        'Hough2D': clustering.run_hough2d,
        'GraphKNN': clustering.run_graph_knn
    }

    results = {}

    for method_name, method_func in methods.items():
        try:
            if method_name == 'RANSAC':
                labels = method_func(points, residual_threshold=0.05, min_samples=40, max_iter=10)
            elif method_name == 'Directional':
                labels = method_func(points, k=10, eps=0.2, min_samples=5)
            elif method_name == 'Agglomerative':
                labels = method_func(points, n_clusters=4, linkage='ward')
            elif method_name == 'Hough2D':
                labels = method_func(points, res=0.05, num_peaks=4)
            elif method_name == 'GraphKNN':
                labels = method_func(points, k=6)

            metrics = evaluate_segmentation(points, labels, target_min, target_max)
            results[method_name] = {
                'labels': labels,
                'metrics': metrics,
                'n_clusters': metrics['n_clusters'],
                'score': metrics['score_global']
            }
            print(f"Score {method_name}: {metrics['score_global']:.2f}")
        except Exception as e:
            results[method_name] = {
                'error': str(e),
                'score': -999
            }
            print(f"Erreur {method_name}: {str(e)}")

    return results


def get_cluster_info(points, labels):
    """Retourne les informations détaillées sur chaque cluster."""
    unique_labels = np.unique(labels[labels != -1])

    clusters_info = {}
    for label in unique_labels:
        cluster_points = points[labels == label]
        clusters_info[label] = {
            'n_points': len(cluster_points),
            'center': cluster_points.mean(axis=0),
            'bounds': {
                'x_min': cluster_points[:, 0].min(),
                'x_max': cluster_points[:, 0].max(),
                'y_min': cluster_points[:, 1].min(),
                'y_max': cluster_points[:, 1].max(),
                'z_min': cluster_points[:, 2].min(),
                'z_max': cluster_points[:, 2].max()
            }
        }

    return clusters_info 