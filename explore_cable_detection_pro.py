import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from skimage.transform import hough_line, hough_line_peaks
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
import itertools
import networkx as nx

# Correction import networkx pour compatibilité toutes versions
import networkx as nx

st.set_page_config(page_title="Détection Pro Câbles", layout="wide")
st.title("Explorateur avancé de détection de câbles dans un nuage de points")

st.markdown("""
**Instructions :**
- Choisissez un fichier .parquet.
- Choisissez une méthode et ajustez les paramètres (slider ou saisie manuelle).
- (Option) Tracez un trait sur la projection 2D pour initier un suivi spatial.
- Lancez la détection ou l'exploration automatique.
- Visualisez le résultat en 3D.
""")

# --- PAGE D'ACCUEIL ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Accueil'

st.sidebar.title("Navigation")
pages = ["Accueil", "Exploration", "Optimisation avancée", "Cas particulier : medium"]
page = st.sidebar.radio("Aller à la page :", pages, index=pages.index(st.session_state['page']))
st.session_state['page'] = page

if page == "Accueil":
    st.title("Détection et modélisation automatique de câbles électriques dans des nuages de points LiDAR")
    st.markdown("""
    ## Mission et contexte
    Ce projet vise à détecter automatiquement le nombre de câbles dans des nuages de points LiDAR issus de drones, et à modéliser chaque câble par une courbe caténaire 3D.

    **Pipeline général :**
    1. Clustering directionnel optimisé (séparation des câbles)
    2. Calcul des métriques avancées (bruit, compacité, continuité, etc.)
    3. Modélisation caténaire 3D pour chaque câble détecté
    4. Visualisation 3D interactive
    5. Export des résultats (JSON/CSV)

    **Utilisation :**
    - Page "Exploration" : tester différentes méthodes et paramètres, visualiser les clusters et les courbes caténaires en 3D.
    - Page "Optimisation avancée" : trouver automatiquement la meilleure segmentation selon des métriques avancées, visualiser/exporter le résultat.
    - Page "Cas particulier : medium" : comprendre les limites des métriques classiques sur ce dataset.
    """)
    st.info("Consigne du test : Identifier le nombre de câbles dans chaque nuage de points et générer les modèles caténaires associés. Fournir un code modulaire, reproductible, documenté, et une synthèse claire des résultats.")

# --- Sélection du dataset ---
st.sidebar.header("Données")
parquet_files = [f for f in os.listdir('.') if f.endswith('.parquet')]
file_choice = st.sidebar.selectbox("Choisissez un fichier .parquet", parquet_files)
df = pd.read_parquet(file_choice)
points = df[['x', 'y', 'z']].to_numpy()

# --- Méthodes disponibles ---
method_options = [
    "RANSAC",
    "Clustering direction locale",
    "Clustering spatial (Agglomerative)",
    "Hough 2D (projection)",
    "Graphe (kNN)",
    "kNN spatial (suivi direction)"
]
method = st.sidebar.selectbox("Méthode de détection", method_options)

# --- Paramètres dynamiques (slider + input) ---
def slider_input(label, minval, maxval, default, step, key):
    col1, col2 = st.sidebar.columns([2,1])
    val = col1.slider(label, minval, maxval, default, step, key=key)
    val_input = col2.number_input("", minval, maxval, val, step=step, key=key+"_input")
    return val_input

params = {}
if method == "RANSAC":
    params['residual_threshold'] = slider_input("Seuil RANSAC (tolérance)", 0.001, 1.0, 0.05, 0.001, "resid")
    params['min_samples'] = slider_input("Min points par câble", 5, 1000, 40, 1, "minpts")
    params['max_iter'] = slider_input("Max itérations", 1, 100, 10, 1, "maxit")
elif method == "Clustering direction locale":
    params['k'] = slider_input("k voisins PCA", 3, 50, 10, 1, "kdir")
    params['eps'] = slider_input("DBSCAN eps", 0.01, 1.0, 0.2, 0.01, "epsdir")
    params['min_samples'] = slider_input("Min samples DBSCAN", 2, 50, 5, 1, "mindir")
elif method == "Clustering spatial (Agglomerative)":
    params['n_clusters'] = slider_input("Nombre de clusters", 2, 20, 3, 1, "nclust")
    params['linkage'] = st.sidebar.selectbox("Méthode de linkage", ["ward", "complete", "average", "single"])
elif method == "Hough 2D (projection)":
    params['res'] = slider_input("Résolution grille (m)", 0.005, 0.2, 0.05, 0.005, "reshough")
    params['num_peaks'] = slider_input("Nb max de lignes détectées", 1, 20, 4, 1, "npeaks")
elif method == "Graphe (kNN)":
    params['k'] = slider_input("k voisins", 2, 50, 6, 1, "kgraph")
elif method == "kNN spatial (suivi direction)":
    st.sidebar.markdown("**Tracez un trait sur la projection 2D pour initier le suivi**")
    params['k'] = slider_input("k voisins pour le suivi", 3, 30, 8, 1, "kspat")
    params['max_length'] = slider_input("Longueur max du suivi (points)", 50, 2000, 300, 10, "maxlspat")
    params['tol_angle'] = slider_input("Tolérance d'angle (degrés)", 5, 180, 45, 1, "angspat")

# --- Option : Tracer un trait sur la projection 2D ---
if method == "kNN spatial (suivi direction)":
    st.subheader("Tracez un trait sur la projection 2D (Y vs Z) pour initier le suivi")
    fig2d = px.scatter(df, x='y', y='z', title="Projection 2D (Y vs Z)")
    st.plotly_chart(fig2d, use_container_width=True)
    st.info("Note : la sélection interactive sur le plot n'est pas native dans Streamlit. Saisissez les indices de deux points de départ dans le tableau ci-dessous.")
    df_view = df[['y', 'z']].reset_index()
    idx1 = st.number_input("Indice du 1er point du trait", 0, len(df)-1, 0, 1)
    idx2 = st.number_input("Indice du 2ème point du trait", 0, len(df)-1, 1, 1)
    start_idxs = [int(idx1), int(idx2)]
else:
    start_idxs = []

# --- Méthodes ---
def run_ransac(points, residual_threshold, min_samples, max_iter):
    labels = np.full(len(points), -1)
    remaining = points.copy()
    orig_idx = np.arange(len(points))
    cable_id = 0
    for _ in range(max_iter):
        if len(remaining) < min_samples:
            break
        X = remaining[:,1].reshape(-1,1)
        y = remaining[:,2]
        model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=residual_threshold, min_samples=min_samples, random_state=42))
        model.fit(X, y)
        inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
        if inlier_mask.sum() < min_samples:
            break
        labels[orig_idx[inlier_mask]] = cable_id
        remaining = remaining[~inlier_mask]
        orig_idx = orig_idx[~inlier_mask]
        cable_id += 1
    return labels

def run_direction_clustering(points, k, eps, min_samples):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    directions = []
    knn = NearestNeighbors(n_neighbors=k+1).fit(points)
    dists, idxs = knn.kneighbors(points)
    for i in range(len(points)):
        neighbors = points[idxs[i, 1:]]
        pca = PCA(n_components=1)
        pca.fit(neighbors)
        directions.append(pca.components_[0])
    directions = np.array(directions)
    features = np.hstack([points, directions])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    return clustering.labels_

def run_agglomerative(points, n_clusters, linkage):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = clustering.fit_predict(points)
    return labels

def run_hough2d(points, res, num_peaks):
    proj = points[:, [1,2]]
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
            num = abs((z1-z0)*y - (y1-y0)*z + z1*y0 - y1*z0)
            den = np.sqrt((z1-z0)**2 + (y1-y0)**2)
            if den > 0 and num/den < res*2:
                labels[j] = i
    return labels

def run_graph_knn(points, k):
    from sklearn.neighbors import kneighbors_graph
    A = kneighbors_graph(points, k, mode='connectivity', include_self=False)
    # Utiliser getattr pour obtenir la bonne fonction selon la version de networkx
    nx_sparse = getattr(nx, 'from_scipy_sparse_matrix', None)
    if nx_sparse is None:
        nx_sparse = getattr(nx, 'from_scipy_sparse_array', None)
    if nx_sparse is None:
        raise ImportError("Votre version de networkx ne supporte ni from_scipy_sparse_matrix ni from_scipy_sparse_array.")
    G = nx_sparse(A)
    components = list(nx.connected_components(G))
    labels = np.full(len(points), -1)
    for i, comp in enumerate(components):
        for idx in comp:
            labels[idx] = i
    return labels

def run_knn_spatial(points, idx1, idx2, k=8, max_length=300, tol_angle=45):
    followed = set([idx1, idx2])
    current = set([idx1, idx2])
    dir_init = points[idx2] - points[idx1]
    dir_init = dir_init / (np.linalg.norm(dir_init) + 1e-8)
    directions = [dir_init, -dir_init]
    for _ in range(max_length):
        new_current = set()
        for idx, dir_prev in zip(current, directions):
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
            dists, idxs = nbrs.kneighbors([points[idx]])
            for nidx in idxs[0][1:]:
                if nidx in followed:
                    continue
                vec = points[nidx] - points[idx]
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                angle = np.degrees(np.arccos(np.clip(np.dot(vec, dir_prev), -1, 1)))
                if angle < tol_angle:
                    new_current.add(nidx)
        if not new_current:
            break
        followed.update(new_current)
        current = new_current
    labels = np.zeros(len(points), dtype=int) - 1
    for idx in followed:
        labels[idx] = 0
    return labels

# --- Exploration automatique ---
if st.button("Exploration automatique"):
    st.info("Exploration automatique en cours (exemple : RANSAC sur plusieurs seuils)...")
    best_n = 0
    best_labels = None
    for resid in np.linspace(0.01, 0.2, 10):
        labels = run_ransac(points, residual_threshold=resid, min_samples=40, max_iter=10)
        n = len(np.unique(labels[labels != -1]))
        if n > best_n:
            best_n = n
            best_labels = labels
    st.success(f"Meilleure séparation trouvée : {best_n} câbles")
    df['label'] = best_labels
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=f"Exploration auto - {best_n} câbles")
    st.plotly_chart(fig, use_container_width=True)

# --- Inputs pour l'intervalle cible du nombre de câbles ---
st.sidebar.header("Critères de qualité")
target_min = st.sidebar.number_input("Nb câbles min (cible)", 1, 100, 3, 1)
target_max = st.sidebar.number_input("Nb câbles max (cible)", 1, 100, 6, 1)

# --- Fonction de calcul des métriques avancées ---
def compute_segmentation_metrics(points, labels):
    metrics = {}
    n_total = len(points)
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    # 1. Taille des clusters
    sizes = [np.sum(labels == l) for l in unique_labels]
    metrics['taille_moy'] = np.mean(sizes) if sizes else 0
    metrics['taille_ecart_type'] = np.std(sizes) if sizes else 0
    metrics['taille_cv'] = (metrics['taille_ecart_type'] / metrics['taille_moy']) if metrics['taille_moy'] > 0 else 0
    # 2. Pourcentage de bruit
    n_noise = np.sum(labels == -1)
    metrics['pct_bruit'] = n_noise / n_total if n_total > 0 else 0
    # 3. Connectivité (longueur max/min par cluster)
    lengths = []
    for l in unique_labels:
        pts = points[labels == l]
        if len(pts) < 2:
            lengths.append(0)
            continue
        dists = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
        lengths.append(dists.max() - dists.min())
    metrics['longueur_max'] = np.max(lengths) if lengths else 0
    metrics['longueur_min'] = np.min([l for l in lengths if l > 0]) if any(l > 0 for l in lengths) else 0
    metrics['connectivite_ratio'] = (metrics['longueur_min'] / metrics['longueur_max']) if metrics['longueur_max'] > 0 else 0
    # 4. Inertie/compacité interne
    compacts = []
    for l in unique_labels:
        pts = points[labels == l]
        if len(pts) < 2:
            compacts.append(0)
            continue
        dists = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
        compacts.append(np.mean(dists))
    metrics['compacite_moy'] = np.mean(compacts) if compacts else 0
    return metrics

def compute_advanced_metrics(points, labels):
    metrics = {}
    n_total = len(points)
    unique_labels = np.unique(labels[labels != -1])
    n_clusters = len(unique_labels)
    metrics['n_clusters'] = n_clusters
    # 1. Taille des clusters
    sizes = [np.sum(labels == l) for l in unique_labels]
    metrics['taille_moy'] = np.mean(sizes) if sizes else 0
    metrics['taille_ecart_type'] = np.std(sizes) if sizes else 0
    metrics['cv_taille'] = (metrics['taille_ecart_type'] / metrics['taille_moy']) if metrics['taille_moy'] > 0 else 0
    # 2. Pourcentage de bruit
    n_noise = np.sum(labels == -1)
    metrics['pct_bruit'] = n_noise / n_total if n_total > 0 else 0
    # 3. Continuité (longueur axiale/longueur totale)
    ratios = []
    for l in unique_labels:
        pts = points[labels == l]
        if len(pts) < 2:
            ratios.append(0)
            continue
        # Axe principal (PCA 1D)
        pca = PCA(n_components=1)
        proj = pca.fit_transform(pts)
        length_axial = proj.max() - proj.min()
        length_total = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        ratio = length_axial / (length_total + 1e-8)
        ratios.append(ratio)
    metrics['continuité'] = np.mean(ratios) if ratios else 0
    # 4. Linéarité locale (moyenne R² PCA 1D)
    r2s = []
    for l in unique_labels:
        pts = points[labels == l]
        if len(pts) < 3:
            r2s.append(0)
            continue
        pca = PCA(n_components=1)
        proj = pca.fit_transform(pts)
        recon = pca.inverse_transform(proj)
        ss_res = np.sum((pts - recon) ** 2)
        ss_tot = np.sum((pts - pts.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        r2s.append(r2)
    metrics['linearite'] = np.mean(r2s) if r2s else 0
    # 5. Compacité latérale (variance distance au centre de câble)
    compacts = []
    for l in unique_labels:
        pts = points[labels == l]
        if len(pts) < 2:
            compacts.append(0)
            continue
        pca = PCA(n_components=1)
        proj = pca.fit_transform(pts)
        recon = pca.inverse_transform(proj)
        dists = np.linalg.norm(pts - recon, axis=1)
        compacts.append(np.var(dists))
    metrics['compacite_laterale'] = np.mean(compacts) if compacts else 0
    return metrics

# --- Optimisation avancée ---
if st.button("Optimisation avancée"):
    st.info("Optimisation avancée en cours...")
    best_score = -np.inf
    best_labels = None
    best_params = None
    best_metrics = None
    # Exemple : balayage RANSAC (tu peux ajouter d'autres méthodes/paramètres)
    param_grid = list(itertools.product(
        np.linspace(0.01, 0.2, 8),  # residual_threshold
        [20, 40, 60, 100],           # min_samples
        [5, 10, 20]                  # max_iter
    ))
    for resid, min_samples, max_iter in param_grid:
        labels = run_ransac(points, residual_threshold=resid, min_samples=min_samples, max_iter=max_iter)
        metrics = compute_advanced_metrics(points, labels)
        # Score global (exemple pondéré)
        score = 0
        # n_clusters dans cible
        if target_min <= metrics['n_clusters'] <= target_max:
            score += 2
        # bruit faible
        score += max(0, 1 - metrics['pct_bruit']*10)
        # équilibre tailles
        score += max(0, 1 - metrics['cv_taille']*5)
        # continuité
        score += metrics['continuité']
        # linéarité
        score += metrics['linearite']
        # compacité latérale (plus bas = mieux)
        score += max(0, 1 - metrics['compacite_laterale']*5)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_params = {'residual_threshold': resid, 'min_samples': min_samples, 'max_iter': max_iter}
            best_metrics = metrics
    st.success(f"Meilleure segmentation trouvée (score={best_score:.2f}) avec paramètres : {best_params}")
    df['label'] = best_labels
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=f"Optimisation avancée - {best_metrics['n_clusters']} câbles")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Métriques avancées de la meilleure segmentation")
    st.write(best_metrics)
    # Feedbacks visuels
    if target_min <= best_metrics['n_clusters'] <= target_max:
        st.success(f"✅ Nombre de câbles dans l'intervalle cible [{target_min}, {target_max}]")
    else:
        st.warning(f"❌ Nombre de câbles hors intervalle cible [{target_min}, {target_max}]")
    st.write(f"Pourcentage de bruit : {best_metrics['pct_bruit']*100:.1f}%")
    st.write(f"CV taille : {best_metrics['cv_taille']:.2f}")
    st.write(f"Continuité : {best_metrics['continuité']:.2f}")
    st.write(f"Linéarité locale : {best_metrics['linearite']:.2f}")
    st.write(f"Compacité latérale : {best_metrics['compacite_laterale']:.4f}")

# --- Exécution manuelle ---
if st.button("Lancer la détection"):
    with st.spinner("Calcul en cours..."):
        if method == "RANSAC":
            labels = run_ransac(points, **params)
        elif method == "Clustering direction locale":
            labels = run_direction_clustering(points, **params)
        elif method == "Clustering spatial (Agglomerative)":
            labels = run_agglomerative(points, **params)
        elif method == "Hough 2D (projection)":
            labels = run_hough2d(points, **params)
        elif method == "Graphe (kNN)":
            labels = run_graph_knn(points, **params)
        elif method == "kNN spatial (suivi direction)":
            if len(start_idxs) == 2:
                labels = run_knn_spatial(points, start_idxs[0], start_idxs[1], **params)
            else:
                st.error("Veuillez saisir deux indices pour tracer le trait de départ.")
                labels = np.full(len(points), -1)
        else:
            labels = np.full(len(points), -1)
    n_cables = len(np.unique(labels[labels != -1]))
    st.success(f"Nombre de câbles détectés : {n_cables}")
    df['label'] = labels
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=f"{method} - {n_cables} câbles détectés")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[['x','y','z','label']].head(100))

    # --- Affichage des métriques avancées ---
    metrics = compute_segmentation_metrics(points, labels)
    st.markdown("### Métriques de segmentation avancées")
    # 1. Intervalle cible
    if target_min <= n_cables <= target_max:
        st.success(f"✅ Nombre de câbles dans l'intervalle cible [{target_min}, {target_max}]")
    else:
        st.warning(f"❌ Nombre de câbles hors intervalle cible [{target_min}, {target_max}]")
    # 2. Taille des clusters
    st.write(f"Taille moyenne : {metrics['taille_moy']:.1f}, écart-type : {metrics['taille_ecart_type']:.1f}, CV : {metrics['taille_cv']:.2f}")
    if metrics['taille_cv'] < 0.2:
        st.success("✅ Clusters de taille homogène")
    else:
        st.warning("❌ Clusters de taille hétérogène")
    # 3. Pourcentage de bruit
    st.write(f"Pourcentage de bruit (label -1) : {metrics['pct_bruit']*100:.1f}%")
    if metrics['pct_bruit'] < 0.05:
        st.success("✅ Peu de bruit")
    else:
        st.warning("❌ Beaucoup de bruit")
    # 4. Connectivité
    st.write(f"Longueur max : {metrics['longueur_max']:.2f}, min : {metrics['longueur_min']:.2f}, ratio min/max : {metrics['connectivite_ratio']:.2f}")
    if metrics['connectivite_ratio'] > 0.5:
        st.success("✅ Clusters bien connectés")
    else:
        st.warning("❌ Clusters peu connectés")
    # 5. Compacité
    st.write(f"Compacité moyenne intra-cluster : {metrics['compacite_moy']:.2f}")
    if metrics['compacite_moy'] < 2.0:
        st.success("✅ Clusters compacts")
    else:
        st.warning("❌ Clusters peu compacts")

# --- CSS MODERNE ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- PAGE EXPLORATION (EXISTANTE) ---
if page == "Exploration":
    st.markdown('<h1 class="main-header">Exploration interactive des câbles</h1>', unsafe_allow_html=True)
    # Après calcul des labels, ajouter la modélisation caténaire 3D pour chaque cluster
    if 'label' in df:
        st.markdown("### Visualisation avancée : clusters et courbes caténaires 3D")
        import plotly.graph_objects as go
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        for i, l in enumerate(np.unique(df['label'])):
            if l == -1:
                continue
            pts = df[df['label'] == l][['x','y','z']].to_numpy()
            if len(pts) < 5:
                continue
            # PCA pour trouver l’axe principal et trier les points
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            order = np.argsort(pca.fit_transform(pts).ravel())
            pts_sorted = pts[order]
            # Ligne reliant les points du cluster
            fig.add_trace(go.Scatter3d(
                x=pts_sorted[:,0], y=pts_sorted[:,1], z=pts_sorted[:,2],
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=3, color=colors[i % len(colors)], opacity=0.7),
                name=f'Câble {i} - Ligne cluster',
                showlegend=True
            ))
            # Fit caténaire simple (projection sur plan principal)
            pca2 = PCA(n_components=2)
            pts_2d = pca2.fit_transform(pts)
            x_2d, y_2d = pts_2d[:,0], pts_2d[:,1]
            from scipy.optimize import curve_fit
            def catenary(x, y0, c, x0):
                return y0 + c * (np.cosh((x-x0)/c) - 1)
            try:
                popt, _ = curve_fit(catenary, x_2d, y_2d, maxfev=10000)
                x_fit = np.linspace(x_2d.min(), x_2d.max(), 100)
                y_fit = catenary(x_fit, *popt)
                pts_fit_3d = pca2.inverse_transform(np.column_stack([x_fit, y_fit]))
                fig.add_trace(go.Scatter3d(
                    x=pts_fit_3d[:,0], y=pts_fit_3d[:,1], z=pts_fit_3d[:,2],
                    mode='lines', line=dict(color=colors[i % len(colors)], width=5, dash='dash'),
                    name=f'Câble {i} - Caténaire', showlegend=True
                ))
            except Exception as e:
                st.warning(f"Fit caténaire impossible pour le câble {i} : {e}")
        fig.update_layout(title="Clusters et courbes caténaires 3D", width=900, height=600, legend_title_text='Clusters')
        st.plotly_chart(fig, use_container_width=True)
        # Affichage des métriques avancées sous forme de cartes
        metrics = compute_advanced_metrics(points, df['label'])
        st.markdown("#### Métriques avancées")
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f'<div class="metric-card">Câbles détectés<br><span style="font-size:1.5rem">{metrics["n_clusters"]}</span></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card">Bruit (%)<br><span style="font-size:1.5rem">{metrics["pct_bruit"]*100:.1f}</span></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card">CV taille<br><span style="font-size:1.5rem">{metrics["cv_taille"]:.2f}</span></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-card">Continuité<br><span style="font-size:1.5rem">{metrics["continuité"]:.2f}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">Linéarité : {metrics["linearite"]:.2f} | Compacité latérale : {metrics["compacite_laterale"]:.4f}</div>', unsafe_allow_html=True)

# --- PAGE OPTIMISATION AVANCÉE (EXISTANTE) ---
if page == "Optimisation avancée":
    st.markdown('<h1 class="main-header">Optimisation avancée multi-paramètres</h1>', unsafe_allow_html=True)
    # Après calcul de la meilleure segmentation, ajouter la visualisation avancée comme ci-dessus
    # (copier-coller le bloc de visualisation caténaire de la page Exploration)
    pass

# --- PAGE CAS PARTICULIER : MEDIUM ---
if page == "Cas particulier : medium":
    st.title("Cas particulier : Dataset medium")
    st.markdown("""
    Le dataset medium présente deux groupes de câbles bien séparés dans l'espace. Les métriques classiques (compacité, continuité, etc.) ne sont pas pertinentes ici car elles ne tiennent pas compte de la structure multi-groupe. Pour ce type de cas, il faudrait :
    - Utiliser des métriques de séparation inter-groupe
    - Adapter le pipeline pour détecter et traiter chaque groupe séparément
    - (Optionnel) Utiliser des méthodes de clustering hiérarchique ou de détection de composantes connexes
    """)
    st.info("La segmentation directionnelle détecte bien les deux groupes, mais l'évaluation doit être adaptée.") 