import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
from src import clustering, metrics, interpolation
import plotly.graph_objects as go

st.set_page_config(page_title="Détection de câbles caténaires LiDAR", layout="wide")

# --- Barre latérale de navigation ---
menu = [
    "🏠 Accueil",
    "⚙️ Exploration & Optimisation",
    "📈 Visualisation 3D",
    "📚 Documentation"
]
page = st.sidebar.radio("Navigation", menu)

# Chercher les fichiers .parquet à la racine du projet
data_dir = "."
parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

# --- Page d'accueil ---
if page == "🏠 Accueil":
    st.title("Détection et Clustering de Câbles Caténaires dans un Nuage de Points LiDAR")
    st.markdown("""
    Ce projet a pour objectif de détecter automatiquement les câbles caténaires dans des nuages de points LiDAR 3D.
    
    **Pipeline général :**
    1. Chargement d'un fichier .parquet contenant les points (x, y, z)
    2. Application de différentes méthodes de clustering (RANSAC, DBSCAN, Agglomératif, etc.)
    3. Optimisation automatique des paramètres pour chaque méthode
    4. Calcul de métriques de qualité (bruit, continuité, linéarité, compacité...)
    5. Interpolation de chaque câble détecté pour générer une courbe continue (caténaire ou linéaire)
    6. Visualisation interactive 3D
    
    **Pourquoi ?**
    - Automatiser l'analyse d'infrastructures électriques
    - Gérer le bruit et la complexité des données LiDAR
    - Fournir des outils d'évaluation et d'amélioration des algorithmes
    
    _Utilisez le menu à gauche pour explorer les fonctionnalités._
    """)
    st.info("Projet réalisé pour un test technique Data Science - 2024")

# --- Exploration & Optimisation ---
elif page == "⚙️ Exploration & Optimisation":
    st.title("Exploration & Optimisation des Méthodes de Clustering")
    st.markdown("""
    - Sélectionnez un fichier de données LiDAR (.parquet)
    - Choisissez une méthode de clustering
    - Lancez l'optimisation automatique des paramètres
    - Visualisez les métriques de qualité et l'interpolation des câbles
    """)

    # 1. Sélection du fichier
    file_choice = st.selectbox("Choisissez un fichier .parquet", parquet_files, key="optim_file")
    df = pd.read_parquet(os.path.join(data_dir, file_choice))
    points = df[['x', 'y', 'z']].to_numpy()

    # 2. Choix de la méthode
    method_options = {
        "RANSAC": "RANSAC (robuste, linéaire)",
        "Directional": "DBSCAN directionnel (densité + orientation)",
        "Agglomerative": "Clustering agglomératif",
        "Hough2D": "Hough 2D (projection)",
        "GraphKNN": "Graphe k-NN (composantes)"
    }
    method = st.selectbox("Méthode de clustering", list(method_options.keys()), format_func=lambda x: method_options[x])

    # 3. Optimisation automatique des paramètres
    st.subheader("Optimisation automatique des paramètres")
    target_min = st.number_input("Nombre de câbles min (cible)", 1, 20, 3, 1)
    target_max = st.number_input("Nombre de câbles max (cible)", 1, 20, 6, 1)
    run_optim = st.button("Lancer l'optimisation automatique")

    best_labels = None
    best_metrics = None
    best_params = None
    best_score = -np.inf
    interpolated_cables = None

    if run_optim:
        with st.spinner("Optimisation en cours..."):
            # Grilles de paramètres simples pour chaque méthode
            if method == "RANSAC":
                param_grid = [
                    (resid, min_s, max_it)
                    for resid in np.linspace(0.01, 0.2, 8)
                    for min_s in [20, 40, 60]
                    for max_it in [5, 10, 20]
                ]
                for resid, min_s, max_it in param_grid:
                    labels = clustering.run_ransac(points, residual_threshold=resid, min_samples=min_s, max_iter=max_it)
                    m = metrics.evaluate_segmentation(points, labels, target_min, target_max)
                    score = m['score_global']
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_metrics = m
                        best_params = {'residual_threshold': resid, 'min_samples': min_s, 'max_iter': max_it}
            elif method == "Directional":
                param_grid = [
                    (k, eps, min_s)
                    for k in [5, 10, 15]
                    for eps in np.linspace(0.05, 0.3, 6)
                    for min_s in [3, 5, 8]
                ]
                for k, eps, min_s in param_grid:
                    labels = clustering.run_direction_clustering(points, k=k, eps=eps, min_samples=min_s)
                    m = metrics.evaluate_segmentation(points, labels, target_min, target_max)
                    score = m['score_global']
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_metrics = m
                        best_params = {'k': k, 'eps': eps, 'min_samples': min_s}
            elif method == "Agglomerative":
                for n_clusters in range(target_min, target_max+1):
                    for linkage in ["ward", "complete", "average", "single"]:
                        try:
                            labels = clustering.run_agglomerative(points, n_clusters=n_clusters, linkage=linkage)
                        except Exception:
                            continue
                        m = metrics.evaluate_segmentation(points, labels, target_min, target_max)
                        score = m['score_global']
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_metrics = m
                            best_params = {'n_clusters': n_clusters, 'linkage': linkage}
            elif method == "Hough2D":
                for res in np.linspace(0.01, 0.1, 5):
                    for num_peaks in range(target_min, target_max+1):
                        labels = clustering.run_hough2d(points, res=res, num_peaks=num_peaks)
                        m = metrics.evaluate_segmentation(points, labels, target_min, target_max)
                        score = m['score_global']
                        if score > best_score:
                            best_score = score
                            best_labels = labels
                            best_metrics = m
                            best_params = {'res': res, 'num_peaks': num_peaks}
            elif method == "GraphKNN":
                for k in range(3, 15):
                    labels = clustering.run_graph_knn(points, k=k)
                    m = metrics.evaluate_segmentation(points, labels, target_min, target_max)
                    score = m['score_global']
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_metrics = m
                        best_params = {'k': k}

        if best_labels is not None:
            st.success(f"Meilleure segmentation trouvée (score={best_score:.2f})")
            st.write("**Paramètres optimaux :**", best_params)
            st.write("**Métriques de qualité :**", best_metrics)
            # Interpolation automatique
            st.subheader("Interpolation des câbles détectés")
            interp_method = st.selectbox("Méthode d'interpolation", ["linear", "catenary"], format_func=lambda x: "Linéaire" if x=="linear" else "Caténaire")
            interpolated_cables = interpolation.interpolate_all_cables(points, best_labels, method=interp_method)
            st.write(f"{len(interpolated_cables)} câbles interpolés.")
            # Affichage rapide (2D)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7,5))
            for label, cable in interpolated_cables.items():
                ax.plot(cable[:,1], cable[:,2], label=f"Câble {label}")
            ax.set_xlabel('y')
            ax.set_ylabel('z')
            ax.set_title("Courbes interpolées (projection YZ)")
            ax.legend()
            st.pyplot(fig)
            # Sauvegarde des résultats pour la page 3D
            st.session_state['file_choice'] = file_choice
            st.session_state['points'] = points
            st.session_state['labels'] = best_labels
            st.session_state['interpolated_cables'] = interpolated_cables
        else:
            st.error("Aucune segmentation satisfaisante trouvée.")

# --- Visualisation 3D ---
elif page == "📈 Visualisation 3D":
    st.title("Visualisation 3D des Câbles Détectés et Interpolés")
    st.markdown("""
    - Sélectionnez le fichier de données à visualiser
    - Affichez les points clusterisés, les courbes interpolées et les caténaires simulées
    - Utilisez les cases à cocher pour activer/désactiver chaque élément
    """)
    # Sélection du fichier à visualiser
    file_choice_3d = st.selectbox("Fichier .parquet à visualiser", parquet_files, key="visu_file")
    # Si on a déjà optimisé ce fichier, on récupère les résultats, sinon on recharge les points
    if st.session_state.get('file_choice') == file_choice_3d:
        points = st.session_state.get('points', None)
        labels = st.session_state.get('labels', None)
        interpolated_cables = st.session_state.get('interpolated_cables', None)
    else:
        df = pd.read_parquet(os.path.join(data_dir, file_choice_3d))
        points = df[['x', 'y', 'z']].to_numpy()
        labels = None
        interpolated_cables = None

    show_points = st.checkbox("Afficher les points clusterisés", value=True)
    show_curves = st.checkbox("Afficher les courbes interpolées", value=True)
    show_catenary = st.checkbox("Afficher les caténaires simulées (extrémités)", value=False)

    if points is not None and labels is not None:
        fig = go.Figure()
        if show_points:
            for label in np.unique(labels[labels != -1]):
                mask = labels == label
                fig.add_trace(go.Scatter3d(
                    x=points[mask,0], y=points[mask,1], z=points[mask,2],
                    mode='markers',
                    marker=dict(size=2),
                    name=f"Câble {label}"
                ))
        if show_curves and interpolated_cables is not None:
            for label, cable in interpolated_cables.items():
                fig.add_trace(go.Scatter3d(
                    x=cable[:,0], y=cable[:,1], z=cable[:,2],
                    mode='lines',
                    line=dict(width=4),
                    name=f"Courbe {label}"
                ))
        if show_catenary and interpolated_cables is not None:
            for label, cable in interpolated_cables.items():
                # Caténaire simulée entre les extrémités
                start, end = cable[0], cable[-1]
                cat_curve = interpolation.generate_catenary_curve(start, end, num_points=50, sag=0.1)
                fig.add_trace(go.Scatter3d(
                    x=cat_curve[:,0], y=cat_curve[:,1], z=cat_curve[:,2],
                    mode='lines',
                    line=dict(width=2, dash='dash'),
                    name=f"Caténaire {label}"
                ))
        fig.update_layout(scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(itemsizing='constant'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Aucun résultat à afficher pour ce fichier. Lancez d'abord une optimisation dans l'onglet précédent.")

# --- Documentation ---
elif page == "📚 Documentation":
    st.title("Documentation & Explications")
    st.markdown("""
    ### Algorithmes utilisés
    - **RANSAC** : Détection robuste de structures linéaires
    - **DBSCAN directionnel** : Clustering basé sur la densité et l'orientation
    - **Agglomératif** : Clustering hiérarchique
    - **Hough 2D** : Détection de lignes dans la projection 2D
    
    ### Métriques de qualité
    - Continuité, linéarité, compacité, bruit
    
    ### Limites & perspectives
    - Sensibilité au bruit
    - Paramétrage automatique
    - Extension à d'autres types d'infrastructures
    """) 