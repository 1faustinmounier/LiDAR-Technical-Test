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
    "📊 Présentation Finale"
]
page = st.sidebar.radio("Navigation", menu)

data_dir = "data"
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

    # 3. Paramètres manuels (affichés en permanence)
    st.subheader("Paramètres de la méthode")
    if method == "RANSAC":
        resid = st.slider("Seuil RANSAC", 0.001, 0.2, 0.05, 0.001)
        min_s = st.slider("Min samples", 10, 100, 40, 5)
        max_it = st.slider("Max itérations", 1, 50, 10, 1)
        manual_params = {'residual_threshold': resid, 'min_samples': min_s, 'max_iter': max_it}
    elif method == "Directional":
        k = st.slider("k voisins", 3, 20, 10, 1)
        eps = st.slider("eps DBSCAN", 0.01, 0.5, 0.2, 0.01)
        min_s = st.slider("Min samples", 2, 20, 5, 1)
        manual_params = {'k': k, 'eps': eps, 'min_samples': min_s}
    elif method == "Agglomerative":
        n_clusters = st.slider("Nombre de clusters", 2, 10, 3, 1)
        linkage = st.selectbox("Méthode de linkage", ["ward", "complete", "average", "single"])
        manual_params = {'n_clusters': n_clusters, 'linkage': linkage}
    elif method == "Hough2D":
        res = st.slider("Résolution grille", 0.005, 0.1, 0.05, 0.005)
        num_peaks = st.slider("Nombre de pics", 1, 10, 4, 1)
        manual_params = {'res': res, 'num_peaks': num_peaks}
    elif method == "GraphKNN":
        k = st.slider("k voisins", 2, 20, 6, 1)
        manual_params = {'k': k}

    # 4. Exécution avec paramètres manuels (affiché en permanence)
    st.subheader("Résultats avec paramètres manuels")
    try:
        if method == "RANSAC":
            labels = clustering.run_ransac(points, **manual_params)
        elif method == "Directional":
            labels = clustering.run_direction_clustering(points, **manual_params)
        elif method == "Agglomerative":
            labels = clustering.run_agglomerative(points, **manual_params)
        elif method == "Hough2D":
            labels = clustering.run_hough2d(points, **manual_params)
        elif method == "GraphKNN":
            labels = clustering.run_graph_knn(points, **manual_params)
        
        # Calcul des métriques
        metrics_result = metrics.evaluate_segmentation(points, labels, 3, 6)
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Métriques de qualité :**")
            st.write(f"- Nombre de clusters : {metrics_result['n_clusters']}")
            st.write(f"- Score global : {metrics_result['score_global']:.2f}")
            st.write(f"- Pourcentage de bruit : {metrics_result['pct_bruit']:.1%}")
            st.write(f"- Courbure moyenne : {metrics_result['courbure']:.3f}")
        
        with col2:
            st.write("**Paramètres utilisés :**")
            for key, value in manual_params.items():
                st.write(f"- {key} : {value}")
        
        # Interpolation automatique
        st.subheader("Interpolation des câbles détectés")
        interp_method = st.selectbox("Méthode d'interpolation", ["linear", "catenary"], format_func=lambda x: "Linéaire" if x=="linear" else "Caténaire")
        interpolated_cables = interpolation.interpolate_all_cables(points, labels, method=interp_method)
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
        st.session_state['labels'] = labels
        st.session_state['interpolated_cables'] = interpolated_cables
        
    except Exception as e:
        st.error(f"Erreur lors de l'exécution : {str(e)}")

    # 5. Optimisation automatique (optionnelle)
    st.subheader("Optimisation automatique des paramètres")
    target_min = st.number_input("Nombre de câbles min (cible)", 1, 20, 3, 1)
    target_max = st.number_input("Nombre de câbles max (cible)", 1, 20, 6, 1)
    run_optim = st.button("Lancer l'optimisation automatique")

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

    # Contrôles d'affichage
    show_clustered = st.checkbox("Afficher les points clusterisés", value=True)
    show_interpolated = st.checkbox("Afficher les courbes interpolées", value=True)
    show_catenaries = st.checkbox("Afficher les caténaires simulées (extrémités)", value=False)

    # Création de la figure 3D
    fig = go.Figure()

    # Points clusterisés
    if show_clustered and labels is not None:
        unique_labels = np.unique(labels[labels != -1])
        for label in unique_labels:
            mask = labels == label
            fig.add_trace(go.Scatter3d(
                x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
                mode='markers', name=f'Cluster {label}',
                marker=dict(size=2, opacity=0.7)
            ))

    # Courbes interpolées
    if show_interpolated and interpolated_cables is not None:
        for label, cable in interpolated_cables.items():
            fig.add_trace(go.Scatter3d(
                x=cable[:, 0], y=cable[:, 1], z=cable[:, 2],
                mode='lines', name=f'Interpolé {label}',
                line=dict(width=3, color='red')
            ))

    # Caténaires simulées
    if show_catenaries and interpolated_cables is not None:
        for label, cable in interpolated_cables.items():
            if len(cable) > 1:
                start_point = cable[0]
                end_point = cable[-1]
                catenary = interpolation.generate_catenary_curve(start_point, end_point, num_points=50, sag=0.1)
                fig.add_trace(go.Scatter3d(
                    x=catenary[:, 0], y=catenary[:, 1], z=catenary[:, 2],
                    mode='lines', name=f'Caténaire {label}',
                    line=dict(width=2, color='green', dash='dash')
                ))

    # Configuration de la figure
    fig.update_layout(
        title=f"Visualisation 3D - {file_choice_3d}",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Présentation Finale ---
elif page == "📊 Présentation Finale":
    st.title("Présentation Finale : Pipeline Adaptatif")
    
    st.markdown("""
    ## 🎯 Objectif du Module
    
    Ce module apporte un **outil de recherche et d'adaptation** pour la détection de câbles dans des nuages de points LiDAR, 
    permettant de s'adapter à différents cas d'usage selon la complexité des données.
    """)
    
    st.markdown("""
    ## 📊 Méthodes de Clustering Disponibles
    
    ### 1. **RANSAC** 
    - **Idéal pour** : Câbles bien séparés, peu de bruit
    - **Paramètres clés** : `residual_threshold`, `min_samples`
    - **Avantages** : Robuste aux outliers, détection automatique du nombre de câbles
    
    ### 2. **Clustering Directionnel (DBSCAN + PCA)**
    - **Idéal pour** : Câbles avec orientations similaires
    - **Paramètres clés** : `k` (voisins), `eps`, `min_samples`
    - **Avantages** : Prend en compte la géométrie locale
    
    ### 3. **Clustering Agglomératif** ⭐
    - **Idéal pour** : Fichiers easy et medium - **MÉTHODE RECOMMANDÉE**
    - **Paramètres clés** : `n_clusters`, `linkage` (ward/complete/average)
    - **Avantages** : Contrôle du nombre de câbles, résultats stables
    
    ### 4. **Hough 2D (projection)**
    - **Idéal pour** : Projections 2D de câbles linéaires
    - **Paramètres clés** : `res` (résolution), `num_peaks`
    - **Avantages** : Détection de lignes droites
    
    ### 5. **Graphe (kNN)**
    - **Idéal pour** : Câbles connectés spatialement
    - **Paramètres clés** : `k` (voisins)
    - **Avantages** : Connectivité naturelle
    """)
    
    st.markdown("""
    ## 🎯 Adaptation selon le Type de Fichier
    
    ### **Fichiers Easy** ✅
    ```
    Méthode recommandée : Clustering Agglomératif
    - n_clusters = 3
    - linkage = "ward"
    - Résultats : Courbes lisses, peu de zigzags
    ```
    
    ### **Fichiers Medium/Hard/Extrahard** ⚠️
    ```
    Méthodes alternatives :
    - RANSAC avec seuil adaptatif
    - Clustering directionnel avec eps réduit
    - Combinaison de méthodes
    ```
    """)
    
    st.markdown("""
    ## 📈 Métriques de Qualité
    
    ### **Métriques Calculées :**
    - **Courbure moyenne** : Angle entre vecteurs successifs
    - **Compacité latérale** : Moyenne des distances perpendiculaires aux reconstructions PCA
    - **Continuité** : Longueur axiale / longueur totale
    - **Linéarité** : R² de la régression PCA
    
    ### **Score Global :**
    ```python
    score = 2 * (clusters_dans_cible) + 
            (1 - pct_bruit * 10) + 
            (1 - cv_taille * 5) + 
            continuite + linearite + 
            (1 - compacite_laterale * 5) +
            (1 - courbure)
    ```
    """)
    
    st.markdown("""
    ## 🔧 Améliorations Apportées
    
    ### **Interpolation Anti-Zigzag :**
    - Tri par distance cumulée le long de la courbe PCA
    - Évite les retours en arrière et les sauts
    - Courbes plus naturelles et lisses
    
    ### **Robustesse :**
    - Try/except individuels pour chaque méthode
    - Gestion des cas limites (clusters vides, points isolés)
    - Messages de debug pour le développement
    """)
    
    st.markdown("""
    ## ⚠️ Limitations et Problèmes Identifiés
    
    ### **Cas Problématiques :**
    
    1. **2+ Sets de Câbles Parallèles**
       - Les méthodes peuvent mélanger les câbles de différents sets
       - Solution : Prétraitement par zones géographiques
    
    2. **Câbles Très Proches**
       - Risque de fusion de clusters
       - Solution : Réduction du `eps` ou augmentation du `min_samples`
    
    3. **Bruit Élevé**
       - Points parasites perturbent le clustering
       - Solution : Filtrage préalable ou RANSAC
    
    4. **Câbles Courbés Complexes**
       - Zigzags possibles même avec tri optimisé
       - Solution : Interpolation caténaire ou splines
    """)
    
    st.markdown("""
    ## 🎯 Conclusion
    
    ### **Points Forts :**
    - ✅ Pipeline modulaire et extensible
    - ✅ Adaptation automatique selon la complexité
    - ✅ Métriques de qualité robustes
    - ✅ Interface utilisateur intuitive
    - ✅ Interpolation anti-zigzag efficace
    
    ### **Améliorations Futures :**
    - 🔄 Algorithme de suivi directionnel (track following)
    - 🔄 Prétraitement par zones géographiques
    - 🔄 Interpolation par splines cubiques
    - 🔄 Validation croisée des méthodes
    
    ### **Recommandations d'Usage :**
    1. **Commencez par le clustering agglomératif** (méthode la plus stable)
    2. **Ajustez les paramètres selon les métriques** affichées
    3. **Testez plusieurs méthodes** si les résultats ne sont pas satisfaisants
    4. **Utilisez l'optimisation automatique** pour trouver les meilleurs paramètres
    
    ---
    
    **Ce module constitue une base solide pour la détection de câbles LiDAR, avec la flexibilité nécessaire pour s'adapter à différents cas d'usage tout en maintenant une qualité de résultats élevée.**
    """) 