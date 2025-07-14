import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from pathlib import Path
import sys

# Ajouter le répertoire courant au path
sys.path.insert(0, str(Path(__file__).parent))

from lidar_catenary import LidarAnalyzer

# Configuration de la page
st.set_page_config(
    page_title="LiDAR Cable Detection - Interactive Demo",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
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

# Données pré-analysées (pour éviter de relancer l'analyse à chaque fois)
@st.cache_data
def load_precomputed_results():
    """Charge les résultats pré-calculés des 4 datasets."""
    datasets = [
        "lidar_cable_points_easy.parquet",
        "lidar_cable_points_medium.parquet", 
        "lidar_cable_points_hard.parquet",
        "lidar_cable_points_extrahard.parquet"
    ]
    
    results = {}
    analyzer = LidarAnalyzer()
    
    for dataset in datasets:
        if Path(dataset).exists():
            try:
                summary = analyzer.analyze_file(dataset, visualize=False)
                results[dataset] = summary
            except Exception as e:
                st.error(f"Erreur lors de l'analyse de {dataset}: {str(e)}")
    
    return results

# Navigation
st.sidebar.title("🔌 LiDAR Cable Detection")
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Accueil", "📊 Comparaison 3D", "📈 Métriques", "🔧 Détails techniques"]
)

# Chargement des données
with st.spinner("Chargement des résultats..."):
    results = load_precomputed_results()

if page == "🏠 Accueil":
    st.markdown('<h1 class="main-header">LiDAR Cable Detection</h1>', unsafe_allow_html=True)
    
    # Introduction avec animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 🎯 **Détection et modélisation automatique de câbles électriques**
        
        
        """)
    
    # Métriques globales avec design moderne
    st.markdown("### 📊 **Résultats globaux**")
    
    if results:
        total_cables = sum(r.get('cables_detected', 0) for r in results.values())
        avg_r2 = np.mean([r.get('mean_r_squared', 0) for r in results.values()])
        total_files = len(results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_files}</h3>
                <p>Datasets analysés</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{total_cables}</h3>
                <p>Câbles détectés</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_r2:.3f}</h3>
                <p>R² moyen</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯</h3>
                <p>Pipeline robuste</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Tableau récapitulatif
    st.markdown("### 📋 **Résumé par dataset**")
    
    if results:
        summary_data = []
        for dataset, result in results.items():
            summary_data.append({
                "Dataset": Path(dataset).stem.replace("lidar_cable_points_", "").title(),
                "Points": result.get('total_points', 0),
                "Câbles": result.get('cables_detected', 0),
                "R² moyen": f"{result.get('mean_r_squared', 0):.3f}",
                "RMSE moyen": f"{result.get('mean_rmse', 0):.3f}",
                "Temps (s)": f"{result.get('analysis_time', 0):.2f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
    
    # Méthode
    st.markdown("### 🔬 **Méthode**")
    
    st.markdown("""
    **Pipeline en 4 étapes :**
    
    1. **🔄 Clustering DBSCAN** : Séparation automatique des câbles
    2. **📐 Plan de projection** : Calcul du plan optimal pour chaque câble  
    3. **📈 Modélisation caténaire** : Ajustement de la courbe y(x) = y₀ + c × [cosh((x-x₀)/c) - 1]
    4. **✅ Validation** : Calcul des métriques de qualité (R², RMSE)
    """)

elif page == "📊 Comparaison 3D":
    st.markdown('<h1 class="main-header">Visualisation 3D Interactive</h1>', unsafe_allow_html=True)
    
    if not results:
        st.error("Aucun résultat disponible. Vérifiez que les fichiers .parquet sont présents.")
        st.stop()
    
    # Sélection du dataset
    dataset_names = list(results.keys())
    selected_dataset = st.selectbox(
        "Choisissez un dataset à visualiser :",
        dataset_names,
        format_func=lambda x: Path(x).stem.replace("lidar_cable_points_", "").title()
    )
    
    if selected_dataset:
        result = results[selected_dataset]
        
        # Animation de chargement
        with st.spinner("Génération de la visualisation 3D..."):
            time.sleep(0.5)  # Simulation d'un calcul
            
            # Création de la figure 3D
            fig = go.Figure()
            
            # Ajout des points originaux
            for i, cable_result in enumerate(result['results']):
                points = cable_result['original_points']
                catenary_points = cable_result['catenary_points']
                
                # Points du câble
                fig.add_trace(go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1], 
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
                        opacity=0.7
                    ),
                    name=f'Câble {i+1} - Points',
                    showlegend=True
                ))
                
                # Courbe caténaire
                fig.add_trace(go.Scatter3d(
                    x=catenary_points[:, 0],
                    y=catenary_points[:, 1],
                    z=catenary_points[:, 2],
                    mode='lines',
                    line=dict(
                        color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)],
                        width=5
                    ),
                    name=f'Câble {i+1} - Caténaire',
                    showlegend=True
                ))
            
            # Mise en forme
            fig.update_layout(
                title=f"Visualisation 3D - {Path(selected_dataset).stem.replace('lidar_cable_points_', '').title()}",
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)', 
                    zaxis_title='Z (m)',
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                ),
                width=800,
                height=600,
                margin=dict(l=0, r=0, b=0, t=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Métriques du dataset sélectionné
        st.markdown("### 📊 **Métriques du dataset**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Câbles détectés", result.get('cables_detected', 0))
        with col2:
            st.metric("R² moyen", f"{result.get('mean_r_squared', 0):.3f}")
        with col3:
            st.metric("RMSE moyen", f"{result.get('mean_rmse', 0):.3f}")

elif page == "📈 Métriques":
    st.markdown('<h1 class="main-header">Analyse Comparative</h1>', unsafe_allow_html=True)
    
    if not results:
        st.error("Aucun résultat disponible pour l'analyse comparative.")
        st.stop()
    
    # Préparation des données pour les graphiques
    datasets = []
    r2_values = []
    rmse_values = []
    cable_counts = []
    
    for dataset, result in results.items():
        dataset_name = Path(dataset).stem.replace("lidar_cable_points_", "").title()
        datasets.append(dataset_name)
        r2_values.append(result.get('mean_r_squared', 0))
        rmse_values.append(result.get('mean_rmse', 0))
        cable_counts.append(result.get('cables_detected', 0))
    
    # Graphiques comparatifs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 **Coefficient de détermination (R²)**")
        fig_r2 = px.bar(
            x=datasets, 
            y=r2_values,
            color=r2_values,
            color_continuous_scale='viridis',
            title="R² par niveau de difficulté"
        )
        fig_r2.update_layout(xaxis_title="Dataset", yaxis_title="R²")
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        st.markdown("### 📏 **Erreur quadratique moyenne (RMSE)**")
        fig_rmse = px.bar(
            x=datasets, 
            y=rmse_values,
            color=rmse_values,
            color_continuous_scale='plasma',
            title="RMSE par niveau de difficulté"
        )
        fig_rmse.update_layout(xaxis_title="Dataset", yaxis_title="RMSE (m)")
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Graphique du nombre de câbles
    st.markdown("### 🔌 **Nombre de câbles détectés**")
    fig_cables = px.bar(
        x=datasets,
        y=cable_counts,
        color=cable_counts,
        color_continuous_scale='inferno',
        title="Nombre de câbles par dataset"
    )
    fig_cables.update_layout(xaxis_title="Dataset", yaxis_title="Nombre de câbles")
    st.plotly_chart(fig_cables, use_container_width=True)
    
    # Analyse des performances
    st.markdown("### 🎯 **Analyse de robustesse**")
    
    if len(r2_values) > 1:
        r2_std = np.std(r2_values)
        rmse_std = np.std(rmse_values)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² moyen global", f"{np.mean(r2_values):.3f}")
        with col2:
            st.metric("Écart-type R²", f"{r2_std:.3f}")
        with col3:
            st.metric("Écart-type RMSE", f"{rmse_std:.3f}")
        
        # Conclusions
        st.markdown("#### 📝 **Conclusions**")
        
        if r2_std < 0.01:
            st.success("✅ **Excellente robustesse** : L'algorithme performe de manière très stable sur tous les niveaux de difficulté.")
        elif r2_std < 0.05:
            st.info("✅ **Bonne robustesse** : L'algorithme maintient des performances acceptables malgré l'augmentation de la difficulté.")
        else:
            st.warning("⚠️ **Robustesse à améliorer** : Les performances varient significativement selon le niveau de difficulté.")

elif page == "🔧 Détails techniques":
    st.markdown('<h1 class="main-header">Détails Techniques</h1>', unsafe_allow_html=True)
    
    st.markdown("### 🔬 **Pipeline d'analyse**")
    
    st.markdown("""
    #### **1. Prétraitement des données**
    - Suppression des points aberrants (méthode MAD - Median Absolute Deviation)
    - Normalisation des coordonnées pour améliorer la stabilité numérique
    
    #### **2. Clustering des câbles**
    - **Algorithme** : DBSCAN (Density-Based Spatial Clustering)
    - **Avantages** : Détection automatique du nombre de clusters, robuste au bruit
    - **Paramètres adaptatifs** : Estimation automatique de `eps` et `min_samples`
    
    #### **3. Modélisation géométrique**
    - **Calcul du plan optimal** : Analyse en composantes principales (PCA)
    - **Projection 2D** : Transformation des points 3D dans le plan du câble
    - **Ajustement caténaire** : Optimisation par moindres carrés non-linéaires
    
    #### **4. Validation et métriques**
    - **R²** : Coefficient de détermination (qualité de l'ajustement)
    - **RMSE** : Erreur quadratique moyenne (précision en mètres)
    """)
    
    st.markdown("### 📐 **Équation caténaire**")
    
    st.latex(r"y(x) = y_0 + c \cdot \left[\cosh\left(\frac{x-x_0}{c}\right) - 1\right]")
    
    st.markdown("""
    **Paramètres :**
    - **c** : Paramètre de courbure (m)
    - **x₀** : Position du point le plus bas (m)  
    - **y₀** : Élévation minimale (m)
    """)
    
    st.markdown("### 🎯 **Choix des métriques**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### **R² (Coefficient de détermination)**
        - **Définition** : Proportion de variance expliquée par le modèle
        - **Interprétation** : 0 = pas d'explication, 1 = explication parfaite
        - **Seuil de qualité** : R² > 0.95 = excellent ajustement
        """)
    
    with col2:
        st.markdown("""
        #### **RMSE (Root Mean Square Error)**
        - **Définition** : Racine carrée de l'erreur quadratique moyenne
        - **Interprétation** : Erreur moyenne en mètres
        - **Seuil de qualité** : RMSE < 1m = très bonne précision
        """)
    
    st.markdown("### 🔗 **Code source**")
    
    st.markdown("""
    Le code source complet est disponible sur GitHub :
    
    **[📁 Repository GitHub](https://github.com/1faustinmounier/LiDAR-Technical-Test)**
    
    **Structure du projet :**
    - `main.py` : Script principal d'analyse
    - `lidar_catenary/` : Package Python modulaire
    - `tests/` : Tests unitaires
    - `requirements.txt` : Dépendances
    """)
    
    st.markdown("### 🚀 **Installation et utilisation**")
    
    st.code("""
# Installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Lancement de l'analyse
python main.py

# Lancement de l'interface web
streamlit run webapp.py
    """, language="bash")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔌 LiDAR Cable Detection - Pipeline de détection et modélisation de câbles électriques</p>
</div>
""", unsafe_allow_html=True) 