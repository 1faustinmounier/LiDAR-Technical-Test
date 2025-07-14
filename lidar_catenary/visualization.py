"""
Module de visualisation pour les résultats d'analyse LiDAR
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
import warnings

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly non disponible. Utilisation de matplotlib uniquement.")


class Visualizer:
    """
    Classe pour visualiser les résultats d'analyse LiDAR.
    """
    
    def __init__(self, use_plotly: bool = True):
        """
        Initialise le visualiseur.
        
        Args:
            use_plotly: Utiliser Plotly pour les visualisations interactives
        """
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE
        
    def plot_3d_scatter(self, points: np.ndarray, 
                       title: str = "Points LiDAR 3D",
                       color: str = 'blue',
                       size: int = 2) -> None:
        """
        Affiche un nuage de points 3D.
        
        Args:
            points: Points 3D (N, 3)
            title: Titre du graphique
            color: Couleur des points
            size: Taille des points
        """
        if self.use_plotly:
            self._plot_3d_scatter_plotly(points, title, color, size)
        else:
            self._plot_3d_scatter_matplotlib(points, title, color, size)
    
    def _plot_3d_scatter_plotly(self, points: np.ndarray, 
                               title: str, color: str, size: int) -> None:
        """Version Plotly de l'affichage 3D."""
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                opacity=0.8
            ),
            name='Points LiDAR'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=800,
            height=600
        )
        
        fig.show()
    
    def _plot_3d_scatter_matplotlib(self, points: np.ndarray, 
                                   title: str, color: str, size: int) -> None:
        """Version Matplotlib de l'affichage 3D."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=color, s=size, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
    
    def plot_clusters(self, points: np.ndarray, 
                     labels: np.ndarray,
                     title: str = "Clusters de câbles") -> None:
        """
        Affiche les clusters de câbles avec des couleurs différentes.
        
        Args:
            points: Points 3D (N, 3)
            labels: Labels des clusters
            title: Titre du graphique
        """
        if self.use_plotly:
            self._plot_clusters_plotly(points, labels, title)
        else:
            self._plot_clusters_matplotlib(points, labels, title)
    
    def _plot_clusters_plotly(self, points: np.ndarray, 
                             labels: np.ndarray, title: str) -> None:
        """Version Plotly de l'affichage des clusters."""
        fig = go.Figure()
        
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set3
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Bruit
                color = 'gray'
                name = 'Bruit'
            else:
                color = colors[i % len(colors)]
                name = f'Câble {label}'
            
            mask = labels == label
            cluster_points = points[mask]
            
            fig.add_trace(go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.8),
                name=name
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=900,
            height=700
        )
        
        fig.show()
    
    def _plot_clusters_matplotlib(self, points: np.ndarray, 
                                 labels: np.ndarray, title: str) -> None:
        """Version Matplotlib de l'affichage des clusters."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Bruit
                color = 'gray'
                name = 'Bruit'
            else:
                color = colors[i]
                name = f'Câble {label}'
            
            mask = labels == label
            cluster_points = points[mask]
            
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                      c=[color], s=3, alpha=0.8, label=name)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_catenary_fit(self, original_points: np.ndarray,
                         catenary_points: np.ndarray,
                         title: str = "Ajustement caténaire") -> None:
        """
        Affiche l'ajustement caténaire avec les points originaux.
        
        Args:
            original_points: Points originaux du câble
            catenary_points: Points de la courbe caténaire ajustée
            title: Titre du graphique
        """
        if self.use_plotly:
            self._plot_catenary_fit_plotly(original_points, catenary_points, title)
        else:
            self._plot_catenary_fit_matplotlib(original_points, catenary_points, title)
    
    def _plot_catenary_fit_plotly(self, original_points: np.ndarray,
                                 catenary_points: np.ndarray, title: str) -> None:
        """Version Plotly de l'affichage de l'ajustement caténaire."""
        fig = go.Figure()
        
        # Points originaux
        fig.add_trace(go.Scatter3d(
            x=original_points[:, 0],
            y=original_points[:, 1],
            z=original_points[:, 2],
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.7),
            name='Points originaux'
        ))
        
        # Courbe caténaire
        fig.add_trace(go.Scatter3d(
            x=catenary_points[:, 0],
            y=catenary_points[:, 1],
            z=catenary_points[:, 2],
            mode='lines',
            line=dict(color='red', width=5),
            name='Courbe caténaire'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=900,
            height=700
        )
        
        fig.show()
    
    def _plot_catenary_fit_matplotlib(self, original_points: np.ndarray,
                                     catenary_points: np.ndarray, title: str) -> None:
        """Version Matplotlib de l'affichage de l'ajustement caténaire."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Points originaux
        ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                  c='blue', s=4, alpha=0.7, label='Points originaux')
        
        # Courbe caténaire
        ax.plot(catenary_points[:, 0], catenary_points[:, 1], catenary_points[:, 2],
                'r-', linewidth=3, label='Courbe caténaire')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_multiple_catenaries(self, results: List[Dict[str, Any]],
                                title: str = "Tous les câbles détectés") -> None:
        """
        Affiche tous les câbles détectés avec leurs courbes caténaire.
        
        Args:
            results: Liste des résultats d'analyse par câble
            title: Titre du graphique
        """
        if self.use_plotly:
            self._plot_multiple_catenaries_plotly(results, title)
        else:
            self._plot_multiple_catenaries_matplotlib(results, title)
    
    def _plot_multiple_catenaries_plotly(self, results: List[Dict[str, Any]], 
                                        title: str) -> None:
        """Version Plotly de l'affichage multiple."""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            
            # Points originaux
            original_points = result['original_points']
            fig.add_trace(go.Scatter3d(
                x=original_points[:, 0],
                y=original_points[:, 1],
                z=original_points[:, 2],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.6),
                name=f'Câble {i+1} - Points',
                showlegend=False
            ))
            
            # Courbe caténaire
            if 'catenary_points' in result:
                catenary_points = result['catenary_points']
                fig.add_trace(go.Scatter3d(
                    x=catenary_points[:, 0],
                    y=catenary_points[:, 1],
                    z=catenary_points[:, 2],
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=f'Câble {i+1} - Caténaire'
                ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=1000,
            height=800
        )
        
        fig.show()
    
    def _plot_multiple_catenaries_matplotlib(self, results: List[Dict[str, Any]], 
                                            title: str) -> None:
        """Version Matplotlib de l'affichage multiple."""
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for i, result in enumerate(results):
            color = colors[i]
            
            # Points originaux
            original_points = result['original_points']
            ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                      c=[color], s=3, alpha=0.6)
            
            # Courbe caténaire
            if 'catenary_points' in result:
                catenary_points = result['catenary_points']
                ax.plot(catenary_points[:, 0], catenary_points[:, 1], catenary_points[:, 2],
                        color=color, linewidth=3, label=f'Câble {i+1}')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_fit_quality(self, results: List[Dict[str, Any]]) -> None:
        """
        Affiche un graphique de la qualité d'ajustement pour chaque câble.
        
        Args:
            results: Liste des résultats d'analyse
        """
        if not results:
            return
        
        cable_ids = [f'Câble {i+1}' for i in range(len(results))]
        r_squared_values = [result.get('fit_quality', {}).get('r_squared', 0) for result in results]
        rmse_values = [result.get('fit_quality', {}).get('rmse', np.inf) for result in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # R²
        ax1.bar(cable_ids, r_squared_values, color='skyblue')
        ax1.set_title('Coefficient de détermination (R²)')
        ax1.set_ylabel('R²')
        ax1.set_ylim(0, 1)
        
        # RMSE
        ax2.bar(cable_ids, rmse_values, color='lightcoral')
        ax2.set_title('Erreur quadratique moyenne (RMSE)')
        ax2.set_ylabel('RMSE')
        
        plt.tight_layout()
        plt.show() 