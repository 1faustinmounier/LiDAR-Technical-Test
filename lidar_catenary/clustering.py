"""
Module de clustering pour séparer les câbles dans les données LiDAR
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any
import warnings

from .utils import remove_outliers


class CableClusterer:
    """
    Classe pour effectuer le clustering des points LiDAR par câble.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialise le clusterer.
        
        Args:
            eps: Distance maximale entre points pour former un cluster
            min_samples: Nombre minimum de points pour former un cluster
        """
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.clusterer = None
        
    def fit_predict(self, points: np.ndarray) -> np.ndarray:
        """
        Effectue le clustering des points.
        
        Args:
            points: Array de points (N, 3)
            
        Returns:
            Labels des clusters (-1 pour le bruit)
        """
        if len(points) == 0:
            return np.array([])
            
        # Supprimer les aberrants
        points_clean = remove_outliers(points, threshold=3.0)
        
        if len(points_clean) < self.min_samples:
            # Pas assez de points pour le clustering
            return np.full(len(points), -1)
        
        # Normaliser les données pour DBSCAN
        points_scaled = self.scaler.fit_transform(points_clean)
        
        # Effectuer le clustering
        self.clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        
        labels = self.clusterer.fit_predict(points_scaled)
        
        # Mapper les labels aux points originaux
        original_labels = np.full(len(points), -1)
        original_labels[:len(points_clean)] = labels
        
        return original_labels
    
    def get_clusters(self, points: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        """
        Extrait les points de chaque cluster.
        
        Args:
            points: Points originaux
            labels: Labels des clusters
            
        Returns:
            Liste des points par cluster
        """
        clusters = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Ignorer le bruit
                continue
                
            cluster_points = points[labels == label]
            if len(cluster_points) >= self.min_samples:
                clusters.append(cluster_points)
        
        return clusters
    
    def estimate_optimal_parameters(self, points: np.ndarray) -> Tuple[float, int]:
        """
        Estime les paramètres optimaux pour DBSCAN basés sur les données.
        
        Args:
            points: Points à analyser
            
        Returns:
            Tuple (eps_optimal, min_samples_optimal)
        """
        if len(points) < 10:
            return self.eps, self.min_samples
        
        # Calculer les distances aux k plus proches voisins
        from sklearn.neighbors import NearestNeighbors
        
        k = min(5, len(points) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, _ = nbrs.kneighbors(points)
        
        # Utiliser la distance moyenne au k-ème voisin
        eps_optimal = np.mean(distances[:, -1])
        
        # Ajuster min_samples basé sur la taille des données
        min_samples_optimal = max(3, min(10, len(points) // 20))
        
        return eps_optimal, min_samples_optimal
    
    def adaptive_clustering(self, points: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Effectue un clustering adaptatif avec estimation automatique des paramètres.
        
        Args:
            points: Points à clusteriser
            
        Returns:
            Tuple (labels, clusters)
        """
        # Estimer les paramètres optimaux
        eps_opt, min_samples_opt = self.estimate_optimal_parameters(points)
        
        # Mettre à jour les paramètres
        self.eps = eps_opt
        self.min_samples = min_samples_opt
        
        # Effectuer le clustering
        labels = self.fit_predict(points)
        clusters = self.get_clusters(points, labels)
        
        return labels, clusters
    
    def validate_clusters(self, clusters: List[np.ndarray]) -> List[np.ndarray]:
        """
        Valide et filtre les clusters selon des critères de qualité.
        
        Args:
            clusters: Liste des clusters
            
        Returns:
            Clusters validés
        """
        valid_clusters = []
        
        for cluster in clusters:
            if len(cluster) < self.min_samples:
                continue
                
            # Vérifier la dispersion du cluster
            center = np.mean(cluster, axis=0)
            distances = np.linalg.norm(cluster - center, axis=1)
            
            # Rejeter les clusters trop dispersés
            if np.std(distances) > np.mean(distances) * 2:
                continue
                
            # Vérifier l'aspect ratio (les câbles doivent être allongés)
            if len(cluster) >= 3:
                # Calculer les composantes principales
                centered = cluster - center
                cov_matrix = np.cov(centered.T)
                eigenvalues, _ = np.linalg.eigh(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                # Ratio entre la plus grande et la plus petite valeur propre
                aspect_ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-8)
                
                if aspect_ratio > 2:  # Le cluster doit être allongé
                    valid_clusters.append(cluster)
            else:
                valid_clusters.append(cluster)
        
        return valid_clusters 