"""
Utilitaires pour le traitement des données LiDAR
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings


def load_lidar_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données LiDAR depuis un fichier parquet.
    
    Args:
        file_path: Chemin vers le fichier parquet
        
    Returns:
        DataFrame avec les colonnes x, y, z
    """
    try:
        df = pd.read_parquet(file_path)
        
        # Vérifier les colonnes attendues
        expected_columns = ['x', 'y', 'z']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"Le fichier doit contenir les colonnes: {expected_columns}")
            
        return df[expected_columns]
    
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier {file_path}: {str(e)}")


def normalize_coordinates(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise les coordonnées en centrant les données.
    
    Args:
        points: Array de points (N, 3)
        
    Returns:
        Tuple (points_normalized, center, scale)
    """
    center = np.mean(points, axis=0)
    points_centered = points - center
    
    # Calculer l'échelle pour normaliser
    scale = np.max(np.abs(points_centered))
    if scale > 0:
        points_normalized = points_centered / scale
    else:
        points_normalized = points_centered
        
    return points_normalized, center, scale


def denormalize_coordinates(points_normalized: np.ndarray, 
                          center: np.ndarray, 
                          scale: float) -> np.ndarray:
    """
    Inverse la normalisation des coordonnées.
    
    Args:
        points_normalized: Points normalisés
        center: Centre original
        scale: Échelle utilisée pour la normalisation
        
    Returns:
        Points dans les coordonnées originales
    """
    return points_normalized * scale + center


def remove_outliers(points: np.ndarray, 
                   threshold: float = 3.0) -> np.ndarray:
    """
    Supprime les points aberrants basés sur la distance statistique.
    
    Args:
        points: Array de points (N, 3)
        threshold: Seuil pour la détection d'aberrants (écarts-types)
        
    Returns:
        Points filtrés
    """
    if len(points) < 4:
        return points
        
    # Calculer les distances euclidiennes au centre
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    
    # Calculer les statistiques robustes
    median_dist = np.median(distances)
    mad = np.median(np.abs(distances - median_dist))
    
    if mad > 0:
        # Score de robustesse
        robust_score = np.abs(distances - median_dist) / mad
        mask = robust_score < threshold
        return points[mask]
    
    return points


def calculate_plane_normal(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calcule le vecteur normal du plan de meilleur ajustement.
    
    Args:
        points: Array de points (N, 3)
        
    Returns:
        Tuple (normal_vector, d_parameter)
    """
    if len(points) < 3:
        raise ValueError("Au moins 3 points requis pour calculer un plan")
    
    # Centrer les points
    center = np.mean(points, axis=0)
    centered_points = points - center
    
    # Calculer la matrice de covariance
    cov_matrix = np.cov(centered_points.T)
    
    # Trouver le vecteur propre correspondant à la plus petite valeur propre
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, 0]  # Plus petite valeur propre
    
    # Normaliser le vecteur
    normal = normal / np.linalg.norm(normal)
    
    # Calculer le paramètre d du plan ax + by + cz + d = 0
    d = -np.dot(normal, center)
    
    return normal, d


def project_to_plane(points: np.ndarray, 
                    normal: np.ndarray, 
                    d: float) -> np.ndarray:
    """
    Projette les points 3D sur un plan 2D.
    
    Args:
        points: Points 3D (N, 3)
        normal: Vecteur normal du plan
        d: Paramètre d du plan
        
    Returns:
        Points projetés 2D (N, 2)
    """
    # Créer une base orthonormale dans le plan
    # Premier vecteur de base (arbitraire, perpendiculaire à normal)
    if abs(normal[0]) < abs(normal[1]):
        v1 = np.array([1, 0, 0])
    else:
        v1 = np.array([0, 1, 0])
    
    v1 = v1 - np.dot(v1, normal) * normal
    v1 = v1 / np.linalg.norm(v1)
    
    # Deuxième vecteur de base (produit vectoriel)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Matrice de projection
    projection_matrix = np.column_stack([v1, v2])
    
    # Projeter les points
    projected_points = points @ projection_matrix
    
    return projected_points 