"""
Module de modélisation caténaire pour les câbles électriques
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Any, Optional
import warnings

from .utils import calculate_plane_normal, project_to_plane


class CatenaryModel:
    """
    Modèle caténaire pour ajuster les courbes de câbles électriques.
    
    L'équation caténaire est: y(x) = y₀ + c × [cosh((x-x₀)/c) - 1]
    """
    
    def __init__(self):
        """Initialise le modèle caténaire."""
        self.parameters = None
        self.plane_normal = None
        self.plane_d = None
        self.projection_matrix = None
        self.fit_quality = None
        
    def catenary_2d(self, x: np.ndarray, c: float, x0: float, y0: float) -> np.ndarray:
        """
        Calcule les valeurs y de la courbe caténaire 2D.
        
        Args:
            x: Coordonnées x
            c: Paramètre de courbure
            x0: Position x du point le plus bas
            y0: Élévation minimale
            
        Returns:
            Coordonnées y de la courbe
        """
        return y0 + c * (np.cosh((x - x0) / c) - 1)
    
    def catenary_residuals(self, params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcule les résidus pour l'ajustement de la courbe caténaire.
        
        Args:
            params: Paramètres [c, x0, y0]
            x: Coordonnées x des points
            y: Coordonnées y des points
            
        Returns:
            Résidus (différences entre points et courbe)
        """
        c, x0, y0 = params
        
        # Éviter les valeurs invalides
        if c <= 0:
            return np.full_like(y, 1e6)
        
        y_pred = self.catenary_2d(x, c, x0, y0)
        return y - y_pred
    
    def fit_2d_catenary(self, points_2d: np.ndarray) -> Dict[str, Any]:
        """
        Ajuste une courbe caténaire 2D aux points donnés.
        
        Args:
            points_2d: Points 2D (N, 2) avec [x, y]
            
        Returns:
            Dictionnaire avec les paramètres ajustés et métriques
        """
        if len(points_2d) < 3:
            raise ValueError("Au moins 3 points requis pour ajuster une caténaire")
        
        x = points_2d[:, 0]
        y = points_2d[:, 1]
        
        # Estimation initiale des paramètres
        x_range = np.max(x) - np.min(x)
        y_range = np.max(y) - np.min(y)
        
        # Paramètres initiaux
        c_init = max(x_range / 4, 1.0)  # Courbure initiale
        x0_init = np.mean(x)  # Centre x
        y0_init = np.min(y)  # Point le plus bas
        
        initial_params = [c_init, x0_init, y0_init]
        
        # Contraintes pour les paramètres
        bounds = [
            (0.1, None),  # c > 0
            (None, None),  # x0 sans contrainte
            (None, None)   # y0 sans contrainte
        ]
        
        try:
            # Ajustement par moindres carrés
            result = least_squares(
                self.catenary_residuals,
                initial_params,
                args=(x, y),
                bounds=([0.1, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                method='trf'
            )
            
            if not result.success:
                warnings.warn("L'ajustement de la caténaire n'a pas convergé")
            
            c_fit, x0_fit, y0_fit = result.x
            
            # Calculer les métriques de qualité
            y_pred = self.catenary_2d(x, c_fit, x0_fit, y0_fit)
            residuals = y - y_pred
            rmse = np.sqrt(np.mean(residuals**2))
            r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
            
            return {
                'c': c_fit,
                'x0': x0_fit,
                'y0': y0_fit,
                'rmse': rmse,
                'r_squared': r_squared,
                'residuals': residuals,
                'success': result.success
            }
            
        except Exception as e:
            warnings.warn(f"Erreur lors de l'ajustement de la caténaire: {str(e)}")
            return {
                'c': c_init,
                'x0': x0_init,
                'y0': y0_init,
                'rmse': np.inf,
                'r_squared': 0.0,
                'residuals': np.full_like(y, np.inf),
                'success': False
            }
    
    def fit_3d_catenary(self, points_3d: np.ndarray) -> Dict[str, Any]:
        """
        Ajuste une courbe caténaire 3D aux points donnés.
        
        Args:
            points_3d: Points 3D (N, 3)
            
        Returns:
            Dictionnaire avec les paramètres et métriques
        """
        if len(points_3d) < 3:
            raise ValueError("Au moins 3 points requis pour ajuster une caténaire 3D")
        
        # Calculer le plan de meilleur ajustement
        self.plane_normal, self.plane_d = calculate_plane_normal(points_3d)
        
        # Projeter les points sur le plan
        points_2d = project_to_plane(points_3d, self.plane_normal, self.plane_d)
        
        # Sauvegarder la matrice de projection pour la reconstruction
        self._create_projection_matrix()
        
        # Ajuster la caténaire 2D
        fit_result = self.fit_2d_catenary(points_2d)
        
        # Ajouter les informations du plan
        fit_result.update({
            'plane_normal': self.plane_normal,
            'plane_d': self.plane_d,
            'projection_matrix': self.projection_matrix
        })
        
        self.parameters = fit_result
        self.fit_quality = fit_result['r_squared']
        
        return fit_result
    
    def _create_projection_matrix(self):
        """Crée la matrice de projection pour la reconstruction 3D."""
        if self.plane_normal is None:
            return
        
        # Créer une base orthonormale dans le plan
        if abs(self.plane_normal[0]) < abs(self.plane_normal[1]):
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])
        
        v1 = v1 - np.dot(v1, self.plane_normal) * self.plane_normal
        v1 = v1 / np.linalg.norm(v1)
        
        v2 = np.cross(self.plane_normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Matrice de projection [v1, v2]^T
        self.projection_matrix = np.row_stack([v1, v2])
    
    def predict_3d(self, x_values: np.ndarray) -> np.ndarray:
        """
        Prédit les points 3D de la courbe caténaire.
        
        Args:
            x_values: Valeurs x pour lesquelles prédire
            
        Returns:
            Points 3D de la courbe
        """
        if self.parameters is None:
            raise ValueError("Le modèle doit être ajusté avant de faire des prédictions")
        
        # Calculer les points 2D de la courbe
        c = self.parameters['c']
        x0 = self.parameters['x0']
        y0 = self.parameters['y0']
        
        y_values = self.catenary_2d(x_values, c, x0, y0)
        points_2d = np.column_stack([x_values, y_values])
        
        # Reconstruire en 3D
        points_3d = self._reconstruct_3d(points_2d)
        
        return points_3d
    
    def _reconstruct_3d(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Reconstruit les points 3D à partir des points 2D projetés.
        
        Args:
            points_2d: Points 2D dans le plan
            
        Returns:
            Points 3D reconstruits
        """
        if self.projection_matrix is None:
            raise ValueError("Matrice de projection non disponible")
        
        # Reconstruire en utilisant la matrice de projection
        # points_3d = points_2d @ projection_matrix.T + offset
        points_3d = points_2d @ self.projection_matrix
        
        # Ajouter l'offset pour respecter l'équation du plan
        if self.plane_normal is not None and self.plane_d is not None:
            # Calculer un point du plan
            plane_point = -self.plane_d * self.plane_normal / np.dot(self.plane_normal, self.plane_normal)
            points_3d += plane_point
        
        return points_3d
    
    def get_catenary_points(self, num_points: int = 100) -> np.ndarray:
        """
        Génère des points équidistants le long de la courbe caténaire.
        
        Args:
            num_points: Nombre de points à générer
            
        Returns:
            Points 3D de la courbe
        """
        if self.parameters is None:
            raise ValueError("Le modèle doit être ajusté avant de générer des points")
        
        # Déterminer la plage x basée sur les données d'origine
        c = self.parameters['c']
        x0 = self.parameters['x0']
        
        # Générer des points x équidistants
        x_range = 4 * c  # Plage typique pour une caténaire
        x_values = np.linspace(x0 - x_range, x0 + x_range, num_points)
        
        return self.predict_3d(x_values)
    
    def evaluate_fit_quality(self) -> Dict[str, float]:
        """
        Évalue la qualité de l'ajustement.
        
        Returns:
            Dictionnaire avec les métriques de qualité
        """
        if self.parameters is None:
            return {}
        
        return {
            'r_squared': self.parameters.get('r_squared', 0.0),
            'rmse': self.parameters.get('rmse', np.inf),
            'success': self.parameters.get('success', False)
        } 