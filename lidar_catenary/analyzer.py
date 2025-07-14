"""
Module principal d'analyse LiDAR pour la détection et modélisation de câbles
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings
import time

from .utils import load_lidar_data, remove_outliers
from .clustering import CableClusterer
from .catenary import CatenaryModel
from .visualization import Visualizer


class LidarAnalyzer:
    """
    Classe principale pour analyser les données LiDAR et détecter les câbles électriques.
    """
    
    def __init__(self, 
                 clustering_eps: float = 0.5,
                 clustering_min_samples: int = 5,
                 use_adaptive_clustering: bool = True,
                 outlier_threshold: float = 3.0):
        """
        Initialise l'analyseur LiDAR.
        
        Args:
            clustering_eps: Distance maximale pour DBSCAN
            clustering_min_samples: Nombre minimum de points pour un cluster
            use_adaptive_clustering: Utiliser le clustering adaptatif
            outlier_threshold: Seuil pour la détection d'aberrants
        """
        self.clusterer = CableClusterer(eps=clustering_eps, min_samples=clustering_min_samples)
        self.catenary_model = CatenaryModel()
        self.visualizer = Visualizer()
        self.use_adaptive_clustering = use_adaptive_clustering
        self.outlier_threshold = outlier_threshold
        
        # Stockage des résultats
        self.raw_data = None
        self.clusters = []
        self.catenary_results = []
        self.analysis_summary = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Charge les données LiDAR depuis un fichier.
        
        Args:
            file_path: Chemin vers le fichier parquet
            
        Returns:
            DataFrame avec les données LiDAR
        """
        print(f"Chargement des données depuis {file_path}...")
        self.raw_data = load_lidar_data(file_path)
        print(f"Données chargées: {len(self.raw_data)} points")
        return self.raw_data
    
    def preprocess_data(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Prétraite les données LiDAR.
        
        Args:
            data: Données à prétraiter (utilise self.raw_data si None)
            
        Returns:
            Points 3D nettoyés
        """
        if data is None:
            data = self.raw_data
            
        if data is None:
            raise ValueError("Aucune donnée chargée. Utilisez load_data() d'abord.")
        
        print("Prétraitement des données...")
        
        # Convertir en array numpy
        points = data[['x', 'y', 'z']].values
        
        # Supprimer les aberrants
        points_clean = remove_outliers(points, threshold=self.outlier_threshold)
        
        print(f"Points après nettoyage: {len(points_clean)} (sur {len(points)})")
        
        return points_clean
    
    def detect_cables(self, points: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Détecte les câbles par clustering.
        
        Args:
            points: Points 3D à analyser
            
        Returns:
            Tuple (labels, clusters)
        """
        print("Détection des câbles par clustering...")
        
        if self.use_adaptive_clustering:
            labels, clusters = self.clusterer.adaptive_clustering(points)
        else:
            labels = self.clusterer.fit_predict(points)
            clusters = self.clusterer.get_clusters(points, labels)
        
        # Valider les clusters
        valid_clusters = self.clusterer.validate_clusters(clusters)
        
        print(f"Câbles détectés: {len(valid_clusters)}")
        
        self.clusters = valid_clusters
        return labels, valid_clusters
    
    def fit_catenary_models(self, clusters: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Ajuste des modèles caténaire pour chaque cluster de câble.
        
        Args:
            clusters: Liste des clusters de câbles
            
        Returns:
            Liste des résultats d'ajustement
        """
        print("Ajustement des modèles caténaire...")
        
        results = []
        
        for i, cluster in enumerate(clusters):
            print(f"  Ajustement du câble {i+1}/{len(clusters)}...")
            
            try:
                # Créer un nouveau modèle pour chaque câble
                model = CatenaryModel()
                
                # Ajuster le modèle 3D
                fit_result = model.fit_3d_catenary(cluster)
                
                # Générer des points de la courbe
                catenary_points = model.get_catenary_points(num_points=100)
                
                # Évaluer la qualité
                quality = model.evaluate_fit_quality()
                
                result = {
                    'cable_id': i + 1,
                    'original_points': cluster,
                    'catenary_points': catenary_points,
                    'model': model,
                    'fit_parameters': fit_result,
                    'fit_quality': quality,
                    'num_points': len(cluster)
                }
                
                results.append(result)
                
                print(f"    R² = {quality.get('r_squared', 0):.3f}, RMSE = {quality.get('rmse', np.inf):.3f}")
                
            except Exception as e:
                print(f"    Erreur lors de l'ajustement du câble {i+1}: {str(e)}")
                continue
        
        self.catenary_results = results
        return results
    
    def analyze_file(self, file_path: str, 
                    visualize: bool = True) -> Dict[str, Any]:
        """
        Analyse complète d'un fichier LiDAR.
        
        Args:
            file_path: Chemin vers le fichier parquet
            visualize: Afficher les visualisations
            
        Returns:
            Résumé de l'analyse
        """
        start_time = time.time()
        
        print(f"=== Analyse du fichier: {file_path} ===")
        
        # 1. Charger les données
        data = self.load_data(file_path)
        
        # 2. Prétraiter
        points = self.preprocess_data(data)
        
        # 3. Détecter les câbles
        labels, clusters = self.detect_cables(points)
        
        # 4. Ajuster les modèles caténaire
        results = self.fit_catenary_models(clusters)
        
        # 5. Créer le résumé
        analysis_time = time.time() - start_time
        
        summary = {
            'file_path': file_path,
            'total_points': len(data),
            'clean_points': len(points),
            'cables_detected': len(clusters),
            'successful_fits': len(results),
            'analysis_time': analysis_time,
            'results': results,
            'clusters': clusters,
            'labels': labels
        }
        
        # Calculer les statistiques de qualité
        if results:
            r_squared_values = [r['fit_quality'].get('r_squared', 0) for r in results]
            rmse_values = [r['fit_quality'].get('rmse', np.inf) for r in results]
            
            summary.update({
                'mean_r_squared': np.mean(r_squared_values),
                'mean_rmse': np.mean(rmse_values),
                'best_r_squared': np.max(r_squared_values),
                'worst_r_squared': np.min(r_squared_values)
            })
        
        self.analysis_summary = summary
        
        # 6. Afficher le résumé
        self._print_summary(summary)
        
        # 7. Visualiser si demandé
        if visualize and results:
            self.visualize_results(results)
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Affiche un résumé de l'analyse."""
        print("\n" + "="*50)
        print("RÉSUMÉ DE L'ANALYSE")
        print("="*50)
        print(f"Fichier: {summary['file_path']}")
        print(f"Points totaux: {summary['total_points']}")
        print(f"Points après nettoyage: {summary['clean_points']}")
        print(f"Câbles détectés: {summary['cables_detected']}")
        print(f"Ajustements réussis: {summary['successful_fits']}")
        print(f"Temps d'analyse: {summary['analysis_time']:.2f} secondes")
        
        if 'mean_r_squared' in summary:
            print(f"R² moyen: {summary['mean_r_squared']:.3f}")
            print(f"RMSE moyen: {summary['mean_rmse']:.3f}")
            print(f"Meilleur R²: {summary['best_r_squared']:.3f}")
            print(f"Pire R²: {summary['worst_r_squared']:.3f}")
        
        print("="*50)
    
    def visualize_results(self, results: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Visualise les résultats d'analyse.
        
        Args:
            results: Résultats à visualiser (utilise self.catenary_results si None)
        """
        if results is None:
            results = self.catenary_results
        
        if not results:
            print("Aucun résultat à visualiser.")
            return
        
        print("Génération des visualisations...")
        
        # Visualiser tous les câbles ensemble
        self.visualizer.plot_multiple_catenaries(
            results, 
            title=f"Analyse LiDAR - {len(results)} câbles détectés"
        )
        
        # Visualiser la qualité d'ajustement
        self.visualizer.plot_fit_quality(results)
        
        # Visualiser chaque câble individuellement
        for i, result in enumerate(results):
            self.visualizer.plot_catenary_fit(
                result['original_points'],
                result['catenary_points'],
                title=f"Câble {i+1} - Ajustement caténaire"
            )
    
    def get_cable_parameters(self) -> List[Dict[str, Any]]:
        """
        Retourne les paramètres de tous les câbles détectés.
        
        Returns:
            Liste des paramètres par câble
        """
        parameters = []
        
        for result in self.catenary_results:
            cable_params = {
                'cable_id': result['cable_id'],
                'num_points': result['num_points'],
                'catenary_parameters': result['fit_parameters'],
                'fit_quality': result['fit_quality']
            }
            parameters.append(cable_params)
        
        return parameters
    
    def export_results(self, output_file: str) -> None:
        """
        Exporte les résultats dans un fichier.
        
        Args:
            output_file: Chemin du fichier de sortie
        """
        import json
        
        # Préparer les données pour l'export (sans références circulaires)
        export_data = {
            'summary': {
                'file_path': self.analysis_summary.get('file_path'),
                'total_points': self.analysis_summary.get('total_points'),
                'clean_points': self.analysis_summary.get('clean_points'),
                'cables_detected': self.analysis_summary.get('cables_detected'),
                'successful_fits': self.analysis_summary.get('successful_fits'),
                'analysis_time': self.analysis_summary.get('analysis_time'),
                'mean_r_squared': self.analysis_summary.get('mean_r_squared'),
                'mean_rmse': self.analysis_summary.get('mean_rmse'),
                'best_r_squared': self.analysis_summary.get('best_r_squared'),
                'worst_r_squared': self.analysis_summary.get('worst_r_squared')
            },
            'cable_parameters': []
        }
        
        # Ajouter les paramètres des câbles
        for result in self.catenary_results:
            cable_data = {
                'cable_id': result['cable_id'],
                'num_points': result['num_points'],
                'fit_quality': result['fit_quality'],
                'catenary_parameters': {
                    'c': result['fit_parameters']['c'],
                    'x0': result['fit_parameters']['x0'],
                    'y0': result['fit_parameters']['y0'],
                    'r_squared': result['fit_parameters']['r_squared'],
                    'rmse': result['fit_parameters']['rmse'],
                    'success': result['fit_parameters']['success']
                }
            }
            export_data['cable_parameters'].append(cable_data)
        
        # Sauvegarder
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Résultats exportés vers {output_file}")
    
    def analyze_multiple_files(self, file_paths: List[str], 
                             visualize: bool = True) -> List[Dict[str, Any]]:
        """
        Analyse plusieurs fichiers LiDAR.
        
        Args:
            file_paths: Liste des chemins de fichiers
            visualize: Afficher les visualisations
            
        Returns:
            Liste des résumés d'analyse
        """
        all_summaries = []
        
        for file_path in file_paths:
            try:
                summary = self.analyze_file(file_path, visualize=visualize)
                all_summaries.append(summary)
            except Exception as e:
                print(f"Erreur lors de l'analyse de {file_path}: {str(e)}")
                continue
        
        return all_summaries 