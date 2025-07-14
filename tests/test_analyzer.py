"""
Tests basiques pour le package lidar_catenary
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from lidar_catenary import LidarAnalyzer, CableClusterer, CatenaryModel


def test_data_loading():
    """Test du chargement des données."""
    print("Test: Chargement des données...")
    
    # Créer des données de test
    test_data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'z': np.random.randn(100)
    })
    
    # Sauvegarder temporairement
    test_file = "test_data.parquet"
    test_data.to_parquet(test_file)
    
    try:
        analyzer = LidarAnalyzer()
        data = analyzer.load_data(test_file)
        
        assert len(data) == 100
        assert all(col in data.columns for col in ['x', 'y', 'z'])
        print("✅ Chargement des données réussi")
        
    finally:
        # Nettoyer
        if os.path.exists(test_file):
            os.remove(test_file)


def test_clustering():
    """Test du clustering."""
    print("Test: Clustering...")
    
    # Créer des données de test avec clusters
    np.random.seed(42)
    
    # Cluster 1
    cluster1 = np.random.multivariate_normal([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 0.1]], 50)
    
    # Cluster 2
    cluster2 = np.random.multivariate_normal([5, 5, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 0.1]], 50)
    
    # Points de bruit
    noise = np.random.randn(20, 3) * 10
    
    # Combiner
    points = np.vstack([cluster1, cluster2, noise])
    
    # Tester le clustering
    clusterer = CableClusterer(eps=1.0, min_samples=5)
    labels, clusters = clusterer.adaptive_clustering(points)
    
    # Vérifier qu'on a au moins 2 clusters
    valid_clusters = [c for c in clusters if len(c) >= 5]
    assert len(valid_clusters) >= 2, f"Attendu au moins 2 clusters, obtenu {len(valid_clusters)}"
    
    print(f"✅ Clustering réussi: {len(valid_clusters)} clusters détectés")


def test_catenary_fit():
    """Test de l'ajustement caténaire."""
    print("Test: Ajustement caténaire...")
    
    # Créer des points suivant une courbe caténaire
    x = np.linspace(-5, 5, 50)
    c, x0, y0 = 2.0, 0.0, 0.0
    y = y0 + c * (np.cosh((x - x0) / c) - 1)
    z = np.zeros_like(x)
    
    # Ajouter du bruit
    noise = np.random.normal(0, 0.1, len(x))
    y += noise
    
    # Créer les points 3D
    points_3d = np.column_stack([x, y, z])
    
    # Tester l'ajustement
    model = CatenaryModel()
    result = model.fit_3d_catenary(points_3d)
    
    # Vérifier la qualité de l'ajustement
    assert result['success'], "L'ajustement a échoué"
    assert result['r_squared'] > 0.8, f"R² trop faible: {result['r_squared']}"
    
    print(f"✅ Ajustement caténaire réussi: R² = {result['r_squared']:.3f}")


def test_full_pipeline():
    """Test du pipeline complet."""
    print("Test: Pipeline complet...")
    
    # Créer des données de test avec plusieurs câbles
    np.random.seed(42)
    
    # Câble 1
    x1 = np.linspace(-3, 3, 30)
    y1 = 2.0 * (np.cosh(x1 / 2.0) - 1) + np.random.normal(0, 0.1, len(x1))
    z1 = np.zeros_like(x1)
    cable1 = np.column_stack([x1, y1, z1])
    
    # Câble 2 (décalé)
    x2 = np.linspace(-3, 3, 30)
    y2 = 1.5 * (np.cosh((x2 - 1) / 1.5) - 1) + np.random.normal(0, 0.1, len(x2))
    z2 = np.ones_like(x2) * 2
    cable2 = np.column_stack([x2, y2, z2])
    
    # Bruit
    noise = np.random.randn(20, 3) * 5
    
    # Combiner
    all_points = np.vstack([cable1, cable2, noise])
    
    # Créer un DataFrame
    df = pd.DataFrame(all_points, columns=['x', 'y', 'z'])
    
    # Sauvegarder
    test_file = "test_pipeline.parquet"
    df.to_parquet(test_file)
    
    try:
        # Tester le pipeline complet
        analyzer = LidarAnalyzer(
            clustering_eps=1.0,
            clustering_min_samples=10,
            use_adaptive_clustering=True
        )
        
        summary = analyzer.analyze_file(test_file, visualize=False)
        
        # Vérifications
        assert summary['cables_detected'] >= 1, "Aucun câble détecté"
        assert summary['successful_fits'] >= 1, "Aucun ajustement réussi"
        
        print(f"✅ Pipeline complet réussi:")
        print(f"   - Câbles détectés: {summary['cables_detected']}")
        print(f"   - Ajustements réussis: {summary['successful_fits']}")
        print(f"   - R² moyen: {summary.get('mean_r_squared', 0):.3f}")
        
    finally:
        # Nettoyer
        if os.path.exists(test_file):
            os.remove(test_file)


def run_all_tests():
    """Exécute tous les tests."""
    print("🧪 Démarrage des tests...\n")
    
    tests = [
        test_data_loading,
        test_clustering,
        test_catenary_fit,
        test_full_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test échoué: {test.__name__} - {str(e)}")
        print()
    
    print(f"📊 Résultats: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés!")
    else:
        print("⚠️  Certains tests ont échoué")


if __name__ == "__main__":
    run_all_tests() 