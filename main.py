#!/usr/bin/env python3
"""
Script principal pour tester l'analyse LiDAR des câbles électriques
"""

import os
import sys
from pathlib import Path
import csv

# Ajouter le répertoire courant au path pour importer le package
sys.path.insert(0, str(Path(__file__).parent))

from lidar_catenary import LidarAnalyzer


def main():
    """Fonction principale pour tester l'analyse LiDAR."""
    print("=== Analyse LiDAR des câbles électriques ===")
    print("Détection et modélisation caténaire des câbles\n")

    # Créer l'analyseur
    analyzer = LidarAnalyzer(
        clustering_eps=0.5,
        clustering_min_samples=5,
        use_adaptive_clustering=True,
        outlier_threshold=3.0
    )

    # Liste des fichiers à analyser (par ordre de difficulté)
    files = [
        "lidar_cable_points_easy.parquet",
        "lidar_cable_points_medium.parquet",
        "lidar_cable_points_hard.parquet",
        "lidar_cable_points_extrahard.parquet"
    ]

    # Vérifier que les fichiers existent
    existing_files = []
    for file_path in files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"Fichier non trouvé: {file_path}")

    if not existing_files:
        print("Aucun fichier de données trouvé!")
        return

    print(f"Fichiers à analyser: {len(existing_files)}")
    for file_path in existing_files:
        print(f"   - {file_path}")
    print()

    # Analyser chaque fichier
    all_results = []
    summary_rows = []

    for i, file_path in enumerate(existing_files):
        print(f"\n{'='*60}")
        print(f"ANALYSE {i+1}/{len(existing_files)}: {file_path}")
        print(f"{'='*60}")

        try:
            # Analyser le fichier (visualisation désactivée)
            summary = analyzer.analyze_file(
                file_path=file_path,
                visualize=False
            )
            all_results.append(summary)

            # Exporter les résultats détaillés
            output_file = f"results_{Path(file_path).stem}.json"
            analyzer.export_results(output_file)

            # Ajouter au résumé agrégé
            row = {
                "fichier": file_path,
                "points_totaux": summary.get('total_points', 0),
                "points_nettoyes": summary.get('clean_points', 0),
                "cables_detectes": summary.get('cables_detected', 0),
                "ajustements_reussis": summary.get('successful_fits', 0),
                "r2_moyen": summary.get('mean_r_squared', 0),
                "rmse_moyen": summary.get('mean_rmse', 0),
                "meilleur_r2": summary.get('best_r_squared', 0),
                "pire_r2": summary.get('worst_r_squared', 0),
                "temps_analyse_s": summary.get('analysis_time', 0)
            }
            summary_rows.append(row)

        except Exception as e:
            print(f"Erreur lors de l'analyse de {file_path}: {str(e)}")
            continue

    # Résumé agrégé
    print(f"\n{'='*60}")
    print("RÉSUMÉ GLOBAL")
    print(f"{'='*60}")
    if summary_rows:
        print(f"{'Fichier':40} | {'Câbles':6} | {'R² moyen':8} | {'RMSE moyen':10} | {'Meilleur R²':10} | {'Pire R²':8} | {'Temps (s)':8}")
        print("-"*100)
        for row in summary_rows:
            print(f"{row['fichier'][:40]:40} | {row['cables_detectes']:6} | {row['r2_moyen']:.3f}    | {row['rmse_moyen']:.3f}     | {row['meilleur_r2']:.3f}      | {row['pire_r2']:.3f}   | {row['temps_analyse_s']:.2f}")
    else:
        print("Aucun résultat à afficher.")

    # Export CSV du résumé global
    summary_csv = "results_summary.csv"
    with open(summary_csv, "w", newline="") as csvfile:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\nRésumé global exporté dans {summary_csv}")

    print("\nAnalyse terminée. Les résultats détaillés sont exportés en JSON.")

if __name__ == "__main__":
    main() 