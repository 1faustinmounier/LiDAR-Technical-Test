# LiDAR Cable Detection and Catenary Modeling

Ce projet implémente un algorithme pour détecter et modéliser les câbles électriques dans des données LiDAR de drones.

## Démo interactive

Vous trouverez ici les résultats : [https://lidar-technical-test.streamlit.app/](https://lidar-technical-test.streamlit.app/)

## Objectif

Analyser des nuages de points LiDAR pour :
1. Identifier le nombre de câbles présents
2. Générer des modèles caténaire 3D pour chaque câble détecté

## Utilisation rapide

1. **Installation des dépendances**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Lancer l'analyse sur tous les fichiers fournis**
   ```bash
   python main.py
   ```
3. **Résultats**
   - Les résultats sont affichés dans le terminal et exportés en JSON (un fichier par jeu de données)
   - Les métriques d'ajustement (R², RMSE) et les paramètres des câbles sont détaillés

## Pipeline de l'algorithme

1. **Clustering des nuages de points**
   - Utilisation de DBSCAN pour séparer les points par câble
   - Prétraitement des données pour éliminer le bruit
2. **Détection du plan de meilleur ajustement**
   - Pour chaque cluster de câble, calcul du plan de régression 3D
   - Projection des points sur ce plan pour obtenir une représentation 2D
3. **Modélisation caténaire**
   - Ajustement de la courbe caténaire : y(x) = y₀ + c × [cosh((x-x₀)/c) - 1]
   - Optimisation des paramètres (c, x₀, y₀) par moindres carrés
   - Extension 3D avec paramètres de rotation et translation
4. **Validation et visualisation**
   - Calcul des métriques de qualité d'ajustement
   - Génération de visualisations 3D des résultats (matplotlib/plotly)

## Structure du projet

```
├── lidar_catenary/
│   ├── __init__.py
│   ├── analyzer.py          # Classe principale d'analyse
│   ├── clustering.py        # Algorithmes de clustering
│   ├── catenary.py          # Modélisation caténaire
│   ├── visualization.py     # Visualisation 3D
│   └── utils.py             # Utilitaires
├── tests/
│   └── test_analyzer.py
├── main.py                  # Script principal (CLI)
├── webapp.py                # Interface web Streamlit
├── requirements.txt
└── README.md
```

## Bonnes pratiques et reproductibilité
- Code modulaire, lisible, orienté objet
- Tests unitaires inclus (`python -m tests.test_analyzer`)
- Résultats reproductibles, pipeline automatisé
- Documentation dans le code et le README

## Dépendances
- pandas
- numpy
- scikit-learn
- matplotlib
- plotly
- pyarrow (pour les fichiers parquet)
- scipy
