# Détection de Câbles LiDAR - Pipeline Adaptatif

## 🎯 Description du Projet

Ce projet implémente un pipeline complet pour la détection et lanalyse de câbles caténaires dans des nuages de points LiDAR. Le système propose plusieurs méthodes de clustering adaptatives et des outils de visualisation 3D pour analyser la qualité des détections.

## 🚀 Fonctionnalités Principales

### **Méthodes de Clustering Multiples**
- **RANSAC** : Détection robuste avec gestion automatique des outliers
- **Clustering Directionnel** : Basé sur DBSCAN + PCA pour la géométrie locale
- **Clustering Agglomératif** ⭐ : Méthode recommandée pour les fichiers easy/medium
- **Hough 2D** : Détection de lignes droites par projection
- **Graphe kNN** : Clustering basé sur la connectivité spatiale

### **Métriques de Qualité Avancées**
- **Courbure moyenne** : Mesure de l'angle entre vecteurs successifs
- **Compacité latérale** : Moyenne des distances perpendiculaires aux reconstructions PCA
- **Continuité** : Ratio longueur axiale / longueur totale
- **Linéarité** : R² de la régression PCA
- **Score global** : Combinaison pondérée de toutes les métriques

### **Interpolation Anti-Zigzag**
- Tri par distance cumulée le long de la courbe principale
- Évite les retours en arrière et les sauts
- Courbes lisses et naturelles
- Support pour interpolation linéaire et caténaire

### **Interface Utilisateur Intuitive**
- Application Streamlit avec navigation par pages
- Sélection de fichiers .parquet
- Paramétrage interactif des méthodes
- Visualisation 3D avec Plotly
- Optimisation automatique des paramètres

## 📁 Structure du Projet

```
lidar-test/
├── src/
│   ├── clustering.py      # Méthodes de clustering
│   ├── metrics.py         # Calcul des métriques de qualité
│   ├── interpolation.py   # Interpolation des courbes
│   └── __init__.py
├── streamlit_app/
│   ├── app.py            # Application principale
│   ├── pages/            # Pages de navigation
│   └── utils.py          # Utilitaires
├── data/                 # Fichiers de données .parquet
├── explore_cable_detection.py  # Script d'exploration legacy
├── requirements.txt      # Dépendances Python
└── README.md
```

## 🛠️ Installation et Utilisation

### **Prérequis**
```bash
pip install -r requirements.txt
```

### **Lancement de lApplication**
```bash
# Application moderne (recommandée)
streamlit run streamlit_app/app.py

# Script d'exploration legacy
streamlit run explore_cable_detection.py
```

### **Utilisation Recommandée**
1. **Sélectionnez un fichier .parquet** dans linterface
2. **Choisissez la méthode de clustering** :
   - **Clustering Agglomératif** pour les fichiers easy/medium
   - **RANSAC** pour les cas complexes avec bruit
   - **Autres méthodes** selon la géométrie des câbles
3. **Ajustez les paramètres** selon les métriques affichées
4. **Visualisez les résultats** en 3D avec interpolation

## 📊 Adaptation selon le Type de Données

### **Fichiers Easy** ✅
```
Méthode recommandée : Clustering Agglomératif
- n_clusters =3
- linkage = ward"
- Résultats : Courbes lisses, peu de zigzags
- Nécessite vérification visuelle pour confirmer le nombre de câbles
```

### **Fichiers Medium/Hard/Extrahard** ⚠️
```
Méthodes alternatives :
- RANSAC avec seuil adaptatif
- Clustering directionnel avec eps réduit
- Combinaison de méthodes
- Vérification visuelle obligatoire
```

## 🔧 Améliorations Récentes

### **Métriques Robustes**
- Ajout de la métrique de courbure moyenne
- Amélioration de la compacité latérale (moyenne vs variance)
- Pénalisation des courbures élevées dans le score global

### **Interpolation Optimisée**
- Tri par distance cumulée pour éviter les zigzags
- Support pour différents types dinterpolation
- Gestion robuste des cas limites

### **Robustesse Générale**
- Try/except individuels pour chaque méthode
- Gestion des clusters vides et points isolés
- Messages de debug pour le développement

## ⚠️ Limitations et Vérification

### **Cas Problématiques**
1. **Sets de câbles parallèles** : Risque de mélange entre sets
2. **Câbles très proches** : Fusion possible de clusters
3. **Bruit élevé** : Points parasites perturbent le clustering
4. **Câbles courbés complexes** : Zigzags possibles même avec tri optimisé

### **Vérification Visuelle**
- **Toujours vérifier visuellement** le nombre de câbles détectés
- L'interface Streamlit permet cette vérification en temps réel
- Ajuster les paramètres selon les résultats observés

## 🎯 Recommandations d'Usage

1. **Commencez par le clustering agglomératif** (méthode la plus stable)
2. **Ajustez les paramètres selon les métriques** affichées
3. **Testez plusieurs méthodes** si les résultats ne sont pas satisfaisants
4. **Utilisez l'optimisation automatique** pour trouver les meilleurs paramètres
5. **Visualisez en 3D** pour valider la qualité des détections
6. **Vérifiez visuellement** le nombre de câbles détectés

## 🔮 Améliorations Futures

- Algorithme de suivi directionnel (track following)
- Prétraitement par zones géographiques
- Interpolation par splines cubiques
- Validation croisée des méthodes
- Interface de comparaison de méthodes

## 📝 Licence

Ce projet est développé dans le cadre dun test technique pour la détection de câbles LiDAR.

---

**Ce module constitue une base solide pour la détection de câbles LiDAR, avec la flexibilité nécessaire pour s'adapter à différents cas d'usage tout en maintenant une qualité de résultats élevée.** 