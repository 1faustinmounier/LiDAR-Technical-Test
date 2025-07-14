"""
LiDAR Cable Detection and Catenary Modeling Package

Ce package fournit des outils pour analyser des données LiDAR de câbles électriques
et générer des modèles caténaire 3D.
"""

from .analyzer import LidarAnalyzer
from .clustering import CableClusterer
from .catenary import CatenaryModel
from .visualization import Visualizer

__version__ = "1.0.0"
__author__ = "Data Science Intern"

__all__ = [
    "LidarAnalyzer",
    "CableClusterer", 
    "CatenaryModel",
    "Visualizer"
] 