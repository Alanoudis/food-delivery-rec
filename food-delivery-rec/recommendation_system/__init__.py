"""
Recommendation System Package for Food Delivery
Automated restaurant recommendation system with ALS and hybrid approaches
"""

__version__ = "1.0.0"
__author__ = "Food Delivery Team"

from .trainer import RecommendationModelTrainer
from .evaluator import RecommendationEvaluator
from .recommender import RecommendationSystem
from .utils import load_data, save_model, load_model, get_available_customers, check_system_quality

__all__ = [
    'RecommendationModelTrainer',
    'RecommendationEvaluator',
    'RecommendationSystem',
    'load_data',
    'save_model',
    'load_model',
    'get_available_customers',
    'check_system_quality'
]
