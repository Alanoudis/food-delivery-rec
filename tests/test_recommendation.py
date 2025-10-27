# tests/test_recommendation.py

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append('../')

from food_delivery_rec.recommendation_system.trainer import RecommendationModelTrainer
from food_delivery_rec.recommendation_system.evaluator import RecommendationEvaluator

class TestRecommendationSystem:
    def test_precision_at_k(self):
        """Test precision calculation"""
        evaluator = RecommendationEvaluator(None)

        recommended = ['a', 'b', 'c', 'd']
        actual = ['a', 'c', 'e']

        precision = evaluator.precision_at_k(recommended, actual, k=4)
        expected = 0.5  # 2 out of 4 are relevant

        assert precision == expected

    def test_ndcg_at_k(self):
        """Test NDCG calculation"""
        evaluator = RecommendationEvaluator(None)

        recommended = ['a', 'b', 'c']
        actual = ['a', 'c']

        ndcg = evaluator.ndcg_at_k(recommended, actual, k=3)
        assert 0 <= ndcg <= 1
