import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class RecommendationModelTrainer:
    def __init__(self, data):
        self.full_data = data
        self.model = None
        self.interaction_matrix = None
        self.vendor_similarity = None
        self.content_matrix = None
        self.reverse_user_map = None
        self.vendor_map = None

    def prepare_data(self, data=None):
        """Prepare and preprocess the data"""
        if data is None:
            data = self.full_data

        print("Preparing data...")
        df = data[['customer_id', 'vendor_id', 'order_frequency', 'product_rating']].copy()
        df['score'] = df['order_frequency'] * df['product_rating']

        # Encode users and vendors
        df['user_code'] = df['customer_id'].astype('category').cat.codes
        df['vendor_code'] = df['vendor_id'].astype('category').cat.codes

        # Build interaction matrix
        interaction_matrix = coo_matrix(
            (df['score'], (df['user_code'], df['vendor_code']))
        ).T.tocsr()

        # Build lookup tables
        user_map = dict(enumerate(df['customer_id'].astype('category').cat.categories))
        vendor_map = dict(enumerate(df['vendor_id'].astype('category').cat.categories))
        reverse_user_map = {v: k for k, v in user_map.items()}

        print(f"Data prepared: {len(user_map)} users, {len(vendor_map)} vendors")
        return df, interaction_matrix, user_map, vendor_map, reverse_user_map

    def build_content_features(self, data=None):
        """Build vendor content-based features"""
        if data is None:
            data = self.full_data

        print("Building content features...")
        vendor_features = data.groupby('vendor_id').agg({
            'cuisine_origin': 'first',
            'unit_price': 'mean',
            'product_rating': 'mean'
        }).reset_index()

        # One-hot encode cuisine
        cuisine_encoded = pd.get_dummies(vendor_features['cuisine_origin'])

        # Normalize numerical features
        vendor_features['unit_price_norm'] = (
            vendor_features['unit_price'] - vendor_features['unit_price'].min()
        ) / (vendor_features['unit_price'].max() - vendor_features['unit_price'].min())

        vendor_features['product_rating_norm'] = (
            vendor_features['product_rating'] - vendor_features['product_rating'].min()
        ) / (vendor_features['product_rating'].max() - vendor_features['product_rating'].min())

        # Combine all features
        content_matrix = pd.concat([
            cuisine_encoded,
            vendor_features[['unit_price_norm', 'product_rating_norm']]
        ], axis=1)
        content_matrix.index = vendor_features['vendor_id']

        # Compute vendor similarity
        vendor_similarity = pd.DataFrame(
            cosine_similarity(content_matrix),
            index=content_matrix.index,
            columns=content_matrix.index
        )
        print("Content features built successfully")
        return content_matrix, vendor_similarity

    def train_als_model(self, interaction_matrix, factors=50, regularization=0.1, iterations=30):
        """Train the ALS model"""
        print("Training ALS model...")
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        model.fit(interaction_matrix.T)
        print("ALS model trained successfully")
        return model

    def train_full_model(self, factors=50, regularization=0.1, iterations=30):
        """Train the complete model on full data"""
        print("🚀 TRAINING FULL RECOMMENDATION MODEL")
        print("=" * 50)

        # Prepare data
        df, interaction_matrix, user_map, vendor_map, reverse_user_map = self.prepare_data()

        # Build content features
        content_matrix, vendor_similarity = self.build_content_features()

        # Train ALS model
        model = self.train_als_model(interaction_matrix, factors, regularization, iterations)

        # Store components
        self.model = model
        self.interaction_matrix = interaction_matrix
        self.vendor_map = vendor_map
        self.reverse_user_map = reverse_user_map
        self.content_matrix = content_matrix
        self.vendor_similarity = vendor_similarity

        model_components = {
            'model': model,
            'interaction_matrix': interaction_matrix,
            'vendor_map': vendor_map,
            'reverse_user_map': reverse_user_map,
            'content_matrix': content_matrix,
            'vendor_similarity': vendor_similarity,
            'model_params': {
                'factors': factors,
                'regularization': regularization,
                'iterations': iterations
            }
        }

        print("✅ Full model training completed!")
        return model_components
