import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationEvaluator:
    def __init__(self, data):
        self.full_data = data
        self.train_data = None
        self.test_data = None

    def split_data_stratified(self, test_ratio=0.2, min_orders=2):
        """Split data with stratification"""
        print("Splitting data for evaluation...")

        train_list, test_list = [], []
        for user_id, user_df in self.full_data.groupby('customer_id'):
            if len(user_df) < min_orders:
                continue
            train, test = train_test_split(user_df, test_size=test_ratio, random_state=42)
            train_list.append(train)
            test_list.append(test)

        self.train_data = pd.concat(train_list)
        self.test_data = pd.concat(test_list)

        print(f"Train: {len(self.train_data)} records, {self.train_data['customer_id'].nunique()} users")
        print(f"Test: {len(self.test_data)} records, {self.test_data['customer_id'].nunique()} users")
        return self.train_data, self.test_data

    def prepare_evaluation_model(self, train_data):
        """Prepare model specifically for evaluation"""
        from .trainer import RecommendationModelTrainer

        trainer = RecommendationModelTrainer(train_data)
        model_components = trainer.train_full_model()
        return model_components

    def precision_at_k(self, recommended, actual, k=10):
        """Precision@K: Percentage of relevant recommendations in top K"""
        if len(recommended) == 0:
            return 0.0
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(actual))
        return hits / len(recommended_k)

    def recall_at_k(self, recommended, actual, k=10):
        """Recall@K: Percentage of actual items found in top K"""
        if len(actual) == 0:
            return 0.0
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(actual))
        return hits / len(actual)

    def ndcg_at_k(self, recommended, actual, k=10):
        """Normalized Discounted Cumulative Gain@K"""
        if len(recommended) == 0:
            return 0.0

        recommended_k = recommended[:k]
        relevance = [1 if item in actual else 0 for item in recommended_k]

        # Calculate DCG
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])

        # Calculate IDCG
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])

        return dcg / idcg if idcg > 0 else 0.0

    def mrr_at_k(self, recommended, actual, k=10):
        """Mean Reciprocal Rank@K"""
        for idx, item in enumerate(recommended[:k]):
            if item in actual:
                return 1.0 / (idx + 1)
        return 0.0

    def map_at_k(self, recommended, actual, k=10):
        """Mean Average Precision@K"""
        if len(actual) == 0:
            return 0.0

        precision_scores = []
        hits = 0
        for i, item in enumerate(recommended[:k]):
            if item in actual:
                hits += 1
                precision_scores.append(hits / (i + 1))

        if not precision_scores:
            return 0.0

        return sum(precision_scores) / min(len(actual), k)

    def evaluate_recommendations(self, k_values=[5, 10, 20]):
        """Comprehensive evaluation at different K values"""
        print("📊 EVALUATING MODEL PERFORMANCE")
        print("=" * 50)

        # Split data
        self.split_data_stratified()

        # Train model on training data
        model_components = self.prepare_evaluation_model(self.train_data)
        model = model_components['model']
        vendor_map = model_components['vendor_map']
        reverse_user_map = model_components['reverse_user_map']
        interaction_matrix = model_components['interaction_matrix']

        results = {}
        all_user_metrics = []

        test_users = self.test_data['customer_id'].unique()
        print(f"Evaluating on {len(test_users)} test users...")

        processed_count = 0
        for user_id in test_users:
            if user_id not in reverse_user_map:
                continue

            # Get actual interactions from test set
            actual = self.test_data[self.test_data['customer_id'] == user_id]['vendor_id'].unique().tolist()

            if not actual:
                continue

            try:
                # Get recommendations
                user_idx = reverse_user_map[user_id]
                user_items = interaction_matrix.T.tocsr()
                recommended = model.recommend(user_idx, user_items[user_idx], N=max(k_values))
                recommended_vendors = [vendor_map[int(i[0])] for i in recommended]

                user_metrics = {'user_id': user_id}
                for k in k_values:
                    user_metrics[f'precision@{k}'] = self.precision_at_k(recommended_vendors, actual, k)
                    user_metrics[f'recall@{k}'] = self.recall_at_k(recommended_vendors, actual, k)
                    user_metrics[f'ndcg@{k}'] = self.ndcg_at_k(recommended_vendors, actual, k)
                    user_metrics[f'mrr@{k}'] = self.mrr_at_k(recommended_vendors, actual, k)
                    user_metrics[f'map@{k}'] = self.map_at_k(recommended_vendors, actual, k)

                all_user_metrics.append(user_metrics)
                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} users...")

            except Exception as e:
                continue

        # Aggregate results
        metrics_df = pd.DataFrame(all_user_metrics)

        for k in k_values:
            results[k] = {
                'Precision': metrics_df[f'precision@{k}'].mean(),
                'Recall': metrics_df[f'recall@{k}'].mean(),
                'NDCG': metrics_df[f'ndcg@{k}'].mean(),
                'MRR': metrics_df[f'mrr@{k}'].mean(),
                'MAP': metrics_df[f'map@{k}'].mean(),
                'Users_Evaluated': len(metrics_df)
            }

        return results, metrics_df

    def print_evaluation_results(self, results):
        """Print formatted evaluation results"""
        print("\n📈 EVALUATION RESULTS")
        print("=" * 70)

        for k, metrics in results.items():
            print(f"\n🎯 Top-{k} Recommendations:")
            print("-" * 40)
            for metric, value in metrics.items():
                if metric == 'Users_Evaluated':
                    print(f"  {metric}: {value}")
                else:
                    print(f"  {metric}: {value:.4f}")

    def plot_evaluation_results(self, results):
        """Plot evaluation metrics"""
        try:
            k_values = list(results.keys())
            metrics = ['Precision', 'Recall', 'NDCG', 'MRR']

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()

            for i, metric in enumerate(metrics):
                values = [results[k][metric] for k in k_values]
                axes[i].bar([str(k) for k in k_values], values, color='skyblue', alpha=0.7)
                axes[i].set_title(f'{metric}@K')
                axes[i].set_xlabel('K')
                axes[i].set_ylabel(metric)

                # Add value labels on bars
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not plot results: {e}")

    def benchmark_against_baselines(self, k=10):
        """Compare against simple baselines"""
        print("\n🔍 BENCHMARKING AGAINST BASELINES")
        print("=" * 50)

        # Popularity baseline
        popular_vendors = self.full_data['vendor_id'].value_counts().head(100).index.tolist()

        # Random baseline
        all_vendors = self.full_data['vendor_id'].unique().tolist()

        baseline_results = {}

        # Evaluate popularity baseline
        pop_scores = []
        for user_id in self.test_data['customer_id'].unique():
            actual = self.test_data[self.test_data['customer_id'] == user_id]['vendor_id'].unique().tolist()
            if actual:
                pop_precision = self.precision_at_k(popular_vendors, actual, k)
                pop_scores.append(pop_precision)

        baseline_results['Popularity'] = np.mean(pop_scores) if pop_scores else 0

        # Evaluate random baseline
        random_scores = []
        for user_id in self.test_data['customer_id'].unique():
            actual = self.test_data[self.test_data['customer_id'] == user_id]['vendor_id'].unique().tolist()
            if actual:
                random_recs = np.random.choice(all_vendors, k, replace=False).tolist()
                random_precision = self.precision_at_k(random_recs, actual, k)
                random_scores.append(random_precision)

        baseline_results['Random'] = np.mean(random_scores) if random_scores else 0

        print("Baseline Performance (Precision@10):")
        for baseline, score in baseline_results.items():
            print(f"  {baseline}: {score:.4f}")

        return baseline_results
