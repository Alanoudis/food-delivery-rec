#!/usr/bin/env python3
"""
Main script to run the recommendation system from ml_logic folder
"""

import sys
import os

# Add the parent directory to Python path to import from ml_logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from food-delivery-rec.recommendation_system import (
    RecommendationModelTrainer,
    RecommendationEvaluator,
    RecommendationSystem,
    get_available_customers,
    check_system_quality
)

def main():
    """Main function to run the complete recommendation system"""
    print("🚀 FOOD DELIVERY RECOMMENDATION SYSTEM")
    print("=" * 50)

    # Load your data (replace with your actual data path)
    try:
        # You need to define full_data in your environment
        if 'full_data' not in globals():
            print("❌ Please load your data into 'full_data' variable first")
            print("   Example: full_data = pd.read_csv('your_data.csv')")
            return

        print(f"Data loaded: {len(full_data)} records")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Step 1: Train the model
    print("\n1. 🔧 TRAINING MODEL...")
    trainer = RecommendationModelTrainer(full_data)
    model_components = trainer.train_full_model()

    # Step 2: Evaluate the model
    print("\n2. 📊 EVALUATING MODEL...")
    evaluator = RecommendationEvaluator(full_data)
    evaluation_results, detailed_metrics = evaluator.evaluate_recommendations(k_values=[5, 10])
    evaluator.print_evaluation_results(evaluation_results)

    # Check system quality
    precision, ndcg = check_system_quality(evaluation_results)

    # Step 3: Run recommendation system
    print("\n3. 🎯 GENERATING RECOMMENDATIONS...")
    recommender = RecommendationSystem(full_data, model_components)

    # Get available customers
    customer_ids = get_available_customers(full_data)

    # Generate recommendations for sample customers
    sample_customers = customer_ids[:3]  # First 3 customers
    print(f"\nGenerating recommendations for {len(sample_customers)} sample customers...")

    reports = recommender.batch_recommendations(sample_customers, N=10)

    print("\n🎉 RECOMMENDATION SYSTEM COMPLETED!")
    print(f"✅ Model trained and evaluated")
    print(f"✅ Recommendations generated for {len(reports)} customers")
    print(f"✅ System quality: Precision@{10}: {precision:.3f}, NDCG@{10}: {ndcg:.3f}")

    return trainer, evaluator, recommender, reports

if __name__ == "__main__":
    main()
