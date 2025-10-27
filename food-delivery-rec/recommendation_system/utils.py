import pandas as pd
import numpy as np
import pickle
import os

def load_data(file_path):
    """Load data from CSV file"""
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Data loaded: {len(data)} records, {data['customer_id'].nunique()} users")
    return data

def save_model(model_components, file_path):
    """Save trained model components to file"""
    print(f"Saving model to {file_path}...")
    with open(file_path, 'wb') as f:
        pickle.dump(model_components, f)
    print("Model saved successfully!")

def load_model(file_path):
    """Load trained model components from file"""
    print(f"Loading model from {file_path}...")
    with open(file_path, 'rb') as f:
        model_components = pickle.load(f)
    print("Model loaded successfully!")
    return model_components

def get_available_customers(data):
    """Get all customer IDs for selection"""
    customer_ids = data['customer_id'].unique()
    print(f"Total customers in dataset: {len(customer_ids)}")
    print("Sample customer IDs:", customer_ids[:10])
    return customer_ids

def check_system_quality(evaluation_results):
    """Check if the recommendation system is good based on metrics"""
    if 10 in evaluation_results:
        results = evaluation_results[10]
    else:
        results = list(evaluation_results.values())[0]

    precision = results['Precision']
    ndcg = results['NDCG']

    print("\n🎯 SYSTEM QUALITY CHECK:")
    print("=" * 40)

    if precision > 0.3:
        print("✅ EXCELLENT! Your Precision@10 is great!")
        print("   Users will find many relevant recommendations")
    elif precision > 0.15:
        print("✅ GOOD! Your system is working well")
        print("   Most recommendations are relevant to users")
    elif precision > 0.05:
        print("⚠️  FAIR - There's room for improvement")
        print("   Some recommendations are relevant, but many aren't")
    else:
        print("❌ NEEDS WORK - The system isn't capturing user preferences well")

    if ndcg > 0.3:
        print("✅ EXCELLENT ranking quality!")
        print("   You're putting the most relevant items first")
    elif ndcg > 0.15:
        print("✅ GOOD ranking - relevant items appear early")
    else:
        print("⚠️  Ranking could be improved")

    return precision, ndcg
