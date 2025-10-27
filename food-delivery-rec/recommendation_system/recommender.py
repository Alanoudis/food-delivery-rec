import pandas as pd
import numpy as np

class RecommendationSystem:
    def __init__(self, full_data, model_components):
        self.full_data = full_data
        self.model = model_components['model']
        self.interaction_matrix = model_components['interaction_matrix']
        self.vendor_map = model_components['vendor_map']
        self.reverse_user_map = model_components['reverse_user_map']
        self.vendor_similarity = model_components['vendor_similarity']

    def get_customer_order_history(self, customer_id):
        """Get detailed order history for a customer"""
        customer_orders = self.full_data[self.full_data['customer_id'] == customer_id]

        if customer_orders.empty:
            return f"No order history found for customer: {customer_id}"

        # Aggregate order history
        order_summary = customer_orders.groupby('vendor_id').agg({
            'name': 'first',
            'cuisine_origin': 'first',
            'order_frequency': 'sum',
            'product_rating': 'mean',
            'unit_price': 'mean'
        }).reset_index()

        return order_summary

    def get_customer_taste_profile(self, customer_id):
        """Analyze customer taste preferences"""
        customer_orders = self.full_data[self.full_data['customer_id'] == customer_id]

        if customer_orders.empty:
            return "No taste profile available (new customer)"

        taste_profile = {
            'total_orders': len(customer_orders),
            'unique_vendors': customer_orders['vendor_id'].nunique(),
            'preferred_cuisines': customer_orders['cuisine_origin'].value_counts().head(3).to_dict(),
            'avg_rating_given': customer_orders['product_rating'].mean(),
            'avg_spending': customer_orders['unit_price'].mean(),
            'favorite_vendors': customer_orders.groupby('vendor_id')['order_frequency']
                                                .sum().sort_values(ascending=False).head(3).to_dict()
        }

        return taste_profile

    def recommend_vendors(self, customer_id, N=10):
        """Get ALS-based recommendations"""
        if customer_id not in self.reverse_user_map:
            return "Cold-start user. Recommend popular restaurants."

        user_idx = self.reverse_user_map[customer_id]
        user_items = self.interaction_matrix.T.tocsr()

        recommended = self.model.recommend(user_idx, user_items[user_idx], N=N)
        recommended_vendors = [self.vendor_map[int(i[0])] for i in recommended]

        return recommended_vendors

    def hybrid_recommend(self, customer_id, N=10, als_weight=0.5, content_weight=0.5):
        """Get hybrid recommendations combining ALS and content-based"""
        # Get user's ordered vendors from full data
        user_vendors = self.full_data[self.full_data['customer_id'] == customer_id]['vendor_id'].unique()

        if len(user_vendors) == 0:
            return "Cold-start user. Recommend popular restaurants."

        # Content-based similarity scores
        content_scores = self.vendor_similarity[user_vendors].mean(axis=1)
        content_scores = content_scores.drop(user_vendors, errors='ignore')

        # ALS recommendations
        als_recs = self.recommend_vendors(customer_id, N=100)

        if isinstance(als_recs, str):
            return als_recs

        als_scores = pd.Series([1 / (i + 1) for i in range(len(als_recs))], index=als_recs)

        # Combine scores
        hybrid_scores = pd.concat([als_scores, content_scores], axis=1).fillna(0)
        hybrid_scores.columns = ['als', 'content']
        hybrid_scores['hybrid'] = (als_weight * hybrid_scores['als'] +
                                 content_weight * hybrid_scores['content'])

        # Return top N
        top_hybrid = hybrid_scores['hybrid'].sort_values(ascending=False).head(N).index.tolist()
        return top_hybrid

    def get_recommendation_details(self, vendor_ids):
        """Get detailed information about recommended vendors"""
        if isinstance(vendor_ids, str):
            return vendor_ids

        vendor_details = self.full_data[self.full_data['vendor_id'].isin(vendor_ids)]

        summary = vendor_details.groupby('vendor_id').agg({
            'name': 'first',
            'cuisine_origin': 'first',
            'unit_price': 'mean',
            'product_rating': 'mean',
            'order_frequency': 'mean'
        }).reset_index()

        return summary

    def generate_customer_report(self, customer_id, N=10):
        """Generate complete automated report for a customer"""
        print(f"🔍 CUSTOMER ANALYSIS REPORT: {customer_id}")
        print("=" * 60)

        # 1. Order History
        print("\n📊 ORDER HISTORY:")
        print("-" * 30)
        order_history = self.get_customer_order_history(customer_id)
        if isinstance(order_history, pd.DataFrame):
            print(order_history.to_string(index=False))
        else:
            print(order_history)

        # 2. Taste Profile
        print("\n👤 TASTE PROFILE:")
        print("-" * 30)
        taste_profile = self.get_customer_taste_profile(customer_id)
        if isinstance(taste_profile, dict):
            for key, value in taste_profile.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        else:
            print(taste_profile)

        # 3. Recommendations
        print(f"\n🎯 HYBRID RECOMMENDATIONS (Top {N}):")
        print("-" * 40)
        recommendations = self.hybrid_recommend(customer_id, N=N)
        rec_details = self.get_recommendation_details(recommendations)

        if isinstance(rec_details, pd.DataFrame):
            print(rec_details.to_string(index=False))
        else:
            print(rec_details)

        return {
            'customer_id': customer_id,
            'order_history': order_history,
            'taste_profile': taste_profile,
            'recommendations': rec_details
        }

    def batch_recommendations(self, customer_ids, N=10):
        """Generate recommendations for multiple customers"""
        reports = {}
        for customer_id in customer_ids:
            try:
                print(f"\n{'='*80}")
                report = self.generate_customer_report(customer_id, N=N)
                reports[customer_id] = report
                print(f"✅ Report completed for {customer_id}")
            except Exception as e:
                print(f"❌ Error processing {customer_id}: {str(e)}")
        return reports
