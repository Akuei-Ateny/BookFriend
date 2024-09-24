import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, n_factors=50, regularization=0.01, n_iterations=20):
        self.n_factors = n_factors
        self.regularization = regularization
        self.n_iterations = n_iterations
        
    def fit(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix.astype(float)
        self.n_users, self.n_items = self.user_item_matrix.shape
        
        # Center the ratings
        self.user_means = np.nanmean(self.user_item_matrix, axis=1)
        self.centered_matrix = self.user_item_matrix - self.user_means[:, np.newaxis]
        
        # Handle potential issues with SVD
        try:
            U, sigma, Vt = svds(self.centered_matrix, k=min(self.n_factors, min(self.n_users, self.n_items) - 1))
        except ValueError:
            print("SVD failed. Using fewer factors.")
            U, sigma, Vt = svds(self.centered_matrix, k=min(10, min(self.n_users, self.n_items) - 1))
        
        self.user_factors = U
        self.item_factors = Vt.T
        self.sigma = sigma

    def predict(self, user_id, item_id):
        if user_id >= self.n_users or item_id >= self.n_items:
            return np.nan
        prediction = self.user_means[user_id] + np.dot(np.dot(self.user_factors[user_id], np.diag(self.sigma)), self.item_factors[item_id].T)
        return max(0, min(5, prediction))  # Clip prediction between 0 and 5

    def recommend(self, user_id, n_recommendations=10):
        if user_id >= self.n_users:
            return []
        user_predictions = self.user_means[user_id] + np.dot(np.dot(self.user_factors[user_id], np.diag(self.sigma)), self.item_factors.T)
        already_interacted = self.user_item_matrix[user_id].nonzero()[0]
        
        user_predictions[already_interacted] = -np.inf
        top_recommendations = user_predictions.argsort()[-n_recommendations:][::-1]
        return top_recommendations

    def similar_items(self, item_id, n_similar=10):
        if item_id >= self.n_items:
            return []
        item_similarity = cosine_similarity(self.item_factors)
        similar_items = item_similarity[item_id].argsort()[-n_similar-1:][::-1][1:]
        return similar_items

# Example usage
if __name__ == "__main__":
    # Create a sample user-item matrix (users as rows, books as columns)
    user_item_matrix = np.array([
        [4, 3, 0, 5, 0],
        [5, 0, 4, 0, 2],
        [3, 1, 2, 4, 1],
        [0, 0, 0, 2, 0],
        [1, 0, 3, 4, 0],
    ])

    cf = CollaborativeFiltering(n_factors=4)
    cf.fit(user_item_matrix)

    # Get recommendations for user 0
    recommendations = cf.recommend(user_id=0, n_recommendations=3)
    print(f"Recommended books for User 0: {recommendations}")

    # Find similar books to book 0
    similar_items = cf.similar_items(item_id=0, n_similar=3)
    print(f"Books similar to Book 0: {similar_items}")

    # Test prediction
    prediction = cf.predict(user_id=0, item_id=2)
    print(f"Predicted rating for User 0, Book 2: {prediction:.2f}")

    # Test error handling
    invalid_recommendations = cf.recommend(user_id=10, n_recommendations=3)
    print(f"Recommendations for invalid user: {invalid_recommendations}")

    invalid_similar_items = cf.similar_items(item_id=10, n_similar=3)
    print(f"Similar items for invalid item: {invalid_similar_items}")

    invalid_prediction = cf.predict(user_id=10, item_id=10)
    print(f"Prediction for invalid user and item: {invalid_prediction}")