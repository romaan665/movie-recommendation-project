# main.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Load the data
def load_data():
    movies = pd.read_csv("C:/Users/Alsheefa/Desktop/romi/project dataset/ml-latest-small/ml-latest-small/movies.csv")
    ratings = pd.read_csv("C:/Users/Alsheefa/Desktop/romi/project dataset/ml-latest-small/ml-latest-small/ratings.csv")
    return movies, ratings

# Step 2: Merge and preprocess the data
def preprocess_data(movies, ratings):
    # Merge movies and ratings datasets
    data = pd.merge(ratings, movies, on="movieId")
    
    # Pivot table: users vs movie ratings
    data_pivot = data.pivot_table(index='userId', columns='title', values='rating')
    return data_pivot

# Step 3: Build the user similarity matrix using cosine similarity
def build_user_similarity_matrix(data_pivot):
    # Handle missing values by filling them with the mean
    imputer = SimpleImputer(strategy='mean')
    data_filled = imputer.fit_transform(data_pivot)
    
    # Calculate user similarity using cosine similarity
    user_similarity = cosine_similarity(data_filled)
    return user_similarity, data_filled

# Step 4: Recommend movies based on user similarity
def recommend_movies(user_id, data_pivot, user_similarity, num_recommendations=5):
    user_index = user_id - 1  # User index starts from 0
    similarity_score = user_similarity[user_index]

    # Weighted ratings based on user similarity
    weighted_ratings = similarity_score @ data_filled
    ratings_mean = weighted_ratings / similarity_score.sum()

    # Get the movies that the user hasn't rated
    user_ratings = data_pivot.iloc[user_index]
    recommendations = pd.Series(ratings_mean, index=data_pivot.columns)
    recommendations = recommendations[user_ratings.isna()].sort_values(ascending=False)

    return recommendations.head(num_recommendations)

# Step 5: Evaluate the model using RMSE
def evaluate_model(data_pivot, user_similarity):
    # Predict ratings using the similarity matrix
    predicted_ratings = user_similarity @ data_filled / np.array([np.abs(user_similarity).sum(axis=1)]).T
    predicted_ratings_df = pd.DataFrame(predicted_ratings, index=data_pivot.index, columns=data_pivot.columns)

    # Compare predicted ratings with actual ratings
    actual = []
    predicted = []
    for user_id in data_pivot.index:
        for movie in data_pivot.columns:
            actual_rating = data_pivot.loc[user_id, movie]
            if not np.isnan(actual_rating):  
                actual.append(actual_rating)
                predicted.append(predicted_ratings_df.loc[user_id, movie])

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return rmse

def main():
    print("=== Movie Recommendation System ===")

    # Load data
    print("Loading data...")
    movies, ratings = load_data()

    # Preprocess data
    print("Preprocessing data...")
    data_pivot = preprocess_data(movies, ratings)

    # Build user similarity matrix
    print("Building user similarity matrix...")
    user_similarity, data_filled = build_user_similarity_matrix(data_pivot)

    # Recommend movies for a sample user (e.g., user with ID 1)
    user_id = 1  # Change this to test with different users
    print(f"Generating recommendations for User {user_id}...")
    recommendations = recommend_movies(user_id, data_pivot, user_similarity)
    print(f"Recommended Movies for User {user_id}:\n", recommendations)

    # Evaluate the model (RMSE)
    print("Evaluating model...")
    rmse = evaluate_model(data_pivot, user_similarity)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

if __name__ == "__main__":
    main()
