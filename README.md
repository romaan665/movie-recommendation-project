# movie-recommendation-project
# Import modules
from data_loader import load_data
from recommender import build_user_similarity_matrix, recommend_movies
from evaluator import calculate_rmse, calculate_precision_recall

def main():
    # Step 1: Load Data
    print("Loading data...")
    movies, ratings, user_movie_matrix = load_data()
    print("Data loaded successfully!")

    # Step 2: Build User Similarity Matrix
    print("Building user similarity matrix...")
    user_similarity_matrix = build_user_similarity_matrix(user_movie_matrix)
    print("User similarity matrix created.")

    # Step 3: Recommend Movies for a User
    user_id = 1  # Example: Recommendations for user 1
    print(f"Generating recommendations for User {user_id}...")
    recommendations = recommend_movies(user_id, user_movie_matrix, user_similarity_matrix, num_recommendations=5)
    print(f"Recommended Movies for User {user_id}:\n")
    print(recommendations)

    # Step 4: Evaluate the Model (Optional)
    print("\nEvaluating the recommendation system...")
    test_actual_ratings 
    test_predicted_ratings
    
    rmse = calculate_rmse(test_actual_ratings, test_predicted_ratings)
    precision, recall = calculate_precision_recall(test_actual_ratings, test_predicted_ratings)
    
    print(f"RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
