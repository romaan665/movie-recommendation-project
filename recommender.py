def recommend_movie(user_id,data1,user_similarity,num_recommendations=5):
    user_index=user_id-1
    similarity_score=user_similarity[user_index]

    weighted_ratings=similarity_score@data1filled
    ratings_mean=weighted_ratings/similarity_score.sum()

    user_ratings=data1.iloc[user_index]
    recommendations=pd.Series(ratings_mean, index=data1.columns)
    recommendations=recommendations[user_ratings.isna()].sort_values(ascending=False)

    return recommendations.head(num_recommendations)

recommended_movies=recommend_movie(1,data1,user_similarity)
print("recommended movies:\n")
print(recommended_movies)

