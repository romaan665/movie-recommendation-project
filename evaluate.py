from sklearn.metrics import mean_squared_error
import numpy as np

predicted_ratings = user_similarity @ data1filled / np.array([np.abs(user_similarity).sum(axis=1)]).T

predicted_ratings_df = pd.DataFrame(predicted_ratings, index=data1.index, columns=data1.columns)

actual = []
predicted = []

for user_id in data1.index:
    for movie in data1.columns:
        actual_rating = data1.loc[user_id, movie]
        if not np.isnan(actual_rating):  
            actual.append(actual_rating)
            predicted.append(predicted_ratings_df.loc[user_id, movie])

rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f"Root Mean Squared Error (RMSE): {rmse}")
