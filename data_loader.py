import pandas as pd

movies=pd.read_csv("C:/Users/Alsheefa/Desktop/romi/project dataset/ml-latest-small/ml-latest-small/movies.csv")
ratings=pd.read_csv("C:/Users/Alsheefa/Desktop/romi/project dataset/ml-latest-small/ml-latest-small/ratings.csv")

print(movies.head())
print(ratings.head())


data=pd.merge(ratings, movies,on="movieId")
print(data.head())

data1=data.pivot_table(index='userId',columns='title',values='rating')
print(data1.head())

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy='mean')
data1filled=imputer.fit_transform(data1)

user_similarity=cosine_similarity(data1filled)
print(user_similarity[:5])
