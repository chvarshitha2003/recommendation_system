import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'User': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Movie': ['Inception', 'The Matrix', 'Inception', 'Interstellar', 'The Matrix', 'Interstellar'],
    'Rating': [5, 4, 4, 5, 2, 4]
}
df = pd.DataFrame(data)
user_item_matrix = df.pivot_table(index='User', columns='Movie', values='Rating')
user_item_matrix = user_item_matrix.fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
user = 'A'
similar_users = user_similarity_df[user].sort_values(ascending=False)
similar_users_ratings = user_item_matrix.loc[similar_users.index]
recommended_movies = similar_users_ratings.loc[similar_users_ratings.index != user].mean(axis=0)
recommended_movies = recommended_movies[recommended_movies > 0].sort_values(ascending=False)
print("Recommended Movies for User 'A':")
print(recommended_movies)
