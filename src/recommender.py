# recommender.py

"""
Netflix Content-Based & Platform-Aware Movie Recommendation System

This script recommends similar movies using genre, description, and platform availability.
Place the following CSV files in a folder named 'data' in the root of your project:
  - data/n_movies.csv
  - data/MoviesOnStreamingPlatforms.csv
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets using relative paths
df_movies = pd.read_csv('data/n_movies.csv')
df_platforms = pd.read_csv('data/MoviesOnStreamingPlatforms.csv')

# Rename and clean titles
df_movies.rename(columns={'title': 'Title'}, inplace=True)
df_movies['Title'] = df_movies['Title'].str.lower().str.strip()
df_platforms['Title'] = df_platforms['Title'].str.lower().str.strip()

# Merge and clean
df = pd.merge(df_movies, df_platforms, on='Title', how='inner')
df['genre'] = df['genre'].fillna('')
df['description'] = df['description'].fillna('')
df['Platforms'] = df.apply(
    lambda row: ', '.join([
        p for p in ['Netflix', 'Hulu', 'Prime Video', 'Disney+'] if p in row and row[p] == 1
    ]), axis=1
)
df['Combined'] = df['genre'] + " " + df['description'] + " " + df['Platforms']

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

# Recommend similar movies
def recommend_movies(title, num_recommendations=5):
    title = title.lower().strip()
    if title not in indices:
        return f"'{title}' not found in the dataset."

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]

    movie_indices = [i[0] for i in sim_scores]
    return df[['Title', 'genre', 'Platforms']].iloc[movie_indices]

# Recommend by genre (optional alternative)
def recommend_by_genre(fav_genres, num_recommendations=5):
    matches = df[df['genre'].str.contains(fav_genres, case=False, na=False)]
    if matches.empty:
        return f"No matches found for genre(s): {fav_genres}"

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(matches['Combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    top_indices = cosine_sim[0].argsort()[::-1][1:num_recommendations+1]

    return matches[['Title', 'genre', 'Platforms']].iloc[top_indices]

# Example usage (can be removed or used for testing)
if __name__ == '__main__':
    print(recommend_by_genre("action|adventure"))
