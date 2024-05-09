import pandas as pd
from scipy.sparse import load_npz
import pickle

# Load the movie recommendation model from the .pkl file
with open('movie_recommendation_model.pkl', 'rb') as file:
    similarity_matrix = pickle.load(file)

# Load your movie dataset into a Pandas dataframe
df = pd.read_csv('TMDB_movie_dataset_v11.csv')

# Select only the required columns
selected_columns = ['title']
df = df[selected_columns]

# Preprocess input title
def preprocess_input(title):
    return title.lower()

# Function to get movie recommendations based on input title
def get_movie_recommendations(input_title, similarity_matrix, df):
    # Preprocess input title
    input_title = preprocess_input(input_title)
    
    # Find the index of the input movie in the dataframe
    input_index = df[df['title'].apply(lambda x: preprocess_input(x)) == input_title].index
    
    if len(input_index) == 0:
        return "Movie not found in the dataset."
    
    input_index = input_index[0]  # Get the first index if multiple matches
    
    # Calculate similarity scores for the input movie with all other movies
    sim_scores = list(enumerate(similarity_matrix[input_index]))
    
    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top recommendations (excluding the input movie itself)
    top_recommendations = sim_scores[1:15]  # Get top 10 recommendations
    
    # Extract movie titles of top recommendations
    top_movie_indices = [x[0] for x in top_recommendations]
    top_movie_titles = df.iloc[top_movie_indices]['title'].tolist()
    
    return top_movie_titles

# Example usage: Get recommendations for a specific movie title
input_title = "parasite"
recommendations = get_movie_recommendations(input_title, similarity_matrix, df)

if isinstance(recommendations, str):
    print(recommendations)
else:
    print(f"Top recommendations for '{input_title}':")
    for idx, movie_title in enumerate(recommendations, 1):
        print(f"{idx}. {movie_title}")
