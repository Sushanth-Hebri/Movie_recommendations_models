import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import numpy as np
import pickle

# Load your movie dataset into a Pandas dataframe
df = pd.read_csv('TMDB_movie_dataset_v11.csv')

# Select only the required columns
selected_columns = ['title', 'genres', 'keywords', 'popularity', 'overview']
df = df[selected_columns]

# Handle missing values if needed
df = df.fillna('')

# Convert numerical columns to strings for text preprocessing
for col in ['genres', 'keywords', 'popularity', 'overview']:
    df[col] = df[col].astype(str)

# Assign weights to features based on their importance
weights = {'genres': 0.85, 'keywords': 0.05, 'popularity': 0.05, 'overview': 0.05}

# Preprocess text data and apply TF-IDF Vectorization
vectorizers = {}
X_weighted = None

for col in ['genres', 'keywords', 'popularity', 'overview']:
    # Preprocess text data (lowercase)
    df[col] = df[col].apply(lambda x: str(x).lower())
    
    # TF-IDF Vectorization for each column
    vectorizers[col] = TfidfVectorizer(stop_words='english')
    X_col = vectorizers[col].fit_transform(df[col])
    
    # Multiply by corresponding weight
    X_weighted_col = X_col * weights[col]
    
    if X_weighted is None:
        X_weighted = X_weighted_col
    else:
        # Concatenate matrices horizontally (along columns)
        X_weighted = hstack((X_weighted, X_weighted_col))

# Train your recommendation model (example using cosine similarity)
similarity_matrix = cosine_similarity(X_weighted, X_weighted)

# Serialize the trained model to a .pkl file
with open('movie_recommendation_model.pkl', 'wb') as file:
    pickle.dump(similarity_matrix, file)
