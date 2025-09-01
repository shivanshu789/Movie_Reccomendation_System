# imdb_movie_clustering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("Raw_data.csv")
df.head()
print(df.shape)
print(df.info())
print(df.isnull().sum())

# Cleaning numeric columns
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
df['Runtime'] = df['Runtime'].str.replace('min', '').astype(float)

# Fill missing Meta_score with mean
df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].mean())

# Clean Gross column
df['Gross'] = df['Gross'].replace(',', '', regex=True).astype(float)
df['Gross'] = df['Gross'].fillna(df['Gross'].median())

# Fill missing Certificate
df['Certificate'] = df['Certificate'].fillna("Unknown")

# Separate categorical and numerical columns
cat_clms = df.select_dtypes(include='object').columns
num_clms = df.select_dtypes(include='number').columns

print("Categorical columns:", cat_clms)
print("Numerical columns:", num_clms)

# SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Encode Tags column into embeddings
embeddings = model.encode(df['Tags'].tolist(), show_progress_bar=True)
df['embeddings'] = list(embeddings)

# Convert embeddings into numpy array
X = np.vstack(df['embeddings'].values)
print("Embedding matrix shape:", X.shape)

# Clustering with KMeans
sse = []
silhouette_scores = []
K = range(2, 30)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)  # Sum of squared errors
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Elbow Method plot
plt.plot(K, sse, 'bx-')
plt.xlabel('k')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method')
plt.show()

# Silhouette Score plot
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()


# Recommendation function
def recommend_by_cluster(movie_Series_Title, n_recs=5):
    """Recommend similar movies based on cluster"""
    if movie_Series_Title not in df['Series_Title'].values:
        return f"Movie '{movie_Series_Title}' not found in dataset!"
    
    # Find cluster of the given movie
    movie_cluster = df.loc[df['Series_Title'] == movie_Series_Title, 'cluster'].values[0]
    
    # Get movies from the same cluster
    cluster_movies = df[df['cluster'] == movie_cluster]
    
    # Exclude the original movie and return top N
    recs = cluster_movies[cluster_movies['Series_Title'] != movie_Series_Title].sample(n=n_recs, random_state=42)
    return recs[['Series_Title', 'Tags', 'cluster']]

print(df.head())
