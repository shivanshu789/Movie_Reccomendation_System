#  IMDb Movie Clustering & Recommendation System  

This project explores **unsupervised learning** by clustering IMDb movies based on their **plot tags** and metadata. Using **SentenceTransformer embeddings** and **KMeans clustering**, it groups similar movies and enables a simple recommendation engine.  

---

##  Features  
- **Data Cleaning & Preprocessing**  
  - Converts `Released_Year`, `Runtime`, and `Gross` into numeric values.  
  - Fills missing values in `Meta_score`, `Gross`, and `Certificate`.  

- **Text Embeddings with Transformers**  
  - Encodes movie **Tags** using `all-MiniLM-L6-v2` from [SentenceTransformers](https://www.sbert.net/).  

- **Clustering with KMeans**  
  - Finds optimal clusters with **Elbow Method (SSE)** and **Silhouette Score**.  
  - Assigns each movie to a cluster for similarity grouping.  

- **Movie Recommendation Function**  
  - Input a movie title and get **N similar movies** from the same cluster.  

---

##  Visualizations  
- **Elbow Method plot** → helps identify optimal number of clusters.  
- **Silhouette Score plot** → evaluates cluster quality.  

---

##  Tech Stack  
- **Python**   
- **Pandas, NumPy** – Data manipulation  
- **Matplotlib, Seaborn** – Visualizations  
- **SentenceTransformers** – Text embeddings  
- **Scikit-learn** – Clustering & evaluation  

---

##  Getting Started  

###  Clone the Repository  
```bash
git clone https://github.com/your-username/imdb-movie-clustering.git
cd imdb-movie-clustering

