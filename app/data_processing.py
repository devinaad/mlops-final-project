import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Model yang digunakan untuk embedding
MODEL_NAME = "all-MiniLM-L6-v2"

# Normalisasi L2 terhadap vektor
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

# Load data dan gabungkan teks
def load_courses(csv_path):
    df = pd.read_csv(csv_path)
    
    # Pastikan kolom yang dibutuhkan ada
    required_cols = ['title', 'skills']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Gabungkan teks dari title dan skills (jika kosong, isi "")
    df['skills'] = df['skills'].fillna("")
    df['text_combined'] = df['title'].astype(str) + " " + df['skills'].astype(str)
    
    return df

# Proses embedding
def embed_texts(texts, model=None):
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

# Simpan FAISS index ke file
def prepare_store(csv_path, index_path):
    df = load_courses(csv_path)
    model = SentenceTransformer(MODEL_NAME)
    
    # Embedding & normalisasi
    embeddings = embed_texts(df['text_combined'].tolist(), model)
    embeddings = normalize(embeddings)
    
    # Buat FAISS index untuk pencarian similarity berbasis cosine
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product ≈ cosine jika dinormalisasi
    index.add(embeddings)
    
    # Simpan index ke file
    faiss.write_index(index, index_path)
    
    return df, model, index
