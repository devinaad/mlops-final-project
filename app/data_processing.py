# app/data_processing.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def load_courses(csv_path):
    df = pd.read_csv(csv_path)
    df['text_combined'] = df['title'] + " " + df['metadata'] + " " + df['skills'].fillna("")
    return df

def embed_texts(texts, model=None):
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return np.array(model.encode(texts))

def prepare_store(csv_path, index_path):
    df = load_courses(csv_path)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = embed_texts(df['text_combined'].tolist(), model)
    embeddings = normalize(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return df, model, index
