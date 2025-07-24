import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def load_and_encode_job_descriptions(path, model):
    df = pd.read_csv(path)
    embeddings = model.encode(df['Description'].tolist())
    embeddings = normalize(embeddings)
    return df, embeddings

def recommend_by_job(job_title, job_df, job_emb, course_df, course_emb, top_k=5):
    # Cari embedding job berdasarkan nama
    idx = job_df[job_df['Job Title'] == job_title].index[0]
    job_vec = job_emb[idx].reshape(1, -1)

    # Hitung similarity ke semua course
    sims = cosine_similarity(job_vec, course_emb)[0]

    # Ambil top-N
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for i in top_indices:
        course = course_df.iloc[i]
        results.append({
            "title": course['title'],
            "link": course['link'],
            "score": float(sims[i])
        })

    return results
