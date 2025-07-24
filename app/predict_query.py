# app/predict.py
import numpy as np

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def search_courses(query, model, index, courses_df, top_k=5):
    query_emb = np.array(model.encode([query]))
    query_emb = normalize(query_emb)
    
    D, I = index.search(query_emb, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        course = courses_df.iloc[idx]
        results.append({
            "title": course['title'],
            "link": course['link'],
            "organization": course['organization'],
            "rating": float(course['rating']),
            "metadata": course['metadata'],
            "skills": course['skills'],
            "score": float(dist)
        })
    return results
