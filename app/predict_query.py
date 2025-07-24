import numpy as np
import math

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def safe_float(value):
    """Ensure float is valid for JSON (no NaN or Infinity)"""
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(float(value), 4)

def search_courses(query, model, index, courses_df, top_k=5):
    """
    Search similar courses for a given user query using FAISS.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = normalize(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue  # skip invalid result

        course = courses_df.iloc[idx]

        results.append({
            "rank": i + 1,
            "title": course.get("title", ""),
            "organization": course.get("organization", ""),
            "rating": safe_float(course.get("rating", 0.0)),
            "skills": course.get("skills", ""),
            "difficulty": course.get("difficulty", ""),
            "duration": course.get("duration", ""),
            "link": course.get("link", ""),
            "score": safe_float(distances[0][i])
        })

    return results
