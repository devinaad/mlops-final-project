# app/predict_jobtitle.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

def normalize(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def extract_skills_from_text(text):
    """Extract skills from job description using keywords"""
    # Common skills keywords - you can expand this list
    skill_keywords = [
        'python', 'java', 'javascript', 'react', 'node.js', 'angular', 'vue.js',
        'html', 'css', 'sql', 'database', 'mongodb', 'postgresql', 'mysql',
        'git', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'cloud',
        'machine learning', 'ai', 'data analysis', 'statistics', 'excel',
        'tableau', 'power bi', 'r programming', 'scala', 'spark',
        'communication', 'leadership', 'project management', 'agile', 'scrum',
        'api', 'rest', 'graphql', 'microservices', 'testing', 'debugging'
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in skill_keywords:
        if skill.lower() in text_lower:
            # Capitalize properly
            found_skills.append(skill.title() if len(skill.split()) == 1 else skill)
    
    # Remove duplicates and limit to top 6 skills
    unique_skills = list(dict.fromkeys(found_skills))[:6]
    
    return unique_skills if unique_skills else ["General Skills", "Problem Solving", "Communication"]

def load_and_encode_job_descriptions(path, model):
    """Load job descriptions and encode them"""
    df = pd.read_csv(path)
    
    # Extract skills for each job
    df['skills'] = df['Job Description'].apply(extract_skills_from_text)
    
    # Combine job title and description for better embedding
    df['combined_text'] = df['Job Title'] + " " + df['Description']
    embeddings = model.encode(df['combined_text'].tolist())
    embeddings = normalize(embeddings)
    return df, embeddings

def get_available_jobs(job_df):
    """Get list of available job titles for dropdown"""
    return sorted(job_df['Job Title'].unique().tolist())

def recommend_courses_by_job(job_title, job_df, job_emb, course_df, course_emb, top_k=5):
    """Recommend courses based on selected job title"""
    try:
        # Find the job in the dataframe
        job_matches = job_df[job_df['Job Title'] == job_title]
        if job_matches.empty:
            return []
        
        # Get the first matching job (in case of duplicates)
        idx = job_matches.index[0]
        job_vec = job_emb[idx].reshape(1, -1)

        # Calculate similarity to all courses
        similarities = cosine_similarity(job_vec, course_emb)[0]

        # Get top-K most similar courses
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for i in top_indices:
            course = course_df.iloc[i]
            results.append({
            "title": course["title"],
            "organization": course.get("organization", ""),
            "rating": course.get("rating", None),
            "skills": course.get("skills", ""),
            "difficulty": course.get("difficulty", ""),
            "duration": course.get("duration", ""),
            "link": course.get("link", ""),
                "similarity_score": float(similarities[i])
            })

        return results
    
    except Exception as e:
        print(f"Error in recommend_courses_by_job: {e}")
        return []

def get_job_details(job_title, job_df):
    """Get comprehensive job details including skills"""
    try:
        job_match = job_df[job_df['Job Title'] == job_title]
        if not job_match.empty:
            job_data = job_match.iloc[0]
            return {
                "title": job_data['Job Title'],
                "description": job_data['Job Description'],
                "skills": job_data['skills']  # This will be a list
            }
        return None
    except:
        return None