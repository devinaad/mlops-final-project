import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

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

def load_job_data(csv_path, embeddings_path=None):
    """Load job descriptions and their pre-computed embeddings"""
    # Load job descriptions CSV
    df = pd.read_csv(csv_path)
    
    # Extract skills for each job
    df['skills'] = df['Job Description'].apply(extract_skills_from_text)
    
    # Load pre-computed embeddings if available
    embeddings = None
    if embeddings_path and os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded pre-computed job embeddings from {embeddings_path}")
    
    return df, embeddings

def load_course_embeddings(embeddings_path):
    """Load pre-computed course embeddings"""
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Loaded pre-computed course embeddings from {embeddings_path}")
        return embeddings
    else:
        raise FileNotFoundError(f"Course embeddings file not found: {embeddings_path}")

def get_available_jobs(job_df):
    """Get list of available job titles for dropdown"""
    return sorted(job_df['Job Title'].unique().tolist())

def recommend_courses_by_job(job_title, job_df, job_embeddings, course_df, course_embeddings, top_k=5):
    """Recommend courses based on selected job title using pre-computed embeddings"""
    try:
        # Find the job in the dataframe
        job_matches = job_df[job_df['Job Title'] == job_title]
        if job_matches.empty:
            return []
        
        # Get the first matching job (in case of duplicates)
        idx = job_matches.index[0]
        job_vec = job_embeddings[idx].reshape(1, -1)

        # Calculate similarity to all courses
        similarities = cosine_similarity(job_vec, course_embeddings)[0]

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
    except Exception as e:
        print(f"Error getting job details: {e}")
        return None

def save_embeddings(embeddings, file_path):
    """Save embeddings to pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {file_path}")

def check_embeddings_exist(job_embeddings_path, course_embeddings_path):
    """Check if pre-computed embeddings exist"""
    job_exists = os.path.exists(job_embeddings_path)
    course_exists = os.path.exists(course_embeddings_path)
    
    return {
        "job_embeddings": job_exists,
        "course_embeddings": course_exists,
        "both_exist": job_exists and course_exists
    }