# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import app.data_processing as dp
import app.predict_query as pred

# Configuration
CSV_PATH = "coursera_courses.csv"
JOB_CSV_PATH = "job_title_des.csv"
COURSE_INDEX_PATH = "app/faiss_store.index"
JOB_INDEX_PATH = "app/job_index.faiss"
JOB_DF_PATH = "app/job_df.pkl"
COURSE_EMBEDDINGS_PATH = "app/course_embeddings.pkl"

app = FastAPI(title="Course Recommender API", version="1.0.0")

# Global variables for loaded data
courses_df, model, course_index = None, None, None
job_df, job_index = None, None
course_embeddings = None

def normalize(vectors):
    """Normalize vectors using L2 normalization"""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

def extract_skills_from_text(text):
    """Extract skills from job description using keywords"""
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
            found_skills.append(skill.title() if len(skill.split()) == 1 else skill)
    
    unique_skills = list(dict.fromkeys(found_skills))[:6]
    return unique_skills if unique_skills else ["General Skills", "Problem Solving", "Communication"]

def get_available_jobs(job_df):
    """Get list of available job titles for dropdown"""
    return sorted(job_df['Job Title'].unique().tolist())

def get_job_details(job_title, job_df):
    """Get comprehensive job details including skills"""
    try:
        job_match = job_df[job_df['Job Title'] == job_title]
        if not job_match.empty:
            job_data = job_match.iloc[0]
            return {
                "title": job_data['Job Title'],
                "description": job_data['Job Description'],
                "skills": extract_skills_from_text(job_data['Job Description'])
            }
        return None
    except Exception as e:
        print(f"Error getting job details: {e}")
        return None

def recommend_courses_by_job(job_title, job_df, job_index, course_df, course_embeddings, model, top_k=5):
    """Recommend courses based on selected job title using FAISS job index"""
    try:
        # Find the job in the dataframe
        job_matches = job_df[job_df['Job Title'] == job_title]
        if job_matches.empty:
            return []
        
        # Get the first matching job (in case of duplicates)
        job_idx = job_matches.index[0]
        
        # Get job embedding by encoding the job text
        job_text = job_matches.iloc[0]['job_text']
        job_embedding = model.encode([job_text])
        job_embedding = normalize(job_embedding)

        # Calculate similarity to all courses
        similarities = cosine_similarity(job_embedding, course_embeddings)[0]

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

def create_and_save_job_embeddings(csv_path, model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   faiss_path="job_index.faiss", df_path="job_df.pkl"):
    """Create and save job embeddings from a CSV file"""
    model = SentenceTransformer(model_name)
    
    job_df = pd.read_csv(csv_path)
    job_df['job_text'] = job_df['Job Title'] + " " + job_df['Job Description']
    
    job_texts = job_df['job_text'].tolist()
    print("🔄 Creating job embeddings...")
    embeddings = model.encode(job_texts, show_progress_bar=True)
    embeddings = normalize(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    
    faiss.write_index(index, faiss_path)
    job_df.to_pickle(df_path)
    print(f"✅ Saved FAISS job index to '{faiss_path}' and job_df to '{df_path}'")

# Load data once at startup
@app.on_event("startup")
async def startup_event():
    global courses_df, model, course_index, job_df, job_index, course_embeddings
    
    try:
        # Load courses and FAISS index for query-based recommendations
        courses_df, model, course_index = dp.prepare_store(CSV_PATH, COURSE_INDEX_PATH)
        print(f"✅ Loaded {len(courses_df)} courses successfully")
        
        # Load or create job data and FAISS index
        if os.path.exists(JOB_INDEX_PATH) and os.path.exists(JOB_DF_PATH):
            # Load existing job data
            job_index = faiss.read_index(JOB_INDEX_PATH)
            job_df = pd.read_pickle(JOB_DF_PATH)
            print(f"✅ Loaded job index and {len(job_df)} job descriptions")
        else:
            print("⚠️ Job embeddings not found. Creating them now...")
            create_and_save_job_embeddings(JOB_CSV_PATH, faiss_path=JOB_INDEX_PATH, df_path=JOB_DF_PATH)
            job_index = faiss.read_index(JOB_INDEX_PATH)
            job_df = pd.read_pickle(JOB_DF_PATH)
            print(f"✅ Created and loaded job index with {len(job_df)} job descriptions")
        
        # Load or create course embeddings for job-based recommendations
        if os.path.exists(COURSE_EMBEDDINGS_PATH):
            with open(COURSE_EMBEDDINGS_PATH, 'rb') as f:
                course_embeddings = pickle.load(f)
            print("✅ Course embeddings loaded successfully")
        else:
            print("⚠️ Course embeddings not found. Creating them now...")
            course_embeddings = normalize(model.encode(courses_df['text_combined'].tolist()))
            with open(COURSE_EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(course_embeddings, f)
            print("✅ Course embeddings created and saved")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        raise e

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, example="I want to learn data analytics and business strategy")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations")

class JobRequest(BaseModel):
    job_title: str = Field(..., example="Data Analyst")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations")

# API Endpoints
@app.get("/")
def root():
    """Welcome endpoint"""
    return {
        "message": "Course Recommender API", 
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "jobs": "/jobs",
            "recommend": "/recommend",
            "recommend-by-job": "/recommend-by-job"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    courses_loaded = courses_df is not None
    jobs_loaded = job_df is not None
    model_loaded = model is not None
    job_index_loaded = job_index is not None
    course_emb_loaded = course_embeddings is not None
    
    return {
        "status": "healthy" if all([courses_loaded, jobs_loaded, model_loaded, job_index_loaded, course_emb_loaded]) else "unhealthy",
        "components": {
            "courses": f"{len(courses_df)} loaded" if courses_loaded else "not loaded",
            "jobs": f"{len(job_df)} loaded" if jobs_loaded else "not loaded",
            "model": "loaded" if model_loaded else "not loaded",
            "job_index": f"{job_index.ntotal} vectors" if job_index_loaded else "not loaded",
            "course_embeddings": f"shape {course_embeddings.shape}" if course_emb_loaded else "not loaded"
        }
    }

@app.get("/jobs", response_model=List[str])
def get_job_list():
    """Get list of available job titles for dropdown"""
    if job_df is None:
        raise HTTPException(status_code=500, detail="Job data not loaded")
    
    try:
        job_list = get_available_jobs(job_df)
        return job_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting job list: {str(e)}")

@app.get("/jobs/{job_title}/details")
def get_job_info(job_title: str):
    """Get comprehensive details for a specific job title"""
    if job_df is None:
        raise HTTPException(status_code=500, detail="Job data not loaded")
    
    job_details = get_job_details(job_title, job_df)
    if not job_details:
        raise HTTPException(status_code=404, detail="Job title not found")
    
    return job_details

@app.post("/recommend")
def get_recommendations(request: QueryRequest):
    """Get course recommendations based on text query"""
    if any(x is None for x in [courses_df, model, course_index]):
        raise HTTPException(status_code=500, detail="Course system not properly initialized")
    
    try:
        results = pred.search_courses(
            request.query, 
            model, 
            course_index, 
            courses_df, 
            top_k=request.top_k
        )
        return {
            "query": request.query,
            "total_results": len(results),
            "recommendations": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/recommend-by-job")
def get_job_based_recommendations(request: JobRequest):
    """Get course recommendations based on selected job title"""
    if any(x is None for x in [job_df, job_index, courses_df, course_embeddings, model]):
        raise HTTPException(status_code=500, detail="Job recommendation system not properly initialized")
    
    try:
        # Get comprehensive job details
        job_details = get_job_details(request.job_title, job_df)
        if not job_details:
            raise HTTPException(status_code=404, detail=f"Job title '{request.job_title}' not found")
        
        # Get course recommendations using FAISS job index
        results = recommend_courses_by_job(
            request.job_title,
            job_df,
            job_index,
            courses_df,
            course_embeddings,
            model,
            top_k=request.top_k
        )
        
        return {
            "job_details": job_details,
            "total_results": len(results),
            "recommendations": results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing job-based request: {str(e)}")

@app.get("/embeddings/status")
def get_embeddings_status():
    """Get status of embeddings files"""
    return {
        "job_index_path": JOB_INDEX_PATH,
        "job_df_path": JOB_DF_PATH,
        "course_embeddings_path": COURSE_EMBEDDINGS_PATH,
        "status": {
            "job_index": os.path.exists(JOB_INDEX_PATH),
            "job_df": os.path.exists(JOB_DF_PATH),
            "course_embeddings": os.path.exists(COURSE_EMBEDDINGS_PATH)
        }
    }

@app.post("/embeddings/regenerate")
def regenerate_embeddings():
    """Regenerate embeddings (useful for development)"""
    global job_df, job_index, course_embeddings
    
    try:
        if model is None or courses_df is None:
            raise HTTPException(status_code=500, detail="Core components not loaded")
        
        # Regenerate job embeddings and FAISS index
        print("🔄 Regenerating job embeddings...")
        create_and_save_job_embeddings(JOB_CSV_PATH, faiss_path=JOB_INDEX_PATH, df_path=JOB_DF_PATH)
        job_index = faiss.read_index(JOB_INDEX_PATH)
        job_df = pd.read_pickle(JOB_DF_PATH)
        
        # Regenerate course embeddings
        print("🔄 Regenerating course embeddings...")
        course_embeddings = normalize(model.encode(courses_df['text_combined'].tolist()))
        with open(COURSE_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(course_embeddings, f)
        
        return {
            "message": "Embeddings regenerated successfully",
            "job_index_vectors": job_index.ntotal,
            "course_embeddings_shape": course_embeddings.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error regenerating embeddings: {str(e)}")