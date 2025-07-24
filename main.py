# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import app.data_processing as dp
import app.predict_query as pred
from app.predict_jobtitle import (
    load_and_encode_job_descriptions, 
    get_available_jobs, 
    recommend_courses_by_job,
    get_job_details
)

# Configuration
CSV_PATH = "coursera_courses.csv"
JOB_CSV_PATH = "job_title_des.csv"
INDEX_PATH = "app/faiss_store.index"

app = FastAPI(title="Course Recommender API", version="1.0.0")

# Global variables for loaded data
courses_df, model, index = None, None, None
job_df, job_embeddings = None, None
course_embeddings = None

# Load data once at startup
@app.on_event("startup")
async def startup_event():
    global courses_df, model, index, job_df, job_embeddings, course_embeddings
    
    try:
        # Load courses and FAISS index
        courses_df, model, index = dp.prepare_store(CSV_PATH, INDEX_PATH)
        print(f"✅ Loaded {len(courses_df)} courses successfully")
        
        # Load job descriptions
        job_df, job_embeddings = load_and_encode_job_descriptions(JOB_CSV_PATH, model)
        print(f"✅ Loaded {len(job_df)} job descriptions successfully")
        
        # Create course embeddings for job-based recommendations
        course_embeddings = dp.normalize(model.encode(courses_df['text_combined'].tolist()))
        print("✅ Course embeddings prepared for job-based recommendations")
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, example="I want to learn data analytics and business strategy")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations: ")

class JobRequest(BaseModel):
    job_title: str = Field(..., example="Data Analyst")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations:")

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
    
    return {
        "status": "healthy" if all([courses_loaded, jobs_loaded, model_loaded]) else "unhealthy",
        "components": {
            "courses": f"{len(courses_df)} loaded" if courses_loaded else "not loaded",
            "jobs": f"{len(job_df)} loaded" if jobs_loaded else "not loaded",
            "model": "loaded" if model_loaded else "not loaded"
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
    if any(x is None for x in [courses_df, model, index]):
        raise HTTPException(status_code=500, detail="Course system not properly initialized")
    
    try:
        results = pred.search_courses(
            request.query, 
            model, 
            index, 
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
    if any(x is None for x in [job_df, job_embeddings, courses_df, course_embeddings]):
        raise HTTPException(status_code=500, detail="Job recommendation system not properly initialized")
    
    try:
        # Get comprehensive job details
        job_details = get_job_details(request.job_title, job_df)
        if not job_details:
            raise HTTPException(status_code=404, detail=f"Job title '{request.job_title}' not found")
        
        # Get course recommendations
        results = recommend_courses_by_job(
            request.job_title,
            job_df,
            job_embeddings,
            courses_df,
            course_embeddings,
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