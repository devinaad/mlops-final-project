# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import app.data_processing as dp
import app.predict_query as pred
# from app.predict_jobtitle import recommend_by_job, load_and_encode_job_descriptions
from sentence_transformers import SentenceTransformer

CSV_PATH = "coursera_courses.csv"
INDEX_PATH = "app/faiss_store.index"

# model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="Course Recommender")

# # Load job description & course description
# job_df, job_emb = load_and_encode_job_descriptions("job_title_des.csv", model)

@app.get("/recommend-by-jobtitle")
# def recommend_job(job_title: str, top_k: int = 5):
#     results = recommend_by_job(job_title, job_df, job_emb, courses_df, course_embeddings, top_k)
#     return {"job_title": job_title, "recommendations": results}

class QueryIn(BaseModel):
    query: str = Field(..., example="I want to learn data analytics and business strategy")
    top_k: int = Field(5, description="Number of recommendations")

courses_df, model, index = dp.prepare_store(CSV_PATH, INDEX_PATH)

@app.post("/recommend")
def recommend(query: QueryIn):
    results = pred.search_courses(query.query, model, index, courses_df, top_k=query.top_k)
    return {"query": query.query, "recommendations": results}
