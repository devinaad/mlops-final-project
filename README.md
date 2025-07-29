# MLOps Course Recommendation System

A comprehensive course recommendation system built with FastAPI, FAISS vector search, and sentence transformers. This system provides intelligent course recommendations based on user queries, job titles, and skills using semantic similarity search.

## 🚀 Features

- **Query-based Recommendations**: Get course suggestions based on free-text queries
- **Job-based Recommendations**: Find relevant courses for specific job titles
- **Semantic Search**: Uses sentence transformers for intelligent matching
- **Fast Vector Search**: FAISS-powered similarity search for scalable performance
- **RESTful API**: Clean FastAPI implementation with automatic documentation
- **Pre-computed Embeddings**: Optimized performance with cached embeddings
- **Comprehensive Data**: 445+ courses from Coursera with detailed metadata

## 📁 Repository Structure

```
mlops-final-project/
│
├── app/                           # Core application modules
│   ├── data_processing.py         # Data loading and FAISS index creation
│   ├── predict_jobtitle.py        # Job-based recommendation logic
│   ├── predict_query.py           # Query-based recommendation logic
│   ├── faiss_store.index          # FAISS vector index (generated)
│   ├── course_embeddings.pkl      # Pre-computed course embeddings
│   ├── job_index.faiss            # Job description index
│   └── job_df.pkl                 # Processed job dataframe
│
├── other_process/                 # Data collection and preprocessing
│   ├── coursera_scrap.ipynb       # Coursera course scraping notebook
│   └── text_preprocessing.ipynb   # Text preprocessing and embedding creation
│
├── main.py                        # FastAPI application entry point
├── save_data_json.py              # Data collection and API testing utility
├── coursera_courses.csv           # Course dataset (445 courses)
├── job_title_des.csv              # Job descriptions dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager
- 4GB+ RAM (for loading models and embeddings)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/devinaad/mlops-final-project.git
   cd mlops-final-project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models**
   The system will automatically download the `all-MiniLM-L6-v2` model on first run.

## 🚀 Quick Start

### Starting the API Server

```bash
python main.py
```

The API will be available at:
- **Base URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Get Available Jobs
```bash
GET /jobs
```

#### 3. Get Job Details
```bash
GET /jobs/{job_title}/details
```

#### 4. Query-based Recommendations
```bash
POST /recommend
Content-Type: application/json

{
  "query": "I want to learn data analytics and business strategy",
  "top_k": 5
}
```

#### 5. Job-based Recommendations
```bash
POST /recommend-by-job
Content-Type: application/json

{
  "job_title": "Data Analyst",
  "top_k": 5
}
```

### Example Usage

#### Python Client Example
```python
import requests

# Query-based recommendation
response = requests.post("http://localhost:8000/recommend", json={
    "query": "I want to learn machine learning and Python",
    "top_k": 3
})

recommendations = response.json()
print(f"Found {recommendations['total_results']} courses")

for course in recommendations['recommendations']:
    print(f"- {course['title']} (Score: {course['score']:.3f})")
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"query": "data science python", "top_k": 3}'
```

## 📊 Data Sources

### Course Dataset (`coursera_courses.csv`)
- **Source**: Coursera courses scraped from various skill and job searches
- **Size**: 445 unique courses
- **Fields**: title, organization, rating, skills, difficulty, duration, link

### Job Dataset (`job_title_des.csv`)
- **Source**: Job descriptions for various tech and business roles
- **Coverage**: 26 different job titles
- **Fields**: Job Title, Job Description

## 🧠 Technical Architecture

### Core Components

1. **Sentence Transformers**: `all-MiniLM-L6-v2` for text embedding
2. **FAISS**: Facebook AI Similarity Search for vector indexing
3. **FastAPI**: Modern web framework for API development
4. **Pandas**: Data manipulation and analysis

### Recommendation Process

1. **Text Preprocessing**: Clean and combine course/job text data
2. **Embedding Generation**: Convert text to 384-dimensional vectors
3. **Vector Normalization**: L2 normalization for cosine similarity
4. **Index Creation**: FAISS index for fast similarity search
5. **Query Processing**: Real-time embedding and similarity matching

### Performance Optimizations

- Pre-computed embeddings for courses and jobs
- FAISS index for O(log n) search complexity
- L2 normalization for accurate cosine similarity
- Efficient memory management with pickle serialization

## 🔧 Configuration

### Model Configuration
```python
MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model
```

### File Paths
```python
CSV_PATH = "coursera_courses.csv"
JOB_CSV_PATH = "job_title_des.csv"
COURSE_INDEX_PATH = "app/faiss_store.index"
COURSE_EMBEDDINGS_PATH = "app/course_embeddings.pkl"
```

## 📈 API Response Format

### Query Recommendation Response
```json
{
  "query": "data science python",
  "total_results": 3,
  "recommendations": [
    {
      "rank": 1,
      "title": "IBM Data Science",
      "organization": "IBM",
      "rating": 4.6,
      "skills": "Python, Machine Learning, Data Analysis",
      "difficulty": "Beginner",
      "duration": "3 - 6 Months",
      "link": "https://coursera.org/...",
      "score": 0.8234
    }
  ]
}
```

### Job Recommendation Response
```json
{
  "job_details": {
    "title": "Data Analyst",
    "description": "Analyze data to help business decisions...",
    "skills": ["Python", "SQL", "Statistics", "Excel"]
  },
  "total_results": 5,
  "recommendations": [...]
}
```

## 🛠️ Development

### Running in Development Mode

```bash
# Start with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Data Collection

Use the data collection utility to gather recommendations:

```bash
python save_data_json.py
```

This will:
- Test API connectivity
- Collect recommendations for skills, jobs, and majors
- Save results to JSON files in `collected_data/` folder

### Regenerating Embeddings

To regenerate embeddings (useful after data updates):

```bash
POST /embeddings/regenerate
```

### Adding New Data

1. Update `coursera_courses.csv` with new course data
2. Restart the application to regenerate embeddings
3. Or use the `/embeddings/regenerate` endpoint

## 🧪 Testing

### Manual Testing

1. **Health Check**: `GET /health`
2. **Job List**: `GET /jobs`
3. **Sample Query**: 
   ```json
   POST /recommend
   {"query": "python programming", "top_k": 3}
   ```

### Data Collection Testing

Run the comprehensive data collector:
```bash
python save_data_json.py
```

## 📋 Requirements

```txt
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.0.3
numpy==1.24.3
faiss-cpu==1.7.4
sentence-transformers==2.2.2
scikit-learn==1.3.0
pydantic==2.4.2
requests==2.31.0
```

## 🚀 Deployment

### Local Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment (Example)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- [@devinaad](https://github.com/devinaad)
- [@salsabilakn](https://github.com/salsabilakn)

## 🙏 Acknowledgments

- Coursera for course data
- Sentence Transformers team for the embedding model
- FAISS team for the vector search library
- FastAPI team for the excellent web framework

