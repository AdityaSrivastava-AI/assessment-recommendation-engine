from fastapi import FastAPI, Query
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load sample SHL dataset
data = {
    "assessment_name": [
        "Data Analysis Test",
        "Machine Learning Test",
        "Communication Skills Test",
        "Leadership Assessment",
        "Logical Reasoning Test"
    ],
    "skills": [
        "Python; Data Analysis; Statistics",
        "Machine Learning; Python; Algorithms",
        "English; Grammar; Speaking",
        "Leadership; Management; Decision Making",
        "Logic; Reasoning; Problem Solving"
    ],
    "role": [
        "Data Analyst",
        "ML Engineer",
        "Customer Support",
        "Manager",
        "All"
    ],
    "description": [
        "Assesses data wrangling and analysis skills",
        "Evaluates ML understanding and model building",
        "Assesses verbal and written communication",
        "Tests leadership qualities and decision making",
        "Assesses analytical and logical thinking ability"
    ]
}

df = pd.DataFrame(data)

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for each assessment
df["embeddings"] = df["description"].apply(lambda x: model.encode(x, convert_to_tensor=True))

@app.get("/")
def home():
    return {"message": "Assessment Recommendation Engine is running!"}

@app.get("/recommend")
def recommend(query: str = Query(..., description="Enter job description or skills")):
    query_emb = model.encode(query, convert_to_tensor=True)
    df["score"] = df["embeddings"].apply(lambda x: float(util.cos_sim(query_emb, x)))
    results = df.sort_values(by="score", ascending=False).head(3)[["assessment_name", "role", "score"]]
    return results.to_dict(orient="records")
