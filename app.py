from fastapi import FastAPI, Query
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

app = FastAPI()

# Sample SHL-style dataset
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

# Use a light model
model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Compute sentence embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(embeddings, p=2, dim=1)

df["embeddings"] = df["description"].apply(lambda x: get_embedding(x))

@app.get("/")
def home():
    return {"message": "Assessment Recommendation Engine (Light) is running!"}

@app.get("/recommend")
def recommend(query: str = Query(..., description="Enter job description or skills")):
    query_emb = get_embedding(query)
    df["score"] = df["embeddings"].apply(lambda x: float(torch.mm(query_emb, x.T)))
    results = df.sort_values(by="score", ascending=False).head(3)[["assessment_name", "role", "score"]]
    return results.to_dict(orient="records")


