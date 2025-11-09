from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="Assessment Recommendation API")

class Recommendation(BaseModel):
    query: str
    recommendations: list

@app.get("/")
def root():
    return {"message": "API is live"}

@app.get("/recommend", response_model=Recommendation)
def recommend(query: str = Query(..., description="Enter assessment name or skill")):
    dummy_results = ["SHL Cognitive Ability Test", "SHL Personality Questionnaire", "SHL Leadership Assessment"]
    return {"query": query, "recommendations": dummy_results[:3]}



