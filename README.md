# Assessment Recommendation Engine

This FastAPI app recommends SHL-style assessments based on job role or skill description.

## Example
GET /recommend?query=data+scientist

Response:
[
  {"assessment_name": "Machine Learning Test", "role": "ML Engineer", "score": 0.82},
  {"assessment_name": "Data Analysis Test", "role": "Data Analyst", "score": 0.76},
  {"assessment_name": "Logical Reasoning Test", "role": "All", "score": 0.64}
]
