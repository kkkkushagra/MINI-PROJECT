import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Sanitize JSON data
def sanitize_for_json(data):
    """Recursively convert NaN, Infinity, etc. into safe JSON values."""
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist())
    else:
        return data


# ✅ Load clause playbook safely
try:
    clause_playbook = pd.read_csv("clause_playbook.csv")
except Exception as e:
    raise RuntimeError(f"Failed to load clause playbook: {e}")

# ✅ Load LegalBERT model and precompute embeddings
model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

clause_embeddings = model.encode(
    clause_playbook["standard_clause"].tolist(), 
    convert_to_numpy=True, 
    show_progress_bar=True
)
clause_embeddings = np.nan_to_num(clause_embeddings, nan=0.0, posinf=0.0, neginf=0.0)

# ✅ Create FastAPI instance
app = FastAPI(
    title="Contract Review & Redlining API",
    description="AI-powered backend for contract clause extraction and review using LegalBERT",
    version="2.0.0"
)

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Import routers
from routers import upload, extract, predict_clause, redline
app.include_router(upload.router)
app.include_router(extract.router)
app.include_router(predict_clause.router)
app.include_router(redline.router)


# ✅ Root endpoint
@app.get("/")
def root():
    return {"message": "LegalBERT-powered backend is running successfully 🚀"}

# ✅ Safe local clause prediction endpoint
@app.post("/predict_clause/")
async def predict_clause_endpoint(data: dict):
    text = data.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is required.")

    query_embedding = model.encode([text], convert_to_numpy=True)
    query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=0.0, neginf=0.0)

    similarities = cosine_similarity(query_embedding, clause_embeddings)
    similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)

    best_idx = int(np.argmax(similarities))
    best_row = clause_playbook.iloc[best_idx]

    response = {
        "matched_clause": str(best_row["standard_clause"]),
        "risk_level": str(best_row["Risk_Level"]),
        "action_required": str(best_row["Action_Required"]),
        "similarity_score": round(float(similarities[0][best_idx]), 3),
    }

    return JSONResponse(content=sanitize_for_json(response))
