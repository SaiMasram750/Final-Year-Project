from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import random
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load models
iso_model = joblib.load('isolation_forest_model.pkl')
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# --- Data Models ---
class SensorData(BaseModel):
    flow: float
    pH: float
    tank_level: float

class ChatRequest(BaseModel):
    query: str

class EmbedRequest(BaseModel):
    text: str

# --- Mock Database ---
logs: List[Dict] = []

# --- Endpoints ---
@app.post("/predict")
def predict(data: SensorData):
    features = np.array([[data.flow, data.pH, data.tank_level]])
    anomaly_score_raw = iso_model.decision_function(features)[0]
    anomaly_score = (anomaly_score_raw + 0.5) * 2
    anomaly_score = max(0, min(1, anomaly_score))

    severity = "High" if anomaly_score > 0.7 else "Low"
    alert = "Possible valve manipulation" if anomaly_score > 0.7 else "Normal"

    result = {
        "anomaly_score": round(anomaly_score, 2),
        "severity": severity,
        "alert": alert
    }
    logs.append({"type": "prediction", "data": result})
    return result

@app.post("/chat")
def chat(request: ChatRequest):
    response_text = f"AI Assistant: Based on your query '{request.query}', anomaly scores are being monitored."
    result = {"response": response_text}
    logs.append({"type": "chat", "data": result})
    return result

@app.post("/embed")
def embed_text(request: EmbedRequest):
    embedding = embed_model.encode(request.text).tolist()
    return {"embedding": embedding}

@app.get("/logs")
def get_logs():
    return {"logs": logs}

@app.get("/health")
def health_check():
    return {"status": "Backend Online"}
