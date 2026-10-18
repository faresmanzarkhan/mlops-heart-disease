import time
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Prometheus metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP Request latency", ["endpoint"])

# Load model
model_path = "models/best_model.joblib"
try:
    model = joblib.load(model_path)
    model_loaded = True
except Exception:
    model = None
    model_loaded = False

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    endpoint = request.url.path
    method = request.method
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
    return response

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_version": "random_forest:572d46d97d4d4cbfb499435c904cf0f9"
    }

@app.post("/predict")
def predict(input_data: HeartInput):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    df = pd.DataFrame([input_data.dict()])
    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(model.predict(df)[0])
    
    return {
        "prediction": pred,
        "risk_label": "lower_risk" if pred == 0 else "higher_risk",
        "confidence": round(1.0 - prob if pred == 0 else prob, 12),
        "disease_probability": round(prob, 12),
        "decision_threshold": 0.5,
        "model_version": "random_forest:572d46d97d4d4cbfb499435c904cf0f9",
        "request_id": str(uuid.uuid4())
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
