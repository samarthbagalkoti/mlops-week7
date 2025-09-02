import os
import time
import json
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn

# ---------- Config ----------
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")   # Staging by default
MODEL_NAME = None
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://54.147.138.39:8081")

# Read model name from params.yaml if present
try:
    import yaml
    with open("params.yaml", "r") as f:
        MODEL_NAME = yaml.safe_load(f)["registered_model_name"]
except Exception:
    MODEL_NAME = os.getenv("MODEL_NAME", "w7d1_cancer_classifier")

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# ---------- App ----------
app = FastAPI(title="W7 Staging Inference", version="0.1.0")

# Globals for model
model = None
n_features = None

class PredictRequest(BaseModel):
    # Rows of feature vectors, e.g. [[v1...vN], [v1...vN]]
    rows: List[List[float]]

class PredictResponse(BaseModel):
    probs: List[float]
    preds: List[int]
    n_features: int
    model_stage: str
    model_name: str

@app.on_event("startup")
def load_model():
    global model, n_features
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Load the sklearn flavor from the Registry using stage
    model = mlflow.sklearn.load_model(MODEL_URI)
    # Infer expected feature count from trained estimator
    n_features = getattr(model, "n_features_in_", None)
    print(f"[BOOT] Loaded {MODEL_URI}; n_features={n_features}")

@app.get("/healthz")
def healthz():
    ok = model is not None
    return {"ok": ok, "model_uri": MODEL_URI, "n_features": n_features}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = np.array(req.rows, dtype=float)
    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="rows must be 2D list")
    if n_features is not None and X.shape[1] != n_features:
        raise HTTPException(status_code=400, detail=f"Expected {n_features} features, got {X.shape[1]}")

    # Predict proba if available; otherwise fallback to decision_function/predict
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1].tolist()
        preds = (np.array(probs) >= 0.5).astype(int).tolist()
    else:
        preds_arr = model.predict(X)
        preds = preds_arr.astype(int).tolist() if hasattr(preds_arr, "astype") else list(preds_arr)
        probs = preds  # not ideal, but keeps schema stable

    return PredictResponse(
        probs=[float(p) for p in probs],
        preds=[int(p) for p in preds],
        n_features=int(n_features) if n_features is not None else X.shape[1],
        model_stage=MODEL_STAGE,
        model_name=MODEL_NAME
    )

