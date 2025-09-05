import os, time, json
from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel
import mlflow, mlflow.sklearn

# --- Prometheus metrics ---
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUESTS = Counter("app_requests_total", "Total HTTP requests", ["path", "method", "code"])
INFER_REQ = Counter("inference_requests_total", "Inference requests", ["code"])
INFER_LAT = Histogram(
    "inference_latency_seconds",
    "Latency for /predict",
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
)

# ---------- Config ----------
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")   # Staging or Production
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://54.147.138.39:8081")
try:
    import yaml
    with open("params.yaml", "r") as f:
        MODEL_NAME = yaml.safe_load(f)["registered_model_name"]
except Exception:
    MODEL_NAME = os.getenv("MODEL_NAME", "w7d1_cancer_classifier")
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

LOG_PATH = "logs/requests.jsonl"
os.makedirs("logs", exist_ok=True)

app = FastAPI(title="W7 Inference Service", version="0.2.0")
model = None
n_features = None

class PredictRequest(BaseModel):
    rows: List[List[float]]

class PredictResponse(BaseModel):
    probs: List[float]
    preds: List[int]
    n_features: int
    model_stage: str
    model_name: str

@app.middleware("http")
async def metrics_mw(request: Request, call_next):
    start = time.perf_counter()
    try:
        resp = await call_next(request)
        code = resp.status_code
    except Exception:
        code = 500
        raise
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(path=request.url.path, method=request.method, code=str(code)).inc()
        # We only observe INFER_LAT inside /predict handler (to avoid bias)
    return resp

@app.on_event("startup")
def load_model():
    global model, n_features
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    n_features = getattr(model, "n_features_in_", None)
    print(f"[BOOT] Loaded {MODEL_URI}; n_features={n_features}")

@app.get("/healthz")
def healthz():
    ok = model is not None
    return {"ok": ok, "model_uri": MODEL_URI, "n_features": n_features}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def _append_request(rows: List[List[float]]):
    # Keep a tiny sample of requests for drift checks (demo only)
    payload = {"ts": time.time(), "rows": rows}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        INFER_REQ.labels(code="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = np.array(req.rows, dtype=float)
    if X.ndim != 2:
        INFER_REQ.labels(code="400").inc()
        raise HTTPException(status_code=400, detail="rows must be 2D list")
    if n_features is not None and X.shape[1] != n_features:
        INFER_REQ.labels(code="400").inc()
        raise HTTPException(status_code=400, detail=f"Expected {n_features} features, got {X.shape[1]}")

    t0 = time.perf_counter()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1].tolist()
        preds = (np.array(probs) >= 0.5).astype(int).tolist()
    else:
        preds_arr = model.predict(X)
        preds = preds_arr.astype(int).tolist() if hasattr(preds_arr, "astype") else list(preds_arr)
        probs = preds
    dt = time.perf_counter() - t0

    INFER_LAT.observe(dt)
    INFER_REQ.labels(code="200").inc()

    # Log a tiny sample for drift (keep it light)
    try:
        _append_request(req.rows[:2])  # sample first 2 rows
    except Exception as e:
        print(f"[WARN] request log failed: {e}")

    return PredictResponse(
        probs=[float(p) for p in probs],
        preds=[int(p) for p in preds],
        n_features=int(n_features) if n_features is not None else X.shape[1],
        model_stage=MODEL_STAGE,
        model_name=MODEL_NAME
    )

