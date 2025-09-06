# app/main.py
import os
import joblib
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Simple token — change in production via env var
INFER_API_TOKEN = os.getenv("INFER_API_TOKEN", "insecure-demo-token")

app = FastAPI(title="Spam Inference Service")

# Serve the static UI from ../static
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load model once (module import time)
MODEL_PATH = os.path.join(os.path.dirname(
    __file__), "..", "models", "sms_spam_pipeline.joblib")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Place sms_spam_pipeline.joblib in the models/ folder.")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


class PredictRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def root():
    # Return the static HTML demo page
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


LABEL_MAP = {0: "HAM", 1: "SPAM"}
THRESHOLD = float(os.getenv("INFER_THRESHOLD", "0.5"))  # default 0.5


@app.post("/predict")
def predict(req: PredictRequest, x_api_key: str | None = Header(None)):
    if x_api_key != INFER_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    text = req.text or ""
    try:
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba([text])[0][1])
            label = 1 if prob >= THRESHOLD else 0
        else:
            label = int(model.predict([text])[0])
        return {"prediction": LABEL_MAP[label], "prob": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
