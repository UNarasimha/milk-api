"""
Milk Composition Prediction API
--------------------------------
Takes 256 NIR absorbance values, returns Fat, Protein, Lactose predictions.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import tensorflow as tf
import joblib

# ---------- 1. Initialize FastAPI app ----------
app = FastAPI(
    title="Milk Composition Predictor",
    description="Predict Fat, Protein, and Lactose content from NIR spectra.",
    version="1.0.0"
)

# ---------- 2. Load model + scaler ONCE at startup ----------
MODEL_PATH  = "milk_model.keras"
SCALER_PATH = "target_scaler.pkl"

try:
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print("Failed to load model/scaler:", e)
    raise

TARGETS = ["Fat", "Prot", "Lact"]

# ---------- 3. Request schema (auto-validated) ----------
class SpectrumInput(BaseModel):
    spectra: List[float] = Field(
        ...,
        min_length=256,
        max_length=256,
        description="256 NIR absorbance values (already SNV-normalized)"
    )

# ---------- 4. Response schema ----------
class PredictionOutput(BaseModel):
    Fat: float
    Prot: float
    Lact: float

# ---------- 5. Helper: apply SNV (same as training) ----------
def apply_snv(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std  = x.std(axis=1, keepdims=True) + 1e-8
    return (x - mean) / std

# ---------- 6. Routes ----------
@app.get("/")
def root():
    return {
        "message": "Milk Composition Predictor API",
        "usage": "POST /predict with {'spectra': [256 floats]}",
        "docs":  "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: SpectrumInput):
    try:
        # 1. Convert to numpy, reshape to (1, 256)
        x = np.array(payload.spectra, dtype=np.float32).reshape(1, -1)

        # 2. Apply SNV (must match training preprocessing)
        x = apply_snv(x)

        # 3. Predict (scaled outputs)
        y_scaled = model.predict(x, verbose=0)

        # 4. Inverse-transform to real units (%)
        y_real = scaler.inverse_transform(y_scaled)[0]

        return PredictionOutput(
            Fat=round(float(y_real[0]), 3),
            Prot=round(float(y_real[1]), 3),
            Lact=round(float(y_real[2]), 3),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")