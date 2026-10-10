
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"

FEATURES = [
    "age", "sex", "bmi", "bp", "s1",
    "s2", "s3", "s4", "s5", "s6",
]

MODEL_VERSION = "v0.1"

app = FastAPI(title="Diabetes Progression Prediction API")

# Load the trained pipeline once when the service starts.
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Run python src/train.py first."
    )

model = joblib.load(MODEL_PATH)


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(allow_inf_nan=False)
    sex: float = Field(allow_inf_nan=False)
    bmi: float = Field(allow_inf_nan=False)
    bp: float = Field(allow_inf_nan=False)
    s1: float = Field(allow_inf_nan=False)
    s2: float = Field(allow_inf_nan=False)
    s3: float = Field(allow_inf_nan=False)
    s4: float = Field(allow_inf_nan=False)
    s5: float = Field(allow_inf_nan=False)
    s6: float = Field(allow_inf_nan=False)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(patient: PatientFeatures):
    try:
        # Keep the feature names and order expected by the trained pipeline.
        features = pd.DataFrame(
            [patient.model_dump()],
            columns=FEATURES,
        )

        prediction = float(model.predict(features)[0])

        return {"prediction": prediction}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc
