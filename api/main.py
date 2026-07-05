import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "models/best_fraud_model.pkl"

app = FastAPI(title="API de Detección de Fraude")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]


class PredictionRequest(BaseModel):
    data: dict[str, Any]


class BatchPredictionRequest(BaseModel):
    data: list[dict[str, Any]]


def prepare_input(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    if "Amount" in df.columns:
        df["Amount_log"] = np.log1p(df["Amount"])

    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24

    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


@app.get("/")
def home():
    return {
        "message": "API de detección de fraude activa",
        "model_name": artifact["model_name"],
        "threshold": threshold,
        "metrics": artifact.get("metrics", {}),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    df = prepare_input([request.data])
    probability = float(model.predict_proba(df)[:, 1][0])
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "fraud_probability": probability,
        "threshold": threshold,
    }


@app.post("/predict-batch")
def predict_batch(request: BatchPredictionRequest):
    df = prepare_input(request.data)
    probabilities = model.predict_proba(df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "predictions": predictions.tolist(),
        "fraud_probabilities": probabilities.tolist(),
        "threshold": threshold,
    }
