from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import json
from pathlib import Path

# Load schema
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "schema_pipeline.json", "r", encoding="utf-8") as f:
    schema = json.load(f)


# Load pipeline (preprocess + model)
pipeline = joblib.load("pipeline_churn_rf.joblib")

app = FastAPI(title="Churn Prediction API")

# Pydantic model (CONTRATO)
class ClienteInput(BaseModel):
    plano_pagamento: str
    chamados_suporte: str
    taxa_skip_musica: float
    taxa_musicas_unicas: float
    notificacoes_clicadas: int
    horas_semanais: float
    tempo_medio_sessao: float
    idade: int

# Healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}

# Schema endpoint
@app.get("/schema")
def get_schema():
    return schema

# Prediction endpoint
@app.post("/predict")
def predict(data: ClienteInput):
    data_dict = data.dict()

    # valida features
    missing = set(schema["required_features"]) - set(data_dict.keys())
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing features",
                "missing_features": list(missing)
            }
        )

    # ordem EXATA usada no treino
    feature_order = schema["required_features"]

    X = pd.DataFrame([data_dict])[feature_order]

    # pipeline completo
    proba = pipeline.predict_proba(X)[0, 1]

    return {
        "churn_probability": float(proba),
        "churn_prediction": int(proba >= 0.5),
        "model_version": schema["model_version"]
    }

@app.post("/v1/predict")
def predict_v1(data: dict):
    if schema["model_version"] != "1.0":
        return {"error": "Model version mismatch"}
    return predict(data)

