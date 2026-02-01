"""FastAPI app exposing prediction endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ARTIFACTS_DIR
from .predict import predict_from_dataframe


class RentRequest(BaseModel):
    date: str = Field(..., description="ISO date like 2025-12-31")
    state: str
    suburb: str
    postcode: str
    bedrooms: int
    dwelling_type: str


class RentResponse(BaseModel):
    prediction_catboost: float
    prediction_baseline: float


app = FastAPI(title="MoveMate Rent Prediction API", version="1.0.0")

# Allow local dev UIs to call the API (OPTIONS preflight).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _ensure_artifacts(artifacts_dir: Path) -> None:
    required = [
        artifacts_dir / "catboost_model.cbm",
        artifacts_dir / "baseline.pkl",
        artifacts_dir / "feature_schema.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing model artifacts: {missing}. Run training first.",
        )


@app.post("/predict", response_model=RentResponse)
def predict_one(payload: RentRequest) -> RentResponse:
    _ensure_artifacts(ARTIFACTS_DIR)
    df = pd.DataFrame([payload.model_dump()])
    result = predict_from_dataframe(df, artifacts_dir=ARTIFACTS_DIR)
    row = result.iloc[0]
    return RentResponse(
        prediction_catboost=float(row["prediction_catboost"]),
        prediction_baseline=float(row["prediction_baseline"]),
    )


@app.post("/predict-batch", response_model=List[RentResponse])
def predict_batch(payloads: List[RentRequest]) -> List[RentResponse]:
    _ensure_artifacts(ARTIFACTS_DIR)
    df = pd.DataFrame([p.model_dump() for p in payloads])
    result = predict_from_dataframe(df, artifacts_dir=ARTIFACTS_DIR)
    responses: List[RentResponse] = []
    for _, row in result.iterrows():
        responses.append(
            RentResponse(
                prediction_catboost=float(row["prediction_catboost"]),
                prediction_baseline=float(row["prediction_baseline"]),
            )
        )
    return responses
