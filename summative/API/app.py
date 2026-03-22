"""
FastAPI service for East Africa CO2 emissions prediction and model retraining.

Purpose:
- Expose HTTP endpoints that score new inputs with the saved regression model.
- Enforce request validation (types and ranges) using Pydantic models.
- Keep preprocessing at inference time consistent with notebook training steps.
- Support a retraining workflow when new labeled records are provided.

High-level structure:
1. Constants and paths:
    Defines target/feature names and file locations for dataset, model, and artifacts.

2. Request/response schemas:
    Pydantic models validate incoming payloads for single prediction, batch prediction,
    and retraining requests.

3. Runtime model state:
    Stores loaded model and predictor used by API handlers.

4. Preprocessing/training helpers:
    - _build_base_training_frame(): delegates training-frame creation to prediction.py.
    - _fit_and_store(): delegates training + artifact persistence to prediction.py.
    - _load_or_initialize(): loads model/predictor on startup or initializes them if missing.

5. API endpoints:
    - GET /            : health/info payload with route hints.
    - POST /predict    : single-record prediction.
    - POST /predict/batch : multi-record prediction.
    - POST /retrain    : append optional new labeled rows and retrain model.

Deployment notes:
- CORS is configured with explicit allowed origins (no wildcard).
- Swagger UI is available at /docs for interactive testing.
"""

# Import required libraries and modules for API routing, validation, and model state persistence.
from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    from .prediction import (
        EmissionsPredictor,
        TARGET_COL,
        build_model_training_frame,
        fit_and_save_linear_model,
    )
except ImportError:
    # Fallback for direct local execution from API folder.
    from prediction import (
        EmissionsPredictor,
        TARGET_COL,
        build_model_training_frame,
        fit_and_save_linear_model,
    )

# Target and feature names are kept identical to notebook training columns.
BASE_DIR = Path(__file__).resolve().parent
LINEAR_REGRESSION_DIR = BASE_DIR.parent / "linear_regression"
DATA_PATH = LINEAR_REGRESSION_DIR / "data" / "africa-co2-emissions.csv"
MODEL_PATH = LINEAR_REGRESSION_DIR / "final_model" / "best_linear_regression_model.joblib"
ARTIFACT_PATH = LINEAR_REGRESSION_DIR / "final_model" / "fastapi_model_artifacts.joblib"
EXTRA_ROWS_PATH = LINEAR_REGRESSION_DIR / "data" / "new_training_rows.csv"

# Prediction request schema defines the expected structure and validation rules for incoming prediction requests.
class PredictionRequest(BaseModel):
    # Pydantic enforces both datatype and value range constraints.
    country: str = Field(min_length=2, max_length=64)
    year: int = Field(ge=1960, le=2100)
    population: float = Field(ge=0, le=2_000_000_000)
    transportation_mt: float = Field(ge=0, le=10_000)
    manufacturing_construction_mt: float = Field(ge=0, le=10_000)
    electricity_heat_mt: float = Field(ge=0, le=10_000)
    building_mt: float = Field(ge=0, le=10_000)

    @field_validator("country")
    @classmethod
    def validate_country(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("country cannot be empty")
        return cleaned

    def to_feature_row(self) -> dict[str, Any]:
        return {
            "Country": self.country,
            "Year": self.year,
            "Population": self.population,
            "Transportation (Mt)": self.transportation_mt,
            "Manufacturing/Construction (Mt)": self.manufacturing_construction_mt,
            "Electricity/Heat (Mt)": self.electricity_heat_mt,
            "Building (Mt)": self.building_mt,
        }

# Batch prediction request allows clients to submit multiple records in one request while still enforcing validation on each record. The endpoint will return a list of predictions corresponding to the input records, and any validation errors will be handled gracefully with appropriate HTTP status codes and messages.
class BatchPredictionRequest(BaseModel):
    # Guardrail to avoid very large payloads in a single request.
    rows: list[PredictionRequest] = Field(min_length=1, max_length=1000)

# Retraining record extends prediction request with the target value, allowing clients to submit new labeled records for model improvement. The to_training_row method converts the record into a format suitable for appending to the training frame, ensuring consistency with the original dataset structure.
class RetrainRecord(PredictionRequest):
    target_mt: float = Field(ge=0, le=20_000)

    def to_training_row(self) -> dict[str, Any]:
        row = self.to_feature_row()
        row[TARGET_COL] = self.target_mt
        return row

# Retraining request allows clients to submit new labeled records for model improvement. The endpoint supports optional persistence of new rows to a local CSV file, which is included in the training frame for future retraining. Validation is performed on each record, and errors are handled gracefully with appropriate HTTP status codes and messages. The retraining process rebuilds the training frame, refits the model, and updates the shared predictor instance to ensure that subsequent predictions use the latest model state.
class RetrainRequest(BaseModel):
    rows: list[RetrainRecord] = Field(min_length=1)
    persist_new_rows: bool = Field(default=True)

# Response schemas for prediction and retraining endpoints. These define the structure of successful responses and can be extended in the future to include additional metadata if needed.
class PredictResponse(BaseModel):
    prediction_mt: float


class BatchPredictResponse(BaseModel):
    predictions_mt: list[float]


class RetrainResponse(BaseModel):
    status: str
    message: str
    rows_used_for_retraining: int

# ModelState class encapsulates the runtime state shared by API handlers, including the loaded model and predictor instance.
class ModelState:
    # Runtime state shared by API handlers.
    def __init__(self) -> None:
        self.model: Any | None = None
        self.predictor: EmissionsPredictor | None = None


state = ModelState()
state_lock = Lock()


def _is_render_runtime() -> bool:
    # Render sets service-level env vars in deployed containers.
    return bool(os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Initialize model artifacts once when application starts.
    with state_lock:
        _load_or_initialize()

    # Print environment-specific startup info for cleaner operational logs.
    if _is_render_runtime():
        render_url = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
        if render_url:
            print(f"\n[startup] Render deployment detected. API URL: {render_url}")
            print(f"[startup] Render Swagger UI: {render_url}/docs\n")
        else:
            print("\n[startup] Render deployment detected. API bound to $PORT.")
            print("[startup] Render Swagger UI: <your-render-service-url>/docs\n")
    else:
        print("\n[startup] Local development mode detected.")
        print("[startup] API URL: http://127.0.0.1:8000")
        print("[startup] Swagger UI: http://127.0.0.1:8000/docs\n")
    yield


app = FastAPI(
    title="East Africa CO2 Prediction API",
    description="Predict CO2 emissions and retrain the model when new labeled data arrives.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration with explicit allowed origins to satisfy stricter CORS policies in production environments. Wildcard origins are intentionally avoided to prevent security risks.
def _allowed_origins() -> list[str]:
    # CORS origins are configurable via env var for deployment environments.
    raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    # Intentionally not using wildcard origins to satisfy stricter CORS policy.
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Helper functions for building training frame, fitting model, and loading/initializing state. These are called from the lifespan context manager to ensure the API is ready to serve predictions immediately after startup, even if model artifacts were not previously saved.
def _build_base_training_frame() -> pd.DataFrame:
    return build_model_training_frame(DATA_PATH, EXTRA_ROWS_PATH)


def _fit_and_store(model_df: pd.DataFrame) -> None:
    # Delegate training + persistence to shared predictor helper.
    model, _ = fit_and_save_linear_model(model_df, MODEL_PATH, ARTIFACT_PATH)
    state.model = model
    # Recreate shared predictor so inference uses the same preprocessing implementation.
    state.predictor = EmissionsPredictor(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        artifact_path=ARTIFACT_PATH,
    )

# On application startup, we attempt to load the model and predictor from disk. If the model file is missing, we build the training frame from the original dataset (and any extra rows), fit a new model, persist it, and initialize the predictor. This ensures that the API is ready to serve predictions immediately after startup, even if the model artifacts were not previously saved.
def _load_or_initialize() -> None:
    model_df = _build_base_training_frame()

    if MODEL_PATH.exists():
        state.model = joblib.load(MODEL_PATH)
        state.predictor = EmissionsPredictor(
            data_path=DATA_PATH,
            model_path=MODEL_PATH,
            artifact_path=ARTIFACT_PATH,
        )
        return

    _fit_and_store(model_df)

# API endpoints are defined below. Each endpoint handler uses the shared model state with thread-safe access to ensure consistent behavior under concurrent requests. Validation errors and unexpected exceptions are handled gracefully with appropriate HTTP status codes and messages.
@app.get("/")
def root() -> dict[str, str]:
    # Service health/info endpoint.
    return {
        "status": "ok",
        "docs": "/docs",
        "predict_endpoint": "/predict",
        "retrain_endpoint": "/retrain",
    }

# Favicon requests are common from browsers but not relevant for API functionality, so we return a 204 No Content to avoid unnecessary logging noise.
@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    # Silence browser favicon requests without polluting API logs.
    return Response(status_code=204)

# Prediction endpoints use shared predictor instance with thread-safe access to ensure consistent preprocessing and model state across concurrent requests.
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictionRequest) -> PredictResponse:
    try:
        with state_lock:
            # Safely read shared predictor under concurrent requests.
            if state.predictor is None:
                raise RuntimeError("Predictor is not initialized")
            prediction = state.predictor.predict_one(payload.to_feature_row())
        return PredictResponse(prediction_mt=prediction)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

# Batch prediction endpoint allows scoring multiple records in one request while still using the shared predictor instance for consistent preprocessing and model state. Validation is performed on each record in the batch, and errors are handled gracefully.
@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictResponse:
    # Score multiple rows in one request using shared preprocessing/model state.
    try:
        with state_lock:
            if state.predictor is None:
                raise RuntimeError("Predictor is not initialized")
            predictions = state.predictor.predict_many([item.to_feature_row() for item in payload.rows])
        return BatchPredictResponse(predictions_mt=predictions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

# Retraining endpoint allows clients to submit new labeled records for model improvement. The endpoint supports optional persistence of new rows to a local CSV file, which is included in the training frame for future retraining. The retraining process rebuilds the training frame, refits the model, and updates the shared predictor instance to ensure that subsequent predictions use the latest model state. Validation errors and unexpected exceptions are handled gracefully with appropriate HTTP status codes and messages.
@app.post("/retrain", response_model=RetrainResponse)
def retrain(payload: RetrainRequest) -> RetrainResponse:
    # Optionally persist new labeled rows, then retrain and refresh runtime predictor.
    try:
        # Pydantic validation is complete before retraining logic runs.
        new_rows_df = pd.DataFrame([row.to_training_row() for row in payload.rows])

        if payload.persist_new_rows:
            EXTRA_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
            if EXTRA_ROWS_PATH.exists():
                existing = pd.read_csv(EXTRA_ROWS_PATH)
                merged = pd.concat([existing, new_rows_df], ignore_index=True)
            else:
                merged = new_rows_df
            merged.to_csv(EXTRA_ROWS_PATH, index=False)

        with state_lock:
            # Rebuild combined training frame, then refit and persist artifacts.
            model_df = _build_base_training_frame()
            _fit_and_store(model_df)

        return RetrainResponse(
            status="success",
            message="Model retrained and artifacts updated.",
            rows_used_for_retraining=len(model_df),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}") from exc
