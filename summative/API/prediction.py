"""Prediction and shared model utilities for East Africa CO2 emissions.

How notebook logic is used in this script:
- The training notebook defines the core preprocessing pipeline: East Africa filtering,
    numeric coercion, sparse-column removal, median imputation, feature selection,
    scaling numeric fields, and one-hot encoding country.
- `build_model_training_frame` re-implements those same data-cleaning and feature
    engineering steps so retraining in the API remains aligned with notebook behavior.
- `fit_and_save_linear_model` applies the same final feature set and preprocessing
    pattern used during notebook training, then saves artifacts (`scaler`, encoded
    training columns, numeric feature list) alongside the model.
- At inference time, `EmissionsPredictor` uses saved artifacts first so request
    preprocessing matches the trained model exactly. If artifacts are unavailable,
    it rebuilds preprocessing references from the notebook-equivalent pipeline.

Function and class purpose map:
- build_model_training_frame: Build cleaned/model-ready training data from raw CSV,
    with optional extra labeled rows.
- fit_and_save_linear_model: Train LinearRegression and persist both model and
    preprocessing artifacts.
- PredictionInput.from_payload: Validate and coerce raw request dictionaries.
- PredictionInput.to_model_row: Convert validated input into model column names.
- EmissionsPredictor: Load model and transform inputs consistently for inference.
- EmissionsPredictor._load_preprocessing_artifacts: Load persisted scaler/schema.
- EmissionsPredictor._fit_preprocessing_reference: Rebuild fallback scaler/schema.
- EmissionsPredictor._prepare_input: Validate, scale, one-hot encode, and align
    payload rows to training columns.
- EmissionsPredictor.predict_one: Score one input row.
- EmissionsPredictor.predict_many: Score multiple input rows.
- _read_payload: Read CLI input from inline JSON or JSON file.
- main: CLI entrypoint for single or batch prediction output.
"""

# import required libraries
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Define constants for column names and features used in the model.
TARGET_COL = "Total CO2 Emission excluding LUCF (Mt)"
FINAL_FEATURES = [
    "Country",
    "Year",
    "Population",
    "Transportation (Mt)",
    "Manufacturing/Construction (Mt)",
    "Electricity/Heat (Mt)",
    "Building (Mt)",
]

# Raw request schema expected by the predictor endpoint/script.
REQUIRED_INPUT_FIELDS = [
    "Country",
    "Year",
    "Population",
    "Transportation (Mt)",
    "Manufacturing/Construction (Mt)",
    "Electricity/Heat (Mt)",
    "Building (Mt)",
]


def build_model_training_frame(data_path: Path, extra_rows_path: Path | None = None) -> pd.DataFrame:
    """Build the model-ready training frame using notebook-equivalent preprocessing."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df = df.replace(["N/A", "na", "NA", ""], np.nan)
    df_east = df[df["Sub-Region"] == "Eastern Africa"].copy()

    numeric_cols = [
        "Year",
        "Population",
        "GDP PER CAPITA (USD)",
        "GDP PER CAPITA PPP (USD)",
        "Area (Km2)",
        "Transportation (Mt)",
        "Other Fuel Combustion (Mt)",
        "Manufacturing/Construction (Mt)",
        "Land-Use Change and Forestry (Mt)",
        "Industrial Processes (Mt)",
        "Fugitive Emissions (Mt)",
        "Energy (Mt)",
        "Electricity/Heat (Mt)",
        "Bunker Fuels (Mt)",
        "Building (Mt)",
        TARGET_COL,
    ]

    for col in numeric_cols:
        if col in df_east.columns:
            df_east[col] = pd.to_numeric(df_east[col], errors="coerce")

    if "Fugitive Emissions (Mt)" in df_east.columns:
        df_east = df_east.drop(columns=["Fugitive Emissions (Mt)"])
        if "Fugitive Emissions (Mt)" in numeric_cols:
            numeric_cols.remove("Fugitive Emissions (Mt)")

    predictor_cols = [c for c in numeric_cols if c != TARGET_COL and c in df_east.columns]
    country_medians = df_east.groupby("Country")[predictor_cols].transform("median")
    df_east[predictor_cols] = df_east[predictor_cols].fillna(country_medians)
    df_east[predictor_cols] = df_east[predictor_cols].fillna(
        df_east[predictor_cols].median(numeric_only=True)
    )
    df_east = df_east.dropna(subset=[TARGET_COL])

    fe_df = df_east.copy()
    for col in [
        "Code",
        "Total CO2 Emission including LUCF (Mt)",
        "Fugitive Emissions (Mt)",
        "GDP PER CAPITA PPP (USD)",
    ]:
        if col in fe_df.columns:
            fe_df = fe_df.drop(columns=[col])

    fe_df["Population Density (people per km2)"] = fe_df["Population"] / fe_df["Area (Km2)"]
    model_df = fe_df[FINAL_FEATURES + [TARGET_COL]].copy()

    if extra_rows_path is not None and extra_rows_path.exists():
        extra_df = pd.read_csv(extra_rows_path)
        missing = [c for c in (FINAL_FEATURES + [TARGET_COL]) if c not in extra_df.columns]
        if missing:
            raise ValueError(f"Extra training rows missing columns: {missing}")
        extra_df = extra_df[FINAL_FEATURES + [TARGET_COL]].copy()
        model_df = pd.concat([model_df, extra_df], ignore_index=True)

    return model_df


def fit_and_save_linear_model(
    model_df: pd.DataFrame,
    model_path: Path,
    artifact_path: Path,
) -> tuple[LinearRegression, dict[str, Any]]:
    """Train the linear model and persist model + preprocessing artifacts."""
    x = model_df[FINAL_FEATURES].copy()
    y = model_df[TARGET_COL].copy()

    scaler = StandardScaler()
    x_scaled = x.copy()
    numeric_features = [c for c in FINAL_FEATURES if c != "Country"]
    x_scaled[numeric_features] = scaler.fit_transform(x[numeric_features])
    x_model = pd.get_dummies(x_scaled, columns=["Country"], drop_first=True)

    model = LinearRegression()
    model.fit(x_model, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    artifacts: dict[str, Any] = {
        "model": model,
        "scaler": scaler,
        "training_columns": x_model.columns.tolist(),
        "numeric_features": numeric_features,
        "final_features": FINAL_FEATURES,
    }
    joblib.dump(artifacts, artifact_path)

    return model, artifacts


@dataclass(frozen=True)
class PredictionInput:
    """Strongly typed input payload for prediction requests."""

    country: str
    year: int
    population: float
    transportation_mt: float
    manufacturing_construction_mt: float
    electricity_heat_mt: float
    building_mt: float

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PredictionInput":
        required = set(REQUIRED_INPUT_FIELDS)
        provided = set(payload.keys())

        missing = sorted(required - provided)
        if missing:
            raise ValueError(f"Missing required input field(s): {missing}")

        unknown = sorted(provided - required)
        if unknown:
            raise ValueError(f"Unknown input field(s): {unknown}")

        country = str(payload["Country"]).strip()
        if not country:
            raise ValueError("Country must be a non-empty string")

        try:
            year = int(payload["Year"])
            population = float(payload["Population"])
            transportation_mt = float(payload["Transportation (Mt)"])
            manufacturing_construction_mt = float(payload["Manufacturing/Construction (Mt)"])
            electricity_heat_mt = float(payload["Electricity/Heat (Mt)"])
            building_mt = float(payload["Building (Mt)"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Numeric fields must contain valid numeric values") from exc

        if year <= 0:
            raise ValueError("Year must be a positive integer")
        if population < 0:
            raise ValueError("Population must be non-negative")
        if transportation_mt < 0 or manufacturing_construction_mt < 0 or electricity_heat_mt < 0 or building_mt < 0:
            raise ValueError("Emission fields must be non-negative")

        return cls(
            country=country,
            year=year,
            population=population,
            transportation_mt=transportation_mt,
            manufacturing_construction_mt=manufacturing_construction_mt,
            electricity_heat_mt=electricity_heat_mt,
            building_mt=building_mt,
        )

    def to_model_row(self) -> dict[str, Any]:
        """Convert typed payload into the model's expected column names."""
        return {
            "Country": self.country,
            "Year": self.year,
            "Population": self.population,
            "Transportation (Mt)": self.transportation_mt,
            "Manufacturing/Construction (Mt)": self.manufacturing_construction_mt,
            "Electricity/Heat (Mt)": self.electricity_heat_mt,
            "Building (Mt)": self.building_mt,
        }

class EmissionsPredictor:
    """Load the model and apply artifact-aligned preprocessing for inference."""

    def __init__(self, data_path: Path, model_path: Path, artifact_path: Path | None = None) -> None:
        self.data_path = data_path
        self.model_path = model_path
        self.artifact_path = artifact_path

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = StandardScaler()
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = ["Country"]
        self.training_columns: list[str] = []

        # Prefer persisted preprocessing artifacts so inference exactly matches training.
        if self.artifact_path is not None and self.artifact_path.exists():
            self._load_preprocessing_artifacts()
        else:
            # Fallback for first-run or local script usage without artifacts.
            self._fit_preprocessing_reference()

    def _load_preprocessing_artifacts(self) -> None:
        """Load scaler/schema artifacts generated during training."""
        artifacts = joblib.load(self.artifact_path)

        scaler = artifacts.get("scaler")
        training_columns = artifacts.get("training_columns")
        numeric_features = artifacts.get("numeric_features")

        if scaler is None or training_columns is None or numeric_features is None:
            raise ValueError("Artifact file is missing required preprocessing keys")

        self.scaler = scaler
        self.training_columns = list(training_columns)
        self.numeric_features = list(numeric_features)

    def _fit_preprocessing_reference(self) -> None:
        """Rebuild preprocessing artifacts from the training dataset."""
        model_df = build_model_training_frame(self.data_path)
        x = model_df[FINAL_FEATURES].copy()
        self.numeric_features = [c for c in x.columns if c not in self.categorical_features]

        # Fit scaler on training-reference data so runtime inputs use same transform.
        x_scaled = x.copy()
        x_scaled[self.numeric_features] = self.scaler.fit_transform(x[self.numeric_features])

        # Capture one-hot output schema to align future payloads safely.
        x_model = pd.get_dummies(
            x_scaled,
            columns=self.categorical_features,
            drop_first=True,
        )
        self.training_columns = x_model.columns.tolist()

    def _prepare_input(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Convert raw input rows into model-ready columns."""
        # Validate schema early for clear API error messages.
        missing = [col for col in REQUIRED_INPUT_FIELDS if col not in rows.columns]
        if missing:
            raise ValueError(f"Missing required input field(s): {missing}")

        rows = rows[REQUIRED_INPUT_FIELDS].copy()

        for col in self.numeric_features:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")

        # Reject payloads with malformed or missing numeric values.
        if rows[self.numeric_features].isna().any().any():
            raise ValueError(
                "Input has non-numeric or missing values in numeric fields: "
                f"{self.numeric_features}"
            )

        # Apply same scaler learned from reference training data.
        rows_scaled = rows.copy()
        rows_scaled[self.numeric_features] = self.scaler.transform(rows[self.numeric_features])

        # One-hot encode country values to match model input expectations.
        rows_model = pd.get_dummies(
            rows_scaled,
            columns=self.categorical_features,
            drop_first=True,
        )

        # Align to training columns (unknown categories become 0 across known dummies)
        rows_model = rows_model.reindex(columns=self.training_columns, fill_value=0.0)
        return rows_model

    def predict_one(self, payload: dict[str, Any]) -> float:
        # Single-row helper for API endpoints that score one request at a time.
        typed_payload = PredictionInput.from_payload(payload)
        prepared = self._prepare_input(pd.DataFrame([typed_payload.to_model_row()]))
        prediction = float(self.model.predict(prepared)[0])
        return prediction

    def predict_many(self, payloads: list[dict[str, Any]]) -> list[float]:
        # Batch helper for bulk scoring.
        typed_payloads = [PredictionInput.from_payload(payload) for payload in payloads]
        prepared = self._prepare_input(pd.DataFrame([item.to_model_row() for item in typed_payloads]))
        return [float(v) for v in self.model.predict(prepared)]


def _read_payload(input_json: str | None, input_file: str | None) -> Any:
    # Support both direct JSON and file-based payloads for local/API testing.
    if input_json:
        return json.loads(input_json)
    if input_file:
        return json.loads(Path(input_file).read_text(encoding="utf-8"))

    # Default example payload for quick local testing
    return {
        "Country": "Kenya",
        "Year": 2020,
        "Population": 53771300,
        "Transportation (Mt)": 5.1,
        "Manufacturing/Construction (Mt)": 2.3,
        "Electricity/Heat (Mt)": 3.8,
        "Building (Mt)": 1.7,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict CO2 emissions using saved best model")
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="JSON object or JSON array with feature payload(s)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to JSON file containing one object or a list of objects",
    )
    args = parser.parse_args()

    # Resolve sibling project folders from the API directory.
    base_dir = Path(__file__).resolve().parent
    linear_regression_dir = base_dir.parent / "linear_regression"
    data_path = linear_regression_dir / "data" / "africa-co2-emissions.csv"
    model_path = linear_regression_dir / "final_model" / "best_linear_regression_model.joblib"
    artifact_path = linear_regression_dir / "final_model" / "fastapi_model_artifacts.joblib"

    predictor = EmissionsPredictor(
        data_path=data_path,
        model_path=model_path,
        artifact_path=artifact_path,
    )
    payload = _read_payload(args.input_json, args.input_file)

    if isinstance(payload, list):
        # Return batch output under a dedicated key for easy API response mapping.
        predictions = predictor.predict_many(payload)
        print(json.dumps({"predictions_mt": predictions}, indent=2))
    elif isinstance(payload, dict):
        # Return single prediction output.
        prediction = predictor.predict_one(payload)
        print(json.dumps({"prediction_mt": prediction}, indent=2))
    else:
        raise ValueError("Input payload must be a JSON object or a JSON array of objects")


if __name__ == "__main__":
    main()
