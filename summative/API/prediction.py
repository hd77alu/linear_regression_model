"""Predict East Africa CO2 emissions using the linear regression model.

This script imitates the notebook preprocessing so predictions are compatible
with the trained model saved in linear_regression/final_model.
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
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Total CO2 Emission excluding LUCF (Mt)"

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
    """Loads the linear regression model and prepares data the same way as the notebook."""

    def __init__(self, data_path: Path, model_path: Path) -> None:
        self.data_path = data_path
        self.model_path = model_path

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = StandardScaler()
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = ["Country"]
        self.training_columns: list[str] = []

        # Fit preprocessing artifacts once so all incoming requests reuse them.
        self._fit_preprocessing_reference()

    def _fit_preprocessing_reference(self) -> None:
        """Rebuild preprocessing artifacts from the training dataset."""
        # Load the same source data used during model development.
        df = pd.read_csv(self.data_path)
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

        # Mirror notebook cleanup for sparse columns.
        if "Fugitive Emissions (Mt)" in df_east.columns:
            df_east = df_east.drop(columns=["Fugitive Emissions (Mt)"])
            if "Fugitive Emissions (Mt)" in numeric_cols:
                numeric_cols.remove("Fugitive Emissions (Mt)")

        # Fill numeric missing values exactly as in training notebook.
        existing_numeric = [c for c in numeric_cols if c in df_east.columns]
        df_east[existing_numeric] = df_east[existing_numeric].fillna(
            df_east[existing_numeric].median(numeric_only=True)
        )

        # Repeat feature-engineering drops and derived column creation.
        fe_df = df_east.copy()
        drop_candidates = [
            "Code",
            "Total CO2 Emission including LUCF (Mt)",
            "Fugitive Emissions (Mt)",
            "GDP PER CAPITA PPP (USD)",
        ]
        for col in drop_candidates:
            if col in fe_df.columns:
                fe_df = fe_df.drop(columns=[col])

        fe_df["Population Density (people per km2)"] = (
            fe_df["Population"] / fe_df["Area (Km2)"]
        )

        final_features = [
            "Country",
            "Year",
            "Population",
            "Transportation (Mt)",
            "Manufacturing/Construction (Mt)",
            "Electricity/Heat (Mt)",
            "Building (Mt)",
        ]

        x = fe_df[final_features].copy()
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

    predictor = EmissionsPredictor(data_path=data_path, model_path=model_path)
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
