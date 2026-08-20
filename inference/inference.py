"""Run Random Forest inference and mandatory anomaly detection.

Example:
    python inference.py \
        --model models/random_forest_regressor.joblib \
        --data data/inference.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "frequency",
    "attack-angle",
    "chord-length",
    "free-stream-velocity",
    "suction-side-displacement-thickness",
]

DEFAULT_OBSERVED_COLUMN = "observed_sound_pressure_db"
ORIGINAL_TARGET_COLUMN = "scaled-sound-pressure"

# Random Forest threshold obtained from the clean validation residuals.
# Replace this with additional decimal places if your notebook displays them.
DEFAULT_ANOMALY_THRESHOLD_DB = 6.137462


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet inference dataset."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("The inference dataset must be a .csv or .parquet file")


def resolve_observed_column(data: pd.DataFrame, requested_column: str) -> str:
    """Find the observed sound-pressure column required for anomaly detection."""

    if requested_column in data.columns:
        return requested_column

    if (
        requested_column == DEFAULT_OBSERVED_COLUMN
        and ORIGINAL_TARGET_COLUMN in data.columns
    ):
        return ORIGINAL_TARGET_COLUMN

    raise ValueError(
        "Anomaly detection requires an observed sound-pressure column. "
        f"Expected '{requested_column}' or '{ORIGINAL_TARGET_COLUMN}'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Random Forest inference and anomaly detection"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to random_forest_regressor.joblib",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Inference CSV or Parquet file",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory (default: results)",
    )
    parser.add_argument(
        "--output-name",
        default="predictions.csv",
        help="Output filename (default: predictions.csv)",
    )
    parser.add_argument(
        "--observed-column",
        default=DEFAULT_OBSERVED_COLUMN,
        help=(
            "Observed dB column (default: observed_sound_pressure_db). "
            "The original scaled-sound-pressure name is also accepted."
        ),
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=DEFAULT_ANOMALY_THRESHOLD_DB,
        help=(
            "Frozen anomaly threshold learned from clean validation residuals "
            f"(default: {DEFAULT_ANOMALY_THRESHOLD_DB})"
        ),
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Inference dataset not found: {args.data}")
    if not np.isfinite(args.threshold_db) or args.threshold_db <= 0:
        raise ValueError("--threshold-db must be a positive finite number")

    data = load_dataset(args.data)
    observed_column = resolve_observed_column(data, args.observed_column)

    required_columns = FEATURE_COLUMNS + [observed_column]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    numeric_data = data[required_columns].apply(pd.to_numeric, errors="coerce")
    invalid_columns = numeric_data.columns[numeric_data.isna().any()].tolist()
    if invalid_columns:
        raise ValueError(
            "Required columns contain missing or non-numeric values: "
            f"{invalid_columns}"
        )

    model = joblib.load(args.model)

    # Random Forest was trained using the original, unscaled feature columns.
    features = numeric_data[FEATURE_COLUMNS]
    expected_sound_pressure_db = np.asarray(
        model.predict(features), dtype=float
    ).reshape(-1)

    observed_sound_pressure_db = numeric_data[observed_column].to_numpy(
        dtype=float
    )
    residual_db = np.abs(
        observed_sound_pressure_db - expected_sound_pressure_db
    )

    results = data.copy()
    results["expected_sound_pressure_db"] = expected_sound_pressure_db
    results["residual_db"] = residual_db
    results["anomaly_threshold_db"] = args.threshold_db
    results["predicted_anomaly"] = (
        residual_db > args.threshold_db
    ).astype(int)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / args.output_name
    results.to_csv(output_path, index=False)

    anomaly_count = int(results["predicted_anomaly"].sum())
    print(f"Saved {len(results)} predictions to {output_path}")
    print(f"Detected {anomaly_count} anomalies")


if __name__ == "__main__":
    main()