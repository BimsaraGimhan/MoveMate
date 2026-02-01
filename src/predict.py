"""Inference helpers for the rent prediction model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostRegressor

from .baseline import BaselineModel, predict_baseline
from .config import ARTIFACTS_DIR, CATEGORICAL_COLS, DATE_COL, NUMERIC_COLS
from .features import add_lag_features, add_time_features


def _load_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(schema_path.read_text())


def _fill_missing_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    filled = df.copy()
    for col in CATEGORICAL_COLS:
        filled[col] = filled[col].fillna("unknown").astype("string")
    return filled


def _fill_missing_numerics(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    filled = df.copy()
    for col in NUMERIC_COLS:
        filled[col] = filled[col].fillna(medians[col])
    return filled


def _prepare_inference_features(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df, allow_missing_target=True)
    df = _fill_missing_categoricals(df)
    df = _fill_missing_numerics(df, medians)
    return df


def predict_from_dataframe(
    df: pd.DataFrame, artifacts_dir: Path = ARTIFACTS_DIR
) -> pd.DataFrame:
    schema = _load_schema(artifacts_dir / "feature_schema.json")
    medians = schema["numeric_medians"]
    feature_cols = schema["feature_cols"]

    model = CatBoostRegressor()
    model.load_model(artifacts_dir / "catboost_model.cbm")
    baseline: BaselineModel = joblib.load(artifacts_dir / "baseline.pkl")

    prepared = _prepare_inference_features(df, medians)
    catboost_preds = model.predict(prepared[feature_cols])
    baseline_preds = predict_baseline(baseline, prepared)

    result = prepared.copy()
    result["prediction_catboost"] = catboost_preds
    result["prediction_baseline"] = baseline_preds
    return result


def _record_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        DATE_COL: args.date,
        "state": args.state,
        "suburb": args.suburb,
        "postcode": str(args.postcode),
        "bedrooms": args.bedrooms,
        "dwelling_type": args.dwelling_type,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rent prediction inference")
    parser.add_argument("--csv", type=Path, help="Path to CSV with input rows")
    parser.add_argument("--json", type=str, help="JSON string with a single input row")

    parser.add_argument("--date", type=str, help="ISO date like 2025-12-31")
    parser.add_argument("--state", type=str)
    parser.add_argument("--suburb", type=str)
    parser.add_argument("--postcode", type=str)
    parser.add_argument("--bedrooms", type=int)
    parser.add_argument("--dwelling_type", type=str)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.json:
        record = json.loads(args.json)
        df = pd.DataFrame([record])
    else:
        required = [args.date, args.state, args.suburb, args.postcode, args.bedrooms, args.dwelling_type]
        if any(item is None for item in required):
            raise ValueError("Provide --csv, --json, or all of: --date --state --suburb --postcode --bedrooms --dwelling_type")
        df = pd.DataFrame([_record_from_args(args)])

    preds = predict_from_dataframe(df)
    print(preds[["prediction_catboost", "prediction_baseline"]].to_string(index=False))


if __name__ == "__main__":
    main()
