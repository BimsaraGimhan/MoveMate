"""Model training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from .baseline import BaselineModel, predict_baseline, train_baseline
from .config import (
    ARTIFACTS_DIR,
    CATEGORICAL_COLS,
    DATA_PATH,
    DATE_COL,
    NUMERIC_COLS,
    RANDOM_SEED,
    TARGET_COL,
)
from .data import DataValidationError, clean_and_validate, load_data
from .features import add_lag_features, add_time_features


EXPECTED_COLUMNS = [
    DATE_COL,
    "state",
    "suburb",
    "postcode",
    "bedrooms",
    "dwelling_type",
    TARGET_COL,
]


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _fill_missing_categoricals(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    filled = df.copy()
    for col in cols:
        filled[col] = filled[col].fillna("unknown").astype("string")
    return filled


def _compute_numeric_medians(df: pd.DataFrame, cols: Iterable[str]) -> dict[str, float]:
    return {col: float(df[col].median()) for col in cols}


def _fill_missing_numerics(
    df: pd.DataFrame, cols: Iterable[str], medians: dict[str, float]
) -> pd.DataFrame:
    filled = df.copy()
    for col in cols:
        filled[col] = filled[col].fillna(medians[col])
    return filled


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    df = df[df["suburb_rent_lag1"].notna()].copy()
    return df


def _split_train_test_by_date(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(df[DATE_COL].unique()))
    if len(unique_dates) < 2:
        raise ValueError("Need at least 2 unique dates for time-based splitting")

    test_size = max(1, int(len(unique_dates) * test_fraction))
    test_dates = set(unique_dates[-test_size:])
    is_test = df[DATE_COL].isin(test_dates)
    train_df = df.loc[~is_test].copy()
    test_df = df.loc[is_test].copy()
    return train_df, test_df


def _time_series_cv_splits(df: pd.DataFrame, n_splits: int = 5):
    unique_dates = np.array(sorted(df[DATE_COL].unique()))
    tss = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tss.split(unique_dates):
        train_dates = set(unique_dates[train_idx])
        val_dates = set(unique_dates[val_idx])
        train_df = df[df[DATE_COL].isin(train_dates)].copy()
        val_df = df[df[DATE_COL].isin(val_dates)].copy()
        yield train_df, val_df


def _train_catboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    target_col: str,
) -> CatBoostRegressor:
    feature_cols = categorical_cols + numeric_cols
    cat_indices = [feature_cols.index(col) for col in categorical_cols]

    train_pool = Pool(train_df[feature_cols], train_df[target_col], cat_features=cat_indices)
    val_pool = Pool(val_df[feature_cols], val_df[target_col], cat_features=cat_indices)

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        random_seed=RANDOM_SEED,
        verbose=200,
    )
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
    return model


def _evaluate_model(model: CatBoostRegressor, df: pd.DataFrame, feature_cols: list[str]) -> float:
    preds = model.predict(df[feature_cols])
    return float(mean_absolute_error(df[TARGET_COL], preds))


def generate_synthetic_sample(path: Path, rows: int = 200) -> None:
    """Generate a tiny synthetic dataset to help with first run."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2023-01-31", periods=12, freq="M")
    suburbs = ["Perth", "Fremantle", "Subiaco", "Sydney", "Melbourne"]
    states = {"Perth": "WA", "Fremantle": "WA", "Subiaco": "WA", "Sydney": "NSW", "Melbourne": "VIC"}
    dwelling_types = ["house", "unit", "apartment"]

    data = []
    for _ in range(rows):
        suburb = rng.choice(suburbs)
        bedrooms = int(rng.integers(1, 5))
        dwelling_type = rng.choice(dwelling_types)
        date = rng.choice(dates)
        base = 350 + bedrooms * 120 + (0 if dwelling_type == "unit" else 80)
        noise = rng.normal(0, 40)
        rent = max(150, base + noise + rng.normal(0, 30))
        data.append(
            {
                "date": date.date().isoformat(),
                "state": states[suburb],
                "suburb": suburb,
                "postcode": str(int(rng.integers(2000, 7000))),
                "bedrooms": bedrooms,
                "dwelling_type": dwelling_type,
                "weekly_rent": round(float(rent), 2),
            }
        )

    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(data_path: Path = DATA_PATH, generate_sample: bool = False) -> None:
    _setup_logging()
    logging.info("Starting training pipeline")

    if generate_sample and not data_path.exists():
        logging.info("Generating synthetic sample data at %s", data_path)
        generate_synthetic_sample(data_path)

    try:
        raw = load_data(data_path)
        cleaned = clean_and_validate(raw)
    except FileNotFoundError:
        logging.error("Data file not found: %s", data_path)
        logging.error("Expected columns: %s", EXPECTED_COLUMNS)
        logging.error("You can generate a small synthetic dataset with --generate-sample")
        return
    except DataValidationError as exc:
        logging.error("Data validation error: %s", exc)
        return

    features = _prepare_features(cleaned)

    train_df, test_df = _split_train_test_by_date(features, test_fraction=0.2)
    logging.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    # Cross-validation on training data only.
    cv_baseline = []
    cv_catboost = []
    for split_idx, (cv_train, cv_val) in enumerate(_time_series_cv_splits(train_df, n_splits=5), start=1):
        logging.info("CV split %d: train=%d val=%d", split_idx, len(cv_train), len(cv_val))

        cv_train = _fill_missing_categoricals(cv_train, CATEGORICAL_COLS)
        cv_val = _fill_missing_categoricals(cv_val, CATEGORICAL_COLS)
        medians = _compute_numeric_medians(cv_train, NUMERIC_COLS)
        cv_train = _fill_missing_numerics(cv_train, NUMERIC_COLS, medians)
        cv_val = _fill_missing_numerics(cv_val, NUMERIC_COLS, medians)

        baseline = train_baseline(cv_train)
        baseline_preds = predict_baseline(baseline, cv_val)
        cv_baseline.append(mean_absolute_error(cv_val[TARGET_COL], baseline_preds))

        model = _train_catboost(cv_train, cv_val, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL)
        feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
        cv_catboost.append(_evaluate_model(model, cv_val, feature_cols))

    # Prepare final training and holdout evaluation.
    train_df = _fill_missing_categoricals(train_df, CATEGORICAL_COLS)
    test_df = _fill_missing_categoricals(test_df, CATEGORICAL_COLS)
    medians = _compute_numeric_medians(train_df, NUMERIC_COLS)
    train_df = _fill_missing_numerics(train_df, NUMERIC_COLS, medians)
    test_df = _fill_missing_numerics(test_df, NUMERIC_COLS, medians)

    # Create an internal validation split from the tail of training dates.
    train_inner, val_inner = _split_train_test_by_date(train_df, test_fraction=0.1)
    model = _train_catboost(train_inner, val_inner, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL)

    baseline = train_baseline(train_df)
    baseline_preds = predict_baseline(baseline, test_df)
    baseline_holdout_mae = float(mean_absolute_error(test_df[TARGET_COL], baseline_preds))

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    catboost_holdout_mae = _evaluate_model(model, test_df, feature_cols)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "catboost_model.cbm"
    baseline_path = ARTIFACTS_DIR / "baseline.pkl"
    schema_path = ARTIFACTS_DIR / "feature_schema.json"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    model.save_model(model_path)
    joblib.dump(baseline, baseline_path)

    feature_schema = {
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "feature_cols": feature_cols,
        "numeric_medians": medians,
        "target_col": TARGET_COL,
        "date_col": DATE_COL,
    }
    schema_path.write_text(json.dumps(feature_schema, indent=2))

    metrics = {
        "cv": {
            "baseline_mae": float(np.mean(cv_baseline)),
            "catboost_mae": float(np.mean(cv_catboost)),
            "baseline_mae_splits": [float(x) for x in cv_baseline],
            "catboost_mae_splits": [float(x) for x in cv_catboost],
        },
        "holdout": {
            "baseline_mae": baseline_holdout_mae,
            "catboost_mae": catboost_holdout_mae,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logging.info("Saved model to %s", model_path)
    logging.info("Saved baseline to %s", baseline_path)
    logging.info("Saved metrics to %s", metrics_path)

    # Feature importance summary.
    importances = model.get_feature_importance()
    importance_table = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    logging.info("Top 20 feature importances:\n%s", importance_table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rent prediction models")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to rents CSV (default: data/rents.csv)",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a small synthetic dataset if data is missing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(data_path=args.data_path, generate_sample=args.generate_sample)
