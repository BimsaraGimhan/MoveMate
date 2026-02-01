"""Configuration constants for the MoveMate rent model project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "rents.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_SEED = 42

CATEGORICAL_COLS = ["suburb", "state", "dwelling_type", "postcode"]
NUMERIC_COLS = [
    "bedrooms",
    "year",
    "quarter",
    "suburb_rent_lag1",
    "rolling_suburb_median_4",
]
TARGET_COL = "weekly_rent"
DATE_COL = "date"
