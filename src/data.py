"""Data loading and validation utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DATE_COL, TARGET_COL

REQUIRED_COLUMNS = [
    DATE_COL,
    "state",
    "suburb",
    "postcode",
    "bedrooms",
    "dwelling_type",
    TARGET_COL,
]


class DataValidationError(ValueError):
    """Raised when input data fails validation."""


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV data from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}. Expected columns: {list(columns)}"
        )


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, coerce types, and drop invalid target rows."""
    _require_columns(df, REQUIRED_COLUMNS)

    cleaned = df.copy()

    # Coerce date to datetime; errors become NaT for later handling.
    cleaned[DATE_COL] = pd.to_datetime(cleaned[DATE_COL], errors="coerce")

    # Coerce numeric columns.
    cleaned["bedrooms"] = pd.to_numeric(cleaned["bedrooms"], errors="coerce")
    cleaned[TARGET_COL] = pd.to_numeric(cleaned[TARGET_COL], errors="coerce")

    # Ensure postcode is string-like.
    cleaned["postcode"] = cleaned["postcode"].astype("string")

    # Drop rows without target or date.
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=[TARGET_COL, DATE_COL])
    after = len(cleaned)
    if after < before:
        logging.info("Dropped %d rows with missing target or date", before - after)

    return cleaned
