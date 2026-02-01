"""Feature engineering helpers."""

from __future__ import annotations

import pandas as pd

from .config import DATE_COL, TARGET_COL


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year and quarter features derived from the date column."""
    enriched = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(enriched[DATE_COL]):
        enriched[DATE_COL] = pd.to_datetime(enriched[DATE_COL], errors="coerce")

    enriched["year"] = enriched[DATE_COL].dt.year
    enriched["quarter"] = enriched[DATE_COL].dt.quarter
    return enriched


def add_lag_features(df: pd.DataFrame, allow_missing_target: bool = False) -> pd.DataFrame:
    """Add lagged rent and rolling median features per suburb.

    If allow_missing_target is True and weekly_rent is missing, lag features
    are filled with NaN to support inference-only rows.
    """
    enriched = df.copy()

    if TARGET_COL not in enriched.columns:
        if allow_missing_target:
            enriched["suburb_rent_lag1"] = pd.NA
            enriched["rolling_suburb_median_4"] = pd.NA
            return enriched
        raise ValueError("weekly_rent column is required to compute lag features")

    enriched = enriched.sort_values(["suburb", DATE_COL])
    grouped = enriched.groupby("suburb", sort=False)[TARGET_COL]

    lag1 = grouped.shift(1)
    rolling = (
        lag1.groupby(enriched["suburb"], sort=False)
        .rolling(window=4, min_periods=1)
        .median()
        .reset_index(level=0, drop=True)
    )

    enriched["suburb_rent_lag1"] = lag1
    enriched["rolling_suburb_median_4"] = rolling
    return enriched
