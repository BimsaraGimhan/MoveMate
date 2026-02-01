"""Baseline model using median rent by group."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TARGET_COL


@dataclass
class BaselineModel:
    group_medians: dict[tuple[str, int, str], float]
    global_median: float


def train_baseline(df_train: pd.DataFrame) -> BaselineModel:
    """Train baseline model: median by (suburb, bedrooms, dwelling_type)."""
    grouped = (
        df_train.groupby(["suburb", "bedrooms", "dwelling_type"])[TARGET_COL]
        .median()
        .dropna()
    )
    group_medians = {idx: float(value) for idx, value in grouped.items()}
    global_median = float(df_train[TARGET_COL].median())
    return BaselineModel(group_medians=group_medians, global_median=global_median)


def predict_baseline(model: BaselineModel, df: pd.DataFrame) -> np.ndarray:
    """Predict using the baseline model."""
    preds = []
    for _, row in df.iterrows():
        key = (row["suburb"], int(row["bedrooms"]), row["dwelling_type"])
        preds.append(model.group_medians.get(key, model.global_median))
    return np.array(preds, dtype=float)
