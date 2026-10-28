"""
src/evaluation/metrics.py
--------------------------
Regression evaluation metrics for time-series forecasting.

All functions accept array-like y_true and y_pred and return floats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.
    Interpretable as the average magnitude of error in µg/m³.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.
    Penalises large errors more heavily than MAE because errors are squared.
    Useful when high PM2.5 spikes are particularly costly to miss.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²).
    R²=1 → perfect; R²=0 → model no better than predicting the mean;
    R²<0 → model worse than the mean (possible on test data).
    Interpret with caution for time-series: it measures variance explained,
    not temporal pattern capture.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.
    WAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100.
    Robust to near-zero values (unlike MAPE).  Returns percentage.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (y_true != 0)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.sum() == 0:
        return np.nan
    return float(np.sum(np.abs(yt - yp)) / np.sum(np.abs(yt)) * 100)


def evaluate_all(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model"
) -> Dict[str, float]:
    """Compute and return all metrics as a dict."""
    return {
        "model": model_name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "WAPE_%": wape(y_true, y_pred),
        "n_samples": int((~np.isnan(y_true)).sum()),
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """Pretty-print a metrics dict."""
    print(f"\n  {'─'*45}")
    print(f"  Model  : {metrics.get('model', '?')}")
    print(f"  Samples: {metrics.get('n_samples', '?')}")
    print(f"  MAE    : {metrics.get('MAE', float('nan')):.4f} µg/m³")
    print(f"  RMSE   : {metrics.get('RMSE', float('nan')):.4f} µg/m³")
    print(f"  R²     : {metrics.get('R2', float('nan')):.4f}")
    print(f"  WAPE   : {metrics.get('WAPE_%', float('nan')):.2f}%")
    print(f"  {'─'*45}\n")
