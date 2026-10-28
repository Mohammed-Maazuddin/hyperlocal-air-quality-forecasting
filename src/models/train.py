"""
src/models/train.py
--------------------
Trains and evaluates the following models for 24-hour-ahead PM2.5
forecasting:

  Baselines:
    1. Persistence  – predict pm25 at t as the forecast for t+24h
    2. Seasonal Naive – predict pm25 at t-24h (same-hour previous day)

  ML Models:
    3. Linear Regression
    4. Random Forest Regressor
    5. HistGradientBoostingRegressor

Chronological train/test split: first 75% for training, last 25% for
test.  No shuffling.  Preprocessing fitted only on training fold.

Usage:
    python -m src.models.train
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Allow importing from src.*
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_features, get_feature_cols, verify_target_alignment
from src.evaluation.metrics import evaluate_all, print_metrics
from src.evaluation.plots import (
    plot_actual_vs_predicted, plot_residuals, plot_model_comparison,
    plot_feature_importance, plot_error_by_hour,
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_ml_pipeline(model, use_scaler: bool = False) -> Pipeline:
    """Wrap a sklearn model in an imputer (and optional scaler) pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def baseline_persistence(X_test: pd.DataFrame) -> pd.Series:
    """Persistence: use pm25 at issue time t as the 24-h forecast."""
    if "pm25_lag0" not in X_test.columns:
        raise ValueError("Missing 'pm25_lag0' for persistence baseline.")
    return X_test["pm25_lag0"].copy()


def baseline_seasonal_naive(X_test: pd.DataFrame) -> pd.Series:
    """Seasonal naive: use pm25 from 24 hours before issue time (lag24h)."""
    if "pm25_lag24h" not in X_test.columns:
        raise ValueError("Missing 'pm25_lag24h' for seasonal naive baseline.")
    return X_test["pm25_lag24h"].copy()


def chronological_split(feat: pd.DataFrame, train_fraction: float):
    """Split feature matrix chronologically, return (train, test) DataFrames."""
    # Use only rows where target is non-null and at least lag features exist
    valid = feat.dropna(subset=["target"])
    n = len(valid)
    split_idx = int(n * train_fraction)
    train = valid.iloc[:split_idx]
    test = valid.iloc[split_idx:]
    return train, test


def main() -> None:
    cfg = load_config()
    seed = cfg["model"]["random_seed"]
    train_frac = cfg["model"]["train_fraction"]
    horizon = cfg["target"]["horizon_hours"]
    lag_hours = cfg["features"]["lag_hours"]
    rolling_windows = cfg["features"]["rolling_windows"]
    artifacts_dir = PROJECT_ROOT / cfg["model"]["artifacts_dir"]
    artifacts_dir.mkdir(exist_ok=True)

    fig_eval = PROJECT_ROOT / cfg["reports"]["figures"]["evaluation"]
    fig_interp = PROJECT_ROOT / cfg["reports"]["figures"]["interpretation"]
    fig_eval.mkdir(parents=True, exist_ok=True)
    fig_interp.mkdir(parents=True, exist_ok=True)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # ── Load processed data ───────────────────────────────────────────────────
    proc_path = PROJECT_ROOT / cfg["data"]["processed_dir"] / cfg["data"]["processed_filename"]
    if not proc_path.exists():
        print(f"[train] Processed file not found: {proc_path}", file=sys.stderr)
        print("  Run:  python -m src.data.prepare_data", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    print(f"[train] Loaded processed data: {df.shape}  {df.index.min()} → {df.index.max()}")

    # ── Verify target alignment ───────────────────────────────────────────────
    print("[train] Verifying target alignment…")
    verify_target_alignment(df, horizon=horizon)

    # ── Build feature matrix ──────────────────────────────────────────────────
    print("[train] Building feature matrix…")
    feat = build_features(df, lag_hours=lag_hours, rolling_windows=rolling_windows, horizon=horizon)
    feature_cols = get_feature_cols(feat)
    print(f"  Features ({len(feature_cols)}): {feature_cols[:10]}…")

    # ── Chronological split ───────────────────────────────────────────────────
    train, test = chronological_split(feat, train_frac)
    print(f"  Train: {len(train):,} rows  ({train.index.min().date()} → {train.index.max().date()})")
    print(f"  Test : {len(test):,} rows  ({test.index.min().date()} → {test.index.max().date()})")

    X_train = train[feature_cols]
    y_train = train["target"]
    X_test = test[feature_cols]
    y_test = test["target"]

    # ── Define models ─────────────────────────────────────────────────────────
    models_cfg = {
        "Persistence": None,  # baseline, handled separately
        "Seasonal Naive": None,  # baseline, handled separately
        "Linear Regression": make_ml_pipeline(LinearRegression(), use_scaler=True),
        "Random Forest": make_ml_pipeline(
            RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=seed)
        ),
        "HistGradientBoosting": make_ml_pipeline(
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          max_depth=6, random_state=seed)
        ),
    }

    all_metrics = []
    test_preds = pd.DataFrame({"actual": y_test})
    fitted_ml_models = {}

    for name, model in models_cfg.items():
        print(f"\n{'─'*55}")
        print(f"  Training / evaluating: {name}")

        if name == "Persistence":
            y_pred = baseline_persistence(X_test)
        elif name == "Seasonal Naive":
            y_pred = baseline_seasonal_naive(X_test)
        else:
            model.fit(X_train, y_train)
            y_pred = pd.Series(model.predict(X_test), index=X_test.index)
            fitted_ml_models[name] = model

        # Align indices
        common_idx = y_test.index.intersection(y_pred.index)
        yt = y_test.loc[common_idx]
        yp = y_pred.loc[common_idx]

        metrics = evaluate_all(yt.values, yp.values, model_name=name)
        all_metrics.append(metrics)
        print_metrics(metrics)

        # Save predictions
        test_preds[name] = y_pred

        # Evaluation plots
        yt_s = pd.Series(yt.values, index=common_idx)
        yp_s = pd.Series(yp.values, index=common_idx)
        plot_actual_vs_predicted(yt_s, yp_s, name, fig_eval)
        plot_residuals(yt_s, yp_s, name, fig_eval)
        plot_error_by_hour(yt_s, yp_s, name, fig_eval)

    # ── Model comparison table ────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = PROJECT_ROOT / cfg["reports"]["metrics_csv"]
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n[train] Model metrics saved: {metrics_csv}")
    print(metrics_df.to_string(index=False))

    # ── Test predictions ──────────────────────────────────────────────────────
    preds_csv = PROJECT_ROOT / cfg["reports"]["test_predictions_csv"]
    test_preds.to_csv(preds_csv)
    print(f"[train] Test predictions saved: {preds_csv}")

    # ── Model comparison plot ─────────────────────────────────────────────────
    plot_model_comparison(metrics_df, fig_eval)

    # ── Select best model (lowest MAE on test, excluding baselines) ───────────
    ml_metrics = metrics_df[~metrics_df["model"].isin(["Persistence", "Seasonal Naive"])]
    best_row = ml_metrics.loc[ml_metrics["MAE"].idxmin()]
    best_name = best_row["model"]
    print(f"\n[train] Best ML model (lowest test MAE): {best_name}  MAE={best_row['MAE']:.4f}")

    best_model = fitted_ml_models[best_name]

    # ── Feature importance for best model ────────────────────────────────────
    final_step = best_model.named_steps["model"]
    if hasattr(final_step, "feature_importances_"):
        importances = final_step.feature_importances_
        plot_feature_importance(feature_cols, importances, best_name, fig_interp)
        print(f"[train] Feature importance plot saved for {best_name}.")

    # ── Save best model artifact ──────────────────────────────────────────────
    artifact_path = artifacts_dir / cfg["model"]["artifact_filename"]
    joblib.dump(best_model, artifact_path)
    print(f"[train] Model saved: {artifact_path}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "model_type": best_name,
        "feature_cols": feature_cols,
        "horizon_hours": horizon,
        "train_period": {
            "start": str(train.index.min()),
            "end": str(train.index.max()),
            "n_samples": int(len(train)),
        },
        "test_period": {
            "start": str(test.index.min()),
            "end": str(test.index.max()),
            "n_samples": int(len(test)),
        },
        "test_metrics": {k: v for k, v in best_row.items() if k != "model"},
        "all_metrics": all_metrics,
        "lag_hours": lag_hours,
        "rolling_windows": rolling_windows,
        "random_seed": seed,
    }
    meta_path = artifacts_dir / cfg["model"]["metadata_filename"]
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[train] Metadata saved: {meta_path}")
    print("\n[train] Training complete.")


if __name__ == "__main__":
    main()
