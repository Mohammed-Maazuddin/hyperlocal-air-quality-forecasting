"""
src/models/predict.py
----------------------
Inference pipeline: load the saved model artifact and produce a
24-hour-ahead PM2.5 forecast for a given issue timestamp.

USAGE:
    python -m src.models.predict --timestamp "2025-06-15 10:00:00"

HOW IT WORKS:
  1. Load the saved model and metadata.
  2. Load the processed hourly dataset.
  3. Slice the dataset up to (and including) the issue timestamp.
  4. Build features from that history using the same logic as training.
  5. Predict pm25 at issue_time + 24h.
  6. Print the forecast.

IMPORTANT CONSTRAINTS:
  - History ending at issue_time must contain enough rows to construct
    lag features (e.g. lag168h requires at least 168 prior observations).
  - No future observations (after issue_time) are used.
  - The model is NOT retrained during inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_features, get_feature_cols, LAG_HOURS, ROLLING_WINDOWS


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_model_and_meta(cfg: dict):
    """Load the saved model pipeline and its metadata."""
    artifacts_dir = PROJECT_ROOT / cfg["model"]["artifacts_dir"]
    artifact_path = artifacts_dir / cfg["model"]["artifact_filename"]
    meta_path = artifacts_dir / cfg["model"]["metadata_filename"]

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}\n"
            "  Run:  python -m src.models.train"
        )
    model = joblib.load(artifact_path)

    meta = {}
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)
    return model, meta


def load_history(cfg: dict, up_to: pd.Timestamp) -> pd.DataFrame:
    """Load processed data up to (and including) up_to timestamp."""
    proc_path = PROJECT_ROOT / cfg["data"]["processed_dir"] / cfg["data"]["processed_filename"]
    if not proc_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {proc_path}\n"
            "  Run:  python -m src.data.prepare_data"
        )
    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Slice up to issue time (inclusive)
    history = df[df.index <= up_to]
    return history


def predict_single(
    issue_time_str: str,
    model=None,
    meta: dict = None,
    cfg: dict = None,
) -> dict:
    """
    Produce a 24-hour-ahead PM2.5 forecast for a single issue timestamp.

    Parameters
    ----------
    issue_time_str : str
        Issue timestamp in any format parseable by pandas (UTC assumed).
    model, meta, cfg : optional pre-loaded objects.

    Returns
    -------
    dict with keys: issue_time, target_time, forecast_pm25
    """
    if cfg is None:
        cfg = load_config()
    if model is None:
        model, meta = load_model_and_meta(cfg)

    horizon = cfg["target"]["horizon_hours"]
    lag_hours = meta.get("lag_hours", cfg["features"]["lag_hours"])
    rolling_windows = meta.get("rolling_windows", cfg["features"]["rolling_windows"])
    feature_cols = meta.get("feature_cols", None)

    # ── Parse issue time ──────────────────────────────────────────────────────
    issue_time = pd.Timestamp(issue_time_str)
    if issue_time.tzinfo is None:
        issue_time = issue_time.tz_localize("UTC")
    else:
        issue_time = issue_time.tz_convert("UTC")

    # ── Load history ──────────────────────────────────────────────────────────
    history = load_history(cfg, up_to=issue_time)

    if len(history) == 0:
        raise ValueError(
            f"No historical data found up to {issue_time}.\n"
            f"  Dataset starts at: (check data/processed/)"
        )

    # Minimum required history: max lag + some buffer
    max_lag = max(lag_hours) if lag_hours else 168
    min_required = max_lag + 1
    if len(history) < min_required:
        raise ValueError(
            f"Insufficient history to construct lag features.\n"
            f"  Required: at least {min_required} hourly observations.\n"
            f"  Available: {len(history)} rows (up to {history.index.max()})."
        )

    # ── Build features ────────────────────────────────────────────────────────
    feat = build_features(
        history,
        lag_hours=lag_hours,
        rolling_windows=rolling_windows,
        horizon=horizon,
    )

    # Get the row corresponding to the issue time
    if issue_time not in feat.index:
        # Find nearest available timestamp
        avail = feat.index
        nearest = avail[avail <= issue_time]
        if nearest.empty:
            raise ValueError(f"No feature row found for or before {issue_time}.")
        issue_time = nearest[-1]
        print(f"  [predict] Issue time snapped to nearest available: {issue_time}")

    row = feat.loc[[issue_time]]

    # Select only the feature columns known to the model
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in row.columns]
        if missing:
            raise ValueError(f"Required feature columns missing from history: {missing}")
        X = row[feature_cols]
    else:
        X = row[[c for c in row.columns if c != "target"]]

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X)[0]
    target_time = issue_time + pd.Timedelta(hours=horizon)

    return {
        "issue_time": str(issue_time),
        "target_time": str(target_time),
        "forecast_pm25": float(y_pred),
        "model_type": meta.get("model_type", "unknown"),
        "horizon_hours": horizon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict PM2.5 24 hours ahead from a given issue timestamp."
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help='Issue timestamp, e.g. "2025-06-15 10:00:00" (UTC assumed)',
    )
    args = parser.parse_args()

    cfg = load_config()
    model, meta = load_model_and_meta(cfg)

    print(f"\n[predict] Issue timestamp : {args.timestamp}")
    print(f"[predict] Model           : {meta.get('model_type', 'unknown')}")
    print(f"[predict] Horizon         : {cfg['target']['horizon_hours']} hours")

    result = predict_single(args.timestamp, model=model, meta=meta, cfg=cfg)

    print("\n" + "=" * 50)
    print(f"  Issue time   : {result['issue_time']}")
    print(f"  Target time  : {result['target_time']}")
    print(f"  Forecast PM2.5 : {result['forecast_pm25']:.2f} µg/m³")
    print("=" * 50)
    print("\n  NOTE: This is a historical-data-based forecast for educational")
    print("  purposes only, not official public-health guidance.")


if __name__ == "__main__":
    main()
