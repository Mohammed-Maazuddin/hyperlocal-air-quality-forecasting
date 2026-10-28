"""
src/features/build_features.py
--------------------------------
Builds the modelling feature matrix for 24-hour-ahead PM2.5 forecasting.

FORECASTING CONSTRAINT:
  Target  : pm25 at time t + 24h
  Features: only information available at time t (the "issue time")

  This means:
  - Lag features are pm25 measured at t, t-1h, t-3h, …
  - Rolling features are computed over the window ending at t
  - Weather/pollutant co-features use values AT t (not t+1, t+2, …)
  - Calendar features derive from t+24h (the forecast target time)

TARGET ALIGNMENT TEST is implemented as `verify_target_alignment()`.

Usage:
    python -m src.features.build_features
    (also importable by training/inference scripts)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Configurable defaults (overridden by config.yaml) ────────────────────────
LAG_HOURS: List[int] = [1, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS: List[int] = [6, 24]
HORIZON: int = 24  # forecast horizon in hours

# Weather/pollutant co-features to include if present in data
CANDIDATE_COFEATURES: List[str] = [
    "pm10", "no", "no2", "nox", "nh3", "so2", "co", "ozone",
    "temp_c", "rh", "ws", "wd", "rf", "sr", "bp",
]


def build_features(
    df: pd.DataFrame,
    lag_hours: List[int] = LAG_HOURS,
    rolling_windows: List[int] = ROLLING_WINDOWS,
    horizon: int = HORIZON,
    cofeature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build the feature matrix from a chronologically sorted hourly DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly resampled data with DatetimeIndex (UTC-aware) and a 'pm25' column.
    lag_hours : list of int
        Lags to compute for pm25 (relative to issue time t).
    rolling_windows : list of int
        Rolling mean/std windows (hours) ending at t.
    horizon : int
        Forecast horizon in hours (default 24).
    cofeature_cols : list of str or None
        Weather/pollutant columns to include.  If None, uses CANDIDATE_COFEATURES
        intersected with df.columns.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned so that index = issue time t,
        and column 'target' = pm25 at t + horizon hours.
        Rows where the target is NaN are retained (the caller can decide
        whether to drop them).
    """
    if "pm25" not in df.columns:
        raise ValueError("DataFrame must contain a 'pm25' column.")

    df = df.sort_index()

    feat = pd.DataFrame(index=df.index)

    # ── A. Target (pm25 at t + horizon) ───────────────────────────────────────
    # We shift pm25 backward by `horizon` positions so that feat.loc[t, "target"]
    # = the pm25 value observed horizon hours after t.
    feat["target"] = df["pm25"].shift(-horizon)

    # ── B. Lag features (pm25 at t-k, k>=0) ──────────────────────────────────
    # lag=0 → current pm25 at issue time t
    feat["pm25_lag0"] = df["pm25"]
    for k in lag_hours:
        feat[f"pm25_lag{k}h"] = df["pm25"].shift(k)

    # ── C. Rolling features (computed over window ending AT t) ─────────────────
    # We must NOT shift before rolling here because we want the window [t-w+1, t].
    # pandas rolling(w) with min_periods already gives this correctly on a sorted
    # series.
    for w in rolling_windows:
        feat[f"pm25_roll{w}h_mean"] = df["pm25"].rolling(w, min_periods=max(1, w // 2)).mean()

    # Rolling standard deviation (24h window)
    feat["pm25_roll24h_std"] = df["pm25"].rolling(24, min_periods=12).std()

    # ── D. Calendar features derived from FORECAST TARGET time (t + horizon) ──
    target_time = df.index + pd.Timedelta(hours=horizon)
    feat["target_hour"] = target_time.hour
    feat["target_dow"] = target_time.dayofweek          # 0=Mon … 6=Sun
    feat["target_month"] = target_time.month
    # Cyclical encoding of hour
    feat["hour_sin"] = np.sin(2 * np.pi * target_time.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * target_time.hour / 24)

    # ── E. Weather / pollutant co-features at issue time t ────────────────────
    if cofeature_cols is None:
        cofeature_cols = [c for c in CANDIDATE_COFEATURES if c in df.columns]

    for col in cofeature_cols:
        feat[col] = df[col]

    return feat


def get_feature_cols(feat: pd.DataFrame) -> List[str]:
    """Return predictor column names (all except 'target')."""
    return [c for c in feat.columns if c != "target"]


def verify_target_alignment(df: pd.DataFrame, horizon: int = HORIZON) -> None:
    """
    Assert that feat['target'] at row i equals df['pm25'] horizon rows later.
    Raises AssertionError if alignment is wrong.
    """
    feat = build_features(df.copy(), horizon=horizon)

    # Find a row where both target and raw pm25 are non-null
    check = feat.dropna(subset=["target", "pm25_lag0"])
    if check.empty:
        print("[verify_target_alignment] No non-null rows to verify.")
        return

    # Take first valid issue time
    t = check.index[0]
    t_future = t + pd.Timedelta(hours=horizon)

    target_val = feat.loc[t, "target"]

    if t_future in df.index:
        raw_val = df.loc[t_future, "pm25"]
        match = np.isclose(target_val, raw_val, equal_nan=True)
        if not (np.isnan(target_val) and np.isnan(raw_val)) and not match:
            raise AssertionError(
                f"Target alignment failed!\n"
                f"  Issue time     : {t}\n"
                f"  Target time    : {t_future}\n"
                f"  feat['target'] : {target_val}\n"
                f"  df['pm25']     : {raw_val}"
            )
    print(
        f"[verify_target_alignment] ✓  issue={t}  target_time={t_future}"
        f"  target_pm25={target_val:.4f}"
    )


def main() -> None:
    cfg = load_config()
    proc_path = PROJECT_ROOT / cfg["data"]["processed_dir"] / cfg["data"]["processed_filename"]

    if not proc_path.exists():
        print(f"[build_features] Processed file not found: {proc_path}", file=sys.stderr)
        print("  Run:  python -m src.data.prepare_data", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    lag_hours = cfg["features"]["lag_hours"]
    rolling_windows = cfg["features"]["rolling_windows"]
    horizon = cfg["target"]["horizon_hours"]

    print("[build_features] Building feature matrix…")
    feat = build_features(df, lag_hours=lag_hours, rolling_windows=rolling_windows, horizon=horizon)

    print(f"  Feature matrix shape : {feat.shape}")
    print(f"  Feature columns ({len(feat.columns)-1}): {get_feature_cols(feat)}")
    print(f"  Rows with non-null target : {feat['target'].notna().sum():,}")
    print(f"  Rows with fully non-null features: {feat.dropna().shape[0]:,}")

    # Run target alignment check
    print("\n[build_features] Running target alignment test…")
    verify_target_alignment(df, horizon=horizon)

    print("[build_features] Done.")


if __name__ == "__main__":
    main()
