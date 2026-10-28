"""
src/data/prepare_data.py
--------------------------
Cleans and resamples the raw sub-hourly CSV to a continuous hourly
time-series with documented transformations.

Transformations applied (in order):
  1. Parse timestamps; drop rows with unparseable timestamps.
  2. Rename columns to ASCII-safe names.
  3. Convert all measurement columns to numeric; non-numeric → NaN.
  4. Drop exact duplicate (Timestamp, PM2.5) rows; keep first.
  5. Resolve remaining duplicate timestamps by taking mean.
  6. Sort chronologically.
  7. Resample to 1-hour frequency by taking mean of all measurements
     in each hour.  Hours with no raw readings remain as NaN rows
     (NOT forward-filled or interpolated).
  8. Save to data/processed/hourly_processed.csv.

Leakage prevention:
  - No imputation is applied in this step; imputation is done inside
    model pipelines, fitted only on training data.
  - Target (PM2.5) NaN rows are preserved so the modelling step can
    decide how to handle them.

Usage:
    python -m src.data.prepare_data
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# ── Column name map: raw (µ may be encoded differently) → canonical ───────────
# We use ASCII-compatible names internally; µ is replaced with u.
COLUMN_RENAME: dict[str, str] = {
    # Pollutants
    "PM2.5 (µg/m³)": "pm25",
    "PM10 (µg/m³)": "pm10",
    "NO (µg/m³)": "no",
    "NO2 (µg/m³)": "no2",
    "NOx (ppb)": "nox",
    "NH3 (µg/m³)": "nh3",
    "SO2 (µg/m³)": "so2",
    "CO (mg/m³)": "co",
    "Ozone (µg/m³)": "ozone",
    "Benzene (µg/m³)": "benzene",
    "Toluene (µg/m³)": "toluene",
    "Xylene (µg/m³)": "xylene",
    "O Xylene (µg/m³)": "o_xylene",
    "Eth-Benzene (µg/m³)": "eth_benzene",
    "MP-Xylene (µg/m³)": "mp_xylene",
    # Meteorological
    "AT (°C)": "temp_c",
    "RH (%)": "rh",
    "WS (m/s)": "ws",
    "WD (deg)": "wd",
    "RF (mm)": "rf",
    "TOT-RF (mm)": "tot_rf",
    "SR (W/mt2)": "sr",
    "BP (mmHg)": "bp",
    "VWS (m/s)": "vws",
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _fuzzy_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using substring matching to handle encoding variants."""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        for raw_name, canonical in COLUMN_RENAME.items():
            # Remove encoding artefacts: compare after stripping special chars
            def _strip(s: str) -> str:
                return (
                    s.replace("µ", "u")
                    .replace("°", "")
                    .replace("³", "3")
                    .replace("\ufffd", "u")
                    .strip()
                )

            if _strip(col) == _strip(raw_name):
                rename_map[col] = canonical
                break
    return df.rename(columns=rename_map)


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    """Load raw CSV and return a cleaned, hourly-resampled DataFrame."""
    print(f"[prepare_data] Loading raw file: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # ── 1. Parse timestamps ───────────────────────────────────────────────────
    # Source timestamps include UTC offset (+0000).  We parse with utc=True
    # so all timestamps are timezone-aware UTC.
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    bad_ts = df["Timestamp"].isna().sum()
    if bad_ts > 0:
        print(f"  [warn] Dropping {bad_ts} rows with unparseable timestamps.")
        df = df.dropna(subset=["Timestamp"])

    # ── 2. Rename columns ─────────────────────────────────────────────────────
    df = _fuzzy_rename(df)

    # ── 3. Convert measurements to numeric ────────────────────────────────────
    meta_cols = ["Station ID", "State", "City", "Station Name", "Timestamp"]
    meas_cols = [c for c in df.columns if c not in meta_cols]
    for col in meas_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clamp negative PM2.5 (instrument artefacts) to NaN
    if "pm25" in df.columns:
        neg_pm25 = (df["pm25"] < 0).sum()
        if neg_pm25 > 0:
            print(f"  [warn] {neg_pm25} negative PM2.5 values → set to NaN.")
            df.loc[df["pm25"] < 0, "pm25"] = np.nan

    # ── 4. Drop exact duplicate rows ──────────────────────────────────────────
    dup_full = df.duplicated().sum()
    if dup_full > 0:
        print(f"  [warn] Dropping {dup_full} fully duplicate rows.")
        df = df.drop_duplicates()

    # ── 5. Set Timestamp as index ─────────────────────────────────────────────
    df = df.set_index("Timestamp").sort_index()
    num_df = df[meas_cols].copy()

    # Resolve remaining duplicate timestamps by mean
    dup_ts = num_df.index.duplicated().sum()
    if dup_ts > 0:
        print(f"  [warn] {dup_ts} duplicate timestamp rows → resolving by mean.")
        num_df = num_df.groupby(num_df.index).mean()

    # ── 6. Resample to hourly ─────────────────────────────────────────────────
    print("  Resampling to 1-hour frequency (mean aggregation)…")
    hourly = num_df.resample("1h").mean()

    # ── 7. Report ─────────────────────────────────────────────────────────────
    print(f"  Hourly shape : {hourly.shape}")
    print(f"  Date range   : {hourly.index.min()} → {hourly.index.max()}")
    if "pm25" in hourly.columns:
        miss_pct = hourly["pm25"].isna().mean() * 100
        print(f"  PM2.5 missing hours: {hourly['pm25'].isna().sum():,}  ({miss_pct:.2f}%)")

    return hourly


def save_processed(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest)
    print(f"[prepare_data] Saved processed data: {dest}  ({dest.stat().st_size:,} bytes)")


def main() -> None:
    cfg = load_config()
    raw_path = PROJECT_ROOT / cfg["data"]["raw_dir"] / cfg["data"]["raw_filename"]
    proc_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    out_path = proc_dir / cfg["data"]["processed_filename"]

    if not raw_path.exists():
        print(f"[prepare_data] ERROR: raw file not found: {raw_path}", file=sys.stderr)
        print("  Run:  python -m src.data.download_data", file=sys.stderr)
        sys.exit(1)

    hourly = load_and_clean(raw_path)
    save_processed(hourly, out_path)
    print("[prepare_data] Done.")


if __name__ == "__main__":
    main()
