"""
src/data/inspect_data.py
-------------------------
Inspects the raw CSV: schema, date range, frequency, missingness,
duplicate timestamps, data quality, and lists available features.

Usage:
    python -m src.data.inspect_data
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def inspect(df: pd.DataFrame, target_col: str) -> dict:
    """Return an inspection summary dict and print a rich report."""
    sep = "=" * 70

    print(sep)
    print("  RAW DATA INSPECTION REPORT")
    print(sep)
    # Force UTF-8 safe output
    import sys as _sys
    import io
    if hasattr(_sys.stdout, 'reconfigure'):
        try: _sys.stdout.reconfigure(encoding='utf-8')
        except Exception: pass

    # ── Basic shape ───────────────────────────────────────────────────────────
    print(f"\n{'-'*40}")
    print(f"  Rows   : {len(df):>10,}")
    print(f"  Columns: {df.shape[1]:>10,}")

    # ── Column overview ───────────────────────────────────────────────────────
    print(f"\n{'-'*40}")
    print("  Column names and dtypes:\n")
    for col in df.columns:
        print(f"    {col:<40}  {str(df[col].dtype):<12}  "
              f"non-null: {df[col].notna().sum():>7,}")

    # ── Timestamp ─────────────────────────────────────────────────────────────
    print(f"\n{'-'*40}")
    ts = df["Timestamp"]
    print(f"  Date range : {ts.min()}  →  {ts.max()}")
    print(f"  Duration   : {ts.max() - ts.min()}")
    diffs = ts.sort_values().diff().dropna()
    mode_diff = diffs.mode()[0]
    print(f"  Sampling frequency (mode gap): {mode_diff}")
    print(f"  Min gap: {diffs.min()},  Max gap: {diffs.max()}")

    # ── Duplicate timestamps ───────────────────────────────────────────────────
    dupes = df.duplicated(subset=["Timestamp"]).sum()
    print(f"\n  Duplicate timestamps: {dupes:,}")

    # ── Target column ─────────────────────────────────────────────────────────
    print(f"\n{'-'*40}")
    print(f"  Target column: '{target_col}'")
    if target_col in df.columns:
        s = df[target_col]
        missing_pct = s.isna().mean() * 100
        neg_count = (s < 0).sum()
        print(f"    non-null  : {s.notna().sum():>8,}  ({100-missing_pct:.2f}%)")
        print(f"    missing   : {s.isna().sum():>8,}  ({missing_pct:.2f}%)")
        print(f"    negative  : {neg_count:>8,}")
        print(f"    min/max   : {s.min():.2f} / {s.max():.2f}")
        print(f"    mean±std  : {s.mean():.2f} ± {s.std():.2f}")
        print(f"    p25/med/p75: {s.quantile(0.25):.2f} / {s.median():.2f} / {s.quantile(0.75):.2f}")
    else:
        print(f"    *** WARNING: '{target_col}' NOT FOUND in dataset! ***")

    # ── Missingness by column ─────────────────────────────────────────────────
    print(f"\n{'-'*40}")
    print("  Missing values by column:\n")
    miss = df.isna().mean().sort_values(ascending=False) * 100
    for col, pct in miss.items():
        bar = "█" * int(pct / 5)
        print(f"    {col:<40}  {pct:6.2f}%  {bar}")

    # ── Descriptive statistics ─────────────────────────────────────────────────
    print(f"\n{'-'*40}")
    print("  Descriptive statistics (numeric columns):\n")
    numeric = df.select_dtypes(include="number")
    desc = numeric.describe().T
    with pd.option_context("display.max_rows", 60, "display.float_format", "{:.3f}".format):
        print(desc.to_string())

    # ── Long gaps in time series ──────────────────────────────────────────────
    print(f"\n{'-'*40}")
    print("  Long gaps (> 2 hours) in raw time series:\n")
    df_sorted = df.sort_values("Timestamp")
    gaps = df_sorted["Timestamp"].diff()
    long_gaps = df_sorted[gaps > pd.Timedelta("2h")][["Timestamp"]].copy()
    long_gaps["gap_before"] = gaps[gaps > pd.Timedelta("2h")].values
    if long_gaps.empty:
        print("    None found.")
    else:
        print(long_gaps.head(20).to_string(index=False))
        if len(long_gaps) > 20:
            print(f"    ... {len(long_gaps) - 20} more gaps not shown.")

    print(f"\n{sep}\n")

    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "date_min": str(ts.min()),
        "date_max": str(ts.max()),
        "sampling_mode": str(mode_diff),
        "n_duplicates": int(dupes),
        "target_missing_pct": float(miss.get(target_col, 0.0)),
    }


def main() -> None:
    cfg = load_config()
    raw_path = PROJECT_ROOT / cfg["data"]["raw_dir"] / cfg["data"]["raw_filename"]

    if not raw_path.exists():
        print(f"[inspect_data] Raw file not found: {raw_path}", file=sys.stderr)
        print("  Run:  python -m src.data.download_data", file=sys.stderr)
        sys.exit(1)

    print(f"[inspect_data] Loading: {raw_path}")
    df = pd.read_csv(raw_path)

    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    print(f"[inspect_data] Rows with unparseable timestamps: {df['Timestamp'].isna().sum()}")

    # Rename target column to ASCII-safe name for inspection display
    target_raw = cfg["target"]["column"]
    # Normalise column name (the CSV may have µ encoded differently)
    col_map = {}
    for c in df.columns:
        if "PM2.5" in c or "PM 2.5" in c:
            col_map[c] = "PM2.5 (µg/m³)"
    df.rename(columns=col_map, inplace=True)
    target_col = "PM2.5 (µg/m³)"
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    inspect(df, target_col)


if __name__ == "__main__":
    main()
