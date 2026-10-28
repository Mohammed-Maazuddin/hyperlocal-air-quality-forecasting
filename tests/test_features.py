"""
tests/test_features.py
-----------------------
Automated tests for feature engineering correctness.
Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_features, get_feature_cols, verify_target_alignment


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_df(n_hours: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic synthetic hourly DataFrame for testing."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01 00:00:00", periods=n_hours, freq="1h", tz="UTC")
    pm25 = rng.uniform(10, 80, n_hours)
    temp_c = rng.uniform(15, 35, n_hours)
    rh = rng.uniform(40, 95, n_hours)
    return pd.DataFrame({"pm25": pm25, "temp_c": temp_c, "rh": rh}, index=index)


@pytest.fixture
def synth_df() -> pd.DataFrame:
    return make_synthetic_df(n_hours=300)


@pytest.fixture
def small_df() -> pd.DataFrame:
    return make_synthetic_df(n_hours=50)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Target alignment
# ─────────────────────────────────────────────────────────────────────────────

def test_target_alignment_24h(synth_df):
    """feat['target'] at row i must equal df['pm25'] at i+24 hours."""
    feat = build_features(synth_df, lag_hours=[1, 6, 24], rolling_windows=[6], horizon=24)

    # Check every 10th row where both target and raw pm25 are available
    for t in feat.index[::10]:
        t_future = t + pd.Timedelta(hours=24)
        if t_future in synth_df.index and not np.isnan(feat.loc[t, "target"]):
            raw_val = synth_df.loc[t_future, "pm25"]
            feat_val = feat.loc[t, "target"]
            assert np.isclose(feat_val, raw_val, rtol=1e-6), (
                f"Alignment failure at t={t}: "
                f"feat['target']={feat_val}, df['pm25'][t+24h]={raw_val}"
            )


def test_target_alignment_utility(synth_df):
    """verify_target_alignment() must not raise for correct data."""
    verify_target_alignment(synth_df, horizon=24)  # should not raise


# ─────────────────────────────────────────────────────────────────────────────
# Test: Lag alignment (no leakage)
# ─────────────────────────────────────────────────────────────────────────────

def test_lag1h_alignment(synth_df):
    """pm25_lag1h at row t must equal pm25 at t-1h."""
    feat = build_features(synth_df, lag_hours=[1], rolling_windows=[], horizon=24)
    for t in feat.index[5:20]:
        t_prev = t - pd.Timedelta(hours=1)
        if t_prev in synth_df.index:
            expected = synth_df.loc[t_prev, "pm25"]
            actual = feat.loc[t, "pm25_lag1h"]
            if not (np.isnan(expected) and np.isnan(actual)):
                assert np.isclose(actual, expected, rtol=1e-6), (
                    f"Lag1h failure at t={t}: expected {expected}, got {actual}"
                )


def test_lag24h_alignment(synth_df):
    """pm25_lag24h at row t must equal pm25 at t-24h."""
    feat = build_features(synth_df, lag_hours=[24], rolling_windows=[], horizon=24)
    for t in feat.index[30:40]:
        t_prev = t - pd.Timedelta(hours=24)
        if t_prev in synth_df.index:
            expected = synth_df.loc[t_prev, "pm25"]
            actual = feat.loc[t, "pm25_lag24h"]
            if not (np.isnan(expected) and np.isnan(actual)):
                assert np.isclose(actual, expected, rtol=1e-6), (
                    f"Lag24h failure at t={t}: expected {expected}, got {actual}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Rolling feature leakage prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_rolling_uses_only_past(synth_df):
    """pm25_roll6h_mean at t must equal mean of df['pm25'][t-5:t+1]."""
    feat = build_features(synth_df, lag_hours=[], rolling_windows=[6], horizon=24)

    # Check row at index 50 (well within range)
    t = feat.index[50]
    window_values = synth_df.loc[:t, "pm25"].iloc[-6:]
    expected = window_values.mean()
    actual = feat.loc[t, "pm25_roll6h_mean"]
    assert np.isclose(actual, expected, rtol=1e-6), (
        f"Rolling mean leakage! at t={t}: expected {expected:.4f}, got {actual:.4f}"
    )


def test_rolling_no_future_values(synth_df):
    """Rolling features at t must not use any value after t."""
    # Modify the value at t+1, re-compute features, and check t is unchanged
    t_idx = 50
    t = synth_df.index[t_idx]

    feat_original = build_features(synth_df.copy(), lag_hours=[], rolling_windows=[6], horizon=24)
    val_original = feat_original.loc[t, "pm25_roll6h_mean"]

    df_modified = synth_df.copy()
    # Change t+1 value dramatically
    df_modified.iloc[t_idx + 1, df_modified.columns.get_loc("pm25")] = 9999.0
    feat_modified = build_features(df_modified, lag_hours=[], rolling_windows=[6], horizon=24)
    val_modified = feat_modified.loc[t, "pm25_roll6h_mean"]

    assert np.isclose(val_original, val_modified, rtol=1e-6), (
        f"Rolling feature at t={t} changed when t+1 was modified – possible future leakage!\n"
        f"  Original: {val_original}, Modified: {val_modified}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Chronological splitting
# ─────────────────────────────────────────────────────────────────────────────

def test_chronological_split_no_overlap(synth_df):
    """Train and test sets must not overlap in time."""
    from src.models.train import chronological_split

    feat = build_features(synth_df, lag_hours=[1, 24], rolling_windows=[6], horizon=24)
    train, test = chronological_split(feat, train_fraction=0.75)

    assert train.index.max() < test.index.min(), (
        f"Train and test overlap! train_max={train.index.max()}, test_min={test.index.min()}"
    )


def test_chronological_split_fractions(synth_df):
    """Train fraction must be approximately 75% of valid rows."""
    from src.models.train import chronological_split

    feat = build_features(synth_df, lag_hours=[1, 24], rolling_windows=[6], horizon=24)
    train, test = chronological_split(feat, train_fraction=0.75)
    total = len(train) + len(test)
    train_pct = len(train) / total
    assert abs(train_pct - 0.75) < 0.01, (
        f"Train fraction expected ~0.75, got {train_pct:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test: Missing required columns
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_pm25_raises():
    """build_features must raise ValueError if pm25 is missing."""
    df_no_pm25 = pd.DataFrame(
        {"temp_c": [25.0] * 50},
        index=pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC"),
    )
    with pytest.raises(ValueError, match="pm25"):
        build_features(df_no_pm25)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Inference timestamp handling
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_cols_exclude_target(synth_df):
    """get_feature_cols() must not include 'target'."""
    feat = build_features(synth_df, lag_hours=[1, 24], rolling_windows=[6], horizon=24)
    cols = get_feature_cols(feat)
    assert "target" not in cols, "'target' must not be in feature columns used for prediction."


def test_features_deterministic(synth_df):
    """Running build_features twice on the same data must give identical results."""
    feat1 = build_features(synth_df.copy(), lag_hours=[1, 24], rolling_windows=[6], horizon=24)
    feat2 = build_features(synth_df.copy(), lag_hours=[1, 24], rolling_windows=[6], horizon=24)
    pd.testing.assert_frame_equal(feat1, feat2)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Calendar features from TARGET time
# ─────────────────────────────────────────────────────────────────────────────

def test_calendar_features_from_target_time(synth_df):
    """target_hour must equal (issue_hour + 24) % 24 = issue_hour (24h horizon)."""
    feat = build_features(synth_df, lag_hours=[1], rolling_windows=[], horizon=24)
    for t in feat.index[30:50]:
        t_future = t + pd.Timedelta(hours=24)
        expected_hour = t_future.hour
        actual_hour = feat.loc[t, "target_hour"]
        assert actual_hour == expected_hour, (
            f"Calendar hour mismatch at t={t}: expected {expected_hour}, got {actual_hour}"
        )


def test_cyclical_encoding_range(synth_df):
    """hour_sin and hour_cos must be in [-1, 1]."""
    feat = build_features(synth_df, lag_hours=[1], rolling_windows=[], horizon=24)
    assert feat["hour_sin"].between(-1.0, 1.0).all()
    assert feat["hour_cos"].between(-1.0, 1.0).all()
