"""
src/evaluation/plots.py
------------------------
Publication-quality evaluation and EDA plots.
All figures are saved to configured output directories.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = "husl"
STYLE = "seaborn-v0_8-whitegrid"
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def save_fig(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_pm25_timeseries(df: pd.DataFrame, out_dir: Path) -> None:
    """Full PM2.5 time-series with 24-hour rolling mean overlay."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df.index, df["pm25"], alpha=0.4, linewidth=0.6, color="#2196F3", label="Hourly PM2.5")
        roll = df["pm25"].rolling(24, min_periods=6).mean()
        ax.plot(df.index, roll, color="#E53935", linewidth=1.2, label="24-h rolling mean")
        ax.set_title("PM2.5 Concentration Over Time — IITH Kandi, Hyderabad (TSPCB)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_xlabel("Date")
        ax.axhline(60, color="orange", linestyle="--", linewidth=0.8, label="WHO 24-h guideline (15 µg/m³ shown as 60 for AQI moderate)")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        fig.autofmt_xdate()
        save_fig(fig, out_dir / "pm25_timeseries.png")


def plot_pm25_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram + KDE + boxplot for PM2.5."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        s = df["pm25"].dropna()
        axes[0].hist(s, bins=60, color="#42A5F5", edgecolor="white", alpha=0.85)
        axes[0].axvline(s.mean(), color="red", linestyle="--", label=f"Mean {s.mean():.1f}")
        axes[0].axvline(s.median(), color="green", linestyle=":", label=f"Median {s.median():.1f}")
        axes[0].set_title("PM2.5 Distribution (Histogram)")
        axes[0].set_xlabel("PM2.5 (µg/m³)")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        axes[1].boxplot(s, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#90CAF9"),
                        medianprops=dict(color="red", linewidth=2))
        axes[1].set_title("PM2.5 Boxplot")
        axes[1].set_ylabel("PM2.5 (µg/m³)")
        save_fig(fig, out_dir / "pm25_distribution.png")


def plot_hourly_patterns(df: pd.DataFrame, out_dir: Path) -> None:
    """Average PM2.5 by hour of day and day of week."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        hourly = df.groupby(df.index.hour)["pm25"].agg(["mean", "std"])
        axes[0].bar(hourly.index, hourly["mean"], color="#42A5F5", alpha=0.8)
        axes[0].errorbar(hourly.index, hourly["mean"], yerr=hourly["std"],
                         fmt="none", color="gray", capsize=3, linewidth=0.8)
        axes[0].set_title("Mean PM2.5 by Hour of Day (UTC)")
        axes[0].set_xlabel("Hour (UTC)")
        axes[0].set_ylabel("PM2.5 (µg/m³)")
        axes[0].set_xticks(range(0, 24, 2))

        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = df.groupby(df.index.dayofweek)["pm25"].mean()
        axes[1].bar(daily.index, daily.values, color="#66BB6A", alpha=0.8)
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(dow_labels)
        axes[1].set_title("Mean PM2.5 by Day of Week")
        axes[1].set_ylabel("PM2.5 (µg/m³)")
        save_fig(fig, out_dir / "pm25_patterns_hour_dow.png")


def plot_monthly_patterns(df: pd.DataFrame, out_dir: Path) -> None:
    """Box plots of PM2.5 by month."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 4))
        data_by_month = [df[df.index.month == m]["pm25"].dropna().values
                         for m in range(1, 13)]
        present = [(i, d) for i, d in enumerate(data_by_month) if len(d) > 0]
        bp = ax.boxplot(
            [d for _, d in present],
            patch_artist=True,
            notch=False,
            medianprops=dict(color="red"),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#BBDEFB")
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels([months[i] for i, _ in present])
        ax.set_title("PM2.5 Distribution by Month")
        ax.set_ylabel("PM2.5 (µg/m³)")
        save_fig(fig, out_dir / "pm25_monthly_patterns.png")


def plot_correlation_matrix(df: pd.DataFrame, out_dir: Path) -> None:
    """Correlation heatmap for numeric features."""
    num_df = df.select_dtypes("number").dropna(axis=1, how="all")
    # Limit to columns with sufficient data
    num_df = num_df.loc[:, num_df.notna().mean() > 0.3]
    corr = num_df.corr()
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(min(16, len(corr) + 2), min(14, len(corr) + 1)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=~mask, annot=len(corr) <= 20, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Pearson Correlation Matrix\n(Note: correlation ≠ causation)")
        save_fig(fig, out_dir / "correlation_matrix.png")


def plot_scatter_vs_pm25(df: pd.DataFrame, cols: List[str], out_dir: Path) -> None:
    """Scatter plots of selected columns vs PM2.5."""
    cols = [c for c in cols if c in df.columns and c != "pm25"]
    if not cols:
        return
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(cols):
            sub = df[["pm25", col]].dropna()
            axes[i].scatter(sub[col], sub["pm25"], alpha=0.1, s=5, color="#42A5F5")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("PM2.5 (µg/m³)")
            axes[i].set_title(f"PM2.5 vs {col}")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        save_fig(fig, out_dir / "scatter_vs_pm25.png")


def plot_acf_pacf(series: pd.Series, out_dir: Path, nlags: int = 48) -> None:
    """ACF and PACF plots."""
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        print("  [plots] statsmodels not installed – skipping ACF/PACF.")
        return
    s = series.dropna()
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        plot_acf(s, lags=nlags, ax=axes[0], title="Autocorrelation Function (PM2.5)")
        plot_pacf(s, lags=min(nlags, len(s) // 2 - 1), ax=axes[1],
                  title="Partial Autocorrelation Function (PM2.5)")
        save_fig(fig, out_dir / "acf_pacf.png")


def plot_missingness(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of missingness percentage by column."""
    miss = df.isna().mean() * 100
    miss = miss[miss > 0].sort_values(ascending=True)
    if miss.empty:
        return
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.4)))
        ax.barh(miss.index, miss.values, color="#EF9A9A")
        ax.set_xlabel("Missing (%)")
        ax.set_title("Missing Values by Column (Hourly Data)")
        for i, v in enumerate(miss.values):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
        save_fig(fig, out_dir / "missingness.png")


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: pd.Series, y_pred: pd.Series, model_name: str, out_dir: Path
) -> None:
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(14, 7))

        # Time plot
        axes[0].plot(y_true.index, y_true.values, label="Actual", color="#1565C0", linewidth=0.8, alpha=0.7)
        axes[0].plot(y_pred.index, y_pred.values, label=f"Predicted ({model_name})",
                     color="#E53935", linewidth=0.8, alpha=0.7)
        axes[0].set_title(f"Actual vs Predicted PM2.5 — {model_name}")
        axes[0].set_ylabel("PM2.5 (µg/m³)")
        axes[0].legend()

        # Scatter
        axes[1].scatter(y_true.values, y_pred.values, alpha=0.2, s=10, color="#42A5F5")
        lim = [min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5]
        axes[1].plot(lim, lim, "r--", linewidth=1, label="Perfect prediction")
        axes[1].set_xlim(lim)
        axes[1].set_ylim(lim)
        axes[1].set_xlabel("Actual PM2.5 (µg/m³)")
        axes[1].set_ylabel("Predicted PM2.5 (µg/m³)")
        axes[1].set_title("Scatter: Actual vs Predicted")
        axes[1].legend()

        save_fig(fig, out_dir / f"actual_vs_predicted_{model_name.replace(' ', '_')}.png")


def plot_residuals(
    y_true: pd.Series, y_pred: pd.Series, model_name: str, out_dir: Path
) -> None:
    residuals = y_true - y_pred
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Residual distribution
        axes[0].hist(residuals.dropna(), bins=50, color="#90CAF9", edgecolor="white")
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].set_title(f"Residual Distribution — {model_name}")
        axes[0].set_xlabel("Residual (µg/m³)")

        # Residuals over time
        axes[1].scatter(y_pred.values, residuals.values, alpha=0.15, s=8, color="#42A5F5")
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Predicted PM2.5 (µg/m³)")
        axes[1].set_ylabel("Residual (µg/m³)")
        axes[1].set_title("Residuals vs Fitted")

        # Residuals over time
        axes[2].plot(residuals.index, residuals.values, alpha=0.4, linewidth=0.6, color="#5C6BC0")
        axes[2].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[2].set_xlabel("Date")
        axes[2].set_ylabel("Residual (µg/m³)")
        axes[2].set_title("Residuals Over Time")
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()

        save_fig(fig, out_dir / f"residuals_{model_name.replace(' ', '_')}.png")


def plot_model_comparison(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart comparing MAE and RMSE across models."""
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        models = metrics_df["model"]

        for ax, metric in zip(axes, ["MAE", "RMSE"]):
            vals = metrics_df[metric]
            bars = ax.bar(models, vals, color=sns.color_palette(PALETTE, len(models)))
            ax.set_title(f"{metric} Comparison (lower is better)")
            ax.set_ylabel(f"{metric} (µg/m³)")
            ax.set_xlabel("")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        save_fig(fig, out_dir / "model_comparison.png")


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    out_dir: Path,
    top_n: int = 25,
) -> None:
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.4)))
        ax.barh(names[::-1], vals[::-1], color="#42A5F5")
        ax.set_title(f"Feature Importance — {model_name}\n(Note: importance ≠ causation)")
        ax.set_xlabel("Importance Score")
        save_fig(fig, out_dir / f"feature_importance_{model_name.replace(' ', '_')}.png")


def plot_error_by_hour(
    y_true: pd.Series, y_pred: pd.Series, model_name: str, out_dir: Path
) -> None:
    err = (y_true - y_pred).abs()
    hourly_err = err.groupby(err.index.hour).mean()
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(hourly_err.index, hourly_err.values, color="#FFA726")
        ax.set_title(f"Mean Absolute Error by Forecast Target Hour — {model_name}")
        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("MAE (µg/m³)")
        ax.set_xticks(range(0, 24, 2))
        save_fig(fig, out_dir / f"error_by_hour_{model_name.replace(' ', '_')}.png")
