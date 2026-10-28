"""
app.py — Streamlit Dashboard
Hyperlocal Air Quality Forecasting: 24-Hour-Ahead PM2.5 Prediction
IITH Kandi, Hyderabad – TSPCB Station

This is a HISTORICAL-DATA FORECASTING DEMO.
It does not use live sensor data and does not constitute
official public-health guidance.

Run: streamlit run app.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hyderabad PM2.5 Forecasting",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


@st.cache_data
def load_processed_data(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


@st.cache_resource
def load_model_and_meta(artifacts_dir: str, model_file: str, meta_file: str):
    try:
        import joblib
        m_path = Path(artifacts_dir) / model_file
        meta_path = Path(artifacts_dir) / meta_file
        if not m_path.exists():
            return None, None
        model = joblib.load(m_path)
        meta = {}
        if meta_path.exists():
            with open(meta_path) as fh:
                meta = json.load(fh)
        return model, meta
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None, None


@st.cache_data
def load_test_predictions(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


@st.cache_data
def load_metrics(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Flag_of_India.svg/120px-Flag_of_India.svg.png", width=80)
        st.title("🌫️ Hyperlocal AQ Forecasting")
        st.markdown("**Station:** IITH Kandi, Hyderabad (TSPCB)")
        st.markdown("**Target:** PM2.5, 24 hours ahead")
        st.markdown("---")
        st.warning(
            "📋 **Disclaimer**: This is a historical-data educational demo only. "
            "Not for public-health decision making."
        )
        st.markdown("---")
        st.markdown("**Project by**: Data Science Student Demo")
        st.markdown("**Data**: [OpenCity India](https://data.opencity.in)")

    # ── Load data ─────────────────────────────────────────────────────────────
    proc_path = str(PROJECT_ROOT / cfg["data"]["processed_dir"] / cfg["data"]["processed_filename"])
    df = load_processed_data(proc_path)

    artifacts_dir = str(PROJECT_ROOT / cfg["model"]["artifacts_dir"])
    model, meta = load_model_and_meta(
        artifacts_dir, cfg["model"]["artifact_filename"], cfg["model"]["metadata_filename"]
    )

    metrics_df = load_metrics(str(PROJECT_ROOT / cfg["reports"]["metrics_csv"]))
    test_preds = load_test_predictions(str(PROJECT_ROOT / cfg["reports"]["test_predictions_csv"]))

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🌫️ Hyperlocal Air Quality Forecasting — Hyderabad")
    st.markdown(
        "**24-hour-ahead PM2.5 prediction** at the IITH Kandi monitoring station, "
        "operated by the Telangana State Pollution Control Board (TSPCB)."
    )

    # ── Dataset coverage banner ───────────────────────────────────────────────
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Hourly Observations", f"{len(df):,}")
        with col2:
            st.metric("Dataset Start", str(df.index.min().date()))
        with col3:
            st.metric("Dataset End", str(df.index.max().date()))
        with col4:
            missing_pct = df["pm25"].isna().mean() * 100 if "pm25" in df.columns else 0
            st.metric("PM2.5 Missing Hours", f"{missing_pct:.1f}%")
    else:
        st.error("⚠️ Processed dataset not found (`data/processed/hourly_processed.csv`).")
        st.info("You can generate it by clicking below or by running `python -m src.data.prepare_data` in your terminal.")
        raw_path = PROJECT_ROOT / cfg["data"]["raw_dir"] / cfg["data"]["raw_filename"]
        btn_label = "⚡ Prepare Dataset from Local Raw Data" if raw_path.exists() else "⬇️ Download & Prepare Dataset"
        if st.button(btn_label):
            with st.spinner("Processing dataset... please wait..."):
                try:
                    if not raw_path.exists():
                        from src.data.download_data import download_csv
                        download_csv(cfg["data"]["source_url"], raw_path)
                    from src.data.prepare_data import load_and_clean, save_processed
                    out_path = PROJECT_ROOT / cfg["data"]["processed_dir"] / cfg["data"]["processed_filename"]
                    hourly_df = load_and_clean(raw_path)
                    save_processed(hourly_df, out_path)
                    st.success("Dataset prepared successfully! Reloading...")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to prepare dataset: {ex}")
        st.stop()

    st.markdown("---")

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Historical Trend", "🔮 Forecast", "📊 Model Evaluation", "📖 About"]
    )

    # ── Tab 1: Historical PM2.5 trend ─────────────────────────────────────────
    with tab1:
        st.subheader("Historical PM2.5 Trend")

        if "pm25" not in df.columns:
            st.warning("pm25 column not found in processed data.")
        else:
            # Date range filter
            min_date = df.index.min().date()
            max_date = df.index.max().date()
            col_a, col_b = st.columns(2)
            with col_a:
                start_d = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
            with col_b:
                end_d = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

            mask = (df.index.date >= start_d) & (df.index.date <= end_d)
            sub = df.loc[mask, "pm25"].dropna()

            if sub.empty:
                st.warning("No data in selected range.")
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(sub.index, sub.values, alpha=0.4, linewidth=0.6, color="#1976D2", label="Hourly")
                roll = sub.rolling(24, min_periods=6).mean()
                ax.plot(roll.index, roll.values, color="#E53935", linewidth=1.2, label="24-h rolling mean")
                ax.set_ylabel("PM2.5 (µg/m³)")
                ax.set_title("PM2.5 Over Time – IITH Kandi, Hyderabad")
                ax.legend(fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                c1, c2, c3 = st.columns(3)
                c1.metric("Mean PM2.5", f"{sub.mean():.1f} µg/m³")
                c2.metric("Max PM2.5", f"{sub.max():.1f} µg/m³")
                c3.metric("Median PM2.5", f"{sub.median():.1f} µg/m³")

    # ── Tab 2: Forecast ────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🔮 24-Hour-Ahead PM2.5 Forecast")

        if model is None:
            st.error("⚠️ Model artifact not found. Run `python -m src.models.train` first.")
        else:
            model_type = meta.get("model_type", "Unknown")
            horizon = meta.get("horizon_hours", 24)
            st.info(f"**Model**: {model_type}  |  **Forecast horizon**: {horizon} hours ahead")

            # Issue timestamp selector
            st.markdown("### Select Forecast Issue Timestamp")
            st.markdown(
                "At the selected time, the model uses all available historical data "
                f"to predict PM2.5 **{horizon} hours later**."
            )

            # Limit to valid range (where we have sufficient history)
            valid_start = df.index.min() + pd.Timedelta(hours=max(meta.get("lag_hours", [168])))
            valid_end = df.index.max() - pd.Timedelta(hours=horizon)
            valid_dates = df.loc[valid_start:valid_end].index

            if valid_dates.empty:
                st.warning("Not enough data to make a forecast.")
            else:
                sel_date = st.date_input(
                    "Issue Date",
                    value=valid_dates[-1].date(),
                    min_value=valid_dates[0].date(),
                    max_value=valid_dates[-1].date(),
                )
                sel_hour = st.slider("Issue Hour (UTC)", 0, 23, int(valid_dates[-1].hour))
                issue_ts = pd.Timestamp(f"{sel_date} {sel_hour:02d}:00:00", tz="UTC")

                if st.button("🔮 Generate Forecast"):
                    try:
                        from src.models.predict import predict_single
                        with st.spinner("Generating forecast…"):
                            result = predict_single(str(issue_ts), model=model, meta=meta, cfg=cfg)

                        st.success("Forecast generated!")
                        col1, col2 = st.columns(2)
                        col1.metric("Issue Time (UTC)", result["issue_time"])
                        col1.metric("Target Time (UTC)", result["target_time"])
                        col2.metric(
                            f"Predicted PM2.5 at {result['target_time'][:16]}",
                            f"{result['forecast_pm25']:.1f} µg/m³",
                        )

                        # Show actual if available
                        tgt_ts = pd.Timestamp(result["target_time"])
                        if tgt_ts in df.index and not pd.isna(df.loc[tgt_ts, "pm25"]):
                            actual_val = df.loc[tgt_ts, "pm25"]
                            col2.metric("Actual PM2.5 (observed)", f"{actual_val:.1f} µg/m³",
                                       delta=f"{result['forecast_pm25'] - actual_val:.1f} µg/m³")
                        else:
                            col2.info("Actual PM2.5 not available for this timestamp.")

                        st.caption(
                            "⚠️ This forecast is generated from historical data for educational purposes only. "
                            "It is not a real-time prediction and must not be used for public-health decisions."
                        )
                    except Exception as e:
                        st.error(f"Forecast error: {e}")

    # ── Tab 3: Model Evaluation ────────────────────────────────────────────────
    with tab3:
        st.subheader("📊 Model Evaluation (Test Set)")

        if metrics_df is not None:
            st.markdown("### Model Comparison Table")
            st.markdown(
                "Evaluation on the **held-out test set** (chronologically last 25% of data). "
                "This is a historical evaluation, not a guarantee of future performance."
            )
            st.dataframe(
                metrics_df.style.format(
                    {"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}", "WAPE_%": "{:.2f}"}
                ).highlight_min(subset=["MAE", "RMSE"], color="#C8E6C9")
                .highlight_max(subset=["R2"], color="#C8E6C9"),
                use_container_width=True,
            )
            st.caption("Green = best performance. MAE/RMSE in µg/m³ (lower is better). R² higher is better.")

        if test_preds is not None and "actual" in test_preds.columns:
            st.markdown("### Actual vs Predicted PM2.5 (Test Period)")
            model_choices = [c for c in test_preds.columns if c != "actual"]
            sel_model = st.selectbox("Select model to visualise", model_choices)

            sub_test = test_preds[["actual", sel_model]].dropna()
            if not sub_test.empty:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(sub_test.index, sub_test["actual"], label="Actual", color="#1565C0",
                        linewidth=0.8, alpha=0.7)
                ax.plot(sub_test.index, sub_test[sel_model], label=f"Predicted ({sel_model})",
                        color="#E53935", linewidth=0.8, alpha=0.7)
                ax.set_ylabel("PM2.5 (µg/m³)")
                ax.set_title(f"Actual vs Predicted — {sel_model}")
                ax.legend(fontsize=9)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # Feature importance
        if meta and model:
            final_step = model.named_steps.get("model", None)
            feature_cols = meta.get("feature_cols", [])
            if final_step and hasattr(final_step, "feature_importances_") and feature_cols:
                st.markdown("### Feature Importance")
                importances = final_step.feature_importances_
                idx = np.argsort(importances)[::-1][:20]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(
                    [feature_cols[i] for i in idx][::-1],
                    importances[idx][::-1],
                    color="#42A5F5",
                )
                ax.set_title(f"Feature Importance — {meta.get('model_type', '')}")
                ax.set_xlabel("Importance Score")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Feature importance measures predictive contribution, not causal influence.")

    # ── Tab 4: About ──────────────────────────────────────────────────────────
    with tab4:
        st.subheader("About This Project")
        st.markdown(f"""
**Project**: Hyperlocal Air Quality Forecasting for Hyderabad — 24-Hour-Ahead PM2.5 Prediction

**Station**: IITH Kandi, Hyderabad – TSPCB (site_5597)

**Data source**: [OpenCity India](https://data.opencity.in/dataset/7a729fcc-f39e-498d-a577-1393a6e873b7/resource/054a52fa-2552-438e-baab-8d9b22fed81b/)
- Station: IITH Kandi, Hyderabad, Telangana
- Operated by: Telangana State Pollution Control Board (TSPCB)
- Raw frequency: 15-minute
- Aggregated to: hourly (mean)
- Coverage: Jan 2024 – Dec 2025

**Forecast horizon**: 24 hours (features from issue time t → predict pm25 at t+24h)

**Models**: Persistence baseline, Seasonal Naive baseline, Linear Regression,
Random Forest, HistGradientBoosting

**Validation**: Chronological train/test split (75%/25%); no data leakage.

**Limitations**:
- ~28% of PM2.5 hours are missing after hourly aggregation.
- Model uses weather observations available AT issue time (not future weather forecasts).
- Sensor outages can create consecutive missing periods that degrade lag features.
- This demo uses historical data only. Real-time deployment would require live sensor integration.
- Not for public-health or regulatory decisions.
        """)


if __name__ == "__main__":
    main()
