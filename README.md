# Hyperlocal Air Quality Forecasting for Hyderabad
## 24-Hour-Ahead PM2.5 Prediction — IITH Kandi Monitoring Station

> **Educational project**: This system is built for learning and demonstration purposes.
> It does **not** constitute official public-health guidance and must not be used
> for regulatory or safety decisions.

---

## Business Problem

Air quality in Indian cities varies significantly by location, season, and time of day.
This project builds a **station-specific 24-hour-ahead PM2.5 forecasting system** for
the **IITH Kandi monitoring station** in Hyderabad, operated by the Telangana State
Pollution Control Board (TSPCB).

**Forecast definition**:
> At time **t** (the _issue time_), use all measurements available up to and
> including **t** to predict PM2.5 at **t + 24 hours**.

This is a **point forecast** from a **single station**. It is not a citywide forecast.

---

## Why Hyderabad? Why IITH Kandi?

Hyderabad is one of India's fastest-growing tech cities.  
The IITH Kandi station is a well-maintained TSPCB station with 15-minute data publicly
available through OpenCity India, covering January 2024 – December 2025 — enough data
to build and evaluate a time-series forecasting model.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [OpenCity India – TSPCB Hyderabad](https://data.opencity.in/dataset/7a729fcc-f39e-498d-a577-1393a6e873b7/) |
| **Station** | IITH Kandi, Hyderabad – TSPCB (site_5597) |
| **State** | Telangana |
| **Raw frequency** | 15-minute |
| **Processed frequency** | Hourly (mean aggregation) |
| **Date range** | 2024-01-01 → 2025-12-31 |
| **Raw rows** | 70,177 (15-min) |
| **Hourly rows** | 17,544 |
| **PM2.5 missing (hourly)** | ~27.87% (4,890 hours) |
| **License** | Verify at OpenCity India portal |

> **Attribution**: Data sourced from OpenCity India. Original measurements by Telangana
> State Pollution Control Board (TSPCB). Please verify the dataset license before
> redistribution.

### Available Features (after processing)

**Pollutants**: PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, Benzene, Toluene, Xylene, Eth-Benzene  
**Meteorological**: Temperature (°C), Relative Humidity (%), Wind Speed (m/s), Wind Direction (°), Rainfall (mm), Solar Radiation (W/m²)

> **Note**: O Xylene, BP (mmHg), and VWS columns are entirely missing and are excluded.

---

## Installation

```bash
git clone https://github.com/Mohammed-Maazuddin/hyperlocal-air-quality-forecasting.git
cd hyperlocal-air-quality-forecasting
pip install -r requirements.txt
```

---

## Project Structure

```
hyperlocal-air-quality-forecasting/
├── app.py                          # Streamlit dashboard
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml                 # Central configuration
├── data/
│   ├── raw/                        # Raw CSV (not in git)
│   └── processed/                  # Hourly processed CSV (not in git)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_model_interpretation.ipynb
├── src/
│   ├── data/
│   │   ├── download_data.py
│   │   ├── inspect_data.py
│   │   └── prepare_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── evaluation/
│       ├── metrics.py
│       └── plots.py
├── tests/
│   └── test_features.py
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
└── reports/
    ├── model_metrics.csv
    ├── test_predictions.csv
    └── figures/
        ├── eda/
        ├── evaluation/
        └── interpretation/
```

---

## Running the Pipeline

### Step 1 — Download Raw Data
```bash
python -m src.data.download_data
```

### Step 2 — Inspect Raw Data
```bash
python -m src.data.inspect_data
```

### Step 3 — Preprocess (15-min → Hourly)
```bash
python -m src.data.prepare_data
```

### Step 4 — Build Features (verification)
```bash
python -m src.features.build_features
```

### Step 5 — Train Models
```bash
python -m src.models.train
```

### Step 6 — Make a Prediction
```bash
python -m src.models.predict --timestamp "2025-06-15 10:00:00"
```

### Step 7 — Run Tests
```bash
pytest tests/ -v
```

### Step 8 — Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## EDA Key Findings

All findings are derived from actual data execution.

### Dataset Overview
- **17,544 hourly observations** spanning 2 years (Jan 2024 – Dec 2025)
- **Raw PM2.5 range**: 2.96 – 147.55 µg/m³ (no negative values)
- **Mean PM2.5**: 31.83 µg/m³ ± 11.77 µg/m³
- **Median PM2.5**: 31.59 µg/m³  (distribution approximately symmetric)
- **Missing PM2.5 hours**: 27.87% (~4,890 hours)
- 3 columns entirely missing: O Xylene, BP, VWS (excluded from modelling)

### Temporal Patterns
- PM2.5 varies throughout the day (UTC hours correspond to IST = UTC+5:30)
- Higher PM2.5 in cooler months (winter inversion trapping pollutants)
- Rolling 24-hour mean reveals multi-day pollution episodes

### Correlation Insights
- PM2.5 is positively correlated with PM10, CO, NO2 (co-emitted pollutants)
- PM2.5 is negatively correlated with temperature and wind speed (dispersion)
- **Correlation does not establish causation**: meteorological variables may be confounded
  by season, land use, traffic patterns, and industrial activity

### Data Quality
- No negative PM2.5 values in the raw data
- 15-minute sampling is consistent (mode gap = 15 min, no long gaps in raw data)
- Missing windows likely correspond to sensor maintenance or power outages
- ~28% missingness after hourly aggregation means lag features will have NaN values
  for some rows — handled via `SimpleImputer(strategy='median')` in the model pipeline

---

## Feature Engineering

All features respect the **no-future-leakage constraint**: only information available
at issue time **t** is used to predict PM2.5 at **t + 24h**.

| Feature Group | Features |
|---|---|
| **PM2.5 lags** | pm25_lag0, lag1h, lag3h, lag6h, lag12h, lag24h, lag48h, lag168h |
| **Rolling mean** | pm25_roll6h_mean, pm25_roll24h_mean |
| **Rolling std** | pm25_roll24h_std |
| **Calendar (from t+24h)** | target_hour, target_dow, target_month, hour_sin, hour_cos |
| **Co-pollutants at t** | pm10, no, no2, nox, nh3, so2, co, ozone |
| **Meteorology at t** | temp_c, rh, ws, wd, rf, sr |

> **Important**: Calendar features are derived from the **forecast target time**
> (t + 24h), not the issue time. Weather features are from issue time t —
> future weather forecasts are not used.

### Target Definition
```
target = pm25[t + 24h]    # using .shift(-24) on the hourly series
```
This is verified by `verify_target_alignment()` and tested in `tests/test_features.py`.

---

## Time-Series Validation

- **No random shuffling** — time-series integrity is preserved throughout
- **Chronological split**: first 75% for training (Jan 2024 – Jun 2025), last 25% for test (Jun 2025 – Dec 2025)
- **Imputer** (`SimpleImputer`) is fitted only on training data, then applied to test
- No hyperparameter tuning on the final test set

---

## Model Results (Actual Measurements)

Evaluated on the **held-out test set** (3,158 samples, Jun–Dec 2025).

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² | WAPE% |
|---|---|---|---|---|
| **Linear Regression** ✅ | **7.02** | **10.57** | **0.167** | **17.82%** |
| Persistence Baseline | 7.33 | 11.57 | 0.009 | 18.51% |
| Random Forest | 7.57 | 11.06 | 0.088 | 19.21% |
| HistGradientBoosting | 7.70 | 11.38 | 0.035 | 19.55% |
| Seasonal Naive Baseline | 7.98 | 12.15 | −0.084 | 20.24% |

### Model Selection
**Linear Regression** was selected as the best model by **lowest test MAE (7.02 µg/m³)**.

It outperforms both baselines (persistence MAE 7.33, seasonal naive MAE 7.98),
demonstrating that the engineered lag and weather features add genuine predictive value.

### Metric Interpretation
- **MAE 7.02 µg/m³**: On average, predictions are ±7 µg/m³ from actual PM2.5.
  Given typical PM2.5 values of 25–40 µg/m³ at this station, this is an error of ~20%.
- **RMSE > MAE**: Suggests the model occasionally makes larger errors (pollution spikes
  are harder to forecast 24 hours ahead).
- **R² = 0.167**: The model explains ~17% of PM2.5 variance on the test set. This is
  modest but positive, reflecting the difficulty of 24-hour-ahead forecasting with
  ~28% missing predictor values.
- All ML models beat the seasonal naive baseline, confirming the features are useful.

> **Interpretation note**: These metrics are historical evaluations. They do not
> guarantee equivalent performance on future unseen data.

---

## Inference Example

```bash
python -m src.models.predict --timestamp "2025-06-15 10:00:00"
```

**Output**:
```
Issue time   : 2025-06-15 10:00:00+00:00
Target time  : 2025-06-16 10:00:00+00:00
Forecast PM2.5 : 26.34 µg/m³
```

---

## Launching the Dashboard

```bash
streamlit run app.py
```

The dashboard includes:
- Historical PM2.5 trend with rolling mean, date-range filter
- Interactive 24-hour forecast tool (select issue date/hour)
- Model comparison table (actual test set metrics)
- Actual vs predicted chart for any model
- Feature importance chart
- Station and attribution information

---

## Tests

```bash
pytest tests/ -v
```

**13 tests, all passing** (Python 3.13.5, pytest 8.3.4):

| Test | Status |
|---|---|
| `test_target_alignment_24h` | ✅ PASSED |
| `test_target_alignment_utility` | ✅ PASSED |
| `test_lag1h_alignment` | ✅ PASSED |
| `test_lag24h_alignment` | ✅ PASSED |
| `test_rolling_uses_only_past` | ✅ PASSED |
| `test_rolling_no_future_values` | ✅ PASSED |
| `test_chronological_split_no_overlap` | ✅ PASSED |
| `test_chronological_split_fractions` | ✅ PASSED |
| `test_missing_pm25_raises` | ✅ PASSED |
| `test_feature_cols_exclude_target` | ✅ PASSED |
| `test_features_deterministic` | ✅ PASSED |
| `test_calendar_features_from_target_time` | ✅ PASSED |
| `test_cyclical_encoding_range` | ✅ PASSED |

---

## Known Limitations and Future Improvements

### Current Limitations
1. **~28% missing PM2.5 hours** — sensor outages degrade lag feature quality
2. **Median imputation** inside the pipeline is a simple strategy; more sophisticated
   methods (e.g., KNN imputation, model-based imputation) may improve accuracy
3. **Weather at issue time only** — real operational forecasts would incorporate
   NWP weather forecasts for t+1 through t+24
4. **Single station** — results may not generalise to other Hyderabad stations
5. **No live data integration** — the app is a historical demo
6. **R² = 0.167** — 24-hour-ahead PM2.5 forecasting is inherently difficult;
   more data, more features, and ensemble methods could improve this
7. **No hyperparameter tuning** — cross-validated grid search was omitted for simplicity

### Suggested Improvements
- Add NWP weather forecast features (ERA5, IMD)
- Implement rolling-origin cross-validation for more robust model selection
- Try LSTM/GRU models for sequence modelling (requires more data)
- Add SHAP explainability for individual predictions
- Integrate live sensor data via API
- Expand to multiple Hyderabad stations

---

## License

**Code**: MIT License (see [LICENSE](LICENSE))  
**Data**: Verify dataset license at [OpenCity India](https://data.opencity.in).  
The data is attributed to TSPCB Hyderabad. Always check the portal for current terms.

---

## Reproducibility

All random seeds are fixed (`random_state=42`).  
Run the pipeline from scratch:
```bash
python -m src.data.download_data
python -m src.data.prepare_data
python -m src.models.train
pytest tests/ -v
streamlit run app.py
```
