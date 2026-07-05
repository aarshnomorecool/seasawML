"""
SeaSaw Streamlit Dashboard
==========================
Phase 8. Reads the artifacts produced by Phases 1-6 (features, trained
models, ensemble weights) and, if present, Phase 7's GRASP comparison -
no training or heavy computation happens here.

Run with:
    streamlit run src/dashboard/app.py
"""

import sys
import json
from pathlib import Path

# streamlit runs this file with its own directory as sys.path[0], not the
# project root, so "from src.models..." would otherwise fail to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.xgb_trainer import XGBTrainer
from src.models.lstm_trainer import LSTMTrainer

st.set_page_config(page_title="SeaSaw - Space Weather Forecast", layout="wide")

FEATURES_PATH = "data/processed/training_features.csv"
MODELS_DIR = Path("models")
DATASET_DIRS = {
    "A": "data/processed/dataset_A_45min",
    "B": "data/processed/dataset_B_6hr",
    "C": "data/processed/dataset_C_12hr",
}
HORIZON_STEPS = {"A": 9, "B": 72, "C": 144}
HORIZON_LABELS = {"A": "30-45 min", "B": "6 hours", "C": "12 hours"}
RESAMPLE_MINUTES = 5
SEQUENCE_LENGTH = 288  # CLAUDE.md LSTM Strategy - fixed 24h lookback for all horizons


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True).sort_index()


@st.cache_data
def load_ensemble_weights() -> dict:
    with open(MODELS_DIR / "ensemble_weights.json") as f:
        return json.load(f)


@st.cache_resource
def load_horizon_models(horizon: str):
    xgb_model = XGBTrainer.load(MODELS_DIR / f"xgb_horizon_{horizon}.pkl")
    lstm_model = LSTMTrainer.load(MODELS_DIR / f"lstm_horizon_{horizon}.h5")
    feature_cols = (Path(DATASET_DIRS[horizon]) / "feature_columns.txt").read_text().splitlines()
    return xgb_model, lstm_model, feature_cols


def forecast_latest(features_df: pd.DataFrame, horizon: str, alpha: float):
    """Run the ensemble on the most recent window only (cheap - one prediction,
    not the full-history inference Phase 7 does for GRASP validation)."""
    xgb_model, lstm_model, feature_cols = load_horizon_models(horizon)
    sequence_length = lstm_model.input_shape[1]

    X = features_df[feature_cols].to_numpy()
    if len(X) < sequence_length:
        return None

    xgb_pred = xgb_model.predict(X[-1:])[0]
    X_seq = X[-sequence_length:].reshape(1, sequence_length, -1).astype(np.float32)
    lstm_pred = lstm_model.predict(X_seq, verbose=0).ravel()[0]

    log_pred = alpha * lstm_pred + (1 - alpha) * xgb_pred
    flux_pred = 10 ** log_pred  # inverse of CLAUDE.md's to_log (Flag 1)
    predicted_time = features_df.index[-1] + pd.Timedelta(minutes=RESAMPLE_MINUTES * HORIZON_STEPS[horizon])
    return flux_pred, log_pred, predicted_time


def confidence_label(test_r2: float) -> str:
    if test_r2 >= 0.5:
        return "High"
    if test_r2 >= 0:
        return "Moderate"
    return "Low (test R2 < 0 - model underperforms a mean-flux baseline on held-out data)"


# ----------------------------------------------------------------------
features_df = load_features()
ensemble_weights = load_ensemble_weights()

st.title("SeaSaw - Space Environment Adaptive Solar Activity Warning")
st.caption("ISRO Hackathon PS-14 - >2 MeV electron flux forecasting at GEO")

# 1. Historical electron flux -------------------------------------------------
st.header("1. Historical Electron Flux (GOES, >2 MeV)")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=features_df.index, y=features_df["goes_electron_flux"],
    mode="lines", name="Electron flux", line=dict(width=1),
))
fig.update_yaxes(type="log", title="Flux (particles / cm2 / s / sr)")
fig.update_xaxes(title="Time")
fig.update_layout(height=350, margin=dict(t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# 2. Current solar wind conditions -------------------------------------------
st.header("2. Current Solar Wind Conditions")
st.caption("Dynamic-lag aligned Wind measurements (raw units, not scaled)")
latest = features_df.iloc[-1]
cols = st.columns(5)
cols[0].metric("Bx (nT)", f"{latest['Bx_lag']:.2f}")
cols[1].metric("By (nT)", f"{latest['By_lag']:.2f}")
cols[2].metric("Bz (nT)", f"{latest['Bz_lag']:.2f}")
cols[3].metric("Solar wind speed (km/s)", f"{latest['solar_wind_speed_lag']:.1f}")
cols[4].metric("Plasma density (cm-3)", f"{latest['plasma_density_lag']:.2f}")
st.caption(f"As of {features_df.index[-1]}")

# 3-5. Forecasts per horizon ---------------------------------------------------
st.header("3-5. Forecasts")
forecast_headers = {"A": "3. 30-45 min forecast", "B": "4. 6-hour forecast", "C": "5. 12-hour forecast"}
forecast_cols = st.columns(3)
for col, horizon in zip(forecast_cols, ["A", "B", "C"]):
    with col:
        st.subheader(forecast_headers[horizon])
        alpha = ensemble_weights[horizon]["alpha"]
        result = forecast_latest(features_df, horizon, alpha)
        if result is None:
            st.warning(f"Need >= {SEQUENCE_LENGTH} rows of history to forecast.")
        else:
            flux_pred, log_pred, predicted_time = result
            st.metric("Predicted flux (particles/cm2/s/sr)", f"{flux_pred:.2e}")
            st.caption(f"Forecast for {predicted_time}")
            test_r2 = ensemble_weights[horizon]["test_ensemble"]["r2"]
            st.caption(f"Confidence: {confidence_label(test_r2)}  (test R2={test_r2:.2f})")

# 6. Model validation metrics --------------------------------------------------
st.header("6. Model Validation Metrics (held-out chronological test split)")
rows = []
for h in ["A", "B", "C"]:
    m = ensemble_weights[h]
    rows.append({
        "Horizon": HORIZON_LABELS[h],
        "alpha (LSTM weight)": m["alpha"],
        "MAE (log10 flux)": round(m["test_ensemble"]["mae"], 4),
        "RMSE (log10 flux)": round(m["test_ensemble"]["rmse"], 4),
        "R2": round(m["test_ensemble"]["r2"], 4),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "Metrics computed on log10(flux) scale against a chronological test split the model "
    "never trained on. Negative R2 means the model underperforms predicting the mean."
)

# 7. GRASP comparison -----------------------------------------------------------
st.header("7. GRASP Validation (ISRO ground truth)")
grasp_metrics_path = MODELS_DIR / "grasp_validation_metrics.json"
if grasp_metrics_path.exists():
    with open(grasp_metrics_path) as f:
        st.json(json.load(f))
else:
    st.info(
        "GRASP validation is pending. GRASP/GSAT data requires manual download from ISRO's "
        "PRADAN portal (auth-gated, not automatable) - place ZIPs in data/raw/grasp/, "
        "re-run Phase 1, then `python run_phase7_validation.py` to populate this panel."
    )
