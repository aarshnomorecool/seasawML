"""
SeaSaw GRASP Validation
========================
Phase 7. GRASP/GSAT (ISRO, Indian longitude) is an independent measurement
that was never used in training — the real test of whether the model
generalizes beyond the specific GOES sensor/location it was trained on.

Why full-history inference, not the Phase 4 test split
--------------------------------------------------------
GRASP's 1-2 year coverage window has no reason to line up with the last
10% (chronologically) of the GOES+Wind training period. So this module
runs the trained ensemble over the FULL engineered feature history
(training_features.csv), labels each prediction with the real calendar
timestamp it is actually a forecast FOR (row time + horizon), and only
keeps whichever of those predictions happen to land near a GRASP
observation.

Persistence baseline
---------------------
"Does the model beat predicting flux stays the same as it is right now?"
The naive forecast for row i is simply that row's own current
log_electron_flux value, carried forward unchanged to t + horizon. Skill
score = 1 - MSE_model / MSE_persistence (matched against the *same* GRASP
ground truth so the comparison is apples-to-apples).

Timestamp alignment
--------------------
GRASP's TXT timestamps are parsed as-is from the source files and are not
guaranteed to fall on the same 5-min grid boundaries as GOES/Wind's
resampled index, so matching uses a tolerance-based nearest-time join
(pd.merge_asof), not exact index equality.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

EPSILON = 1e-10
RESAMPLE_FREQ_MINUTES = 5  # matches the frozen resample cadence used since Phase 1
MATCH_TOLERANCE = pd.Timedelta(minutes=2, seconds=30)  # half of the 5-min cadence

TARGET_COL = "log_electron_flux"


def log_transform(flux: pd.Series) -> pd.Series:
    return np.log10(flux + EPSILON)


def build_predictions(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    xgb_model,
    lstm_model,
    alpha: float,
    horizon_steps: int,
) -> pd.DataFrame:
    """
    Run the ensemble over every row of features_df, returning a DataFrame
    indexed by the timestamp each prediction is a forecast FOR (row time +
    horizon), with columns:
        model_pred        - alpha-blended XGBoost + LSTM prediction (log flux)
        persistence_pred  - that row's own current log flux, held constant
    """
    from src.models.lstm_trainer import build_sequences  # local import avoids a hard TF dependency for callers that only need metrics helpers

    sequence_length = lstm_model.input_shape[1]
    X_all = features_df[feature_cols].to_numpy()

    xgb_preds = xgb_model.predict(X_all)

    dummy_y = np.zeros(len(X_all))
    X_seq, _ = build_sequences(X_all, dummy_y, sequence_length)
    lstm_preds = lstm_model.predict(X_seq, verbose=0).ravel()

    xgb_aligned = xgb_preds[sequence_length - 1:]
    ensemble_preds = alpha * lstm_preds + (1 - alpha) * xgb_aligned

    aligned_index = features_df.index[sequence_length - 1:]
    predicted_time = aligned_index + pd.Timedelta(minutes=RESAMPLE_FREQ_MINUTES * horizon_steps)

    persistence_preds = features_df[TARGET_COL].to_numpy()[sequence_length - 1:]

    return pd.DataFrame(
        {"model_pred": ensemble_preds, "persistence_pred": persistence_preds},
        index=predicted_time,
    ).sort_index()


def match_against_grasp(
    pred_df: pd.DataFrame, grasp_log_flux: pd.Series, tolerance: pd.Timedelta = MATCH_TOLERANCE
) -> pd.DataFrame:
    """
    Nearest-time join of pred_df (model_pred, persistence_pred) against
    GRASP's own log-flux series, within `tolerance`. One shared "truth"
    column so model and persistence are scored against identical rows.
    """
    # merge_asof requires both sides to share the same datetime64 resolution.
    # date_range, CSV parse_dates, and Timedelta arithmetic can each land on
    # a different resolution (ns vs us) in pandas >= 2, so normalize both.
    pred_df = pred_df.copy()
    pred_df.index = pd.DatetimeIndex(pred_df.index).as_unit("ns")

    grasp_frame = grasp_log_flux.rename("truth").to_frame().sort_index()
    grasp_frame.index = pd.DatetimeIndex(grasp_frame.index).as_unit("ns")

    merged = pd.merge_asof(
        pred_df, grasp_frame, left_index=True, right_index=True,
        direction="nearest", tolerance=tolerance,
    ).dropna()
    return merged


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_persistence: np.ndarray) -> float:
    """1 - MSE_model / MSE_persistence. >0 means the model beats persistence."""
    mse_model = mean_squared_error(y_true, y_pred)
    mse_persistence = mean_squared_error(y_true, y_persistence)
    if mse_persistence == 0:
        return float("nan")
    return float(1 - mse_model / mse_persistence)
