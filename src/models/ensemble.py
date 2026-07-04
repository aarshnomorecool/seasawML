"""
SeaSaw Ensemble
================
Phase 6. Combines the per-horizon XGBoost and LSTM predictions:

    P_ensemble = alpha * P_LSTM + (1 - alpha) * P_XGB

alpha is grid-searched over [0.0, 0.1, ..., 1.0] on the validation set
(CLAUDE.md Phase 6 Spec), picking the value that minimizes validation RMSE.

Alignment
---------
XGBoost predicts on every row of X. The LSTM only produces a prediction
from row (sequence_length - 1) onward, since each prediction needs
sequence_length rows of history. `align_predictions()` slices the XGBoost
predictions and ground truth down to that same range so the two models'
outputs can be blended 1:1.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

ALPHA_GRID = tuple(round(a, 1) for a in np.arange(0.0, 1.01, 0.1))


def align_predictions(
    xgb_preds: np.ndarray, lstm_preds: np.ndarray, y: np.ndarray, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xgb_aligned = xgb_preds[sequence_length - 1:]
    y_aligned = y[sequence_length - 1:]
    if not (len(xgb_aligned) == len(lstm_preds) == len(y_aligned)):
        raise ValueError(
            f"Length mismatch after alignment: xgb={len(xgb_aligned)}, "
            f"lstm={len(lstm_preds)}, y={len(y_aligned)}"
        )
    return xgb_aligned, lstm_preds, y_aligned


def find_best_alpha(
    xgb_preds: np.ndarray, lstm_preds: np.ndarray, y_true: np.ndarray
) -> Tuple[float, float]:
    """Grid search alpha over ALPHA_GRID, return (best_alpha, best_val_rmse)."""
    best_alpha, best_rmse = None, np.inf
    for alpha in ALPHA_GRID:
        blend = alpha * lstm_preds + (1 - alpha) * xgb_preds
        rmse = float(np.sqrt(mean_squared_error(y_true, blend)))
        logger.info(f"  alpha={alpha:.1f} val_rmse={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
    return best_alpha, best_rmse


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def ensemble_predict(xgb_preds: np.ndarray, lstm_preds: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * lstm_preds + (1 - alpha) * xgb_preds
