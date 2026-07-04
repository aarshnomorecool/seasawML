"""
Phase 6 — Weighted Ensemble
============================
Run this script AFTER Phase 5 has produced models/xgb_horizon_{A,B,C}.pkl
and models/lstm_horizon_{A,B,C}.h5.

For each horizon:
    P_ensemble = alpha * P_LSTM + (1 - alpha) * P_XGB

alpha is grid-searched over [0.0, 0.1, ..., 1.0] against the validation
split, then the chosen alpha is reported on the held-out test split
alongside each standalone model's test metrics for comparison.

Usage
-----
    python run_phase6_ensemble.py
    python run_phase6_ensemble.py --horizon A

Output:
    models/ensemble_weights.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phase6_ensemble.log"),
    ],
)
logger = logging.getLogger(__name__)

import numpy as np

from src.models.xgb_trainer import XGBTrainer
from src.models.lstm_trainer import LSTMTrainer, build_sequences
from src.models.ensemble import align_predictions, find_best_alpha, regression_metrics, ensemble_predict

DATASET_DIRS = {
    "A": "data/processed/dataset_A_45min",
    "B": "data/processed/dataset_B_6hr",
    "C": "data/processed/dataset_C_12hr",
}

MODELS_DIR = Path("models")


def load_horizon_data(horizon: str):
    d = Path(DATASET_DIRS[horizon])
    return (
        np.load(d / "X_val.npy"), np.load(d / "y_val.npy"),
        np.load(d / "X_test.npy"), np.load(d / "y_test.npy"),
    )


def run_horizon(horizon: str) -> dict:
    logger.info(f"[{horizon}] Loading models and data ...")
    X_val, y_val, X_test, y_test = load_horizon_data(horizon)

    xgb_model = XGBTrainer.load(MODELS_DIR / f"xgb_horizon_{horizon}.pkl")
    lstm_model = LSTMTrainer.load(MODELS_DIR / f"lstm_horizon_{horizon}.h5")
    sequence_length = lstm_model.input_shape[1]

    # --- Validation: find best alpha ---
    xgb_val_preds = xgb_model.predict(X_val)
    X_val_seq, y_val_seq = build_sequences(X_val, y_val, sequence_length)
    lstm_val_preds = lstm_model.predict(X_val_seq, verbose=0).ravel()
    xgb_val_aligned, lstm_val_aligned, y_val_aligned = align_predictions(
        xgb_val_preds, lstm_val_preds, y_val, sequence_length
    )

    logger.info(f"[{horizon}] Grid searching alpha on validation set:")
    best_alpha, best_val_rmse = find_best_alpha(xgb_val_aligned, lstm_val_aligned, y_val_aligned)
    logger.info(f"[{horizon}] BEST alpha={best_alpha} val_rmse={best_val_rmse:.4f}")

    # --- Test: report ensemble vs standalone baselines ---
    xgb_test_preds = xgb_model.predict(X_test)
    X_test_seq, y_test_seq = build_sequences(X_test, y_test, sequence_length)
    lstm_test_preds = lstm_model.predict(X_test_seq, verbose=0).ravel()
    xgb_test_aligned, lstm_test_aligned, y_test_aligned = align_predictions(
        xgb_test_preds, lstm_test_preds, y_test, sequence_length
    )

    ensemble_test_preds = ensemble_predict(xgb_test_aligned, lstm_test_aligned, best_alpha)

    result = {
        "alpha": best_alpha,
        "val_rmse": best_val_rmse,
        "sequence_length": int(sequence_length),
        "test_ensemble": regression_metrics(y_test_aligned, ensemble_test_preds),
        "test_xgb_only": regression_metrics(y_test_aligned, xgb_test_aligned),
        "test_lstm_only": regression_metrics(y_test_aligned, lstm_test_aligned),
    }
    logger.info(f"[{horizon}] Test metrics: {json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeaSaw Phase 6 - Weighted Ensemble")
    parser.add_argument("--horizon", choices=["A", "B", "C"], default=None,
                        help="Run only this horizon (default: all three)")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else ["A", "B", "C"]

    results = {h: run_horizon(h) for h in horizons}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "ensemble_weights.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE 6 COMPLETE")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("\nSaved -> models/ensemble_weights.json")
    print("Next step: run Phase 7 - Validation against GRASP")
