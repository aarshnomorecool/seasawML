"""
Phase 5 — Model Training
========================
Run this script AFTER Phase 4 has produced the per-horizon datasets in
data/processed/dataset_{A_45min,B_6hr,C_12hr}/.

Trains, for each requested horizon:
  - XGBoost  (grid search + early stopping)  -> models/xgb_horizon_{H}.pkl
  - LSTM     (direct single-output)          -> models/lstm_horizon_{H}.h5

See TRAINING_GUIDE.md: XGBoost is fine on a laptop CPU; LSTM should be
trained on a GPU (Colab/Kaggle/local NVIDIA) — this script will still run
on CPU (e.g. for a quick structural smoke test) but warns if no GPU is found.

Usage
-----
    python run_phase5_training.py                     # both models, all 3 horizons
    python run_phase5_training.py --xgb-only          # XGBoost only, all horizons
    python run_phase5_training.py --xgb-only --horizon A
    python run_phase5_training.py --lstm-only --epochs 30 --sequence-length 288
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
        logging.FileHandler("phase5_training.log"),
    ],
)
logger = logging.getLogger(__name__)

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.xgb_trainer import XGBTrainer
from src.models.lstm_trainer import LSTMTrainer, build_sequences

DATASET_DIRS = {
    "A": "data/processed/dataset_A_45min",
    "B": "data/processed/dataset_B_6hr",
    "C": "data/processed/dataset_C_12hr",
}

MODELS_DIR = Path("models")


def load_horizon_data(horizon: str):
    d = Path(DATASET_DIRS[horizon])
    return (
        np.load(d / "X_train.npy"), np.load(d / "y_train.npy"),
        np.load(d / "X_val.npy"), np.load(d / "y_val.npy"),
        np.load(d / "X_test.npy"), np.load(d / "y_test.npy"),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_xgb_for_horizon(horizon: str, X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    logger.info(f"[{horizon}] Training XGBoost ...")
    trainer = XGBTrainer(horizon=horizon)
    trainer.train(X_train, y_train, X_val, y_val)

    test_metrics = regression_metrics(y_test, trainer.model.predict(X_test))
    logger.info(f"[{horizon}] XGBoost test metrics: {test_metrics}")

    trainer.save(MODELS_DIR / f"xgb_horizon_{horizon}.pkl")
    return {"best_params": trainer.best_params, "val_rmse": trainer.best_val_rmse, "test": test_metrics}


def train_lstm_for_horizon(
    horizon: str, X_train, y_train, X_val, y_val, X_test, y_test, epochs: int, sequence_length: int
) -> dict:
    logger.info(f"[{horizon}] Training LSTM (sequence_length={sequence_length}) ...")
    trainer = LSTMTrainer(horizon=horizon, sequence_length=sequence_length)
    trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)

    X_test_seq, y_test_seq = build_sequences(X_test, y_test, sequence_length)
    preds = trainer.model.predict(X_test_seq, verbose=0).ravel()
    test_metrics = regression_metrics(y_test_seq, preds)
    logger.info(f"[{horizon}] LSTM test metrics: {test_metrics}")

    trainer.save(MODELS_DIR / f"lstm_horizon_{horizon}.h5")
    return {"final_val_loss": float(min(trainer.history.history["val_loss"])), "test": test_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeaSaw Phase 5 - Model Training")
    parser.add_argument("--horizon", choices=["A", "B", "C"], default=None,
                        help="Train only this horizon (default: all three)")
    parser.add_argument("--xgb-only", action="store_true", help="Train XGBoost only")
    parser.add_argument("--lstm-only", action="store_true", help="Train LSTM only")
    parser.add_argument("--epochs", type=int, default=50, help="Max LSTM epochs")
    parser.add_argument("--sequence-length", type=int, default=288, help="LSTM input sequence length")
    args = parser.parse_args()

    if args.xgb_only and args.lstm_only:
        parser.error("--xgb-only and --lstm-only are mutually exclusive")

    run_xgb = not args.lstm_only
    run_lstm = not args.xgb_only

    if run_lstm:
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"):
            logger.warning(
                "No GPU detected. LSTM training on CPU will be very slow for a full "
                "dataset (see TRAINING_GUIDE.md) - fine for a quick smoke test only."
            )

    horizons = [args.horizon] if args.horizon else ["A", "B", "C"]

    results = {}
    for horizon in horizons:
        X_train, y_train, X_val, y_val, X_test, y_test = load_horizon_data(horizon)
        results[horizon] = {}

        if run_xgb:
            results[horizon]["xgb"] = train_xgb_for_horizon(
                horizon, X_train, y_train, X_val, y_val, X_test, y_test
            )
        if run_lstm:
            results[horizon]["lstm"] = train_lstm_for_horizon(
                horizon, X_train, y_train, X_val, y_val, X_test, y_test,
                epochs=args.epochs, sequence_length=args.sequence_length,
            )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "training_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("\nFiles saved to models/ (xgb_horizon_*.pkl, lstm_horizon_*.h5, training_metrics.json)")
    print("Next step: run Phase 6 - Weighted Ensemble")
