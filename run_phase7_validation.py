"""
Phase 7 — Validation against GRASP
====================================
Run this script AFTER:
  - Phase 3 has produced data/processed/training_features.csv
  - Phase 1 has produced data/processed/grasp_validation.csv
  - Phase 4 has produced data/processed/dataset_{A,B,C}_*/feature_columns.txt
  - Phase 5/6 have produced models/{xgb,lstm}_horizon_*.{pkl,h5} and
    models/ensemble_weights.json

For each horizon, runs the trained ensemble over the FULL feature history
(not just the Phase 4 test split — GRASP's 1-2 year coverage has no reason
to overlap with the internal GOES test period), matches predictions to
GRASP's independent ground-truth flux by nearest timestamp, and reports:
  - MAE / RMSE / R² (log scale) of model vs GRASP
  - MAE / RMSE / R² (log scale) of a persistence baseline vs GRASP
  - Skill score = 1 - MSE_model / MSE_persistence
    (>0 means the model beats "predict flux stays the same as it is now")

Usage
-----
    python run_phase7_validation.py
    python run_phase7_validation.py --horizon A

Output:
    models/grasp_validation_metrics.json
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
        logging.FileHandler("phase7_validation.log"),
    ],
)
logger = logging.getLogger(__name__)

import pandas as pd

from src.models.xgb_trainer import XGBTrainer
from src.models.lstm_trainer import LSTMTrainer
from src.validation.grasp_validator import (
    log_transform, build_predictions, match_against_grasp,
    regression_metrics, skill_score, MATCH_TOLERANCE,
)

FEATURES_PATH = "data/processed/training_features.csv"
GRASP_PATH = "data/processed/grasp_validation.csv"
MODELS_DIR = Path("models")

HORIZON_DATASET_DIRS = {
    "A": "dataset_A_45min",
    "B": "dataset_B_6hr",
    "C": "dataset_C_12hr",
}
HORIZON_STEPS = {"A": 9, "B": 72, "C": 144}


def load_feature_cols(horizon: str) -> list:
    path = Path("data/processed") / HORIZON_DATASET_DIRS[horizon] / "feature_columns.txt"
    return path.read_text().splitlines()


def validate_horizon(horizon: str, features_df: pd.DataFrame, grasp_log_flux: pd.Series, alpha: float) -> dict:
    feature_cols = load_feature_cols(horizon)
    xgb_model = XGBTrainer.load(MODELS_DIR / f"xgb_horizon_{horizon}.pkl")
    lstm_model = LSTMTrainer.load(MODELS_DIR / f"lstm_horizon_{horizon}.h5")

    pred_df = build_predictions(
        features_df, feature_cols, xgb_model, lstm_model, alpha, HORIZON_STEPS[horizon]
    )
    matched = match_against_grasp(pred_df, grasp_log_flux, MATCH_TOLERANCE)

    if matched.empty:
        logger.warning(f"[{horizon}] No overlapping timestamps between predictions and GRASP data.")
        return {"n_matched": 0}

    result = {
        "n_matched": int(len(matched)),
        "model_vs_grasp": regression_metrics(matched["truth"].to_numpy(), matched["model_pred"].to_numpy()),
        "persistence_vs_grasp": regression_metrics(matched["truth"].to_numpy(), matched["persistence_pred"].to_numpy()),
        "skill_score_vs_persistence": skill_score(
            matched["truth"].to_numpy(), matched["model_pred"].to_numpy(), matched["persistence_pred"].to_numpy()
        ),
    }
    logger.info(f"[{horizon}] {json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeaSaw Phase 7 - Validation against GRASP")
    parser.add_argument("--horizon", choices=["A", "B", "C"], default=None,
                        help="Validate only this horizon (default: all three)")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else ["A", "B", "C"]

    features_df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True).sort_index()
    grasp_df = pd.read_csv(GRASP_PATH, index_col=0, parse_dates=True).sort_index()
    grasp_log_flux = log_transform(grasp_df["electron_flux"].dropna())

    logger.info(f"Feature history: {len(features_df):,} rows")
    logger.info(f"GRASP ground truth: {len(grasp_log_flux):,} valid records")

    with open(MODELS_DIR / "ensemble_weights.json") as f:
        ensemble_weights = json.load(f)

    results = {}
    for horizon in horizons:
        alpha = ensemble_weights[horizon]["alpha"]
        results[horizon] = validate_horizon(horizon, features_df, grasp_log_flux, alpha)

    with open(MODELS_DIR / "grasp_validation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PHASE 7 COMPLETE")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("\nSaved -> models/grasp_validation_metrics.json")
    print("Next step: run Phase 8 - Streamlit Dashboard")
