"""
Phase 4 — Multi-Horizon Dataset Builder
========================================
Run this script AFTER Phase 3 (feature engineering) has produced
data/processed/training_features.csv.

Builds 3 separate supervised datasets (see CLAUDE.md Phase 4 Spec), one
per forecast horizon:

    Dataset A — target = log_electron_flux at t + 9   steps (45 min)
    Dataset B — target = log_electron_flux at t + 72  steps (6 hours)
    Dataset C — target = log_electron_flux at t + 144 steps (12 hours)

Each dataset gets a chronological (NOT random) 80/10/10 train/val/test
split, since shuffling would leak future information into training.

Feature selection
------------------
X uses every engineered column except:
  - goes_electron_flux      : Flag 1 mandates the model only ever sees the
                               log-transformed flux, no exceptions.
  - <wind>_lag (unscaled)    : redundant with <wind>_scaled_lag — a
                               RobustScaler is just an affine transform, so
                               keeping both duplicates the same information.
log_electron_flux itself is kept as an input feature: at prediction time
the *current* flux is known, only the value at t + horizon is the target.

Usage
-----
    python run_phase4_dataset_builder.py

Output:
    data/processed/dataset_A_45min/{X,y}_{train,val,test}.npy
    data/processed/dataset_B_6hr/...
    data/processed/dataset_C_12hr/...
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phase4_dataset_builder.log"),
    ],
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/training_features.csv"
OUTPUT_DIR = Path("data/processed")

TARGET_COL = "log_electron_flux"

EXCLUDE_FROM_X = [
    "goes_electron_flux",
    "Bx_lag", "By_lag", "Bz_lag", "solar_wind_speed_lag", "plasma_density_lag",
]

HORIZONS = {
    "A_45min": 9,
    "B_6hr": 72,
    "C_12hr": 144,
}

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1  # remainder (0.1) goes to test


def build_horizon_dataset(
    df: pd.DataFrame, horizon_steps: int, feature_cols: List[str]
) -> Dict[str, np.ndarray]:
    y = df[TARGET_COL].shift(-horizon_steps)
    valid = y.notna()

    X = df.loc[valid, feature_cols].to_numpy()
    y = y.loc[valid].to_numpy()

    n = len(X)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    return {
        "X_train": X[:train_end], "y_train": y[:train_end],
        "X_val": X[train_end:val_end], "y_val": y[train_end:val_end],
        "X_test": X[val_end:], "y_test": y[val_end:],
    }


def save_dataset(splits: Dict[str, np.ndarray], out_dir: Path, feature_cols: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in splits.items():
        np.save(out_dir / f"{name}.npy", arr)
    (out_dir / "feature_columns.txt").write_text("\n".join(feature_cols))


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_X]

    logger.info("=" * 60)
    logger.info("  Phase 4 - Multi-Horizon Dataset Builder")
    logger.info("=" * 60)
    logger.info(f"Input rows: {len(df):,}")
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)

    for name, horizon_steps in HORIZONS.items():
        splits = build_horizon_dataset(df, horizon_steps, feature_cols)
        out_dir = OUTPUT_DIR / f"dataset_{name}"
        save_dataset(splits, out_dir, feature_cols)

        logger.info(
            f"Dataset {name} (horizon={horizon_steps} steps): "
            f"train={splits['X_train'].shape}, val={splits['X_val'].shape}, "
            f"test={splits['X_test'].shape} -> saved to {out_dir}"
        )
        print(f"\n{name} (t+{horizon_steps} steps):")
        print(f"  X_train {splits['X_train'].shape}  y_train {splits['y_train'].shape}")
        print(f"  X_val   {splits['X_val'].shape}  y_val   {splits['y_val'].shape}")
        print(f"  X_test  {splits['X_test'].shape}  y_test  {splits['y_test'].shape}")
        print(f"  Saved -> {out_dir}")

    print("\nNext step: run Phase 5 - Model Training")
