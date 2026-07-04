"""
Phase 3 - Feature Engineering
=============================
Run this script AFTER Phase 2 (preprocessing) has produced
data/processed/training_preprocessed.csv.

Steps applied (see CLAUDE.md Phase 3 Spec):
  1. Dynamic Lag - shift Wind features by dt = DELTA_X / Vsw per row
  2. Lag features on log_electron_flux: t-1, t-2, t-6, t-12, t-24
  3. Rolling mean/std of log_electron_flux over windows 6, 12, 24
  4. Delta features: delta_Bz, delta_speed, delta_flux
  5. Drop rows with NaN created by the above

Usage
-----
    python run_phase3_features.py

Output:
    data/processed/training_features.csv
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phase3_features.log"),
    ],
)

import pandas as pd

from src.features.feature_engineer import SeaSawFeatureEngineer

INPUT_PATH = "data/processed/training_preprocessed.csv"
OUTPUT_PATH = "data/processed/training_features.csv"

if __name__ == "__main__":
    preprocessed_df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)

    engineer = SeaSawFeatureEngineer()
    features_df = engineer.run(preprocessed_df)

    features_df.to_csv(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)
    print(f"\nFeatures DataFrame:")
    print(f"  Shape   : {features_df.shape}")
    print(f"  Columns : {list(features_df.columns)}")
    print(f"  Date range: {features_df.index.min()} -> {features_df.index.max()}")
    print(f"\nSample (first 5 rows):")
    print(features_df.head())

    print("\nFile saved:")
    print(f"  {OUTPUT_PATH}")
    print("\nNext step: run Phase 4 - Multi-Horizon Dataset Builder")
