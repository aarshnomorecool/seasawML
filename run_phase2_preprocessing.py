"""
Phase 2 - Preprocessing
=======================
Run this script AFTER Phase 1 (data ingestion) has produced
data/processed/training_raw.csv.

Steps applied (see CLAUDE.md Phase 2 Spec):
  1. Drop rows with missing target (goes_electron_flux)
  2. Spike removal (>5sigma from rolling median, window=12 steps)
  3. Linear interpolation for gaps <= 6 steps; drop rows for longer gaps
  4. Log transform (log_electron_flux = log10(goes_electron_flux + 1e-10))
  5. RobustScaler normalization of Wind features
  6. Save scalers to models/scalers.pkl

Usage
-----
    python run_phase2_preprocessing.py

Outputs:
    data/processed/training_preprocessed.csv
    models/scalers.pkl
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phase2_preprocessing.log"),
    ],
)

import pandas as pd

from src.preprocessing.preprocessor import SeaSawPreprocessor

INPUT_PATH = "data/processed/training_raw.csv"
OUTPUT_PATH = "data/processed/training_preprocessed.csv"
SCALERS_PATH = "models/scalers.pkl"

if __name__ == "__main__":
    raw_df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)

    preprocessor = SeaSawPreprocessor()
    processed_df, scalers = preprocessor.run(raw_df)

    processed_df.to_csv(OUTPUT_PATH)
    preprocessor.save_scalers(scalers, SCALERS_PATH)

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"\nProcessed DataFrame:")
    print(f"  Shape   : {processed_df.shape}")
    print(f"  Columns : {list(processed_df.columns)}")
    print(f"  Date range: {processed_df.index.min()} -> {processed_df.index.max()}")
    print(f"\nSample (first 5 rows):")
    print(processed_df.head())

    print("\nFiles saved:")
    print(f"  {OUTPUT_PATH}")
    print(f"  {SCALERS_PATH}")
    print("\nNext step: run Phase 3 - Feature Engineering")
