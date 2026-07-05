"""
Phase 1 — Data Ingestion
========================
Run this script AFTER placing your data files in the expected directories:

  data/raw/goes/       → all GOES .cdf files
  data/raw/wind_mfi/   → all Wind MFI .cdf files  (magnetic field)
  data/raw/wind_swe/   → all Wind SWE .cdf files  (plasma/speed)
  data/raw/grasp/      → all GRASP .zip files

STEP 0: Inspect a CDF file first
---------------------------------
Before running this script, inspect one CDF file from each source
to verify the variable names are correctly auto-detected:

    python -m src.ingestion.cdf_inspector data/raw/goes/your_goes_file.cdf
    python -m src.ingestion.cdf_inspector data/raw/wind_mfi/your_mfi_file.cdf
    python -m src.ingestion.cdf_inspector data/raw/wind_swe/your_swe_file.cdf

If the auto-detected variable names are wrong, pass them explicitly
in the SeaSawDataPipeline constructor below.

STEP 1: Run this script
------------------------
    python run_phase1_ingestion.py

    Add --cleanup-raw to delete the raw GOES/Wind CDF files afterward, once
    they're safely merged into training_raw.csv. Off by default because
    re-running Phase 1 (e.g. after a bugfix) needs the raw files again -
    only pass it once you're done iterating on ingestion for this batch.
    GRASP ZIPs in data/raw/grasp/ are never deleted (manual download, no
    programmatic way to re-fetch them).

Outputs saved to:
    data/processed/training_raw.csv
    data/processed/grasp_validation.csv
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Configure logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phase1_ingestion.log"),
    ],
)

from src.ingestion.data_pipeline import SeaSawDataPipeline

# -----------------------------------------------------------------------
# Configure paths — edit these if your folder structure is different
# -----------------------------------------------------------------------
GOES_DIR     = "data/raw/goes/"
WIND_MFI_DIR = "data/raw/wind_mfi/"
WIND_SWE_DIR = "data/raw/wind_swe/"
GRASP_DIR    = "data/raw/grasp/"

# If you know the exact CDF variable names (from cdf_inspector.py), set them here.
# Leave as None for auto-detection.
GOES_FLUX_VAR  = None   # e.g. "e2_flux" or "AvgDiffElectronFlux"
GOES_EPOCH_VAR = None   # almost always "Epoch"

# -----------------------------------------------------------------------

def cleanup_raw_files(dirs: list) -> None:
    """Delete downloaded CDF files (not the directories themselves) once
    they're safely merged into data/processed/training_raw.csv."""
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        freed = 0
        n = 0
        for f in d.glob("*.cdf"):
            freed += f.stat().st_size
            f.unlink()
            n += 1
        if n:
            print(f"  Deleted {n} file(s) from {d}  ({freed / 1024 / 1024:.1f} MB freed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeaSaw Phase 1 - Data Ingestion")
    parser.add_argument(
        "--cleanup-raw", action="store_true",
        help="Delete raw GOES/Wind CDF files after a successful run to save disk space "
             "(GRASP ZIPs are never touched - they require manual re-download)."
    )
    args = parser.parse_args()

    pipeline = SeaSawDataPipeline(
        goes_dir     = GOES_DIR,
        wind_mfi_dir = WIND_MFI_DIR,
        wind_swe_dir = WIND_SWE_DIR,
        grasp_dir    = GRASP_DIR,
        resample_freq="5min",
        goes_flux_var  = GOES_FLUX_VAR,
        goes_epoch_var = GOES_EPOCH_VAR,
    )

    result = pipeline.run(save_processed=True, processed_dir="data/processed/")

    training   = result["training"]
    validation = result["validation"]

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"\nTraining DataFrame:")
    print(f"  Shape   : {training.shape}")
    print(f"  Columns : {list(training.columns)}")
    print(f"  Date range: {training.index.min()} -> {training.index.max()}")
    print(f"\nSample (first 5 rows):")
    print(training.head())

    print(f"\nValidation DataFrame (GRASP):")
    if validation.empty:
        print("  (skipped - no GRASP ZIPs found in data/raw/grasp/)")
    else:
        print(f"  Shape   : {validation.shape}")
        print(f"  Date range: {validation.index.min()} -> {validation.index.max()}")
        print(f"\nSample (first 5 rows):")
        print(validation.head())

    print("\nFiles saved:")
    print("  data/processed/training_raw.csv")
    if not validation.empty:
        print("  data/processed/grasp_validation.csv")

    if args.cleanup_raw:
        print("\nCleaning up raw CDF files (--cleanup-raw)...")
        cleanup_raw_files([GOES_DIR, WIND_MFI_DIR, WIND_SWE_DIR])

    print("\nNext step: run Phase 2 - Preprocessing")
