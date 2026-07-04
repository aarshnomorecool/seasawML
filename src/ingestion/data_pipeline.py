"""
SeaSaw Unified Data Pipeline
============================
Orchestrates all data ingestion and produces two outputs:

  1. training_df  — merged GOES + Wind (GOES gives target flux,
                    Wind gives model input features)

  2. validation_df — GRASP electron flux (used later to validate
                     model predictions at Indian longitude)

IMPORTANT — Dynamic Lag is NOT applied here.
Dynamic lag (shifting Wind data by Δt = ΔX / Vsw) is applied in the
Feature Engineering phase (Phase 3), after preprocessing.

Usage
-----
    from src.ingestion.data_pipeline import SeaSawDataPipeline

    pipeline = SeaSawDataPipeline(
        goes_dir="data/raw/goes/",
        wind_mfi_dir="data/raw/wind_mfi/",
        wind_swe_dir="data/raw/wind_swe/",
        grasp_dir="data/raw/grasp/",
    )

    result = pipeline.run()

    training_df    = result["training"]    # shape: (N, 5)  columns: electron_flux, Bx, By, Bz, solar_wind_speed, plasma_density
    validation_df  = result["validation"]  # shape: (M, 1)  column: electron_flux
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Union, Optional

from .goes_reader import GOESReader
from .wind_reader import WindMFIReader, WindSWEReader
from .grasp_reader import GRASPReader

logger = logging.getLogger(__name__)


class SeaSawDataPipeline:
    """
    Full data ingestion pipeline for SeaSaw.

    Parameters
    ----------
    goes_dir     : directory with GOES CDF files
    wind_mfi_dir : directory with Wind MFI CDF files
    wind_swe_dir : directory with Wind SWE CDF files
    grasp_dir    : directory with GRASP ZIP files
    resample_freq: common time resolution to resample all datasets to.
                   Default '5min' (matches GRASP cadence and is fine-grained
                   enough for the 30-45 min forecast horizon).
    goes_flux_var : if you know your GOES flux variable name from the
                    inspector, pass it here. Otherwise auto-detected.
    """

    def __init__(
        self,
        goes_dir: Union[str, Path],
        wind_mfi_dir: Union[str, Path],
        wind_swe_dir: Union[str, Path],
        grasp_dir: Union[str, Path],
        resample_freq: str = "5min",
        goes_flux_var: Optional[str] = None,
        goes_epoch_var: Optional[str] = None,
    ):
        self.goes_dir     = Path(goes_dir)
        self.wind_mfi_dir = Path(wind_mfi_dir)
        self.wind_swe_dir = Path(wind_swe_dir)
        self.grasp_dir    = Path(grasp_dir)
        self.resample_freq = resample_freq

        # Readers
        self._goes_reader  = GOESReader(epoch_var=goes_epoch_var, flux_var=goes_flux_var)
        self._mfi_reader   = WindMFIReader()
        self._swe_reader   = WindSWEReader()
        self._grasp_reader = GRASPReader()

    # ------------------------------------------------------------------
    # Individual loaders
    # ------------------------------------------------------------------

    def load_goes(self) -> pd.DataFrame:
        logger.info(">>> Loading GOES data")
        return self._goes_reader.read_directory(self.goes_dir)

    def load_wind_mfi(self) -> pd.DataFrame:
        logger.info(">>> Loading Wind MFI data")
        return self._mfi_reader.read_directory(self.wind_mfi_dir)

    def load_wind_swe(self) -> pd.DataFrame:
        logger.info(">>> Loading Wind SWE data")
        return self._swe_reader.read_directory(self.wind_swe_dir)

    def load_grasp(self) -> pd.DataFrame:
        logger.info(">>> Loading GRASP data")
        return self._grasp_reader.read_zip_directory(self.grasp_dir)

    # ------------------------------------------------------------------
    # Merge & Align
    # ------------------------------------------------------------------

    def _merge_wind(
        self, mfi: pd.DataFrame, swe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Resample both Wind datasets to self.resample_freq and merge.
        Uses mean aggregation within each time bin.
        """
        mfi_r = mfi.resample(self.resample_freq).mean()
        swe_r = swe.resample(self.resample_freq).mean()

        wind = pd.concat([mfi_r, swe_r], axis=1)
        wind.sort_index(inplace=True)

        logger.info(
            f"Wind merged: {len(wind):,} records  |  "
            f"cols: {list(wind.columns)}"
        )
        return wind

    def _align_goes_wind(
        self, goes: pd.DataFrame, wind: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Resample GOES to self.resample_freq, then inner-join with Wind
        on the shared time index.

        NOTE: We do NOT apply dynamic lag here. Raw timestamps only.
              Lag correction happens in Feature Engineering (Phase 3).
        """
        goes_r = goes.resample(self.resample_freq).mean()
        wind_r = wind.resample(self.resample_freq).mean()

        merged = goes_r.join(wind_r, how="inner")
        merged.sort_index(inplace=True)

        # Rename GOES column to make clear it's the target
        merged.rename(columns={"electron_flux": "goes_electron_flux"}, inplace=True)

        logger.info(
            f"Training DataFrame: {len(merged):,} records  |  "
            f"cols: {list(merged.columns)}  |  "
            f"{merged.index.min().date()} → {merged.index.max().date()}"
        )

        # Report NaN rates
        nan_rates = merged.isna().mean() * 100
        for col, rate in nan_rates.items():
            if rate > 0:
                logger.info(f"  NaN rate — {col}: {rate:.1f}%")

        return merged

    # ------------------------------------------------------------------
    # Full Run
    # ------------------------------------------------------------------

    def run(self, save_processed: bool = True, processed_dir: str = "data/processed") -> dict:
        """
        Load, merge, and align all data sources.

        Parameters
        ----------
        save_processed : if True, save CSVs to processed_dir
        processed_dir  : where to save outputs

        Returns
        -------
        {
          "training"   : pd.DataFrame  (GOES + Wind, aligned to resample_freq)
          "validation" : pd.DataFrame  (GRASP flux only)
          "raw"        : {
              "goes"      : pd.DataFrame,
              "wind_mfi"  : pd.DataFrame,
              "wind_swe"  : pd.DataFrame,
              "grasp"     : pd.DataFrame,
          }
        }
        """
        logger.info("=" * 60)
        logger.info("  SeaSaw Data Pipeline — starting")
        logger.info("=" * 60)

        # Load
        raw_goes     = self.load_goes()
        raw_wind_mfi = self.load_wind_mfi()
        raw_wind_swe = self.load_wind_swe()
        raw_grasp    = self.load_grasp()

        # Merge Wind
        wind = self._merge_wind(raw_wind_mfi, raw_wind_swe)

        # Align GOES + Wind
        training = self._align_goes_wind(raw_goes, wind)

        logger.info("=" * 60)
        logger.info("  Pipeline complete")
        logger.info(f"  Training rows  : {len(training):,}")
        logger.info(f"  Validation rows: {len(raw_grasp):,}")
        logger.info("=" * 60)

        # Optional save
        if save_processed:
            out = Path(processed_dir)
            out.mkdir(parents=True, exist_ok=True)

            training.to_csv(out / "training_raw.csv")
            raw_grasp.to_csv(out / "grasp_validation.csv")

            logger.info(f"  Saved → {out}/training_raw.csv")
            logger.info(f"  Saved → {out}/grasp_validation.csv")

        return {
            "training": training,
            "validation": raw_grasp,
            "raw": {
                "goes"     : raw_goes,
                "wind_mfi" : raw_wind_mfi,
                "wind_swe" : raw_wind_swe,
                "grasp"    : raw_grasp,
            },
        }
