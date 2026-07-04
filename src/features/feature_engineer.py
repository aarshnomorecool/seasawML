"""
SeaSaw Feature Engineering
==========================
Phase 3 of the pipeline. Takes the preprocessed DataFrame produced by
Phase 2 (data/processed/training_preprocessed.csv) and builds the
model-ready feature set:

  1. Dynamic Lag  (signature innovation - see CLAUDE.md Flag 2)
  2. Lag features on the target
  3. Rolling statistics on the target
  4. Delta features
  5. Drop rows with NaN created by the above (edge effects of shifts/windows)

Dynamic Lag detail
------------------
The solar wind measured at the Wind spacecraft (L1) at time t does not
affect Earth until t + dt, where dt = DELTA_X / Vsw. For each row i we
estimate dt from that row's own (contemporaneous) solar wind speed and
look *backward* that many steps for the Wind features:

    aligned_row[i] = source_row[i - steps(i)]

This is a first-order approximation (the true dt is governed by the
speed of the parcel at the moment it left L1, not the speed measured
when it arrives), but it is the standard, deployable approach: a
forward-scatter using departure-time speed is not surjective (variable
per-row shifts leave scattered destination rows with no source and
others with collisions), which loses the majority of rows once
combined with the later dropna. A backward gather is only undefined
for the first max(steps) rows of the series and mirrors exactly what a
live nowcasting system would compute (current Vsw -> look back into
the history buffer).

Usage
-----
    from src.features.feature_engineer import SeaSawFeatureEngineer

    fe = SeaSawFeatureEngineer()
    features_df = fe.run(preprocessed_df)
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DELTA_X_KM = 1.5e6  # Wind (L1) to Earth's bow shock

RAW_WIND_FEATURES = ["Bx", "By", "Bz", "solar_wind_speed", "plasma_density"]
SCALED_WIND_FEATURES = [f"{c}_scaled" for c in RAW_WIND_FEATURES]

TARGET_LOG_COL = "log_electron_flux"


def _gather_backward(values: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """
    For each destination index i, pull the value from i - steps[i].
    Destination indices whose source would be negative are left as NaN.
    """
    n = len(values)
    idx = np.arange(n)
    src_idx = idx - steps
    valid = src_idx >= 0

    result = np.full(n, np.nan)
    result[valid] = values[src_idx[valid]]
    return result


class SeaSawFeatureEngineer:
    """
    Phase 3 feature engineering pipeline.

    Parameters
    ----------
    resample_freq_minutes : cadence of the time series (matches Phase 1/2, 5 min)
    lag_steps             : lags (in steps) to add for the target flux
    rolling_windows       : window sizes (in steps) for rolling mean/std of the target flux
    """

    def __init__(
        self,
        resample_freq_minutes: int = 5,
        lag_steps: List[int] = [1, 2, 6, 12, 24],
        rolling_windows: List[int] = [6, 12, 24],
    ):
        self.resample_freq_minutes = resample_freq_minutes
        self.lag_steps = lag_steps
        self.rolling_windows = rolling_windows

    # ------------------------------------------------------------------
    # Step 1 - Dynamic Lag
    # ------------------------------------------------------------------

    def _compute_lag_steps(self, speed: pd.Series) -> np.ndarray:
        delta_t_minutes = DELTA_X_KM / (speed * 60.0)
        steps = np.round(delta_t_minutes / self.resample_freq_minutes)
        steps = steps.to_numpy()
        steps = np.nan_to_num(steps, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(steps, 0, None).astype(int)

    def apply_dynamic_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        steps = self._compute_lag_steps(df["solar_wind_speed"])

        for col in RAW_WIND_FEATURES + SCALED_WIND_FEATURES:
            df[f"{col}_lag"] = _gather_backward(df[col].to_numpy(), steps)

        df = df.drop(columns=RAW_WIND_FEATURES + SCALED_WIND_FEATURES)

        logger.info(
            f"Step 1 - dynamic lag applied "
            f"(mean shift {steps.mean():.1f} steps, max {steps.max()} steps)"
        )
        return df

    # ------------------------------------------------------------------
    # Step 2 - Lag features
    # ------------------------------------------------------------------

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for lag in self.lag_steps:
            df[f"log_flux_lag_{lag}"] = df[TARGET_LOG_COL].shift(lag)
        logger.info(f"Step 2 - added lag features {self.lag_steps}")
        return df

    # ------------------------------------------------------------------
    # Step 3 - Rolling statistics
    # ------------------------------------------------------------------

    def add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for window in self.rolling_windows:
            rolling = df[TARGET_LOG_COL].rolling(window)
            df[f"log_flux_roll_mean_{window}"] = rolling.mean()
            df[f"log_flux_roll_std_{window}"] = rolling.std()
        logger.info(f"Step 3 - added rolling mean/std for windows {self.rolling_windows}")
        return df

    # ------------------------------------------------------------------
    # Step 4 - Delta features
    # ------------------------------------------------------------------

    def add_delta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["delta_Bz"] = df["Bz_lag"].diff()
        df["delta_speed"] = df["solar_wind_speed_lag"].diff()
        df["delta_flux"] = df[TARGET_LOG_COL].diff()
        logger.info("Step 4 - added delta_Bz, delta_speed, delta_flux")
        return df

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("  Phase 3 - Feature Engineering")
        logger.info("=" * 60)

        df = self.apply_dynamic_lag(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_stats(df)
        df = self.add_delta_features(df)

        before = len(df)
        df = df.dropna()
        logger.info(f"Step 5 - dropped {before - len(df):,} rows with NaN from lag/rolling operations")

        logger.info(f"Phase 3 complete - {len(df):,} rows, {len(df.columns)} columns")
        logger.info("=" * 60)
        return df
