"""
SeaSaw Preprocessing
=====================
Phase 2 of the pipeline. Takes the raw merged GOES + Wind DataFrame
produced by Phase 1 (data/processed/training_raw.csv) and applies,
in order:

  1. Drop rows where the target (goes_electron_flux) is NaN
  2. Spike removal on the target (>5sigma from a rolling median)
  3. Gap interpolation (linear, only for gaps <= max_gap_steps; longer
     gaps are left as NaN and dropped)
  4. Log transform of the target (mandatory - see CLAUDE.md Flag 1)
  5. RobustScaler normalization of the Wind features
  6. Persist the fitted scaler so it can be re-applied to test/live data

Usage
-----
    from src.preprocessing.preprocessor import SeaSawPreprocessor

    pre = SeaSawPreprocessor()
    processed_df, scalers = pre.run(raw_df)
    pre.save_scalers(scalers, "models/scalers.pkl")
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

EPSILON = 1e-10

TARGET_COL = "goes_electron_flux"
WIND_FEATURES = ["Bx", "By", "Bz", "solar_wind_speed", "plasma_density"]


def _interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """
    Linearly interpolate only NaN runs of length <= max_gap.
    Longer runs are left untouched (as NaN) so they get dropped downstream.
    """
    is_na = series.isna()
    if not is_na.any():
        return series

    # group id per consecutive NaN run: increments only on non-NaN values
    run_id = (~is_na).cumsum()
    run_length = is_na.groupby(run_id).transform("sum")

    interpolated = series.interpolate(method="linear", limit_direction="both")
    fillable = is_na & (run_length <= max_gap)

    result = series.copy()
    result[fillable] = interpolated[fillable]
    return result


class SeaSawPreprocessor:
    """
    Phase 2 preprocessing pipeline.

    Parameters
    ----------
    spike_window   : rolling window (in 5-min steps) for spike detection median/std
    spike_threshold: number of standard deviations from the rolling median
                     beyond which a value is treated as a spike
    max_gap_steps  : maximum consecutive-NaN run length (in steps) eligible
                     for linear interpolation; longer gaps are dropped
    """

    def __init__(
        self,
        spike_window: int = 12,
        spike_threshold: float = 5.0,
        max_gap_steps: int = 6,
    ):
        self.spike_window = spike_window
        self.spike_threshold = spike_threshold
        self.max_gap_steps = max_gap_steps

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def remove_missing_target(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.loc[df[TARGET_COL].notna()].copy()
        logger.info(f"Step 1 - dropped {before - len(df):,} rows with missing target")
        return df

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------

    def remove_spikes(self, df: pd.DataFrame, column: str = TARGET_COL) -> pd.DataFrame:
        df = df.copy()
        series = df[column]
        min_periods = max(1, self.spike_window // 2)
        rolling_median = series.rolling(self.spike_window, center=True, min_periods=min_periods).median()

        # Median absolute deviation, not rolling std: a spike sitting inside its
        # own window inflates plain std enough to hide itself, but barely moves
        # the median-based MAD. 1.4826 rescales MAD to a normal-consistent sigma.
        abs_dev = (series - rolling_median).abs()
        rolling_mad = abs_dev.rolling(self.spike_window, center=True, min_periods=min_periods).median()
        robust_sigma = rolling_mad * 1.4826

        spike_mask = abs_dev > (self.spike_threshold * robust_sigma)
        spike_mask = spike_mask.fillna(False)

        n_spikes = int(spike_mask.sum())
        df.loc[spike_mask, column] = np.nan
        logger.info(f"Step 2 - flagged {n_spikes:,} spikes (>{self.spike_threshold}sigma) in '{column}'")
        return df

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------

    def interpolate_gaps(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            df[col] = _interpolate_short_gaps(df[col], self.max_gap_steps)

        before = len(df)
        df = df.dropna(subset=columns)
        logger.info(
            f"Step 3 - interpolated gaps <= {self.max_gap_steps} steps; "
            f"dropped {before - len(df):,} rows with remaining (longer) gaps"
        )
        return df

    # ------------------------------------------------------------------
    # Step 4
    # ------------------------------------------------------------------

    def log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["log_electron_flux"] = np.log10(df[TARGET_COL] + EPSILON)
        logger.info("Step 4 - added 'log_electron_flux'")
        return df

    # ------------------------------------------------------------------
    # Step 5
    # ------------------------------------------------------------------

    def normalize_wind_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = WIND_FEATURES,
    ) -> Tuple[pd.DataFrame, RobustScaler]:
        df = df.copy()
        scaler = RobustScaler()
        scaled = scaler.fit_transform(df[feature_cols])

        for i, col in enumerate(feature_cols):
            df[f"{col}_scaled"] = scaled[:, i]

        logger.info(f"Step 5 - normalized {feature_cols} with RobustScaler")
        return df, scaler

    # ------------------------------------------------------------------
    # Step 6
    # ------------------------------------------------------------------

    def save_scalers(self, scalers: Dict, path: Union[str, Path] = "models/scalers.pkl") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(scalers, f)
        logger.info(f"Step 6 - saved scalers to {path}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        logger.info("=" * 60)
        logger.info("  Phase 2 - Preprocessing")
        logger.info("=" * 60)

        df = self.remove_missing_target(df)
        df = self.remove_spikes(df, TARGET_COL)
        df = self.interpolate_gaps(df, [TARGET_COL] + WIND_FEATURES)
        df = self.log_transform(df)
        df, wind_scaler = self.normalize_wind_features(df, WIND_FEATURES)

        scalers = {"wind_scaler": wind_scaler, "wind_features": WIND_FEATURES}

        logger.info(f"Phase 2 complete - {len(df):,} rows, {len(df.columns)} columns")
        logger.info("=" * 60)
        return df, scalers
