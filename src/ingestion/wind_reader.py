"""
Wind Spacecraft CDF Reader
==========================
Wind data from CDAWeb comes in two separate instrument files:

  MFI  (Magnetic Field Investigation)
       → Bx, By, Bz  in GSE coordinates

  SWE  (Solar Wind Experiment)
       → Solar wind speed  (km/s)
       → Proton density    (cm⁻³)

Both readers follow the same pattern as GOESReader:
  - Auto-detect variable names if not specified
  - Replace CDF fill values with NaN
  - Return a clean DatetimeIndex DataFrame

Common variable names:
  MFI epoch : 'Epoch'
  MFI field : 'BGSE', 'BGSEc', 'B_GSE'  (shape N×3 → [Bx, By, Bz])

  SWE epoch  : 'Epoch'
  SWE speed  : 'Proton_V_nonlin', 'V_GSE', 'Vp', 'bulk_speed'
  SWE density: 'Proton_Np_nonlin', 'Np', 'np', 'Density'

Run cdf_inspector.py on your actual files to confirm.
"""

import cdflib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

_EPOCH_CANDIDATES = ["Epoch", "epoch", "TIME", "time", "Time", "Epoch_0", "EPOCH"]

_BGSE_CANDIDATES = [
    "BGSE", "BGSEc", "B_GSE", "b_gse", "bgsec",
    "BF1", "B", "BFIELD", "IMF",
]

_SPEED_CANDIDATES = [
    "Proton_V_nonlin", "V_GSE", "Vp", "vp", "bulk_speed",
    "SW_V", "V", "velocity", "Speed",
]

_DENSITY_CANDIDATES = [
    "Proton_Np_nonlin", "Np", "np", "Density", "density",
    "proton_density", "N_p", "Np_moment",
]

# Sanity bounds for decoded timestamps. A handful of records in real CDAWeb
# files decode to garbage dates (observed: one Wind SWE record at 2055-02-24,
# another at 1998-05-31 in a file that should only span March 2015) due to
# corrupted epoch values that aren't caught by cdflib's own fill-value
# handling. Left unfiltered, these blow up any later pd.resample() call into
# millions of empty rows spanning decades. An absolute mission-lifetime bound
# alone isn't enough (1998 is a real Wind date, just the wrong file) - each
# CDAWeb file covers one bounded fetch chunk (typically a month, see
# auto_fetcher.fetch_dataset), so anything far from *that file's own median*
# timestamp is corrupt regardless of whether it also happens to be a
# plausible date in isolation.
_MIN_VALID_TIMESTAMP = pd.Timestamp("1994-11-01")  # Wind launch date
_MAX_VALID_TIMESTAMP = pd.Timestamp.now() + pd.Timedelta(days=1)
_MAX_DEVIATION_FROM_MEDIAN = pd.Timedelta(days=60)


def _drop_corrupt_timestamps(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df.index >= _MIN_VALID_TIMESTAMP) & (df.index <= _MAX_VALID_TIMESTAMP)
    median_ts = df.index[mask].to_series().median() if mask.any() else df.index.to_series().median()
    mask &= np.abs(df.index - median_ts) <= _MAX_DEVIATION_FROM_MEDIAN
    n_bad = (~mask).sum()
    if n_bad:
        logger.warning(f"  {label}: dropping {n_bad} record(s) with corrupt epoch/timestamp")
        df = df.loc[mask]
    return df


def _auto_detect(cdf: cdflib.CDF, candidates: list, label: str) -> str:
    info = cdf.cdf_info()
    # cdflib >=1.0 returns cdf_info() as a CDFInfo dataclass, not a dict
    all_vars = set(info.rVariables + info.zVariables)
    for c in candidates:
        if c in all_vars:
            return c
    raise ValueError(
        f"Could not auto-detect {label}.\n"
        f"Available: {sorted(all_vars)}\n"
        f"Run 'python -m src.ingestion.cdf_inspector <file.cdf>' to check."
    )


# ======================================================================
# Wind MFI Reader
# ======================================================================

class WindMFIReader:
    """
    Reads Wind MFI CDF files.

    Output DataFrame columns:
        Bx, By, Bz  — interplanetary magnetic field components (nT, GSE frame)

    Parameters
    ----------
    epoch_var : CDF epoch variable name (None = auto-detect)
    bgse_var  : CDF variable name for 3-component IMF vector (None = auto-detect)
    """

    def __init__(
        self,
        epoch_var: Optional[str] = None,
        bgse_var: Optional[str] = None,
    ):
        self._epoch_var = epoch_var
        self._bgse_var = bgse_var

    def read_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        filepath = Path(filepath)
        logger.info(f"Wind MFI: reading {filepath.name}")

        cdf = cdflib.CDF(str(filepath))

        epoch_var = self._epoch_var or _auto_detect(cdf, _EPOCH_CANDIDATES, "epoch")
        epoch = cdf.varget(epoch_var)
        times = cdflib.cdfepoch.to_datetime(epoch)

        bgse_var = self._bgse_var or _auto_detect(cdf, _BGSE_CANDIDATES, "IMF BGSE")
        bgse = cdf.varget(bgse_var).astype(float)

        # BGSE must be Nx3 — warn if shape is unexpected
        if bgse.ndim == 1:
            raise ValueError(
                f"Expected IMF variable '{bgse_var}' to be Nx3 but got shape {bgse.shape}. "
                f"Check cdf_inspector output."
            )
        if bgse.shape[1] != 3:
            logger.warning(
                f"  '{bgse_var}' has {bgse.shape[1]} columns instead of 3. "
                f"Using first 3."
            )
            bgse = bgse[:, :3]

        df = pd.DataFrame(
            {"Bx": bgse[:, 0], "By": bgse[:, 1], "Bz": bgse[:, 2]},
            index=pd.DatetimeIndex(times),
        )
        df.index.name = "timestamp"

        # Replace CDF fill values with NaN
        for col in ["Bx", "By", "Bz"]:
            df.loc[df[col].abs() > 1e20, col] = np.nan

        df = _drop_corrupt_timestamps(df, "Wind MFI")
        df.sort_index(inplace=True)
        logger.info(
            f"  {len(df)} records  |  "
            f"{df.index.min().date()} -> {df.index.max().date()}"
        )
        return df

    def read_directory(
        self,
        dirpath: Union[str, Path],
        pattern: str = "*.cdf",
    ) -> pd.DataFrame:
        dirpath = Path(dirpath)
        files = sorted(dirpath.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No '{pattern}' files in {dirpath}")
        logger.info(f"Wind MFI: found {len(files)} files")

        dfs = []
        for f in files:
            try:
                dfs.append(self.read_file(f))
            except Exception as e:
                logger.error(f"  Skipping {f.name}: {e}")

        combined = pd.concat(dfs)
        combined = combined.loc[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)
        logger.info(f"Wind MFI combined: {len(combined):,} records")
        return combined


# ======================================================================
# Wind SWE Reader
# ======================================================================

class WindSWEReader:
    """
    Reads Wind SWE CDF files.

    Output DataFrame columns:
        solar_wind_speed  — proton bulk speed (km/s)
        plasma_density    — proton number density (cm⁻³)

    Parameters
    ----------
    epoch_var   : CDF epoch variable name (None = auto-detect)
    speed_var   : CDF variable name for solar wind speed (None = auto-detect)
    density_var : CDF variable name for plasma density (None = auto-detect)
    """

    def __init__(
        self,
        epoch_var: Optional[str] = None,
        speed_var: Optional[str] = None,
        density_var: Optional[str] = None,
    ):
        self._epoch_var = epoch_var
        self._speed_var = speed_var
        self._density_var = density_var

    def read_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        filepath = Path(filepath)
        logger.info(f"Wind SWE: reading {filepath.name}")

        cdf = cdflib.CDF(str(filepath))

        epoch_var = self._epoch_var or _auto_detect(cdf, _EPOCH_CANDIDATES, "epoch")
        epoch = cdf.varget(epoch_var)
        times = cdflib.cdfepoch.to_datetime(epoch)

        speed_var = self._speed_var or _auto_detect(cdf, _SPEED_CANDIDATES, "solar wind speed")
        speed = cdf.varget(speed_var).astype(float)

        density_var = self._density_var or _auto_detect(cdf, _DENSITY_CANDIDATES, "plasma density")
        density = cdf.varget(density_var).astype(float)

        # Speed may be a vector (Vx, Vy, Vz) — compute magnitude
        if speed.ndim == 2:
            logger.info(
                f"  Speed variable '{speed_var}' is a vector. "
                f"Computing magnitude."
            )
            speed = np.sqrt(np.sum(speed ** 2, axis=1))

        df = pd.DataFrame(
            {"solar_wind_speed": speed, "plasma_density": density},
            index=pd.DatetimeIndex(times),
        )
        df.index.name = "timestamp"

        # Replace fill values and physically impossible values with NaN
        df.loc[df["solar_wind_speed"].abs() > 1e20, "solar_wind_speed"] = np.nan
        df.loc[df["plasma_density"].abs() > 1e20, "plasma_density"] = np.nan
        df.loc[df["solar_wind_speed"] <= 0, "solar_wind_speed"] = np.nan
        df.loc[df["plasma_density"] <= 0, "plasma_density"] = np.nan

        df = _drop_corrupt_timestamps(df, "Wind SWE")
        df.sort_index(inplace=True)
        logger.info(
            f"  {len(df)} records  |  "
            f"{df.index.min().date()} -> {df.index.max().date()}"
        )
        return df

    def read_directory(
        self,
        dirpath: Union[str, Path],
        pattern: str = "*.cdf",
    ) -> pd.DataFrame:
        dirpath = Path(dirpath)
        files = sorted(dirpath.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No '{pattern}' files in {dirpath}")
        logger.info(f"Wind SWE: found {len(files)} files")

        dfs = []
        for f in files:
            try:
                dfs.append(self.read_file(f))
            except Exception as e:
                logger.error(f"  Skipping {f.name}: {e}")

        combined = pd.concat(dfs)
        combined = combined.loc[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)
        logger.info(f"Wind SWE combined: {len(combined):,} records")
        return combined
