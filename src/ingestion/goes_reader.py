"""
GOES Satellite CDF Reader
=========================
Reads >2 MeV electron flux from GOES CDF files.

IMPORTANT:
  GOES-13/14/15 (EPEAD instrument) and GOES-16/17/18 (SEISS instrument)
  use different variable names. Run cdf_inspector.py on your actual files
  first to confirm the correct variable name for 'electron_flux_var'.

Common variable names to look for (check inspector output):
  GOES-13/15: 'e2_flux', 'E2', 'gt2mev', or similar
  GOES-16+  : 'AvgDiffElectronFlux', 'electron_flux', or similar

CDF fill values (representing bad/missing data) are typically ~-1e31.
"""

import cdflib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

# Candidate variable names to try in order if no explicit name is given
_FLUX_CANDIDATES = [
    "E2W_COR_FLUX", "E2E_COR_FLUX",  # GOES-13/15 EPEAD >2 MeV (CDAWeb, background/dead-time corrected)
    "e2_flux", "E2", "e2", "gt2mev", "mev2",
    "AvgDiffElectronFlux", "electron_flux", "Flux", "flux",
    "p8", "p9", "FLUX",
]

_EPOCH_CANDIDATES = [
    "Epoch", "epoch", "TIME", "time", "Time",
    "Epoch_0", "EPOCH", "time_tags",
]


def _auto_detect(cdf: cdflib.CDF, candidates: list, label: str) -> str:
    """Return the first candidate variable that exists in the CDF."""
    info = cdf.cdf_info()
    # cdflib >=1.0 returns cdf_info() as a CDFInfo dataclass, not a dict
    all_vars = set(info.rVariables + info.zVariables)
    for c in candidates:
        if c in all_vars:
            return c
    raise ValueError(
        f"Could not auto-detect {label} variable.\n"
        f"Available variables: {sorted(all_vars)}\n"
        f"Run 'python -m src.ingestion.cdf_inspector <your_file.cdf>' to inspect."
    )


class GOESReader:
    """
    Reads GOES CDF files and returns a DataFrame with:
        index       : UTC timestamp (DatetimeIndex)
        electron_flux : >2 MeV electron flux (particles / cm² / s / sr / MeV)

    Parameters
    ----------
    epoch_var : CDF variable name for time/epoch.
                Leave None for auto-detection.
    flux_var  : CDF variable name for >2 MeV electron flux.
                Leave None for auto-detection.
    """

    def __init__(
        self,
        epoch_var: Optional[str] = None,
        flux_var: Optional[str] = None,
    ):
        self._epoch_var = epoch_var
        self._flux_var = flux_var

    # ------------------------------------------------------------------
    # Single File
    # ------------------------------------------------------------------

    def read_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Read a single GOES CDF file."""
        filepath = Path(filepath)
        logger.info(f"GOES: reading {filepath.name}")

        cdf = cdflib.CDF(str(filepath))

        # --- Epoch --------------------------------------------------------
        epoch_var = self._epoch_var or _auto_detect(cdf, _EPOCH_CANDIDATES, "epoch")
        epoch = cdf.varget(epoch_var)
        times = cdflib.cdfepoch.to_datetime(epoch)

        # --- Flux ---------------------------------------------------------
        flux_var = self._flux_var or _auto_detect(cdf, _FLUX_CANDIDATES, "electron flux")
        flux = cdf.varget(flux_var).astype(float)

        # Some GOES files have multi-channel flux arrays (shape: N × channels)
        # The >2 MeV channel index varies by file — log a warning so the user
        # can verify.
        if flux.ndim == 2:
            logger.warning(
                f"  '{flux_var}' has shape {flux.shape} — "
                f"using column 0 as the >2 MeV channel. "
                f"Verify with cdf_inspector.py if results look wrong."
            )
            flux = flux[:, 0]

        # --- Build DataFrame ----------------------------------------------
        df = pd.DataFrame({"electron_flux": flux}, index=pd.DatetimeIndex(times))
        df.index.name = "timestamp"

        # Replace CDF fill values and physically impossible values with NaN
        df.loc[df["electron_flux"] < -1e20, "electron_flux"] = np.nan
        df.loc[df["electron_flux"] <= 0, "electron_flux"] = np.nan

        df.sort_index(inplace=True)

        logger.info(
            f"  {len(df)} records  |  "
            f"{df.index.min().date()} → {df.index.max().date()}"
        )
        return df

    # ------------------------------------------------------------------
    # Directory (multiple files)
    # ------------------------------------------------------------------

    def read_directory(
        self,
        dirpath: Union[str, Path],
        pattern: str = "*.cdf",
    ) -> pd.DataFrame:
        """
        Read all CDF files matching pattern in dirpath and concatenate.

        Parameters
        ----------
        dirpath : path to folder containing GOES CDF files
        pattern : glob pattern, e.g. '*.cdf' or 'go13*.cdf'
        """
        dirpath = Path(dirpath)
        files = sorted(dirpath.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' in {dirpath}"
            )

        logger.info(f"GOES: found {len(files)} CDF files in {dirpath.name}/")

        dfs = []
        for f in files:
            try:
                dfs.append(self.read_file(f))
            except Exception as e:
                logger.error(f"  Skipping {f.name}: {e}")

        if not dfs:
            raise ValueError("No GOES files could be read.")

        combined = pd.concat(dfs)
        combined = combined.loc[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)

        logger.info(
            f"GOES combined: {len(combined):,} records  |  "
            f"{combined.index.min().date()} → {combined.index.max().date()}"
        )
        return combined
