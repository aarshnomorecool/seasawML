"""
GRASP / GSAT Data Reader
========================
Handles bulk reading from ZIP archives containing:
  .txt → 5-minute averaged electron flux  ← what we actually need
  .xml → metadata / instrument info       ← parsed for completeness
  .png → visualization plots              ← NOT parsed (images only)

Each ZIP typically covers one day or one pass.
This reader scans an entire directory of ZIPs, extracts and parses all TXT/XML,
and returns a single combined DataFrame for use in validation.

TXT Format Detection
--------------------
ISRO PRADAN TXT files vary across missions. Two common layouts:

  Layout A — space-separated with DOY:
    YYYY  DOY  HH  MM  SS  flux  ...

  Layout B — ISO datetime:
    YYYY-MM-DD  HH:MM:SS  flux  ...

The reader tries both. If neither works it logs the raw first 5 lines
so you can extend _assign_columns() for your specific format.

PNG Files
---------
PNG files inside the ZIPs are ISRO-generated visualization plots.
They are NOT parsed — the model uses TXT flux values for validation.
"""

import zipfile
import io
import os
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# TXT Parser helpers
# ======================================================================

def _try_layout_a(lines: List[str]) -> Optional[pd.DataFrame]:
    """
    Layout A: YYYY DOY HH MM SS flux1 [flux2 ...]
    Common in older ISRO / NOAA text products.
    """
    records = []
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            year = int(parts[0])
            doy  = int(parts[1])
            hh   = int(parts[2])
            mm   = int(parts[3])
            ss   = float(parts[4])
            flux = float(parts[5])   # first flux column = primary channel

            ts = (
                pd.Timestamp(f"{year}-01-01")
                + pd.Timedelta(days=doy - 1)
                + pd.Timedelta(hours=hh, minutes=mm, seconds=ss)
            )
            records.append({"timestamp": ts, "electron_flux": flux})
        except (ValueError, IndexError):
            continue

    if records:
        df = pd.DataFrame(records).set_index("timestamp")
        return df
    return None


def _try_layout_b(lines: List[str]) -> Optional[pd.DataFrame]:
    """
    Layout B: YYYY-MM-DD HH:MM:SS flux1 [flux2 ...]
    Or:       YYYY-MM-DDTHH:MM:SS flux1 ...
    """
    records = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            # Handle both "date time" and ISO "date_Ttime"
            if len(parts) >= 3:
                ts_str = parts[0] + " " + parts[1]
                flux   = float(parts[2])
            else:
                ts_str = parts[0].replace("T", " ")
                flux   = float(parts[1])

            ts = pd.Timestamp(ts_str)
            records.append({"timestamp": ts, "electron_flux": flux})
        except (ValueError, IndexError):
            continue

    if records:
        df = pd.DataFrame(records).set_index("timestamp")
        return df
    return None


def parse_grasp_txt(content: str, filename: str = "") -> pd.DataFrame:
    """
    Parse raw text content of a GRASP TXT file into a DataFrame.

    Tries Layout A then Layout B. If both fail, logs the first 5 lines
    so you can inspect and extend this function.

    Returns DataFrame with:
        index          : UTC timestamp (DatetimeIndex)
        electron_flux  : primary electron flux channel
    """
    raw_lines = content.strip().split("\n")

    # Strip comment lines (starting with # or %)
    data_lines = [
        l.strip()
        for l in raw_lines
        if l.strip() and not l.strip().startswith(("#", "%", "//", "!"))
    ]

    if not data_lines:
        logger.warning(f"No data lines in {filename}")
        return pd.DataFrame()

    # --- Try Layout A ------------------------------------------------
    df = _try_layout_a(data_lines)
    if df is not None and len(df) > 0:
        logger.info(f"  Parsed {filename} as Layout A ({len(df)} records)")
        return df

    # --- Try Layout B ------------------------------------------------
    df = _try_layout_b(data_lines)
    if df is not None and len(df) > 0:
        logger.info(f"  Parsed {filename} as Layout B ({len(df)} records)")
        return df

    # --- Both failed — log raw lines for manual inspection -----------
    logger.warning(
        f"Could not parse {filename}. "
        f"First 5 raw lines:\n"
        + "\n".join(f"  | {l}" for l in raw_lines[:5])
    )
    return pd.DataFrame()


# ======================================================================
# XML Parser
# ======================================================================

def parse_grasp_xml(content: bytes, filename: str = "") -> dict:
    """
    Parse GRASP XML metadata into a flat dictionary.

    Returns a dict of  tag_path → value  for every leaf element.
    Useful for checking satellite name, time range, quality flags etc.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        logger.warning(f"XML parse error in {filename}: {e}")
        return {}

    result = {}

    def _flatten(element, prefix=""):
        key = f"{prefix}/{element.tag}" if prefix else element.tag
        text = element.text.strip() if element.text and element.text.strip() else None
        if text:
            result[key] = text
        for attr, val in element.attrib.items():
            result[f"{key}@{attr}"] = val
        for child in element:
            _flatten(child, key)

    _flatten(root)
    logger.info(f"  XML metadata: {len(result)} fields parsed from {filename}")
    return result


# ======================================================================
# Main Reader Class
# ======================================================================

class GRASPReader:
    """
    Reads GRASP/GSAT ZIP archives and returns combined electron flux DataFrame.

    Usage
    -----
    reader = GRASPReader()

    # Single ZIP:
    df, meta = reader.read_zip("path/to/grasp_20200101.zip")

    # Entire directory of ZIPs:
    df = reader.read_zip_directory("data/raw/grasp/")
    """

    # ------------------------------------------------------------------
    # Single ZIP
    # ------------------------------------------------------------------

    def read_zip(
        self, zip_path: Union[str, Path]
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Read one ZIP file.

        Returns
        -------
        (flux_df, metadata_dict)
            flux_df       : DataFrame indexed by timestamp, column 'electron_flux'
            metadata_dict : flattened XML metadata (empty dict if no XML)
        """
        zip_path = Path(zip_path)
        logger.info(f"GRASP: reading {zip_path.name}")

        all_flux = []
        metadata = {}

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

            txt_files = [n for n in names if n.lower().endswith(".txt")]
            xml_files = [n for n in names if n.lower().endswith(".xml")]
            png_files = [n for n in names if n.lower().endswith(".png")]

            logger.info(
                f"  Contents: {len(txt_files)} TXT, "
                f"{len(xml_files)} XML, {len(png_files)} PNG "
                f"(PNG files skipped - visualization only)"
            )

            # ---- TXT -----------------------------------------------
            for name in txt_files:
                with zf.open(name) as f:
                    content = f.read().decode("utf-8", errors="replace")
                df = parse_grasp_txt(content, filename=name)
                if not df.empty:
                    all_flux.append(df)

            # ---- XML -----------------------------------------------
            for name in xml_files:
                with zf.open(name) as f:
                    content = f.read()
                meta = parse_grasp_xml(content, filename=name)
                metadata.update(meta)

        if not all_flux:
            logger.warning(f"No flux data extracted from {zip_path.name}")
            return pd.DataFrame(), metadata

        flux_df = pd.concat(all_flux)
        flux_df = flux_df.loc[~flux_df.index.duplicated(keep="first")]
        flux_df.sort_index(inplace=True)

        # Remove fill values / negatives
        flux_df.loc[flux_df["electron_flux"] < -1e20, "electron_flux"] = np.nan
        flux_df.loc[flux_df["electron_flux"] <= 0, "electron_flux"] = np.nan

        return flux_df, metadata

    # ------------------------------------------------------------------
    # Directory of ZIPs
    # ------------------------------------------------------------------

    def read_zip_directory(
        self,
        dirpath: Union[str, Path],
        pattern: str = "*.zip",
    ) -> pd.DataFrame:
        """
        Read all ZIP files in a directory and return combined DataFrame.

        Parameters
        ----------
        dirpath : folder containing GRASP ZIP archives
        pattern : glob pattern (default '*.zip')
        """
        dirpath = Path(dirpath)
        zip_files = sorted(dirpath.glob(pattern))

        if not zip_files:
            raise FileNotFoundError(
                f"No ZIP files matching '{pattern}' found in {dirpath}"
            )

        logger.info(f"GRASP: found {len(zip_files)} ZIP files")

        all_dfs = []
        failed = []

        for zf in zip_files:
            try:
                df, _ = self.read_zip(zf)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"  Failed to read {zf.name}: {e}")
                failed.append(zf.name)

        if failed:
            logger.warning(f"Skipped {len(failed)} files due to errors: {failed}")

        if not all_dfs:
            raise ValueError("No data could be read from GRASP ZIP files.")

        combined = pd.concat(all_dfs)
        combined = combined.loc[~combined.index.duplicated(keep="first")]
        combined.sort_index(inplace=True)

        logger.info(
            f"GRASP combined: {len(combined):,} records  |  "
            f"{combined.index.min().date()} -> {combined.index.max().date()}"
        )
        return combined
