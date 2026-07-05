"""
SeaSaw — Automatic Data Fetcher
================================
Automatically downloads GOES and Wind spacecraft data from public archives.

What is automatable:
  ✅ Wind MFI (IMF: Bx, By, Bz)     — NASA CDAWeb via cdasws
  ✅ Wind SWE (speed, density)        — NASA CDAWeb via cdasws
  ✅ GOES-13/15 electron flux         — NASA CDAWeb via cdasws
  ✅ GOES-16/17/18 electron flux      — NOAA HTTPS archive
  ❌ GRASP/GSAT (ISRO PRADAN)         — Requires login, see note below

GRASP / PRADAN note:
  Register at https://pradan.issdc.gov.in/
  After login: Data → GSAT → Select dates → Download ZIPs
  Place downloaded ZIPs in data/raw/grasp/
  The reader handles extraction automatically.
  For the hackathon, ISRO will provide this data directly.

Dependencies:
  pip install cdasws requests tqdm

Usage:
  python -m src.ingestion.auto_fetcher --start 2013-01-01 --end 2024-01-01
  python -m src.ingestion.auto_fetcher --start 2013-01-01 --end 2024-01-01 --list-datasets
"""

import os
import sys
import time
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# CDAWeb Dataset IDs  (verified from https://cdaweb.gsfc.nasa.gov/)
# ──────────────────────────────────────────────────────────────────────

CDAWEB_DATASETS = {
    # Wind spacecraft — Magnetic Field Investigation (IMF)
    # WI_H0_MFI (1-min cadence) is used instead of WI_H2_MFI (3-sec) because
    # everything downstream resamples to 5-min anyway (see resample_freq in
    # data_pipeline.py) — H2 buys no accuracy here but is ~20x the raw storage
    # (500MB/month vs ~25MB/month), which matters over an 11-year fetch.
    "wind_mfi_1min": {
        "dataset_id": "WI_H0_MFI",
        "variables": ["BGSE"],          # 3-component [Bx, By, Bz] in GSE (nT)
        "description": "Wind MFI 1-min averaged IMF (1994–present)",
    },
    # Wind spacecraft — Solar Wind Experiment (plasma)
    "wind_swe": {
        "dataset_id": "WI_K0_SWE",
        "variables": ["V_GSE", "Np"],
        "description": "Wind SWE proton velocity vector (km/s) and density (cm⁻³)",
    },
    # GOES-13 EPEAD >2 MeV electron flux (background/dead-time corrected)
    "goes13_epead": {
        "dataset_id": "GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN",
        "variables": ["E2W_COR_FLUX", "E2E_COR_FLUX"],
        "description": "GOES-13 >2 MeV electron flux, EPEAD science-quality (2010-05 to 2017-12)",
    },
    # GOES-15 EPEAD >2 MeV electron flux (background/dead-time corrected)
    "goes15_epead": {
        "dataset_id": "GOES15_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN",
        "variables": ["E2W_COR_FLUX", "E2E_COR_FLUX"],
        "description": "GOES-15 >2 MeV electron flux, EPEAD science-quality (2010-03 to 2020-03)",
    },
}

# NOAA HTTPS base for GOES-R series (16, 17, 18)
NOAA_GOES_R_BASE = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
    "/goes/goes{sat}/l2/data/sgps/seiss-sgps-g{sat}_"
)

# ──────────────────────────────────────────────────────────────────────
# CDAWeb Fetcher (Wind + older GOES)
# ──────────────────────────────────────────────────────────────────────

class CDAWebFetcher:
    """
    Downloads CDF files from NASA CDAWeb using the cdasws library.

    Each month is fetched and saved as a separate CDF file to keep
    individual file sizes manageable.
    """

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self._cdas = None

    def _get_cdas(self):
        """Lazy-load cdasws to avoid import error if not installed."""
        if self._cdas is None:
            try:
                from cdasws import CdasWs
                self._cdas = CdasWs()
                logger.info("CDAWeb connection established.")
            except ImportError:
                raise ImportError(
                    "cdasws not installed. Run: pip install cdasws"
                )
        return self._cdas

    def list_available_datasets(self, keyword: str = "GOES") -> None:
        """
        List all CDAWeb datasets whose label matches a keyword.
        Use this to find the correct dataset ID for your data.
        """
        cdas = self._get_cdas()
        logger.info(f"Searching CDAWeb for '{keyword}'...")
        try:
            # get_datasets() takes keyword args only; labelPattern is a Java
            # regex matched server-side against the dataset label.
            datasets = cdas.get_datasets(labelPattern=f"(?i).*{keyword}.*")
            print(f"\nFound {len(datasets)} datasets matching '{keyword}':\n")
            for ds in datasets[:30]:   # cap output
                print(f"  ID: {ds.get('Id', '')}")
                print(f"  Description: {ds.get('Label', '')}")
                print(f"  Time range: {ds.get('TimeInterval', {}).get('Start', '')} "
                      f"-> {ds.get('TimeInterval', {}).get('End', '')}")
                print()
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")

    def list_variables(self, dataset_id: str) -> None:
        """Print all variables in a CDAWeb dataset."""
        cdas = self._get_cdas()
        try:
            variables = cdas.get_variables(dataset_id)
            print(f"\nVariables in {dataset_id}:")
            for v in variables:
                print(f"  {v.get('Name', '')} - {v.get('LongDescription', '')}")
        except Exception as e:
            logger.error(f"Failed to get variables for {dataset_id}: {e}")

    def fetch_dataset(
        self,
        dataset_key: str,
        start_date: str,
        end_date: str,
        subdir: Optional[str] = None,
    ) -> list:
        """
        Download a dataset month-by-month from CDAWeb.

        Parameters
        ----------
        dataset_key : one of the keys in CDAWEB_DATASETS dict
        start_date  : 'YYYY-MM-DD'
        end_date    : 'YYYY-MM-DD'
        subdir      : override output subdirectory name

        Returns
        -------
        list of paths to downloaded CDF files
        """
        if dataset_key not in CDAWEB_DATASETS:
            raise ValueError(
                f"Unknown dataset key '{dataset_key}'. "
                f"Valid options: {list(CDAWEB_DATASETS.keys())}"
            )

        config = CDAWEB_DATASETS[dataset_key]
        dataset_id = config["dataset_id"]
        variables = config["variables"]

        out_subdir = subdir or dataset_key.split("_")[0] + "_" + dataset_key.split("_")[1]
        out_dir = self.output_dir / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        cdas = self._get_cdas()

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")

        downloaded = []
        current = start

        logger.info(f"Fetching {dataset_id} from {start_date} to {end_date}")
        logger.info(f"Variables: {variables}")
        logger.info(f"Output: {out_dir}")

        while current < end:
            # Fetch one month at a time
            month_end = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            if month_end > end:
                month_end = end

            t_start = current.strftime("%Y-%m-%dT00:00:00Z")
            t_end   = month_end.strftime("%Y-%m-%dT00:00:00Z")

            logger.info(f"  Requesting {current.strftime('%Y-%m')} ...")

            try:
                # get_data() downloads into memory (SpaceData/xarray); we only
                # want the generated CDF file(s), so use get_data_file() which
                # returns (status_code, {"FileDescription": [...]})
                status, result = cdas.get_data_file(dataset_id, variables, t_start, t_end)

                if status != 200 or not result:
                    logger.warning(f"  No data for {current.strftime('%Y-%m')} (status {status})")
                    current = month_end
                    continue

                # CDAWeb may split a month into more than one file (e.g. for
                # high-resolution datasets) — each gets its own output path so
                # later files don't silently overwrite earlier ones.
                urls = result.get("FileDescription", [])
                if not urls:
                    logger.warning(f"  No data for {current.strftime('%Y-%m')}")
                    current = month_end
                    continue

                for url_info in urls:
                    url = url_info.get("Name", "")
                    if not url:
                        continue
                    fname = url.split("/")[-1]
                    file_out_path = out_dir / fname
                    if file_out_path.exists():
                        logger.info(f"  {fname} already exists - skipping")
                        downloaded.append(file_out_path)
                        continue
                    self._download_file(url, file_out_path)
                    downloaded.append(file_out_path)

            except Exception as e:
                logger.error(f"  Failed {current.strftime('%Y-%m')}: {e}")

            current = month_end
            time.sleep(1)   # be polite to the server

        logger.info(f"Done. {len(downloaded)} files downloaded to {out_dir}")
        return downloaded

    @staticmethod
    def _download_file(url: str, out_path: Path) -> None:
        """Download a file from URL with progress logging."""
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

            size_mb = out_path.stat().st_size / 1024 / 1024
            logger.info(f"  OK: {out_path.name}  ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"  Download failed for {url}: {e}")
            if out_path.exists():
                out_path.unlink()


# ──────────────────────────────────────────────────────────────────────
# NOAA HTTPS Fetcher (GOES-16/17/18)
# ──────────────────────────────────────────────────────────────────────

class NOAAGoesRFetcher:
    """
    Downloads GOES-R series (16, 17, 18) energetic particle CDF files
    directly from NOAA's public HTTPS archive.

    URL format:
    https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites
    /goes/goes16/l2/data/sgps/
    """

    NOAA_BASE = (
        "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites"
        "/goes/goes{sat}/l2/data/sgps/"
    )

    def __init__(self, output_dir: str = "data/raw/goes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_url(self, sat: int, year: int, month: int) -> str:
        base = self.NOAA_BASE.format(sat=sat)
        return f"{base}{year}/{month:02d}/"

    def _list_remote_files(self, url: str) -> list:
        """Scrape CDF filenames from a NOAA directory listing."""
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            import re
            # Find all CDF filenames in the HTML directory listing
            cdf_files = re.findall(r'href="([^"]+\.cdf)"', r.text, re.IGNORECASE)
            return [url + f for f in cdf_files if not f.startswith("http")]
        except Exception as e:
            logger.warning(f"Could not list {url}: {e}")
            return []

    def fetch(
        self,
        satellite: int,
        start_date: str,
        end_date: str,
    ) -> list:
        """
        Download GOES-R CDF files.

        Parameters
        ----------
        satellite  : 16, 17, or 18
        start_date : 'YYYY-MM-DD'
        end_date   : 'YYYY-MM-DD'
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")

        downloaded = []
        current = start

        logger.info(f"NOAA GOES-{satellite}: fetching {start_date} -> {end_date}")

        while current <= end:
            yr, mo = current.year, current.month
            remote_url = self._build_url(satellite, yr, mo)

            logger.info(f"  Listing {yr}-{mo:02d} from {remote_url}")
            file_urls = self._list_remote_files(remote_url)

            if not file_urls:
                logger.warning(f"  No CDF files found for {yr}-{mo:02d}")
            else:
                logger.info(f"  Found {len(file_urls)} CDF files")
                for furl in file_urls:
                    fname = furl.split("/")[-1]
                    out_path = self.output_dir / fname
                    if out_path.exists():
                        logger.info(f"  {fname} already exists - skipping")
                        downloaded.append(out_path)
                        continue
                    try:
                        r = requests.get(furl, stream=True, timeout=180)
                        r.raise_for_status()
                        with open(out_path, "wb") as f:
                            for chunk in r.iter_content(8192):
                                f.write(chunk)
                        size_mb = out_path.stat().st_size / 1024 / 1024
                        logger.info(f"  OK: {fname} ({size_mb:.1f} MB)")
                        downloaded.append(out_path)
                    except Exception as e:
                        logger.error(f"  Failed {fname}: {e}")
                    time.sleep(0.5)

            # Advance to next month
            if mo == 12:
                current = current.replace(year=yr + 1, month=1, day=1)
            else:
                current = current.replace(month=mo + 1, day=1)

        logger.info(f"GOES-{satellite}: {len(downloaded)} files downloaded")
        return downloaded


# ──────────────────────────────────────────────────────────────────────
# Master Fetch Orchestrator
# ──────────────────────────────────────────────────────────────────────

class SeaSawDataFetcher:
    """
    Single entry point for fetching all SeaSaw training data.

    What is fetched:
      Wind MFI   → data/raw/wind_mfi/   (via CDAWeb)
      Wind SWE   → data/raw/wind_swe/   (via CDAWeb)
      GOES-13/15 → data/raw/goes/       (via CDAWeb)
      GOES-16+   → data/raw/goes/       (via NOAA HTTPS)

    GRASP/PRADAN:
      Manual download required (see module docstring).
      Place ZIPs in data/raw/grasp/
    """

    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)
        self.cdaweb   = CDAWebFetcher(output_dir=str(self.base_dir))
        self.noaa     = NOAAGoesRFetcher(output_dir=str(self.base_dir / "goes"))

    def fetch_all(
        self,
        start_date: str,
        end_date: str,
        include_goes16: bool = True,
    ) -> None:
        """
        Fetch the complete training dataset for a date range.

        Parameters
        ----------
        start_date     : 'YYYY-MM-DD'  (suggest '2013-01-01' for 11 years)
        end_date       : 'YYYY-MM-DD'  (suggest '2024-01-01')
        include_goes16 : also fetch GOES-16 from NOAA (recommended)
        """
        logger.info("=" * 60)
        logger.info("  SeaSaw Auto-Fetcher")
        logger.info(f"  Range: {start_date} -> {end_date}")
        logger.info("=" * 60)

        # ── Wind MFI ──────────────────────────────────────────────────
        logger.info("\n[1/4] Wind MFI (IMF Bx, By, Bz)")
        self.cdaweb.fetch_dataset(
            "wind_mfi_1min", start_date, end_date,
            subdir="wind_mfi"
        )

        # ── Wind SWE ──────────────────────────────────────────────────
        logger.info("\n[2/4] Wind SWE (speed, density)")
        self.cdaweb.fetch_dataset(
            "wind_swe", start_date, end_date,
            subdir="wind_swe"
        )

        # ── GOES-13/15 via CDAWeb ─────────────────────────────────────
        logger.info("\n[3/4] GOES-13 and GOES-15 (CDAWeb)")
        for key in ["goes13_epead", "goes15_epead"]:
            try:
                self.cdaweb.fetch_dataset(key, start_date, end_date, subdir="goes")
            except Exception as e:
                logger.warning(f"Skipping {key}: {e}")

        # ── GOES-16 via NOAA ──────────────────────────────────────────
        if include_goes16:
            logger.info("\n[4/4] GOES-16 (NOAA HTTPS archive)")
            self.noaa.fetch(16, start_date, end_date)

        logger.info("\n" + "=" * 60)
        logger.info("  Fetch complete.")
        logger.info("  Place GRASP ZIPs in data/raw/grasp/ (manual download from PRADAN)")
        logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("auto_fetch.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="SeaSaw Auto Data Fetcher")
    parser.add_argument("--start",         default="2013-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",           default="2024-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--base-dir",      default="data/raw",   help="Base output directory")
    parser.add_argument("--list-datasets", action="store_true",  help="List available CDAWeb datasets")
    parser.add_argument("--list-vars",     default="",           help="List variables in a dataset ID")
    parser.add_argument("--goes16-only",   action="store_true",  help="Only fetch GOES-16 (NOAA)")
    parser.add_argument("--wind-only",     action="store_true",  help="Only fetch Wind data (CDAWeb)")
    args = parser.parse_args()

    fetcher = SeaSawDataFetcher(base_dir=args.base_dir)

    if args.list_datasets:
        fetcher.cdaweb.list_available_datasets("GOES")
        fetcher.cdaweb.list_available_datasets("Wind")
        sys.exit(0)

    if args.list_vars:
        fetcher.cdaweb.list_variables(args.list_vars)
        sys.exit(0)

    if args.goes16_only:
        fetcher.noaa.fetch(16, args.start, args.end)
    elif args.wind_only:
        fetcher.cdaweb.fetch_dataset("wind_mfi_1min", args.start, args.end, subdir="wind_mfi")
        fetcher.cdaweb.fetch_dataset("wind_swe",      args.start, args.end, subdir="wind_swe")
    else:
        fetcher.fetch_all(args.start, args.end)
