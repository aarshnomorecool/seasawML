"""
CDF File Inspector
==================
Run this FIRST on your actual GOES and Wind CDF files before anything else.
It prints every variable name, shape, dtype, and a sample value.
This tells you the correct variable names to plug into the readers.

Usage:
    python -m src.ingestion.cdf_inspector path/to/your/file.cdf
"""

import sys
import cdflib
import numpy as np


def inspect_cdf(filepath: str) -> dict:
    """
    Inspect a CDF file — prints all variables with shape, type, and sample values.

    Parameters
    ----------
    filepath : path to a .cdf file

    Returns
    -------
    dict mapping variable name → {'shape', 'dtype', 'sample'}
    """
    cdf = cdflib.CDF(filepath)
    info = cdf.cdf_info()

    print(f"\n{'=' * 70}")
    print(f"  CDF File: {filepath}")
    print(f"{'=' * 70}")

    # cdflib >=1.0 returns cdf_info() as a CDFInfo dataclass, not a dict
    all_vars = info.rVariables + info.zVariables
    print(f"\n  Total variables: {len(all_vars)}\n")

    results = {}

    for var in all_vars:
        try:
            data = cdf.varget(var)

            if data is None:
                print(f"  [{var}]  ->  None")
                continue

            shape = data.shape if hasattr(data, "shape") else "(scalar)"
            dtype = str(data.dtype) if hasattr(data, "dtype") else type(data).__name__

            # Get a short sample
            if hasattr(data, "__len__") and len(data) > 0:
                sample = data[:3]
                # If it's an epoch, convert to readable datetime
                if "epoch" in var.lower() or "time" in var.lower():
                    try:
                        dt_sample = cdflib.cdfepoch.to_datetime(data[:3])
                        sample = dt_sample
                    except Exception:
                        pass
            else:
                sample = data

            print(f"  [{var}]")
            print(f"      shape  : {shape}")
            print(f"      dtype  : {dtype}")
            print(f"      sample : {sample}")
            print()

            results[var] = {"shape": shape, "dtype": dtype, "sample": str(sample)}

        except Exception as e:
            print(f"  [{var}]  →  ERROR: {e}\n")

    return results


def print_epoch_range(filepath: str, epoch_var: str = "Epoch"):
    """Print the time range of a CDF file."""
    cdf = cdflib.CDF(filepath)
    try:
        epoch = cdf.varget(epoch_var)
        times = cdflib.cdfepoch.to_datetime(epoch)
        print(f"\n  Time range:")
        print(f"    Start : {times[0]}")
        print(f"    End   : {times[-1]}")
        print(f"    Count : {len(times)} records")
    except Exception as e:
        print(f"  Could not read epoch '{epoch_var}': {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.cdf_inspector <path_to_cdf_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    inspect_cdf(filepath)

    epoch_var = sys.argv[2] if len(sys.argv) > 2 else "Epoch"
    print_epoch_range(filepath, epoch_var)
