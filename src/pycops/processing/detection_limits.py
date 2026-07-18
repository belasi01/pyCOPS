"""Per-instrument radiometric detection limits, interpolated to a cast's bands.

Port of the ``detection.limit`` table lookup in ``process.cops.R`` (the table
itself is instrument calibration reference data, bundled from
``cops.detection.limit.dat`` in the R package).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    path = resources.files("pycops.processing") / "data" / "detection_limit.csv"
    with resources.as_file(path) as f:
        return pd.read_csv(f)


def detection_limit_for_waves(instrument: str, waves: np.ndarray) -> np.ndarray:
    """Interpolate the reference detection-limit table to ``waves`` (nm).

    ``instrument`` must be one of the table's columns (``EdZ``, ``EuZ``,
    ``LuZ`` -- Ed0 has no detection limit, it's the above-water reference).
    """
    table = _table()
    if instrument not in table.columns:
        raise KeyError(f"No detection limit data for instrument {instrument!r}")
    spline = CubicSpline(table["waves"].to_numpy(dtype=float), table[instrument].to_numpy(dtype=float))
    return spline(np.asarray(waves, dtype=float))
