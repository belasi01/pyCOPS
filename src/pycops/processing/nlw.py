"""Normalized water-leaving radiance (nLw), port of ``etirr``/``etirrwindow`` (``copsutils.R``).

``etirr`` reads the Thuillier extraterrestrial solar spectrum (bundled from
``thuillier.completed.by.AM0AM1.RData`` as ``data/thuillier.csv``, scaled by
``0.1`` -- a unit conversion baked into the R source); ``etirrwindow``
band-averages it over a rectangular window around each target wavelength,
matching a radiometer channel's bandwidth (``init.cops.dat``'s
``bandwidth``, default 10 nm). ``compute_nlw`` is
``compute.aops.R``'s ``nLw.0p <- Lw.0p / Ed0.0p * etirrwindow(waves.d,
bandwidth)``.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _thuillier() -> pd.DataFrame:
    path = resources.files("pycops.processing") / "data" / "thuillier.csv"
    with resources.as_file(path) as f:
        return pd.read_csv(f)


def etirr(waves: np.ndarray) -> np.ndarray:
    """Extraterrestrial solar irradiance at ``waves`` (nm), linearly interpolated."""
    table = _thuillier()
    return 0.1 * np.interp(np.asarray(waves, dtype=float), table["wave"].to_numpy(), table["F0"].to_numpy())


def etirrwindow(waves: np.ndarray, bandwidth: float) -> np.ndarray:
    """Extraterrestrial irradiance averaged over a ``bandwidth``-wide window around each of ``waves``.

    ``NaN`` for a wavelength with no Thuillier table point inside its window (matches R's
    ``mean(numeric(0))``, e.g. for a target far outside the table's ~320-998 nm range).
    """
    table = _thuillier()
    thuillier_wave = table["wave"].to_numpy()
    thuillier_f0 = table["F0"].to_numpy()

    waves = np.atleast_1d(np.asarray(waves, dtype=float))
    half = bandwidth / 2.0
    out = np.empty(waves.shape)
    for i, wave in enumerate(waves):
        mask = (thuillier_wave >= wave - half) & (thuillier_wave <= wave + half)
        out[i] = np.mean(0.1 * thuillier_f0[mask]) if np.any(mask) else np.nan
    return out


def compute_nlw(lw_0p: np.ndarray, ed0_0p: np.ndarray, waves: np.ndarray, bandwidth: float) -> np.ndarray:
    """Normalized water-leaving radiance: ``Lw(0+) / Ed0(0+) * etirrwindow(waves, bandwidth)``."""
    lw_0p = np.asarray(lw_0p, dtype=float)
    ed0_0p = np.asarray(ed0_0p, dtype=float)
    return lw_0p / ed0_0p * etirrwindow(waves, bandwidth)
