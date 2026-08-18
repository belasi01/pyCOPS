"""Forel-Ule water-color class from a remote-sensing reflectance spectrum.

Port of ``Rrs2FU.R``: spline ``Rrs`` onto the CIE 1931 color-matching
functions (``data/cie.csv``, bundled from ``CIE.RData``), integrate against
each of ``x``/``y``/``z`` over 390-830 nm to get CIE tristimulus values,
convert to chromaticity coordinates, and classify the angle from white point
(0.33, 0.33) into one of 21 Forel-Ule bins using the reference hue angles
from Novoa et al. (2013) Table 5 (``FU1``..``FU21``, hardcoded here exactly
as in the R source -- small enough not to need a bundled data file).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline

# Novoa et al. 2013 Table 5: reference (x, y) chromaticity coordinates for FU 1-21.
_FU_XY = np.array(
    [
        [0.191, 0.167], [0.199, 0.200], [0.210, 0.240], [0.227, 0.288], [0.246, 0.335],
        [0.266, 0.376], [0.291, 0.412], [0.315, 0.440], [0.337, 0.462], [0.363, 0.476],
        [0.386, 0.487], [0.402, 0.481], [0.416, 0.474], [0.431, 0.466], [0.446, 0.458],
        [0.461, 0.449], [0.475, 0.441], [0.489, 0.433], [0.503, 0.425], [0.516, 0.416],
        [0.528, 0.408],
    ]
)


def _hue_angle_deg(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = x - 0.33
    dy = y - 0.33
    alpha_rad = np.arctan(dy / dx)
    return np.where(dx >= 0, np.degrees(alpha_rad), 180.0 + np.degrees(alpha_rad))


_FU_ALPHA = _hue_angle_deg(_FU_XY[:, 0], _FU_XY[:, 1])
_FU_ALPHA_T = (_FU_ALPHA[:-1] + _FU_ALPHA[1:]) / 2.0  # 20 boundary angles between the 21 classes


@lru_cache(maxsize=1)
def _cie() -> pd.DataFrame:
    path = resources.files("pycops.processing") / "data" / "cie.csv"
    with resources.as_file(path) as f:
        return pd.read_csv(f)


@dataclass(frozen=True)
class ForelUleResult:
    x: float  # CIE 1931 chromaticity coordinate
    y: float
    fu: int  # Forel-Ule class, 1-21


def forel_ule_class(waves: np.ndarray, rrs: np.ndarray) -> ForelUleResult:
    """Forel-Ule water-color class (1-21) from an ``Rrs`` spectrum, port of ``Rrs2FU``."""
    waves = np.asarray(waves, dtype=float)
    rrs = np.asarray(rrs, dtype=float)
    finite = np.isfinite(rrs)
    waves, rrs = waves[finite], rrs[finite]
    cie = _cie()
    cie_waves = cie["waves"].to_numpy(dtype=float)

    order = np.argsort(waves)
    spline = CubicSpline(waves[order], rrs[order], bc_type="natural")
    rrs_int = spline(cie_waves)
    rrs_int = np.clip(rrs_int, 0.0, None)

    in_range = (cie_waves >= 390) & (cie_waves <= 830)
    w = cie_waves[in_range]
    X = trapezoid(cie["x"].to_numpy(dtype=float)[in_range] * rrs_int[in_range], w)
    Y = trapezoid(cie["y"].to_numpy(dtype=float)[in_range] * rrs_int[in_range], w)
    Z = trapezoid(cie["z"].to_numpy(dtype=float)[in_range] * rrs_int[in_range], w)

    total = X + Y + Z
    x = X / total
    y = Y / total

    alpha_m = float(_hue_angle_deg(np.array([x]), np.array([y]))[0])

    if alpha_m > _FU_ALPHA_T[0]:
        fu = 1
    elif alpha_m < _FU_ALPHA_T[-1]:
        fu = 21
    else:
        fu = next(i + 2 for i in range(19) if alpha_m <= _FU_ALPHA_T[i] and alpha_m > _FU_ALPHA_T[i + 1])

    return ForelUleResult(x=x, y=y, fu=fu)


# CIE XYZ (D65 white point) -> linear sRGB, the standard 3x3 matrix (IEC 61966-2-1).
_XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)
_NOT_VISIBLE_GRAY = 0.6  # neutral swatch for wavelengths outside the CIE table's own domain


def _srgb_gamma(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def wavelength_to_rgb(waves_nm: np.ndarray) -> np.ndarray:
    """Approximate true display color for each wavelength (nm), via the same bundled CIE 1931
    color-matching functions used for :func:`forel_ule_class` -- a monochromatic "spectrum" is
    just a delta function, so its own (x, y, z) color-matching values *are* its CIE XYZ tristimulus
    (up to a scale factor); converting to sRGB gives a physically grounded approximation of the
    perceived color, rather than a hand-tuned visible-spectrum palette.

    Wavelengths outside the CIE table's own domain (360-830 nm -- i.e. genuinely outside what a
    human eye perceives; some COPS deployments carry UV/NIR bands beyond that) get a neutral gray
    swatch, since assigning them a fake hue would misrepresent them as visible colors.

    Returns an ``(n, 3)`` array of RGB values in ``[0, 1]``, gamma-corrected for display.
    Out-of-gamut monochromatic colors (a well-known real limitation of sRGB, especially near
    380-480 nm) are clipped rather than reproduced exactly. Each wavelength's own brightness is
    normalized to its own peak channel -- the human eye's hugely uneven photopic sensitivity
    across the spectrum would otherwise render the violet/red edges almost black relative to the
    555 nm green peak, which isn't the point of a "what color is this band" swatch.
    """
    waves_nm = np.asarray(waves_nm, dtype=float)
    cie = _cie()
    cie_waves = cie["waves"].to_numpy(dtype=float)
    in_range = (waves_nm >= cie_waves.min()) & (waves_nm <= cie_waves.max())

    xyz = np.stack(
        [np.interp(waves_nm, cie_waves, cie[c].to_numpy(dtype=float)) for c in ("x", "y", "z")], axis=-1
    )
    linear_rgb = np.clip(xyz @ _XYZ_TO_LINEAR_SRGB.T, 0.0, None)
    peak = linear_rgb.max(axis=-1, keepdims=True)
    linear_rgb = linear_rgb / np.where(peak == 0, 1.0, peak)

    rgb = _srgb_gamma(linear_rgb)
    rgb[~in_range] = _NOT_VISIBLE_GRAY
    return rgb
