"""QWIP: Quality Water Index Polynomial (Dierssen et al. 2022, Front. Rem. Sens.).

Port of the scoring computation in ``QWIP.R`` (the plotting half of that
function isn't ported -- see :mod:`pycops.io.netcdf` for how ``pycops``
persists diagnostics instead of plotting them). The Apparent Visible
Wavelength (AVW) and Normalized Difference Index (NDI) characterize an
``Rrs`` spectrum's shape; the QWIP polynomial predicts the NDI a "normal"
water spectrum should have at that AVW, and the score (``NDI - QWIP``) flags
spectra whose shape looks physically implausible (``|score| >= 0.1``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from pycops.processing.color import forel_ule_class

# Dierssen et al. 2022 QWIP polynomial coefficients (NDI as a quartic in AVW).
_P1 = -8.399885e-9
_P2 = 1.715532e-5
_P3 = -1.301670e-2
_P4 = 4.357838e0
_P5 = -5.449532e2

_WAVES_INT = np.arange(400, 701)
_IX_492 = int(np.where(_WAVES_INT == 492)[0][0])
_IX_560 = int(np.where(_WAVES_INT == 560)[0][0])
_IX_665 = int(np.where(_WAVES_INT == 665)[0][0])


def _qwip_polynomial(avw: np.ndarray) -> np.ndarray:
    return _P1 * avw**4 + _P2 * avw**3 + _P3 * avw**2 + _P4 * avw + _P5


@dataclass(frozen=True)
class QWIPResult:
    avw: float  # Apparent Visible Wavelength (nm)
    ndi: float  # Normalized Difference Index between 665/492 nm
    predicted_ndi: float  # QWIP polynomial's NDI prediction at this AVW
    score: float  # ndi - predicted_ndi
    passed: bool  # abs(score) < 0.1
    water_class: str  # "Red", "Blue", or "Green"
    fu: int  # Forel-Ule class, 1-21 (see pycops.processing.color.forel_ule_class)


def compute_qwip(waves: np.ndarray, rrs: np.ndarray) -> QWIPResult:
    """QWIP quality-control score for an ``Rrs`` spectrum, port of ``QWIP.R``'s scoring logic."""
    waves = np.asarray(waves, dtype=float)
    rrs = np.asarray(rrs, dtype=float)
    finite = np.isfinite(rrs)
    waves, rrs = waves[finite], rrs[finite]

    order = np.argsort(waves)
    spline = CubicSpline(waves[order], rrs[order], bc_type="natural")
    rrs_int = spline(_WAVES_INT)

    avw = float(np.sum(rrs_int) / np.sum(rrs_int / _WAVES_INT))
    ndi = float((rrs_int[_IX_665] - rrs_int[_IX_492]) / (rrs_int[_IX_665] + rrs_int[_IX_492]))

    predicted_ndi = float(_qwip_polynomial(np.array(avw)))
    score = ndi - predicted_ndi
    passed = abs(score) < 0.1

    if rrs_int[_IX_665] > rrs_int[_IX_560] or rrs_int[_IX_665] > 0.025:
        water_class = "Red"
    elif rrs_int[_IX_560] < rrs_int[_IX_492]:
        water_class = "Blue"
    else:
        water_class = "Green"

    fu = forel_ule_class(waves, rrs).fu

    return QWIPResult(
        avw=avw,
        ndi=ndi,
        predicted_ndi=predicted_ndi,
        score=score,
        passed=passed,
        water_class=water_class,
        fu=fu,
    )
