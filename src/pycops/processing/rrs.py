"""Basic (non-shadow-corrected) water-leaving radiance and remote-sensing reflectance.

Port of the unconditional part of ``compute.aops.R``'s ``Lw.0p``/``Rrs.0p``/
``nLw.0p`` formulas. Shadow correction, the Gregg & Carder diffuse/direct
split, and Q/f BRDF factors adjust ``LuZ.0m`` upstream of this and are not
yet fully ported (see :mod:`pycops.processing.shadow`/``qfactor``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycops.processing.nlw import compute_nlw


@dataclass(frozen=True)
class RrsResult:
    lw_0p: np.ndarray
    rrs_0p: np.ndarray
    nlw_0p: np.ndarray | None = None  # normalized water-leaving radiance, if waves/bandwidth given


def compute_rrs(
    luz_0m: np.ndarray,
    ed0_0p: np.ndarray,
    indice_water: float,
    rau_fresnel: float,
    waves: np.ndarray | None = None,
    bandwidth: float | None = None,
) -> RrsResult:
    """Water-leaving radiance and reflectance from surface-extrapolated LuZ and Ed0.

    ``indice_water`` and ``rau_fresnel`` come from ``init.cops.dat`` (typically
    ``1.34`` and ``0.043``). When ``waves``/``bandwidth`` (``init.cops.dat``'s
    ``bandwidth``) are both given, also computes ``nlw_0p`` (see
    :func:`pycops.processing.nlw.compute_nlw`); otherwise ``nlw_0p`` is ``None``.
    """
    luz_0m = np.asarray(luz_0m, dtype=float)
    ed0_0p = np.asarray(ed0_0p, dtype=float)
    lw_0p = luz_0m * (1 - rau_fresnel) / indice_water**2
    rrs_0p = lw_0p / ed0_0p
    nlw_0p = compute_nlw(lw_0p, ed0_0p, waves, bandwidth) if waves is not None and bandwidth is not None else None
    return RrsResult(lw_0p=lw_0p, rrs_0p=rrs_0p, nlw_0p=nlw_0p)
