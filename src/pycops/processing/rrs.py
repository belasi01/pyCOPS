"""Basic (non-shadow-corrected) water-leaving radiance and remote-sensing reflectance.

Port of the unconditional part of ``compute.aops.R``'s ``Lw.0p``/``Rrs.0p``
formulas. Shadow correction, the Gregg & Carder diffuse/direct split, and Q/f
BRDF factors adjust ``LuZ.0m`` upstream of this and are not yet ported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RrsResult:
    lw_0p: np.ndarray
    rrs_0p: np.ndarray


def compute_rrs(luz_0m: np.ndarray, ed0_0p: np.ndarray, indice_water: float, rau_fresnel: float) -> RrsResult:
    """Water-leaving radiance and reflectance from surface-extrapolated LuZ and Ed0.

    ``indice_water`` and ``rau_fresnel`` come from ``init.cops.dat`` (typically
    ``1.34`` and ``0.043``).
    """
    luz_0m = np.asarray(luz_0m, dtype=float)
    ed0_0p = np.asarray(ed0_0p, dtype=float)
    lw_0p = luz_0m * (1 - rau_fresnel) / indice_water**2
    rrs_0p = lw_0p / ed0_0p
    return RrsResult(lw_0p=lw_0p, rrs_0p=rrs_0p)
