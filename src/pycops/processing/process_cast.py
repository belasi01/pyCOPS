"""Run the whole per-cast pipeline: fit every depth-profiled instrument
present and derive Rrs/Lw from LuZ and Ed0.

Ties :func:`pycops.processing.cast_fit.fit_ed0_for_cast` and
:func:`pycops.processing.cast_fit.fit_cast` together across all instruments
in one cast, plus :func:`pycops.processing.rrs.compute_rrs`. Equivalent to one
iteration of ``process.cops.R``'s per-cast loop followed by the unconditional
part of ``compute.aops.R`` -- shadow correction, the Gregg & Carder
diffuse/direct split, and Q/f BRDF factors are not yet ported, so a EuZ-only
cast (no LuZ) can't yet produce Rrs (that requires the Q/f-based EuZ-to-LuZ
conversion in ``compute.aops.R``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from pycops.processing.cast_fit import InstrumentFit, fit_cast, fit_ed0_for_cast
from pycops.processing.ed0 import Ed0Fit
from pycops.processing.rrs import RrsResult, compute_rrs

_DEPTH_PROFILED_INSTRUMENTS = ("EdZ", "LuZ", "EuZ")


@dataclass(frozen=True)
class CastResult:
    """Full result of processing one cast: every instrument's fit plus Rrs/Lw."""

    waves: np.ndarray
    ed0_fit: Ed0Fit
    instrument_fits: dict[str, InstrumentFit]
    rrs_loess: RrsResult | None  # from LuZ's LOESS-fitted surface value, if LuZ present
    rrs_linear: RrsResult | None  # from LuZ's linear-fitted surface value, if LuZ present


def process_cast(ds: xr.Dataset, init: dict[str, object]) -> CastResult:
    """Fit Ed0 plus every depth-profiled instrument present in ``ds``, and Rrs/Lw.

    ``ds`` is one cast from :func:`pycops.io.raw.read_cast` (or
    :func:`pycops.io.discovery.read_deployment_casts`); ``init`` is the parsed
    ``init.cops.dat`` dict from :func:`pycops.io.config.read_init_cops`.
    Instruments the R package's ``instruments.optics`` lists but that aren't
    actually in ``ds`` (e.g. no EuZ sensor on that deployment) are skipped.
    """
    ed0_fit = fit_ed0_for_cast(ds, init)

    instrument_fits = {instr: fit_cast(ds, init, instr, ed0_fit) for instr in _DEPTH_PROFILED_INSTRUMENTS if instr in ds}

    rrs_loess = rrs_linear = None
    if "LuZ" in instrument_fits:
        luz_fit = instrument_fits["LuZ"]
        indice_water = init["indice.water"]
        rau_fresnel = init["rau.Fresnel"]
        rrs_loess = compute_rrs(luz_fit.value_at_0, ed0_fit.value_at_0, indice_water, rau_fresnel)
        rrs_linear = compute_rrs(
            luz_fit.surface_linear.value_at_surface, ed0_fit.value_at_0, indice_water, rau_fresnel
        )

    return CastResult(
        waves=ds["wavelength"].values,
        ed0_fit=ed0_fit,
        instrument_fits=instrument_fits,
        rrs_loess=rrs_loess,
        rrs_linear=rrs_linear,
    )
