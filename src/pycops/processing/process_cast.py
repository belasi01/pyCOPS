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

# select.cops.dat's "method" column names which Rrs a researcher already vetted
# as the right one for a given cast (see discovery.CastSelection.method) --
# some casts fit better with the LOESS surface value, others with the linear
# one (e.g. a non-monotonic depth start near the surface fails the linear
# fit's Kolmogorov-Smirnov gate but the LOESS fit tolerates it fine).
_METHOD_LOESS = "Rrs.0p"
_METHOD_LINEAR = "Rrs.0p.linear"


@dataclass(frozen=True)
class CastResult:
    """Full result of processing one cast: every instrument's fit plus Rrs/Lw."""

    waves: np.ndarray
    ed0_fit: Ed0Fit
    instrument_fits: dict[str, InstrumentFit]
    rrs_loess: RrsResult | None  # from LuZ's LOESS-fitted surface value, if LuZ present
    rrs_linear: RrsResult | None  # from LuZ's linear-fitted surface value, if LuZ present
    rrs_method: str | None  # select.cops.dat's method for this cast, if known
    recommended_rrs: RrsResult | None  # rrs_loess or rrs_linear per rrs_method, whichever is available


def process_cast(ds: xr.Dataset, init: dict[str, object]) -> CastResult:
    """Fit Ed0 plus every depth-profiled instrument present in ``ds``, and Rrs/Lw.

    ``ds`` is one cast from :func:`pycops.io.raw.read_cast` (or
    :func:`pycops.io.discovery.read_deployment_casts`); ``init`` is the parsed
    ``init.cops.dat`` dict from :func:`pycops.io.config.read_init_cops`.
    Instruments the R package's ``instruments.optics`` lists but that aren't
    actually in ``ds`` (e.g. no EuZ sensor on that deployment) are skipped.

    If ``ds`` came from :func:`~pycops.io.discovery.read_deployment_casts`, its
    ``rrs_method`` attr (from ``select.cops.dat``) picks ``recommended_rrs``
    between ``rrs_loess`` and ``rrs_linear``, falling back to whichever is
    available if the preferred one is missing (e.g. no LuZ) or ``None`` (the
    surface fit failed for every wavelength).
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

    rrs_method = ds.attrs.get("rrs_method")
    preferred, fallback = (rrs_loess, rrs_linear) if rrs_method == _METHOD_LOESS else (rrs_linear, rrs_loess)
    recommended_rrs = preferred if preferred is not None else fallback

    return CastResult(
        waves=ds["wavelength"].values,
        ed0_fit=ed0_fit,
        instrument_fits=instrument_fits,
        rrs_loess=rrs_loess,
        rrs_linear=rrs_linear,
        rrs_method=rrs_method,
        recommended_rrs=recommended_rrs,
    )
