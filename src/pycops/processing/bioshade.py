"""Extract the diffuse/total downwelling irradiance split from a BioShade cast.

A BioShade cast (``select.cops.dat`` flag ``2``) is a time series of the
above-water Ed0 sensor while a rotating shading band sweeps across the sky,
periodically occluding the sun disc. Port of ``process.Ed0.BioShade.R``: fit
a LOESS curve through the scans where the band is swung clear (giving a
smooth estimate of the total, direct+diffuse irradiance through the cast),
then find the scan where the band was actually blocking the sun (a global
irradiance minimum near 490 nm, "point M" in Morrow et al. 2012) and read the
raw (diffuse-only) reading there. The ratio of that to the smoothed total at
the same moment is the diffuse fraction, used by
:mod:`pycops.processing.shadow` as a measured alternative to the Gregg &
Carder clear-sky estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from pycops.processing.profile_fit import fit_profile_loess
from pycops.processing.tilt import compute_tilt

# uProfile has exported this column under several different names over the years.
_BIOSHADE_POSITION_NAMES = ("BioShade_Position", "BioShade.Position", "BioShade:Position", "BioShadePosition")

# Encoder-position thresholds marking "band swung clear of the sky" -- ported
# as-is from process.Ed0.BioShade.R, including its "ix.shade" condition being
# an near-tautology (`< 24000 | > 2000` is true for virtually any position;
# only `ix.no.shade`, the complement-shaped `> 24000 | < 2000`, meaningfully
# narrows anything). Kept for fidelity: in practice `ix.shade` is only ever
# used to search for a global minimum among already tilt/time-window-filtered
# scans, so the redundancy doesn't change the result.
_NO_SHADE_HIGH = 24000.0
_NO_SHADE_LOW = 2000.0


def _find_bioshade_position(ds: xr.Dataset) -> np.ndarray:
    for name in _BIOSHADE_POSITION_NAMES:
        if name in ds:
            return ds[name].values
    raise KeyError(f"none of {_BIOSHADE_POSITION_NAMES} found in cast; is this really a BioShade cast?")


@dataclass(frozen=True)
class BioShadeResult:
    """Diffuse/total downwelling irradiance split from a BioShade cast."""

    waves: np.ndarray
    ed0_tot: np.ndarray  # smoothed total (direct+diffuse) Ed0 at the sun-occlusion moment
    ed0_dif: np.ndarray  # raw diffuse-only Ed0 at that same moment
    ed0_diffuse_fraction: np.ndarray  # ed0_dif / ed0_tot


def process_bioshade(ds: xr.Dataset, init: dict[str, object]) -> BioShadeResult:
    """Process a BioShade cast into a diffuse/total Ed0 split.

    ``ds`` is a BioShade cast read with :func:`pycops.io.raw.read_cast`
    (``instruments=("Ed0",)`` is enough -- BioShade casts only need the
    above-water sensor); ``init`` is the parsed ``init.cops.dat`` dict.
    """
    waves = ds["wavelength"].values
    ed0_all = ds["Ed0"].values
    time_seconds = (ds["time"].values - ds["time"].values[0]) / np.timedelta64(1, "s")

    tilt = compute_tilt(ds["Ed0_Roll"].values, ds["Ed0_Pitch"].values)
    tilt_ok = tilt < init["tiltmax.optics"]["Ed0"]

    time_window = init.get("time.window", [0.0, 10000.0])
    time_ok = (time_seconds >= time_window[0]) & (time_seconds <= time_window[1])

    kept = tilt_ok & time_ok
    ed0 = ed0_all[kept]
    time_kept = time_seconds[kept]
    position = _find_bioshade_position(ds)[kept]

    no_shade = (position > _NO_SHADE_HIGH) | (position < _NO_SHADE_LOW)
    shade = (position < _NO_SHADE_HIGH) | (position > _NO_SHADE_LOW)

    fitted = fit_profile_loess(
        waves,
        time_kept[no_shade],
        ed0[no_shade],
        span=0.8,
        depth_grid=time_kept,
        idx_depth_0=0,
        span_wave_correction=False,
        depth_span=False,
    )
    ed0_fitted = fitted.fitted

    ix_490 = int(np.argmin(np.abs(waves - 490.0)))
    shade_indices = np.flatnonzero(shade)
    ix_m = shade_indices[np.argmin(ed0[shade_indices, ix_490])]

    ed0_tot = ed0_fitted[ix_m, :]
    ed0_dif = ed0[ix_m, :]
    ed0_diffuse_fraction = ed0_dif / ed0_tot

    return BioShadeResult(waves=waves, ed0_tot=ed0_tot, ed0_dif=ed0_dif, ed0_diffuse_fraction=ed0_diffuse_fraction)
