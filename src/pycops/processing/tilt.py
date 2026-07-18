"""Tilt (from roll/pitch) and tilt-based scan filtering, ported from ``derived.data.R``."""

from __future__ import annotations

import numpy as np
import xarray as xr

# LuZ is mounted on the same frame as EdZ (or, failing that, EuZ) and has no
# inclinometer of its own in the raw files -- process.LuZ.R falls back to
# EdZ's tilt, then EuZ's. Ed0/EdZ/EuZ always carry their own Roll/Pitch.
_TILT_FALLBACK = {"LuZ": ("EdZ", "EuZ")}


def compute_tilt(roll_deg: np.ndarray, pitch_deg: np.ndarray) -> np.ndarray:
    """Combine roll and pitch (degrees) into a single tilt-from-vertical angle (degrees)."""
    roll = np.deg2rad(np.asarray(roll_deg, dtype=float))
    pitch = np.deg2rad(np.asarray(pitch_deg, dtype=float))
    return np.degrees(np.arctan(np.sqrt(np.tan(roll) ** 2 + np.tan(pitch) ** 2)))


def _roll_pitch_source(ds: xr.Dataset, instrument: str) -> str:
    """Which instrument's Roll/Pitch to use for ``instrument``'s tilt."""
    for candidate in (instrument, *_TILT_FALLBACK.get(instrument, ())):
        if f"{candidate}_Roll" in ds and f"{candidate}_Pitch" in ds:
            return candidate
    raise KeyError(
        f"No Roll/Pitch found for {instrument!r} or its fallbacks "
        f"{_TILT_FALLBACK.get(instrument, ())}"
    )


def add_tilt(ds: xr.Dataset, instrument: str) -> xr.Dataset:
    """Return ``ds`` with a ``<instrument>_Tilt`` variable (re)computed from roll/pitch.

    Matches the R package, which always derives tilt from ``Roll``/``Pitch``
    rather than trusting a ``Tilt`` column some uProfile exports also carry,
    and falls back to a co-mounted instrument's Roll/Pitch when ``instrument``
    has none of its own (see ``_TILT_FALLBACK``).
    """
    source = _roll_pitch_source(ds, instrument)
    tilt = compute_tilt(ds[f"{source}_Roll"].values, ds[f"{source}_Pitch"].values)
    return ds.assign({f"{instrument}_Tilt": ("time", tilt)})


def tilt_mask(ds: xr.Dataset, instrument: str, tiltmax: float) -> xr.DataArray:
    """Boolean mask of scans within the tilt limit for ``instrument`` (strict ``<``)."""
    source = _roll_pitch_source(ds, instrument)
    tilt = compute_tilt(ds[f"{source}_Roll"].values, ds[f"{source}_Pitch"].values)
    return xr.DataArray(tilt < tiltmax, dims="time", coords={"time": ds["time"]})
