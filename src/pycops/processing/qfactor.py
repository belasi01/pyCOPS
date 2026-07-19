"""The Q factor relating upwelling irradiance (EuZ) to upwelling radiance (LuZ).

Port of ``Q.and.f.factors.R``, restricted to the piece it actually feeds
into elsewhere (``compute.aops.R``'s ``Q.sun.nadir``, used to convert
between ``EuZ.0m``/``LuZ.0m`` when a cast has only one of the two
instruments): ``Q.sun.nadir`` defaults to the constant ``pi`` at every
wavelength unless ``info.cops.dat``'s ``chl`` field is a genuine positive
chlorophyll concentration, in which case the R package looks it up from a
bidirectional-reflectance model (``popt.f.Q``, in ``popt.R``) instead of
using the constant. ``popt.R`` isn't ported yet (see
:func:`pycops.processing.shadow.resolve_absorption`, which has the same
``chl > 0`` gap for the shadow-correction absorption model) -- so this
raises the same way, rather than silently pretending ``chl > 0`` works.

Deliberate deviation from the R source: ``Q.and.f.factors.R`` only special-
cases ``chl == 999`` before calling ``popt.f.Q`` -- for ``chl == 0`` it
still calls ``popt.f.Q(..., log(0), ...)``, i.e. ``log(-Inf)``, seemingly an
unexercised/unintended edge case (``chl == 0`` means "read absorption from
``absorption.cops.dat``", per ``resolve_absorption``, not "no chlorophyll
estimate available" in any sense ``popt.f.Q`` was designed around). Since
``popt.R`` isn't ported anyway, there's nothing meaningful to reproduce
there -- ``chl == 0`` defaults to the constant here too, consistently with
every other ``chl`` value pycops's shadow correction already supports.

The rest of ``Q.and.f.factors.R`` (``Q.0``, ``f.sun``, ``f.0``) isn't ported:
nothing else in the pieces of the R package ported so far consumes them.
"""

from __future__ import annotations

import numpy as np

_PI_DEFAULT_CHL = (0, 999)


def compute_q_factor(chl: float | None, n_waves: int) -> np.ndarray:
    """``Q.sun.nadir``: the EuZ/LuZ ratio, one value per wavelength.

    ``chl`` is ``info.cops.dat``'s ``chl`` field (``None``/``NaN`` -- no
    shadow correction --, ``0`` -- absorption from a file --, or ``999``
    -- Kd-derived absorption -- all default to the constant ``pi``, matching
    every ``chl`` value pycops's shadow correction (see
    :mod:`pycops.processing.shadow`) already supports. ``chl > 0`` (an actual
    chlorophyll concentration, the Morel & Maritorena / Loisel-Morel
    bidirectional model in ``popt.R``) raises ``NotImplementedError``.
    """
    if chl is None or (isinstance(chl, float) and np.isnan(chl)) or chl in _PI_DEFAULT_CHL:
        return np.full(n_waves, np.pi)

    raise NotImplementedError(
        "chlorophyll-based Q factor (Morel & Maritorena / Loisel-Morel, info.cops.dat chl > 0, "
        "popt.f.Q) is not yet ported"
    )
