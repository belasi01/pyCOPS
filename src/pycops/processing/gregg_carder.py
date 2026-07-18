"""Clear-sky spectral solar irradiance (direct + diffuse), Gregg & Carder (1990).

Port of ``GreggCarder.R``/``GreggCarder.data.R``. Used by shadow correction to
estimate the sky-to-sun diffuse/direct irradiance ratio when no BioShade
shadow-band measurements are available for a cast.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _reference_table() -> pd.DataFrame:
    path = resources.files("pycops.processing") / "data" / "gregg_carder.csv"
    with resources.as_file(path) as f:
        return pd.read_csv(f)


def _approx(x: np.ndarray, y: np.ndarray, xout: np.ndarray) -> np.ndarray:
    """Linear interpolation returning NaN outside ``x``'s range (R's ``approx`` default)."""
    xout = np.asarray(xout, dtype=float)
    out = np.interp(xout, x, y, left=np.nan, right=np.nan)
    return out


@dataclass(frozen=True)
class ClearSkyIrradiance:
    """Direct/diffuse spectral irradiance just below a clear-sky surface."""

    waves: np.ndarray
    edir: np.ndarray
    edif: np.ndarray
    ed: np.ndarray


def _navaer(rel_humidity: float, air_mass: float, wind_speed_mean: float, wind_speed: float, visibility_km: float):
    rh = 99.9 if rel_humidity >= 100.0 else rel_humidity
    frh = ((2.0 - rh / 100.0) / (6.0 * (1.0 - rh / 100.0))) ** 0.333

    a = np.empty(3)
    a[0] = 2000.0 * air_mass * air_mass
    a[1] = max(5.866 * (wind_speed_mean - 2.2), 0.5)
    a[2] = max(0.01527 * (wind_speed - 2.2) * 0.05, 1.4e-5)

    ro = np.array([0.03, 0.24, 2.0])
    r = np.array([0.1, 1.0, 10.0])
    dndr = np.empty(3)
    for n in range(3):
        rden = frh * ro
        arg = np.log(r[n] / rden) ** 2
        dndr[n] = np.sum(a * np.exp(-arg) / frh)

    log_r = np.log10(r)
    log_dndr = np.log10(dndr)
    sumx = log_r.sum()
    sumy = log_dndr.sum()
    sumxy = (log_r * log_dndr).sum()
    sumx2 = (log_r * log_r).sum()

    gama = sumxy / sumx2
    rlogc = sumy / 3.0 - gama * sumx / 3.0
    alpha = -(gama + 3.0)

    rlam = 0.55
    cext = 3.91 / visibility_km
    beta = cext * rlam**alpha

    if alpha > 1.2:
        asymp = 0.65
    elif alpha < 0.0:
        asymp = 0.82
    else:
        asymp = -0.14167 * alpha + 0.82

    w0 = (-0.0032 * air_mass + 0.972) * np.exp(3.06e-4 * rh)

    return {"beta": beta, "alpha": alpha, "wa": w0, "asymp": asymp, "rlogc": rlogc}


def surface_reflectance(sun_zenith_deg: float, wind_speed: float) -> tuple[float, float]:
    """Direct (``rod``) and diffuse (``ros``) sea-surface reflectance for a given
    sun zenith angle and wind speed. Port of ``GreggCarder.sfcrfl``.
    """
    rn = 1.341  # refractive index of pure seawater
    roair = 1.2e3

    if wind_speed > 4.0:
        if wind_speed <= 7.0:
            cn = 6.2e-4 + 1.56e-3 / wind_speed
            rof = roair * cn * 2.2e-5 * wind_speed**2 - 4.0e-4
        else:
            cn = 0.49e-3 + 0.065e-3 * wind_speed
            rof = (roair * cn * 4.5e-5 - 4.0e-5) * wind_speed**2
        rosps = 0.057
    else:
        rof = 0.0
        rosps = 0.066

    theta = sun_zenith_deg
    if theta < 50.0 or wind_speed < 2.0:
        if theta == 0.0:
            rospd = 0.0211
        else:
            rtheta = np.radians(theta)
            sintr = np.sin(rtheta) / rn
            rthetar = np.arcsin(sintr)
            rmin = rtheta - rthetar
            rpls = rtheta + rthetar
            sinp = (np.sin(rmin) ** 2) / (np.sin(rpls) ** 2)
            tanp = (np.tan(rmin) ** 2) / (np.tan(rpls) ** 2)
            rospd = 0.5 * (sinp + tanp)
    else:
        a = 5.25e-4 * wind_speed + 0.065
        b = -1.67e-3 * wind_speed + 0.074
        rospd = a * np.exp(b * (theta - 60.0))

    rod = rospd + rof
    ros = rosps + rof
    return rod, ros


def clear_sky_irradiance(
    julian_day: int,
    lon: float,
    lat: float,
    waves: np.ndarray,
    sun_zenith_deg: float,
    visibility_km: float = 15.0,
    air_mass: float = 1.0,
    wind_speed_mean: float = 4.0,
    wind_speed: float = 6.0,
    pressure_mb: float = 1013.25,
    rel_humidity: float = 80.0,
    water_vapor_cm: float = 1.5,
) -> ClearSkyIrradiance:
    """Clear-sky direct+diffuse spectral irradiance just below the surface.

    ``sun_zenith_deg`` should come from :func:`pycops.processing.solar.sun_position`
    for the cast's date/time/location (the R package always calls this with an
    explicit zenith angle rather than computing it internally from hour/lon/lat).
    Returns NaN at any wavelength outside the reference table's 320-900 nm range.
    """
    waves = np.asarray(waves, dtype=float)
    if sun_zenith_deg > 90.0:
        raise ValueError("sun is below the horizon (zenith > 90 deg); irradiance is zero/undefined")

    table = _reference_table()
    fobar = _approx(table["wave"].to_numpy(), table["Fobar"].to_numpy(), waves)
    oza = _approx(table["wave"].to_numpy(), table["oza"].to_numpy(), waves)
    ag = _approx(table["wave"].to_numpy(), table["ag"].to_numpy(), waves)
    aw = _approx(table["wave"].to_numpy(), table["aw"].to_numpy(), waves)

    fo = fobar * (1.0 + 1.67e-2 * np.cos(2.0 * np.pi * (julian_day - 3) / 365.0)) ** 2

    theta = sun_zenith_deg
    cosunz = np.cos(np.radians(theta))
    rm = 1.0 / (cosunz + 0.15 * (93.885 - theta) ** -1.253)
    rmp = pressure_mb / 1013.25 * rm
    rmo = (1.0 + 22.0 / 6370.0) / (cosunz**2 + 44.0 / 6370.0) ** 0.5

    nav = _navaer(rel_humidity, air_mass, wind_speed_mean, wind_speed, visibility_km)
    eta = -nav["alpha"]
    alg = np.log(1.0 - nav["asymp"])
    afs = alg * (1.459 + alg * (0.1595 + alg * 0.4129))
    bfs = alg * (0.0783 + alg * (-0.3824 - alg * 0.5874))
    fa = 1.0 - 0.5 * np.exp((afs + bfs * cosunz) * cosunz)

    rlam = waves * 1.0e-3
    tr = 1.0 / (115.6406 * rlam**4 - 1.335 * rlam**2)
    rtra = np.exp(-tr * rmp)

    to3 = 235.0 + (150.0 + 40.0 * np.sin(0.9865 * (julian_day - 30.0)) + 20.0 * np.sin(3.0 * lon)) * np.sin(
        1.28 * lat
    ) ** 2
    sco3 = to3 * 1.0e-3
    to = oza * sco3
    otra = np.exp(-to * rmo)

    ta = nav["beta"] * rlam**eta
    atra = np.exp(-ta * rm)
    taa = np.exp(-(1.0 - nav["wa"]) * ta * rm)
    tas = np.exp(-nav["wa"] * ta * rm)

    gtmp = (1.0 + 118.3 * ag * rmp) ** 0.45
    gtra = np.exp(-1.41 * ag * rmp / gtmp)

    wtmp = (1.0 + 20.07 * aw * water_vapor_cm * rm) ** 0.45
    wtra = np.exp(-0.2385 * aw * water_vapor_cm * rm / wtmp)

    edir = fo * cosunz * rtra * otra * atra * gtra * wtra
    dray = fo * cosunz * gtra * wtra * otra * taa * 0.5 * (1.0 - rtra**0.95)
    daer = fo * cosunz * gtra * wtra * otra * rtra**1.5 * taa * fa * (1.0 - tas)
    edif = dray + daer
    ed = edir + edif

    return ClearSkyIrradiance(waves=waves, edir=edir, edif=edif, ed=ed)
