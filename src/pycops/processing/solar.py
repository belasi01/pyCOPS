"""Solar position (zenith/azimuth) from date, time UTC, and location.

Port of ``possol.R`` (Bernard Gentili), used to compute a cast's sun-zenith
angle for the clear-sky irradiance model and shadow correction.
"""

from __future__ import annotations

import math


def sun_position(month: int, day: int, hour_utc: float, lon: float, lat: float) -> tuple[float, float]:
    """Solar zenith and azimuth angles (degrees) for a date/time/location.

    ``hour_utc`` is decimal UTC time (0.0-23.999). ``lon`` is in decimal
    degrees (east positive); ``lat`` in decimal degrees (north positive).
    Returns ``(zenith_deg, azimuth_deg)``; zenith is ``-1`` when the sun is
    below the horizon (matching the R package's sentinel).
    """
    if month <= 2:
        day_of_year = 31 * (month - 1) + day
    elif month > 8:
        day_of_year = 31 * (month - 1) - (month - 2) // 2 - 2 + day
    else:
        day_of_year = 31 * (month - 1) - (month - 1) // 2 - 2 + day

    tsm = hour_utc + lon / 15.0
    lat_rad = math.radians(lat)
    day_angle = 2.0 * math.pi * day_of_year / 365.0

    # equation of time (minutes)
    eq_time = (
        0.000075
        + 0.001868 * math.cos(day_angle)
        - 0.032077 * math.sin(day_angle)
        - 0.014615 * math.cos(2.0 * day_angle)
        - 0.040849 * math.sin(2.0 * day_angle)
    )
    eq_time = eq_time * 12.0 * 60.0 / math.pi

    true_solar_time = tsm + eq_time / 60.0 - 12.0
    hour_angle = math.radians(true_solar_time * 15.0)

    declination = (
        0.006918
        - 0.399912 * math.cos(day_angle)
        + 0.070257 * math.sin(day_angle)
        - 0.006758 * math.cos(2.0 * day_angle)
        + 0.000907 * math.sin(2.0 * day_angle)
        - 0.002697 * math.cos(3.0 * day_angle)
        + 0.001480 * math.sin(3.0 * day_angle)
    )

    cos_elevation_sin_part = math.sin(lat_rad) * math.sin(declination) + math.cos(lat_rad) * math.cos(
        declination
    ) * math.cos(hour_angle)
    elevation = math.asin(cos_elevation_sin_part)

    az_sin = math.cos(declination) * math.sin(hour_angle) / math.cos(elevation)
    az_sin = max(-1.0, min(1.0, az_sin))
    az_cos = (
        -math.cos(lat_rad) * math.sin(declination) + math.sin(lat_rad) * math.cos(declination) * math.cos(hour_angle)
    ) / math.cos(elevation)

    azimuth = math.asin(az_sin)
    if az_cos <= 0.0:
        azimuth = math.pi - azimuth
    elif az_sin <= 0.0:
        azimuth = 2.0 * math.pi + azimuth
    azimuth += math.pi
    if azimuth > 2.0 * math.pi:
        azimuth -= 2.0 * math.pi

    zenith_deg = 90.0 - math.degrees(elevation)
    azimuth_deg = math.degrees(azimuth)
    if zenith_deg > 90.0:
        zenith_deg = -1.0

    return zenith_deg, azimuth_deg
