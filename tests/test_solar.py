from __future__ import annotations

import pytest

from pycops.processing.solar import sun_position


def test_equator_equinox_local_noon_near_zero_zenith():
    # March equinox, local solar noon at lon=0 (hour_utc=12) on the equator:
    # sun should be very close to overhead.
    zenith, _ = sun_position(month=3, day=20, hour_utc=12.0, lon=0.0, lat=0.0)
    assert zenith == pytest.approx(0.0, abs=2.0)


def test_matches_real_cast_order_of_magnitude():
    # WISEMan station MAN-F05, 2019-08-18, ~18:18 UTC, documented sunzen ~41.8 deg
    # in the R package's own README for this exact cast.
    zenith, _ = sun_position(month=8, day=18, hour_utc=18.31, lon=-68.108833, lat=49.13445)
    assert 30.0 < zenith < 55.0


def test_night_side_gives_sentinel():
    zenith, _ = sun_position(month=1, day=1, hour_utc=0.0, lon=0.0, lat=80.0)
    assert zenith == -1.0


def test_matches_r_possol_exactly():
    # Cross-checked by running possol() directly in R (source possol.R) for
    # the real WISE_MAN_F5_CAST_003 cast: zenith=41.78139, azimuth=219.0824.
    zenith, azimuth = sun_position(
        month=8, day=18, hour_utc=18 + 18 / 60 + 38.37 / 3600, lon=-68.108833, lat=49.13445
    )
    assert zenith == pytest.approx(41.78139, abs=1e-4)
    assert azimuth == pytest.approx(219.0824, abs=1e-3)
