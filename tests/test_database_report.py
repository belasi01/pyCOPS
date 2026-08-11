from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pycops.io.database_report import build_kd_comparison_figure, build_rrs_comparison_figure
from pycops.processing.database import STANDARD_WAVELENGTHS, MeanSd, ScalarMeanSd, StationAggregate

N_WAVES = len(STANDARD_WAVELENGTHS)


def _nan_mean_sd() -> MeanSd:
    return MeanSd(mean=np.full(N_WAVES, np.nan), sd=np.full(N_WAVES, np.nan))


def _make_station(station_id: str, rrs_scale: float = 1.0, kd_scale: float = 1.0) -> StationAggregate:
    rrs_mean = np.full(N_WAVES, np.nan)
    rrs_mean[5:10] = np.linspace(0.001, 0.005, 5) * rrs_scale
    rrs = MeanSd(mean=rrs_mean, sd=np.nan_to_num(rrs_mean * 0.1))

    def kd_for(base: float) -> MeanSd:
        mean = np.full(N_WAVES, np.nan)
        mean[5:10] = np.linspace(base, base * 2, 5) * kd_scale
        return MeanSd(mean=mean, sd=np.nan_to_num(mean * 0.1))

    return StationAggregate(
        station_id=station_id,
        directory=Path(f"/data/{station_id}/cops"),
        n_casts=3,
        date_mean=pd.Timestamp("2019-08-17T12:00:00"),
        sun_zenith_mean=45.0,
        longitude_mean=-68.1,
        latitude_mean=49.1,
        forel_ule_mean=6.0,
        shadow_correction_method="abs.Kd.method",
        bottom_depth_mean=None,
        ed0_diffuse_fraction=None,
        rrs=rrs,
        nlw=_nan_mean_sd(),
        rb=_nan_mean_sd(),
        kd_1pct=kd_for(1.0),
        kd_10pct=kd_for(0.5),
        kd_pd=kd_for(0.3),
        ed0_0p=_nan_mean_sd(),
        par_0=ScalarMeanSd(mean=500.0, sd=10.0),
        kd_par_1pct=ScalarMeanSd(mean=0.5, sd=0.05),
        kd_par_10pct=ScalarMeanSd(mean=0.6, sd=0.06),
        kd_par_pd=ScalarMeanSd(mean=0.55, sd=0.055),
    )


def test_build_rrs_comparison_figure_none_for_empty_list():
    assert build_rrs_comparison_figure([]) is None


def test_build_rrs_comparison_figure_none_when_all_stations_nan():
    station = StationAggregate(
        station_id="A",
        directory=Path("/data/A"),
        n_casts=1,
        date_mean=None,
        sun_zenith_mean=None,
        longitude_mean=None,
        latitude_mean=None,
        forel_ule_mean=None,
        shadow_correction_method=None,
        bottom_depth_mean=None,
        ed0_diffuse_fraction=None,
        rrs=_nan_mean_sd(),
        nlw=_nan_mean_sd(),
        rb=_nan_mean_sd(),
        kd_1pct=_nan_mean_sd(),
        kd_10pct=_nan_mean_sd(),
        kd_pd=_nan_mean_sd(),
        ed0_0p=_nan_mean_sd(),
        par_0=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_1pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_10pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_pd=ScalarMeanSd(mean=np.nan, sd=np.nan),
    )

    assert build_rrs_comparison_figure([station]) is None


def test_build_rrs_comparison_figure_has_side_by_side_linear_and_log_axes_with_mean_and_band():
    stations = [_make_station("A"), _make_station("B", rrs_scale=2.0)]

    fig = build_rrs_comparison_figure(stations)

    assert fig is not None
    assert len(fig.axes) == 2
    scales = {ax.get_yscale() for ax in fig.axes}
    assert scales == {"linear", "log"}
    for ax in fig.axes:
        assert len(ax.lines) == 2  # one mean line per station
        assert len(ax.collections) == 2  # one shaded SD band per station
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_build_kd_comparison_figure_invalid_metric_raises():
    with pytest.raises(ValueError, match="metric must be one of"):
        build_kd_comparison_figure([_make_station("A")], metric="not_a_real_metric")


def test_build_kd_comparison_figure_uses_the_requested_metric():
    import matplotlib.pyplot as plt

    station = _make_station("A")

    fig_1pct = build_kd_comparison_figure([station], metric="kd_1pct")
    fig_pd = build_kd_comparison_figure([station], metric="kd_pd")

    linear_ax_1pct = fig_1pct.axes[0]
    linear_ax_pd = fig_pd.axes[0]
    y_1pct = linear_ax_1pct.lines[0].get_ydata()
    y_pd = linear_ax_pd.lines[0].get_ydata()

    np.testing.assert_allclose(y_1pct, station.kd_1pct.mean[np.isfinite(station.kd_1pct.mean)])
    np.testing.assert_allclose(y_pd, station.kd_pd.mean[np.isfinite(station.kd_pd.mean)])
    assert not np.allclose(y_1pct, y_pd)
    plt.close(fig_1pct)
    plt.close(fig_pd)


def test_build_kd_comparison_figure_none_for_empty_list():
    assert build_kd_comparison_figure([], metric="kd_pd") is None
