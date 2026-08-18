from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pycops.io.database_report import (
    build_kd_comparison_figure,
    build_pd_depth_comparison_figure,
    build_rrs_comparison_figure,
)
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
        pd_depth=kd_for(0.3),
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
        pd_depth=_nan_mean_sd(),
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


def test_build_rrs_comparison_figure_accepts_a_trimmed_waves_grid():
    """A caller that pre-trims stations via trim_unused_wavelengths() must pass the matching
    (shorter) waves array -- the figure should plot on that grid, not silently fall back to the
    full 25-band STANDARD_WAVELENGTHS (which would misalign with the trimmed MeanSd arrays)."""
    import matplotlib.pyplot as plt

    trimmed_waves = np.array([443.0, 555.0])
    station = StationAggregate(
        station_id="A",
        directory=Path("/data/A/cops"),
        n_casts=1,
        date_mean=None,
        sun_zenith_mean=None,
        longitude_mean=None,
        latitude_mean=None,
        forel_ule_mean=None,
        shadow_correction_method=None,
        bottom_depth_mean=None,
        ed0_diffuse_fraction=None,
        rrs=MeanSd(mean=np.array([1.0, 2.0]), sd=np.array([0.1, 0.2])),
        nlw=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        rb=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        kd_1pct=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        kd_10pct=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        kd_pd=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        pd_depth=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        ed0_0p=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        par_0=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_1pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_10pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_pd=ScalarMeanSd(mean=np.nan, sd=np.nan),
    )

    fig = build_rrs_comparison_figure([station], waves=trimmed_waves)

    assert fig is not None
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), trimmed_waves)
    plt.close(fig)


def test_build_rrs_comparison_figure_sd_band_has_no_fabricated_floor_when_sd_exceeds_mean():
    """Regression test for a real bug (Amundsen 2026, station UMQ3): 3 kept casts' Rrs at
    875 nm ranged ~5.8e-6 to 4.8e-5 -- an SD (~2.3e-5) bigger than the mean (~2.1e-5), so
    mean - sd goes negative. The old code clamped that to a fixed 1e-12 floor, which rendered
    as a physically meaningless multi-order-of-magnitude plunge on the log-scale axis (nothing
    in the real data was anywhere near 1e-12) -- Simon spotted this comparing the plot against
    his own casts' real values."""
    real_values = np.array([4.765e-5, 5.76e-6, 9.08e-6])
    mean = np.full(N_WAVES, np.nan)
    sd = np.full(N_WAVES, np.nan)
    # a neighboring, well-behaved wavelength (normal sd < mean) so fill_between has an actual
    # band to draw -- otherwise a single all-problematic-point station has nothing to fill at
    # all, which trivially "passes" without exercising the fix.
    mean[7] = 2e-5
    sd[7] = 3e-6
    mean[8] = float(real_values.mean())
    sd[8] = float(real_values.std(ddof=1))
    assert mean[8] - sd[8] < 0  # reproduces the exact real-data condition

    station = StationAggregate(
        station_id="UMQ3",
        directory=Path("/data/UMQ3/cops"),
        n_casts=3,
        date_mean=None,
        sun_zenith_mean=None,
        longitude_mean=None,
        latitude_mean=None,
        forel_ule_mean=None,
        shadow_correction_method=None,
        bottom_depth_mean=None,
        ed0_diffuse_fraction=None,
        rrs=MeanSd(mean=mean, sd=sd),
        nlw=_nan_mean_sd(),
        rb=_nan_mean_sd(),
        kd_1pct=_nan_mean_sd(),
        kd_10pct=_nan_mean_sd(),
        kd_pd=_nan_mean_sd(),
        pd_depth=_nan_mean_sd(),
        ed0_0p=_nan_mean_sd(),
        par_0=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_1pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_10pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_pd=ScalarMeanSd(mean=np.nan, sd=np.nan),
    )

    fig = build_rrs_comparison_figure([station])

    assert fig is not None
    for ax in fig.axes:
        poly = ax.collections[0]
        y_values = np.concatenate([path.vertices[:, 1] for path in poly.get_paths()])
        finite_y = y_values[np.isfinite(y_values)]
        # nothing anywhere near the old 1e-12 floor -- the real data's own smallest value is
        # ~5.76e-6, so the shaded band should never dip more than a couple orders of magnitude
        # below that, not seven.
        assert finite_y.min() > 1e-8
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

    np.testing.assert_allclose(y_1pct, station.kd_1pct.mean, equal_nan=True)
    np.testing.assert_allclose(y_pd, station.kd_pd.mean, equal_nan=True)
    assert not np.allclose(y_1pct, y_pd, equal_nan=True)
    plt.close(fig_1pct)
    plt.close(fig_pd)


def test_build_kd_comparison_figure_none_for_empty_list():
    assert build_kd_comparison_figure([], metric="kd_pd") is None


def test_plot_does_not_draw_a_line_across_a_missing_green_band():
    """Regression test: Simon's report -- when a station doesn't reach the 1%/10% light level
    in the green band within the cast's own measured depth (common, since Kd is smallest
    there), that gap must not be silently dropped before plotting: doing so would let
    matplotlib draw a straight line directly from the last blue-band point to the first
    red-band point, a line that doesn't correspond to any real data."""
    import matplotlib.pyplot as plt

    from pycops.io.database_report import _plot_mean_sd_band

    mean = np.full(N_WAVES, np.nan)
    mean[5] = 1.0  # a blue-ish band, present
    mean[6] = 1.1
    # indices 7-17 (green-ish) intentionally left NaN -- the gap
    mean[18] = 5.0  # a red-ish band, present
    mean[19] = 5.2
    sd = np.where(np.isfinite(mean), mean * 0.05, np.nan)

    fig, ax = plt.subplots()
    _plot_mean_sd_band(ax, np.asarray(STANDARD_WAVELENGTHS, dtype=float), MeanSd(mean=mean, sd=sd), "blue", "station")

    line = ax.lines[0]
    ydata = line.get_ydata()
    # the gap survives into the plotted data -- matplotlib itself (not pre-filtering) is what
    # must decide to break the line there.
    assert np.isnan(ydata[10])
    plt.close(fig)


def test_build_pd_depth_comparison_figure_none_for_empty_list():
    assert build_pd_depth_comparison_figure([]) is None


def test_build_pd_depth_comparison_figure_inverted_axis_one_band_per_station():
    import matplotlib.pyplot as plt

    stations = [_make_station("A"), _make_station("B", kd_scale=2.0)]

    fig = build_pd_depth_comparison_figure(stations)

    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    ylim = ax.get_ylim()
    assert ylim[0] > ylim[1]  # inverted -- depth 0 at the top
    assert len(ax.lines) == 2  # one mean line per station
    assert len(ax.collections) == 2  # one shaded SD band per station
    plt.close(fig)


def test_build_pd_depth_comparison_figure_accepts_a_trimmed_waves_grid():
    import matplotlib.pyplot as plt

    trimmed_waves = np.array([443.0, 555.0])
    station = StationAggregate(
        station_id="A",
        directory=Path("/data/A/cops"),
        n_casts=1,
        date_mean=None,
        sun_zenith_mean=None,
        longitude_mean=None,
        latitude_mean=None,
        forel_ule_mean=None,
        shadow_correction_method=None,
        bottom_depth_mean=None,
        ed0_diffuse_fraction=None,
        rrs=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        nlw=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        rb=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        kd_1pct=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        kd_10pct=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        kd_pd=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        pd_depth=MeanSd(mean=np.array([2.0, 3.0]), sd=np.array([0.1, 0.2])),
        ed0_0p=MeanSd(mean=np.full(2, np.nan), sd=np.full(2, np.nan)),
        par_0=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_1pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_10pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_pd=ScalarMeanSd(mean=np.nan, sd=np.nan),
    )

    fig = build_pd_depth_comparison_figure([station], waves=trimmed_waves)

    assert fig is not None
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), trimmed_waves)
    plt.close(fig)


def test_build_pd_depth_comparison_figure_none_when_all_nan():
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
        pd_depth=_nan_mean_sd(),
        ed0_0p=_nan_mean_sd(),
        par_0=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_1pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_10pct=ScalarMeanSd(mean=np.nan, sd=np.nan),
        kd_par_pd=ScalarMeanSd(mean=np.nan, sd=np.nan),
    )

    assert build_pd_depth_comparison_figure([station]) is None
