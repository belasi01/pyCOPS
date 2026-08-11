from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pycops.io.config import CastInfo
from pycops.io.netcdf import cast_result_to_dataset
from pycops.io.pdf_report import (
    _effective_tiltmax,
    _effective_time_window,
    _k0_at_adaptive_depth,
    _kept_mask,
    _mask_negligible_rb,
    _raw_scan_values,
    _visible_band_ylim,
    build_cast_report_figures,
    build_cover_page,
    build_ed0_stability_figure,
    build_extrapolation_grid_figure,
    build_par_kd_par_figures,
    build_qfactor_figure,
    build_rrs_figure,
    build_spectral_kd_figure,
    build_station_kd_penetration_depth_figure,
    build_station_par_depth_table,
    build_station_par_profile_figure,
    build_station_qfactor_figure,
    build_tilt_figures_combined,
    write_cast_pdf_report,
    write_station_pdf_reports,
    write_station_summary_pdf,
)
from pycops.processing.process_cast import process_cast

WAVES = (340.0, 380.0, 443.0, 555.0)
K_TRUE = (2.0, 0.9, 0.3, 0.1)
LUZ_X0 = (0.0015, 0.006, 0.05, 0.3)
EDZ_X0 = (50.0, 80.0, 150.0, 300.0)
EUZ_X0 = (0.002, 0.008, 0.06, 0.35)
DELTA_LUZ = 0.238
DELTA_EDZ = -0.05
DELTA_EUZ = 0.238


def _make_full_dataset(n=300):
    """EdZ+LuZ+EuZ, a real position/chl (shadow correction) and shallow=True (bottom
    reflectance) -- exercises every optional section build_cast_report_figures() can produce."""
    depth = np.linspace(0.05, 8.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)
    luz = np.array(LUZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_LUZ)[:, None])
    edz = np.array(EDZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_EDZ)[:, None])
    euz = np.array(EUZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_EUZ)[:, None])
    ed0 = np.full((n, len(waves)), 100.0)
    zeros = np.zeros(n)
    times = pd.date_range("2019-08-17T22:08:56", periods=n, freq="s")
    ds = xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "LuZ": (("time", "wavelength"), luz),
            "EdZ": (("time", "wavelength"), edz),
            "EuZ": (("time", "wavelength"), euz),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "EdZ_Roll": ("time", zeros),
            "EdZ_Pitch": ("time", zeros),
            "EuZ_Roll": ("time", zeros),
            "EuZ_Pitch": ("time", zeros),
            "LuZ_Depth": ("time", depth),
            "EuZ_Depth": ("time", depth),
            "LuZ_Temp": ("time", np.full(n, 10.0)),
            "EuZ_Temp": ("time", np.full(n, 10.0)),
        },
        coords={"time": times, "wavelength": waves},
    )
    ds.attrs["chl_flag"] = 999.0
    ds.attrs["longitude"] = -68.11626
    ds.attrs["latitude"] = 49.24872
    ds.attrs["shallow"] = True
    ds.attrs["rrs_method"] = "Rrs.0p.linear"
    return ds


def _make_full_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0, "EuZ": 7.0},
        "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035, "EuZ": 0.035},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": DELTA_EDZ, "LuZ": DELTA_LUZ, "EuZ": DELTA_EUZ},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0, "EuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0, "EuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5, "EuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5, "EuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
        "bandwidth": 10.0,
        "ed0.correction.method": "raw",
    }


def _make_minimal_dataset(n=200):
    """LuZ+EdZ only, no position/chl (no shadow correction), not shallow (no bottom)."""
    depth = np.linspace(0.05, 5.0, n)
    waves = np.array(WAVES)
    K = np.array(K_TRUE)
    luz = np.array(LUZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_LUZ)[:, None])
    edz = np.array(EDZ_X0)[None, :] * np.exp(-K[None, :] * (depth + DELTA_EDZ)[:, None])
    ed0 = np.full((n, len(waves)), 100.0)
    zeros = np.zeros(n)
    times = pd.date_range("2019-08-17T22:08:56", periods=n, freq="s")
    return xr.Dataset(
        {
            "Ed0": (("time", "wavelength"), ed0),
            "LuZ": (("time", "wavelength"), luz),
            "EdZ": (("time", "wavelength"), edz),
            "Ed0_Roll": ("time", zeros),
            "Ed0_Pitch": ("time", zeros),
            "EdZ_Roll": ("time", zeros),
            "EdZ_Pitch": ("time", zeros),
            "LuZ_Depth": ("time", depth),
            "LuZ_Temp": ("time", np.full(n, 10.0)),
        },
        coords={"time": times, "wavelength": waves},
    )


def _make_minimal_init():
    nan = float("nan")
    return {
        "depth.is.on": "LuZ",
        "indice.water": 1.34,
        "rau.Fresnel": 0.043,
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 7.0, "LuZ": 7.0},
        "radius.instrument.optics": {"Ed0": 0.035, "EdZ": 0.035, "LuZ": 0.035},
        "delta.capteur.optics": {"Ed0": 0.0, "EdZ": DELTA_EDZ, "LuZ": DELTA_LUZ},
        "sub.surface.removed.layer.optics": {"Ed0": 0.0, "EdZ": 0.3, "LuZ": 0.0},
        "depth.interval.for.smoothing.optics": {"Ed0": 10.0, "EdZ": 3.0, "LuZ": 3.0},
        "linear.fit.Rsquared.threshold.optics": {"Ed0": nan, "EdZ": 0.5, "LuZ": 0.5},
        "linear.fit.max.delta.depth.optics": {"Ed0": nan, "EdZ": 3.0, "LuZ": 2.5},
        "depth.discretization": [0, 0.01, 1, 0.02, 2, 0.05, 5, 0.1, 10, 0.2, 20, 0.5, 50, 1, 100, 2, 200, 5, 500],
        "bandwidth": 10.0,
        "ed0.correction.method": "raw",
    }


def _make_nc(ds, init):
    result = process_cast(ds, init)
    return cast_result_to_dataset(result, ds=ds)


# ---- moved from test_analyze_app.py: these functions now live in pdf_report.py ----


def test_raw_scan_values_applies_delta_capteur_offset():
    """Regression test: the raw depth column is the reference sensor's own depth (depth_is_on),
    not this instrument's true depth -- delta_capteur_optics (the sensor-to-sensor offset already
    applied by cast_fit.py before fitting) must be added, or the raw scatter and the fitted curve
    end up systematically offset from each other (Simon: EdZ's fit sat above its raw points, LuZ's
    below -- exactly what a missing, oppositely-signed offset per instrument would cause)."""
    raw_ds = xr.Dataset(
        {"LuZ": (("time", "wavelength"), np.array([[1.0], [2.0]])), "LuZ_Depth": ("time", np.array([1.0, 2.0]))},
        coords={"wavelength": [340.0]},
    )
    nc = xr.Dataset(coords={"wavelength": [340.0]})  # no ed0_correction -- corrected_values stays None

    values, corrected_values, depth = _raw_scan_values(raw_ds, nc, "LuZ", 0.238, "LuZ", 340.0)
    np.testing.assert_allclose(depth, [1.238, 2.238])
    assert corrected_values is None

    values, corrected_values, depth = _raw_scan_values(raw_ds, nc, "LuZ", None, "LuZ", 340.0)
    np.testing.assert_allclose(depth, [1.0, 2.0])


def test_raw_scan_values_applies_ed0_correction_when_available():
    """{instrument}_fitted is fit from Ed0-corrected values, not the plain raw ones (Simon: the
    fitted curve sat visibly away from the raw points on a cast with unstable Ed0) -- the raw
    overlay should multiply in the same per-scan correction actually applied, so it lines up with
    what the fitted curve was really fit against."""
    raw_ds = xr.Dataset(
        {"LuZ": (("time", "wavelength"), np.array([[1.0], [2.0]])), "LuZ_Depth": ("time", np.array([1.0, 2.0]))},
        coords={"wavelength": [340.0]},
    )
    nc = xr.Dataset(
        {"ed0_correction": (("time", "wavelength"), np.array([[1.5], [0.8]]))},
        coords={"time": np.arange(2), "wavelength": [340.0]},
    )

    values, corrected_values, depth = _raw_scan_values(raw_ds, nc, "LuZ", None, "LuZ", 340.0)

    np.testing.assert_allclose(values, [1.0, 2.0])
    np.testing.assert_allclose(corrected_values, [1.5, 1.6])


def test_kept_mask_filters_out_excluded_scans():
    nc = xr.Dataset(
        {"LuZ_kept": ("time", np.array([1, 0, 1], dtype=np.int8))},
        coords={"time": np.arange(3)},
    )
    raw_ds = xr.Dataset(coords={"time": np.arange(3)})
    raw_depth = np.array([1.0, 2.0, 3.0])

    kept = _kept_mask(nc, raw_ds, "LuZ", raw_depth)

    np.testing.assert_array_equal(kept, [True, False, True])


def test_kept_mask_defaults_to_all_kept_when_var_or_raw_ds_missing():
    raw_depth = np.array([1.0, 2.0])
    nc_without_var = xr.Dataset(coords={"time": np.arange(2)})

    assert _kept_mask(nc_without_var, xr.Dataset(coords={"time": np.arange(2)}), "LuZ", raw_depth).all()
    assert _kept_mask(nc_without_var, None, "LuZ", raw_depth).all()


def test_build_extrapolation_grid_figure_only_plots_kept_scans():
    """Regression test: the small-multiples LOESS-vs-linear grid (PDF report) previously plotted
    every raw scan within a near-surface depth window regardless of whether the fit actually used
    it -- Simon: the extrapolation pages "ne devraient presenter que les points valides qui sont
    utilises par le fit." A scan flagged excluded ({instrument}_kept == False) must not appear as
    a point in the grid, even if it falls within the near-surface display window."""
    n = 5
    depth = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    raw_ds = xr.Dataset(
        {
            "LuZ": (("time", "wavelength"), np.tile(np.array([[10.0]]), (n, 1))),
            "LuZ_Depth": ("time", depth),
        },
        coords={"time": np.arange(n), "wavelength": [340.0]},
    )
    kept = np.array([1, 1, 0, 1, 1], dtype=np.int8)  # scan index 2 excluded
    nc = xr.Dataset(
        {
            "LuZ_fitted": (("LuZ_depth", "wavelength"), np.array([[9.0], [8.0]])),
            "LuZ_surface_value_at_surface": ("wavelength", np.array([10.0])),
            "LuZ_surface_k_surf": ("wavelength", np.array([1.0])),
            "LuZ_surface_z_interval": ("wavelength", np.array([1.0])),
            "LuZ_kept": ("time", kept),
        },
        coords={"time": np.arange(n), "wavelength": [340.0], "LuZ_depth": [0.0, 1.0]},
    )

    fig = build_extrapolation_grid_figure(nc, raw_ds, "LuZ", None, "LuZ")

    scatter_line = fig.axes[0].lines[0]  # the raw-scan scatter is plotted first
    plotted_depths = scatter_line.get_ydata()
    assert 0.3 not in plotted_depths  # the excluded scan's own depth
    assert len(plotted_depths) == n - 1
    plt.close(fig)


def test_mask_negligible_rb_flags_near_zero_denominator_and_nan():
    rb = np.array([1.0, 2.0, 3.0, 4.0])
    rb_extrapolated = np.array([1.1, 2.1, 3.1, 4.1])
    edz_surface = np.array([100.0, 100.0, 100.0, 100.0])
    # wave 0: NaN at bottom (fast-attenuating channel with no valid fitted point that deep);
    # wave 1: 0.5% of surface (below the 1% threshold); wave 2/3: comfortably above it.
    edz_bottom = np.array([np.nan, 0.5, 9.0, 45.0])

    masked_rb, masked_rb_extrap = _mask_negligible_rb(rb, rb_extrapolated, edz_bottom, edz_surface)

    assert np.isnan(masked_rb[0]) and np.isnan(masked_rb_extrap[0])
    assert np.isnan(masked_rb[1]) and np.isnan(masked_rb_extrap[1])
    assert masked_rb[2] == 3.0 and masked_rb[3] == 4.0
    assert masked_rb_extrap[2] == 3.1 and masked_rb_extrap[3] == 4.1
    # the inputs aren't mutated in place
    assert not np.isnan(rb[0])


def test_visible_band_ylim_ignores_nir_fluorescence_spike():
    waves = np.array([443.0, 555.0, 683.0, 780.0])
    rb = np.array([0.05, 0.08, 0.06, 1.4])  # 780 nm: >100%, a fluorescence artifact
    rb_extrapolated = np.array([0.06, 0.09, 0.07, 1.5])

    ylim = _visible_band_ylim(rb, rb_extrapolated, waves)

    assert ylim is not None
    assert ylim < 1.0  # scaled from the <=700 nm bands only, not the 780 nm spike


def test_visible_band_ylim_none_when_nothing_finite():
    waves = np.array([443.0, 555.0])
    nan_array = np.full(2, np.nan)
    assert _visible_band_ylim(nan_array, nan_array, waves) is None


def test_k0_at_adaptive_depth_uses_per_wavelength_z_interval():
    depth_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    k0 = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)
    z_interval = np.array([1.0, 3.0])

    result = _k0_at_adaptive_depth(k0, depth_grid, z_interval)

    np.testing.assert_allclose(result, [2.0, 40.0])


def test_k0_at_adaptive_depth_falls_back_to_2m_when_any_z_interval_nan():
    """Port of plot.Rrs.Kd.for.station.R's all-or-nothing fallback: one invalid linear fit
    forces every wavelength (not just that one) onto the fixed ~2 m depth."""
    depth_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    k0 = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)
    z_interval = np.array([1.0, np.nan])

    result = _k0_at_adaptive_depth(k0, depth_grid, z_interval)

    np.testing.assert_allclose(result, [3.0, 30.0])  # depth nearest 2 m -> index 2, both bands


def test_effective_tiltmax_uses_override_when_present():
    init = {
        "instruments.optics": ("Ed0", "EdZ", "LuZ"),
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 5.0, "LuZ": 5.0},
    }
    info = CastInfo(
        file="x",
        longitude=None,
        latitude=None,
        chl_flag=None,
        time_window=None,
        sub_surface_removed_layer=None,
        tiltmax=[10.0, 2.0, 3.0],
        depth_interval_for_smoothing=None,
        dark_files=[],
    )

    assert _effective_tiltmax(init, info, "EdZ") == 2.0


def test_effective_tiltmax_falls_back_to_init_default_when_no_override():
    init = {
        "instruments.optics": ("Ed0", "EdZ", "LuZ"),
        "tiltmax.optics": {"Ed0": 10.0, "EdZ": 5.0, "LuZ": 5.0},
    }

    assert _effective_tiltmax(init, None, "EdZ") == 5.0


def test_effective_time_window_uses_info_override_when_present():
    init = {"time.window": [0.0, 100.0]}
    info = CastInfo(
        file="x",
        longitude=None,
        latitude=None,
        chl_flag=None,
        time_window=(3.5, 22.5),
        sub_surface_removed_layer=None,
        tiltmax=None,
        depth_interval_for_smoothing=None,
        dark_files=[],
    )

    assert _effective_time_window(init, info) == (3.5, 22.5)


def test_effective_time_window_falls_back_to_init_default_when_no_override():
    init = {"time.window": [0.0, 100.0]}

    assert _effective_time_window(init, None) == (0.0, 100.0)


def test_effective_time_window_none_when_neither_set():
    assert _effective_time_window({}, None) is None


# ---- new: PDF report assembly ----


def test_build_cover_page_includes_cast_file_and_key_init_params():
    ds = _make_minimal_dataset()
    init = _make_minimal_init()
    nc = _make_nc(ds, init)

    fig = build_cover_page(nc, init, None, "MY_CAST_001.csv")

    text = " ".join(t.get_text() for t in fig.axes[0].texts)
    assert "MY_CAST_001.csv" in text
    assert "depth.is.on" in text
    plt.close(fig)


def test_build_cast_report_figures_full_case_has_more_sections_than_minimal():
    full_nc = _make_nc(_make_full_dataset(), _make_full_init())
    minimal_nc = _make_nc(_make_minimal_dataset(), _make_minimal_init())

    full_figures = build_cast_report_figures(
        full_nc, None, _make_full_init(), None, "full.csv", ("EdZ", "LuZ", "EuZ"), None, {}, None
    )
    minimal_figures = build_cast_report_figures(
        minimal_nc, None, _make_minimal_init(), None, "minimal.csv", ("EdZ", "LuZ"), None, {}, None
    )

    # full has EuZ (an extra depth-profile/attenuation/extrapolation-grid page), shadow
    # correction, and bottom reflectance (shallow=True) on top of what minimal has.
    assert len(full_figures) > len(minimal_figures)
    for fig in (*full_figures, *minimal_figures):
        plt.close(fig)


def test_build_cast_report_figures_skips_bottom_reflectance_when_not_shallow():
    ds = _make_full_dataset()
    ds.attrs["shallow"] = False  # otherwise identical to the "full" fixture
    nc = _make_nc(ds, _make_full_init())

    figures = build_cast_report_figures(
        nc, None, _make_full_init(), None, "cast.csv", ("EdZ", "LuZ", "EuZ"), None, {}, None
    )

    titles = [t for fig in figures for t in _figure_titles(fig)]
    assert not any("bottom reflectance" in t for t in titles)
    for fig in figures:
        plt.close(fig)


def _figure_titles(fig) -> list[str]:
    titles = []
    if fig._suptitle is not None:
        titles.append(fig._suptitle.get_text().lower())
    for ax in fig.axes:
        title = ax.get_title()
        if title:
            titles.append(title.lower())
    return titles


def test_write_cast_pdf_report_produces_a_real_multi_page_pdf(tmp_path):
    ds = _make_full_dataset()
    init = _make_full_init()
    nc = _make_nc(ds, init)
    path = tmp_path / "pdf" / "cast.pdf"

    write_cast_pdf_report(nc, None, init, None, "cast.csv", ("EdZ", "LuZ", "EuZ"), None, {}, None, path)

    assert path.exists()
    content = path.read_bytes()
    assert content.startswith(b"%PDF")
    # A crude but dependency-free page count: matplotlib's PdfPages emits one "/Type /Page"
    # object (not "/Pages", the parent tree node) per page.
    page_count = content.count(b"/Type /Page") - content.count(b"/Type /Pages")
    assert page_count > 5  # cover page + Ed0 stability + 3x(depth profile+attenuation) + ...


def test_write_cast_pdf_report_minimal_case_does_not_crash(tmp_path):
    ds = _make_minimal_dataset()
    init = _make_minimal_init()
    nc = _make_nc(ds, init)
    path = tmp_path / "pdf" / "cast.pdf"

    write_cast_pdf_report(nc, None, init, None, "cast.csv", ("EdZ", "LuZ"), None, {}, None, path)

    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


# ---- new: the 4 report-quality improvements Simon asked for after reviewing a real PDF ----


def test_build_ed0_stability_figure_shades_excluded_time_window():
    ds = _make_minimal_dataset()
    nc = _make_nc(ds, _make_minimal_init())

    fig_unshaded = build_ed0_stability_figure(nc)
    fig_shaded = build_ed0_stability_figure(nc, time_window=(20.0, 150.0))

    assert len(fig_unshaded.axes[0].patches) == 0
    # axvspan adds one Polygon patch per shaded region -- one before start, one after end.
    assert len(fig_shaded.axes[0].patches) == 2
    plt.close(fig_unshaded)
    plt.close(fig_shaded)


def test_build_tilt_figures_combined_one_subplot_per_instrument():
    ds = _make_minimal_dataset()  # has Ed0_Roll/Pitch and EdZ_Roll/Pitch, not LuZ's own (falls back to EdZ's)
    init = _make_minimal_init()

    fig = build_tilt_figures_combined(ds, "LuZ", init["delta.capteur.optics"], init, None, ("Ed0", "EdZ", "LuZ"))

    assert fig is not None
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert "Ed0 tilt" in titles
    assert "EdZ tilt" in titles
    assert "LuZ tilt" in titles  # via EdZ's own inclinometer fallback (tilt.py's _TILT_FALLBACK)
    plt.close(fig)


def test_build_tilt_figures_combined_none_when_no_instrument_has_tilt():
    raw_ds = xr.Dataset(
        {"LuZ_Depth": ("time", np.array([1.0, 2.0]))},
        coords={"time": pd.date_range("2020-01-01", periods=2, freq="s")},
    )
    init = {"delta.capteur.optics": {}, "tiltmax.optics": {"LuZ": 5.0}, "instruments.optics": ("LuZ",)}

    fig = build_tilt_figures_combined(raw_ds, "LuZ", {}, init, None, ("LuZ",))

    assert fig is None


def test_build_par_kd_par_figures_labels_include_crossing_depth():
    """Simon's request: label the 50/10/1/0.1% light-level lines with their actual depth, not
    just the bare percentage. A clean exponential-decay profile deep enough to cross all four
    levels (unlike the shared minimal/full fixtures, which may not reach some of them within
    their own shallow depth range) makes this a deterministic check of the annotation itself."""
    depth = np.linspace(0.0, 20.0, 100)
    par_0 = 1000.0
    par_d = par_0 * np.exp(-0.5 * depth)  # ratio crosses 50%/10%/1%/0.1% at ~1.4/4.6/9.2/13.8 m
    nc = xr.Dataset(
        {
            "par_d_profile": ("EdZ_depth", par_d),
            "k0_par": ("EdZ_depth", np.concatenate([[np.nan], np.full(len(depth) - 1, 0.5)])),
        },
        coords={"EdZ_depth": depth},
        attrs={"par_0": par_0},
    )

    fig = build_par_kd_par_figures(nc)[0]

    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("%" in t for t in texts)
    assert any("@" in t and "m" in t for t in texts)
    plt.close(fig)


def test_build_rrs_figure_has_side_by_side_linear_and_log_axes():
    ds = _make_minimal_dataset()
    nc = _make_nc(ds, _make_minimal_init())

    fig = build_rrs_figure(nc)

    assert len(fig.axes) == 2
    scales = {ax.get_yscale() for ax in fig.axes}
    assert scales == {"linear", "log"}
    plt.close(fig)


def test_build_qfactor_figure_none_without_euz():
    ds = _make_minimal_dataset()  # LuZ+EdZ only, no EuZ
    nc = _make_nc(ds, _make_minimal_init())

    assert build_qfactor_figure(nc) is None


def test_build_qfactor_figure_has_loess_and_linear_lines_when_both_present():
    ds = _make_full_dataset()  # EdZ+LuZ+EuZ
    nc = _make_nc(ds, _make_full_init())

    fig = build_qfactor_figure(nc)

    assert fig is not None
    assert len(fig.axes[0].lines) == 2  # one loess line, one linear line
    plt.close(fig)


def test_build_spectral_kd_figure_none_without_kd_vars():
    nc = xr.Dataset(coords={"wavelength": np.array([400.0, 500.0])})

    assert build_spectral_kd_figure(nc) is None


def test_build_spectral_kd_figure_has_side_by_side_linear_and_log_axes_with_three_lines():
    ds = _make_minimal_dataset()  # LuZ+EdZ -- kd_1pct/kd_10pct/kd_pd only need EdZ
    nc = _make_nc(ds, _make_minimal_init())

    fig = build_spectral_kd_figure(nc)

    assert fig is not None
    assert len(fig.axes) == 2
    scales = {ax.get_yscale() for ax in fig.axes}
    assert scales == {"linear", "log"}
    for ax in fig.axes:
        assert len(ax.lines) == 3  # kd_1pct, kd_10pct, kd_pd
    plt.close(fig)


# ---- new: station-wide PAR penetration-depth table + comparison figures ----


def _write_synthetic_station_nc(nc_dir, stem, par_0, decay_rate, n=100, max_depth=20.0):
    """A minimal, deliberately synthetic (not process_cast()-derived) .nc file: just enough
    (par_d_profile, par_0, kd_par_pd, kd_pd) for the station-level PAR/Kd comparison functions,
    with a clean exponential decay guaranteed to cross every _STATION_PAR_FRACTIONS level within
    max_depth -- unlike the shared full/minimal fixtures, whose real fitted PAR profile may not
    reach some of those levels within their own shallow depth range (see the single-cast PAR
    label test above for the same reasoning)."""
    nc_dir.mkdir(parents=True, exist_ok=True)
    depth = np.linspace(0.0, max_depth, n)
    par_d = par_0 * np.exp(-decay_rate * depth)
    ds = xr.Dataset(
        {
            "par_d_profile": ("EdZ_depth", par_d),
            # spectral Kd at the penetration depth (kd_pd), one value per wavelength -- distinct
            # from the broadband par_0/kd_par_pd scalars above.
            "kd_pd": ("wavelength", np.array([decay_rate])),
            # empirical Q-factor (EuZ.0m/LuZ.0m), one value per wavelength -- unrelated to PAR/Kd,
            # just piggy-backing on this same fixture like kd_pd above.
            "q_factor_loess": ("wavelength", np.array([decay_rate])),
            "q_factor_linear": ("wavelength", np.array([decay_rate * 1.1])),
        },
        # "wavelength" is a bare placeholder -- every real .nc always has one (fundamental to the
        # whole pipeline), but build_station_comparison_figures's Rrs/K0 pass reads it
        # unconditionally for every kept file, so this PAR-only fixture needs it too even though
        # it carries no actual Rrs/K0 data (rrs_0p_*/EdZ_K0 stay absent, so that pass just skips
        # this file, same as a real EuZ-only cast with neither instrument would).
        coords={"EdZ_depth": depth, "wavelength": np.array([555.0])},
        attrs={"par_0": par_0, "kd_par_pd": decay_rate},
    )
    ds.to_netcdf(nc_dir / f"{stem}.nc")


def test_build_station_par_depth_table_has_expected_columns_and_values(tmp_path):
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_001", par_0=1000.0, decay_rate=0.5)
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_002", par_0=800.0, decay_rate=1.0)

    table = build_station_par_depth_table(tmp_path)

    assert list(table["cast"]) == ["CAST_001", "CAST_002"]
    assert list(table.columns) == ["cast", "50%", "10%", "5%", "1%", "0.1%"]
    # ratio = exp(-decay_rate * z) = fraction  =>  z = -ln(fraction) / decay_rate
    expected_50pct = -np.log(0.5) / 0.5
    assert abs(table.loc[0, "50%"] - expected_50pct) < 0.1


def test_build_station_par_depth_table_empty_without_nc_dir(tmp_path):
    assert build_station_par_depth_table(tmp_path).empty


def test_build_station_par_profile_figure_overlays_every_kept_cast(tmp_path):
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_001", par_0=1000.0, decay_rate=0.5)
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_002", par_0=800.0, decay_rate=1.0)

    fig = build_station_par_profile_figure(tmp_path)

    assert fig is not None
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_build_station_par_profile_figure_none_without_nc_dir(tmp_path):
    assert build_station_par_profile_figure(tmp_path) is None


def test_build_station_kd_penetration_depth_figure_has_side_by_side_linear_and_log_axes(tmp_path):
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_001", par_0=1000.0, decay_rate=0.5)
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_002", par_0=800.0, decay_rate=1.0)

    fig = build_station_kd_penetration_depth_figure(tmp_path)

    assert fig is not None
    assert len(fig.axes) == 2
    scales = {ax.get_yscale() for ax in fig.axes}
    assert scales == {"linear", "log"}
    for ax in fig.axes:
        assert len(ax.lines) == 2  # one spectral Kd line per cast
    plt.close(fig)


def test_build_station_kd_penetration_depth_figure_none_without_nc_dir(tmp_path):
    assert build_station_kd_penetration_depth_figure(tmp_path) is None


def test_build_station_qfactor_figure_two_lines_per_cast(tmp_path):
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_001", par_0=1000.0, decay_rate=0.5)
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_002", par_0=800.0, decay_rate=1.0)

    fig = build_station_qfactor_figure(tmp_path)

    assert fig is not None
    assert len(fig.axes[0].lines) == 4  # loess (solid) + linear (dashed), per cast
    plt.close(fig)


def test_build_station_qfactor_figure_none_without_nc_dir(tmp_path):
    assert build_station_qfactor_figure(tmp_path) is None


def test_write_station_pdf_reports_writes_one_pdf_per_kept_cast_plus_summary(tmp_path):
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()
    for stem in ("CAST_001", "CAST_002"):
        nc = _make_nc(_make_full_dataset(), _make_full_init())
        nc.to_netcdf(nc_dir / f"{stem}.nc")

    written, failures = write_station_pdf_reports(tmp_path)

    assert written == 2
    assert failures == []
    assert (tmp_path / "pdf" / "CAST_001.pdf").exists()
    assert (tmp_path / "pdf" / "CAST_002.pdf").exists()
    assert (tmp_path / "pdf" / f"{tmp_path.name}_station_summary.pdf").exists()


def test_write_station_pdf_reports_include_station_summary_false_skips_it(tmp_path):
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()
    nc = _make_nc(_make_full_dataset(), _make_full_init())
    nc.to_netcdf(nc_dir / "CAST_001.nc")

    written, failures = write_station_pdf_reports(tmp_path, include_station_summary=False)

    assert written == 1
    assert failures == []
    assert not (tmp_path / "pdf" / f"{tmp_path.name}_station_summary.pdf").exists()


def test_write_station_pdf_reports_isolates_one_bad_cast(tmp_path):
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()
    nc = _make_nc(_make_full_dataset(), _make_full_init())
    nc.to_netcdf(nc_dir / "GOOD_CAST.nc")
    (nc_dir / "BAD_CAST.nc").write_bytes(b"not a real netcdf file")

    written, failures = write_station_pdf_reports(tmp_path, include_station_summary=False)

    assert written == 1
    assert len(failures) == 1
    assert "BAD_CAST" in failures[0]
    assert (tmp_path / "pdf" / "GOOD_CAST.pdf").exists()
    assert not (tmp_path / "pdf" / "BAD_CAST.pdf").exists()


def test_write_station_pdf_reports_calls_progress_callback_per_cast(tmp_path):
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()
    for stem in ("CAST_001", "CAST_002"):
        nc = _make_nc(_make_full_dataset(), _make_full_init())
        nc.to_netcdf(nc_dir / f"{stem}.nc")

    calls = []
    write_station_pdf_reports(
        tmp_path, include_station_summary=False, progress_callback=lambda i, total: calls.append((i, total))
    )

    assert calls == [(1, 2), (2, 2)]


def test_write_station_pdf_reports_none_without_nc_dir(tmp_path):
    written, failures = write_station_pdf_reports(tmp_path)

    assert written == 0
    assert len(failures) == 1
    assert "process this station" in failures[0]


def test_write_station_pdf_reports_flags_empty_station_summary(tmp_path):
    """A station whose only kept cast carries no Rrs/K0/PAR/Kd/Q-factor data still gets a
    (page-less) station_summary.pdf written -- write_station_pdf_reports must surface that as a
    failure/warning rather than silently reporting success, so a researcher isn't left thinking
    the comparison PDF has real content when it's actually empty."""
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()
    xr.Dataset(coords={"wavelength": np.array([555.0])}).to_netcdf(nc_dir / "CAST_001.nc")

    written, failures = write_station_pdf_reports(tmp_path)

    assert written == 1  # the per-cast report (cover page only) still succeeds
    assert any("station summary" in f and "no comparison data" in f for f in failures)


def test_write_station_summary_pdf_includes_par_sections(tmp_path):
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_001", par_0=1000.0, decay_rate=0.5)
    _write_synthetic_station_nc(tmp_path / "nc", "CAST_002", par_0=800.0, decay_rate=1.0)
    path = tmp_path / "pdf" / "station_summary.pdf"

    n_pages = write_station_summary_pdf(tmp_path, path)

    assert path.exists()
    content = path.read_bytes()
    assert content.startswith(b"%PDF")
    page_count = content.count(b"/Type /Page") - content.count(b"/Type /Pages")
    # this fixture has no Rrs/K0 data (no rrs_0p_*/EdZ_K0 vars) -- only the PAR/Kd sections plus
    # the Q-factor comparison (this fixture also carries q_factor_loess/linear, see
    # _write_synthetic_station_nc).
    assert page_count == 4
    assert n_pages == 4  # the function's own returned count matches the real PDF page count
