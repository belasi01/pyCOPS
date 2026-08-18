"""Static PDF processing-report export: the same diagnostics tab 4 (``ui/analyze_app.py``) shows
interactively, assembled into one portable file per cast -- plus a station-wide summary --
matching the shape of Simon's R workflow's own per-cast PDF (``process.cops.R``'s ``pdf(...)``/
``dev.off()`` block driving ``process.Ed0/LuZ/EdZ/EuZ.R``'s and ``compute.aops.R``'s ``plot()``/
``matplot()`` calls, one file per cast in ``dirpdf/``) and its separate station-comparison script
(``plot.Rrs.Kd.for.station.R``).

Every function here builds a ``matplotlib.figure.Figure`` from already-computed ``.nc`` content
(and, for a few sections, the reopened raw cast file) -- no Streamlit dependency, so this lives in
``io/`` (mirrors ``io/netcdf.py``'s role: turning already-computed output into a file format)
rather than ``ui/``. ``ui/analyze_app.py`` imports these back and wraps each one with
``st.pyplot()`` for the interactive tab, so there is exactly one implementation of each plot, not
two.

Deliberately not a literal page-for-page port of R's PDF: reuses pycops's own existing "all
wavelengths overlaid" convention (depth profile, attenuation) as-is rather than R's small-multiples
grids, except for :func:`build_extrapolation_grid_figure` (LOESS vs. linear near the surface),
where per-band detail genuinely matters and pycops had no overlay mode for it yet -- ported from
``process.LuZ.R``'s own 4x5-subplot-grid convention for that one section. Tables (per-wavelength
fit statistics) shown in the interactive tab are not included here -- this first version is
figures only, matching R's own PDF, which is almost entirely plots.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from pycops.io.config import CastInfo, read_info_cops, read_init_cops
from pycops.io.discovery import kept_nc_files
from pycops.io.raw import read_cast
from pycops.io.scaffold import discover_l1_casts
from pycops.processing.attenuation import depth_at_light_fraction
from pycops.processing.color import wavelength_to_rgb
from pycops.processing.depth import time_window_mask
from pycops.processing.par import percent_par_at_depth
from pycops.processing.qwip import _qwip_polynomial
from pycops.processing.tilt import add_tilt

_DEPTH_INSTRUMENTS = ("EdZ", "LuZ", "EuZ")
_SHADOW_INSTRUMENTS = ("LuZ", "EuZ")
_VISIBLE_MAX_NM = 700.0  # visible-band cutoff, matching QWIP's own 400-700 nm convention
_RB_NEGLIGIBLE_EDZ_FRACTION = 0.01  # Simon's own starting suggestion ("e.g. inferieur a 1%?")
_DEFAULT_ED0_CORRECTION_METHOD = "raw"
_PAR_LEVEL_FRACTIONS = (0.5, 0.1, 0.01, 0.001)  # 50% / 10% / 1% / 0.1% of PAR_0
# Kd(PAR) integrated from the surface is noisy/not meaningful in the first few cm (near-surface
# scan sparsity, extrapolation sensitivity) -- Simon: "j'aurais tendance a commencer a partir de
# 0.5 metres" for the integrated Kd(PAR)-vs-depth plot specifically (not the PAR profile itself).
_KD_PAR_MIN_DEPTH_M = 0.5
# Station-wide PAR-depth comparison table -- Simon's own requested fraction set, one more (5%)
# than the single-cast PAR profile page's own dashed-line annotations (_PAR_LEVEL_FRACTIONS).
_STATION_PAR_FRACTIONS = (0.5, 0.1, 0.05, 0.01, 0.001)

_INIT_COVER_KEYS = (
    "instruments.optics",
    "depth.is.on",
    "number.of.fields.before.date",
    "tiltmax.optics",
    "sub.surface.removed.layer.optics",
    "depth.interval.for.smoothing.optics",
    "linear.fit.Rsquared.threshold.optics",
    "linear.fit.max.delta.depth.optics",
    "bandwidth",
    "windspeed_ms",
    "ed0.correction.method",
)
_INFO_COVER_FIELDS = (
    ("sub_surface_removed_layer", "sub.surface.removed.layer override"),
    ("tiltmax", "tiltmax override"),
    ("depth_interval_for_smoothing", "depth.interval.for.smoothing override"),
    ("linear_r2_threshold", "linear.fit.Rsquared.threshold override"),
    ("linear_max_delta_depth", "linear.fit.max.delta.depth override"),
)


def _instruments_present(nc: xr.Dataset) -> tuple[str, ...]:
    return tuple(instr for instr in _DEPTH_INSTRUMENTS if f"{instr}_fitted" in nc.data_vars)


def _wavelength_dim(ds: xr.Dataset, instrument: str) -> str:
    return "wavelength" if "wavelength" in ds[instrument].dims else f"wavelength_{instrument}"


def _wavelength_colors(waves: np.ndarray) -> np.ndarray:
    return plt.cm.viridis(np.linspace(0, 1, len(waves)))


def _new_fig(figsize: tuple[float, float] = (9, 4.5)):
    return plt.subplots(figsize=figsize)


def _raw_scan_values(
    raw_ds: xr.Dataset | None,
    nc: xr.Dataset,
    depth_is_on: str | None,
    delta_capteur: float | None,
    instrument: str,
    w: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray] | None:
    """Raw per-scan (value, Ed0-corrected value or ``None``, depth) for ``instrument`` at the
    wavelength nearest ``w`` -- see ``ui/analyze_app.py``'s own copy of this function (before this
    module existed) for the full rationale; identical body, moved here so both the interactive tab
    and the PDF report share one implementation.
    """
    if raw_ds is None or depth_is_on is None:
        return None
    if instrument not in raw_ds or f"{depth_is_on}_Depth" not in raw_ds:
        return None
    wdim = _wavelength_dim(raw_ds, instrument)
    raw_waves = raw_ds[wdim].values
    nearest_raw_wave = raw_waves[int(np.argmin(np.abs(raw_waves - w)))]
    values = raw_ds[instrument].sel({wdim: nearest_raw_wave}).values
    depth = raw_ds[f"{depth_is_on}_Depth"].values
    if delta_capteur is not None and np.isfinite(delta_capteur):
        depth = depth + delta_capteur

    corrected_values = None
    if "ed0_correction" in nc.data_vars and nc.sizes["time"] == raw_ds.sizes["time"]:
        nc_waves = nc["wavelength"].values
        wi = int(np.argmin(np.abs(nc_waves - w)))
        correction = nc["ed0_correction"].isel(wavelength=wi).values
        corrected_values = values * correction

    return values, corrected_values, depth


def _kept_mask(nc: xr.Dataset, raw_ds: xr.Dataset | None, instrument: str, raw_depth: np.ndarray) -> np.ndarray:
    """Boolean mask, aligned with a raw-scan array (e.g. :func:`_raw_scan_values`'s own return),
    of scans ``instrument``'s fit actually used (``{instrument}_kept``, written by
    :func:`pycops.io.netcdf.cast_result_to_dataset`) -- everything unmasked ("kept") when that
    var or the raw dataset itself is unavailable, or their scan counts don't match (e.g. an
    older ``.nc`` file written before ``_kept`` existed).
    """
    kept_var = f"{instrument}_kept"
    if kept_var in nc.data_vars and raw_ds is not None and nc.sizes["time"] == raw_ds.sizes["time"]:
        return nc[kept_var].values.astype(bool)
    return np.ones(raw_depth.shape, dtype=bool)


def _effective_time_window(
    init: dict[str, object], info: CastInfo | None
) -> tuple[float, float] | None:
    """The ``time.window`` actually used by processing -- see ``analyze_app.py``'s copy for the
    full rationale."""
    if info is not None and info.time_window is not None:
        return info.time_window
    time_window = init.get("time.window")
    return tuple(time_window) if time_window is not None else None


def _effective_tiltmax(init: dict[str, object], info: CastInfo | None, instrument: str) -> float:
    """``tiltmax.optics`` for ``instrument`` -- see ``analyze_app.py``'s copy for the full
    rationale."""
    if info is not None and info.tiltmax is not None:
        instruments_list = list(init["instruments.optics"])
        if instrument in instruments_list:
            idx = instruments_list.index(instrument)
            if idx < len(info.tiltmax):
                return info.tiltmax[idx]
    return init["tiltmax.optics"][instrument]


def _mask_negligible_rb(
    rb: np.ndarray,
    rb_extrapolated: np.ndarray,
    edz_at_bottom: np.ndarray,
    edz_at_surface: np.ndarray,
    threshold: float = _RB_NEGLIGIBLE_EDZ_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """NaN out wavelengths where EdZ at the bottom is negligible relative to the surface -- see
    ``analyze_app.py``'s copy for the full rationale."""
    rb = rb.copy()
    rb_extrapolated = rb_extrapolated.copy()
    with np.errstate(invalid="ignore"):
        negligible = ~(edz_at_bottom > threshold * edz_at_surface)
    rb[negligible] = np.nan
    rb_extrapolated[negligible] = np.nan
    return rb, rb_extrapolated


def _visible_band_ylim(rb: np.ndarray, rb_extrapolated: np.ndarray, waves: np.ndarray) -> float | None:
    """Y-axis max from visible-band (<=700 nm) values only -- see ``analyze_app.py``'s copy for
    the full rationale."""
    visible = waves <= _VISIBLE_MAX_NM
    values = np.concatenate([rb[visible], rb_extrapolated[visible]])
    values = values[np.isfinite(values)]
    return float(np.max(values)) * 1.15 if len(values) else None


def _k0_at_adaptive_depth(k0: np.ndarray, depth_grid: np.ndarray, z_interval: np.ndarray) -> np.ndarray:
    """K0(EdZ) sampled at each wavelength's own near-surface linear-fit ``z_interval`` depth --
    see ``analyze_app.py``'s copy for the full rationale."""
    if np.any(np.isnan(z_interval)):
        ix = int(np.argmin(np.abs(depth_grid - 2.0)))
        return k0[ix, :]
    values = np.empty(len(z_interval))
    for w in range(len(z_interval)):
        ix = int(np.argmin(np.abs(depth_grid - z_interval[w])))
        values[w] = k0[ix, w]
    return values


def build_cover_page(
    nc: xr.Dataset,
    init: dict[str, object] | None,
    info: CastInfo | None,
    cast_file: str,
) -> Figure:
    """Direct port of ``plot.init.info.R``: a blank-axes text page listing key ``init.cops.dat``
    parameters (left column) and this cast's own info (right column)."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    left_lines = [f"Cast: {cast_file}", ""]
    if init is not None:
        for key in _INIT_COVER_KEYS:
            if key in init:
                left_lines.append(f"{key}: {init[key]}")

    lon, lat = nc.attrs.get("longitude"), nc.attrs.get("latitude")
    right_lines = [
        f"longitude: {lon if lon is not None and not np.isnan(lon) else 'NA'}",
        f"latitude: {lat if lat is not None and not np.isnan(lat) else 'NA'}",
        f"chl flag: {nc.attrs.get('chl_flag', 'NA')}",
        f"QC flag: {nc.attrs.get('qc_flag', '-')}",
        f"Rrs method: {nc.attrs.get('rrs_method') or '-'}",
        f"Rrs source: {nc.attrs.get('rrs_source') or '-'}",
        f"shallow: {'yes' if nc.attrs.get('shallow') else 'no'}",
        f"Ed0 correction method: {nc.attrs.get('ed0_correction_method', '-')}",
        f"excluded wavelengths: {nc.attrs.get('excluded_wavelengths') or '-'}",
    ]
    time_window = _effective_time_window(init, info) if init is not None else None
    if time_window is not None:
        right_lines.append(f"time.window: {time_window[0]:.2f} - {time_window[1]:.2f} s")
    if info is not None:
        for attr, label in _INFO_COVER_FIELDS:
            value = getattr(info, attr)
            if value is not None:
                right_lines.append(f"{label}: {value}")

    y = 0.97
    for line in left_lines:
        ax.text(0.02, y, line, transform=ax.transAxes, fontsize=8, va="top")
        y -= 0.03
    y = 0.97
    for line in right_lines:
        ax.text(0.55, y, line, transform=ax.transAxes, fontsize=8, va="top")
        y -= 0.03

    fig.suptitle("pycops processing report")
    return fig


def build_ed0_stability_figure(
    nc: xr.Dataset, time_window: tuple[float, float] | None = None
) -> Figure | None:
    if "ed0_correction" not in nc.data_vars:
        return None
    active_method = nc.attrs.get("ed0_correction_method") or _DEFAULT_ED0_CORRECTION_METHOD

    fig, ax = _new_fig((9, 3))
    time_values = nc["time"].values
    if time_window is not None:
        # time_window is elapsed seconds from the cast's first scan (see time_window_mask) --
        # convert back to the same real-time axis this plot's x-axis uses, matching
        # build_depth_vs_time_figure's own gray-shading convention for the excluded regions.
        start, end = time_window
        t0 = time_values.min()
        start_dt = t0 + np.timedelta64(int(round(start * 1000)), "ms")
        end_dt = t0 + np.timedelta64(int(round(end * 1000)), "ms")
        ax.axvspan(time_values.min(), start_dt, color="gray", alpha=0.3)
        ax.axvspan(end_dt, time_values.max(), color="gray", alpha=0.3)

    all_finite = []
    if "ed0_correction_raw" in nc.data_vars:
        raw = nc["ed0_correction_raw"].mean(dim="wavelength").values
        all_finite.append(raw)
        ax.plot(
            nc["time"].values,
            raw,
            color="tab:orange",
            lw=2.5 if active_method == "raw" else 1,
            ls="-" if active_method == "raw" else "--",
            label="Raw" + (" (active)" if active_method == "raw" else ""),
        )
    if "ed0_correction_smoothed" in nc.data_vars:
        smoothed = nc["ed0_correction_smoothed"].mean(dim="wavelength").values
        all_finite.append(smoothed)
        ax.plot(
            nc["time"].values,
            smoothed,
            color="tab:blue",
            lw=2.5 if active_method == "smoothed" else 1,
            ls="-" if active_method == "smoothed" else "--",
            label="Smoothed" + (" (active)" if active_method == "smoothed" else ""),
        )
    if not all_finite:
        correction = nc["ed0_correction"].mean(dim="wavelength").values
        all_finite.append(correction)
        ax.plot(nc["time"].values, correction, color="tab:orange", label="Active")

    ax.axhline(1.05, color="red", ls="--", lw=1)
    ax.axhline(0.95, color="red", ls="--", lw=1)
    combined = np.concatenate(all_finite)
    ax.set_ylim(min(0.9, float(np.nanmin(combined))), max(1.1, float(np.nanmax(combined))))
    ax.set_xlabel("Time")
    ax.set_ylabel("Ed0 correction (mean over wavelengths)")
    ax.set_title("Ed0 stability")
    ax.legend(loc="best", fontsize="small")
    fig.autofmt_xdate()

    active_var = f"ed0_correction_{active_method}"
    active_correction = nc[active_var] if active_var in nc.data_vars else nc["ed0_correction"]
    active_values = active_correction.mean(dim="wavelength").values
    outside = int(np.sum((active_values < 0.95) | (active_values > 1.05)))
    if outside:
        fig.text(
            0.5, 0.01, f"Outside +/-5% for {outside} scan(s) -- possible illumination instability.",
            ha="center", color="red", fontsize=8,
        )
    return fig


def build_depth_vs_time_figure(
    raw_ds: xr.Dataset, depth_is_on: str, time_window: tuple[float, float] | None
) -> Figure:
    elapsed = (raw_ds["time"].values - raw_ds["time"].values.min()) / np.timedelta64(1, "s")
    depth = raw_ds[f"{depth_is_on}_Depth"].values

    fig, ax = _new_fig((9, 3.2))
    ax.plot(elapsed, depth, ".", markersize=2, color="tab:blue")
    if time_window is not None:
        start, end = time_window
        ax.axvspan(0, start, color="gray", alpha=0.3)
        ax.axvspan(end, float(elapsed.max()), color="gray", alpha=0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(f"{depth_is_on} depth (m)")
    ax.set_title(f"{depth_is_on} depth vs time")
    return fig


def _plot_tilt_on_ax(
    ax,
    raw_ds: xr.Dataset,
    depth_is_on: str,
    delta_capteur: float | None,
    init: dict[str, object],
    info: CastInfo | None,
    instrument: str,
    time_window: tuple[float, float] | None,
) -> bool:
    """Draw one instrument's tilt-vs-depth onto ``ax`` (shared by :func:`build_tilt_figure` --
    one instrument, its own standalone figure, for the interactive tab -- and
    :func:`build_tilt_figures_combined` -- every instrument as subplots of one figure, for the
    PDF report). Returns ``False`` (and leaves ``ax`` untouched) if ``instrument`` has no usable
    Roll/Pitch to derive tilt from."""
    try:
        tilt = add_tilt(raw_ds, instrument)[f"{instrument}_Tilt"].values
    except KeyError:
        return False

    depth = raw_ds[f"{depth_is_on}_Depth"].values
    if delta_capteur is not None and np.isfinite(delta_capteur):
        depth = depth + delta_capteur
    tiltmax = _effective_tiltmax(init, info, instrument)

    if time_window is not None:
        in_window = time_window_mask(raw_ds["time"].values, time_window)
    else:
        in_window = np.ones(tilt.shape, dtype=bool)
    within = in_window & (tilt < tiltmax)
    exceeds = in_window & ~(tilt < tiltmax)
    excluded = ~in_window

    ax.plot(tilt[excluded], depth[excluded], ".", markersize=3, color="gray", label="excluded by time.window")
    ax.plot(tilt[within], depth[within], ".", markersize=3, color="tab:blue", label="within limit")
    ax.plot(tilt[exceeds], depth[exceeds], ".", markersize=4, color="red", label="exceeds limit")
    if np.isfinite(tiltmax):
        ax.axvline(tiltmax, color="red", ls="--", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel(f"{instrument} tilt (degrees)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{instrument} tilt")
    return True


def build_tilt_figure(
    raw_ds: xr.Dataset,
    depth_is_on: str,
    delta_capteur: float | None,
    init: dict[str, object],
    info: CastInfo | None,
    instrument: str,
    time_window: tuple[float, float] | None = None,
) -> Figure | None:
    fig, ax = _new_fig((9, 4))
    if not _plot_tilt_on_ax(ax, raw_ds, depth_is_on, delta_capteur, init, info, instrument, time_window):
        plt.close(fig)
        return None
    ax.legend(fontsize="small")
    return fig


def build_tilt_figures_combined(
    raw_ds: xr.Dataset,
    depth_is_on: str,
    delta_capteur_optics: dict[str, float],
    init: dict[str, object],
    info: CastInfo | None,
    instruments: tuple[str, ...],
    time_window: tuple[float, float] | None = None,
) -> Figure | None:
    """Every instrument's tilt-vs-depth as subplots of *one* figure (Simon: gather the tilt pages
    onto a single PDF page rather than one per instrument) -- same per-instrument plotting as
    :func:`build_tilt_figure`, just laid out as a grid instead of separate figures. Returns
    ``None`` if not one of ``instruments`` has usable Roll/Pitch."""
    n = len(instruments)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    any_ok = False
    for ax, instrument in zip(axes_flat, instruments):
        if _plot_tilt_on_ax(ax, raw_ds, depth_is_on, delta_capteur_optics.get(instrument), init, info, instrument, time_window):
            any_ok = True
            ax.legend(fontsize="small")
        else:
            ax.axis("off")
    for ax in axes_flat[n:]:
        ax.axis("off")

    if not any_ok:
        plt.close(fig)
        return None
    fig.suptitle("Tilt vs depth")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def build_depth_profile_figure(
    nc: xr.Dataset,
    raw_ds: xr.Dataset | None,
    depth_is_on: str | None,
    delta_capteur: float | None,
    instrument: str,
    wavelength_choice: str = "All",
) -> Figure:
    depth_dim = f"{instrument}_depth"
    waves = nc["wavelength"].values
    fitted = nc[f"{instrument}_fitted"]
    depth = nc[depth_dim].values

    fig, ax = _new_fig()
    if wavelength_choice == "All":
        colors = _wavelength_colors(waves)
        for i, w in enumerate(waves):
            ax.plot(fitted.isel(wavelength=i).values, depth, color=colors[i], lw=1.5)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(waves.min(), waves.max()))
        fig.colorbar(sm, ax=ax, label="Wavelength (nm)")
    else:
        w = float(wavelength_choice)
        wi = int(np.argmin(np.abs(waves - w)))
        raw = _raw_scan_values(raw_ds, nc, depth_is_on, delta_capteur, instrument, w)
        if raw is not None:
            raw_values, corrected_values, raw_depth = raw
            kept = _kept_mask(nc, raw_ds, instrument, raw_depth)
            ax.plot(
                raw_values, raw_depth, ".", markersize=3, color="navajowhite", alpha=0.7,
                label="raw (Ed0-uncorrected)",
            )
            plot_values = corrected_values if corrected_values is not None else raw_values
            kept_label = "kept scans (Ed0-corrected)" if corrected_values is not None else "kept scans"
            excluded_label = "excluded scans (Ed0-corrected)" if corrected_values is not None else "excluded scans"
            ax.plot(plot_values[kept], raw_depth[kept], ".", markersize=3, color="tab:blue", label=kept_label)
            ax.plot(plot_values[~kept], raw_depth[~kept], ".", markersize=3, color="lightgray", label=excluded_label)
        ax.plot(fitted.isel(wavelength=wi).values, depth, color="tab:red", lw=2, label="fitted")
        detection_limit = nc[f"{instrument}_detection_limit"].values[wi]
        if np.isfinite(detection_limit):
            ax.axvline(detection_limit, color="black", ls="--", lw=1, label="detection limit")
        ax.legend(loc="best", fontsize="small")

    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel(f"{instrument} (log scale)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{instrument} depth profile")
    return fig


def build_attenuation_figure(nc: xr.Dataset, instrument: str) -> Figure:
    depth_dim = f"{instrument}_depth"
    waves = nc["wavelength"].values
    depth = nc[depth_dim].values
    colors = _wavelength_colors(waves)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, w in enumerate(waves):
        ax1.plot(nc[f"{instrument}_KZ"].isel(wavelength=i).values, depth, color=colors[i], lw=1.5)
        ax2.plot(nc[f"{instrument}_K0"].isel(wavelength=i).values, depth, color=colors[i], lw=1.5)
    for ax, title in ((ax1, "KZ (local)"), (ax2, "K0 (depth-integrated)")):
        ax.invert_yaxis()
        ax.set_xlabel(f"{title} (m⁻¹)")
        ax.set_ylabel("Depth (m)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(waves.min(), waves.max()))
    fig.colorbar(sm, ax=(ax1, ax2), label="Wavelength (nm)")
    fig.suptitle(f"{instrument} attenuation")
    return fig


def build_spectral_kd_figure(nc: xr.Dataset) -> Figure | None:
    """Spectral Kd for this cast -- mean diffuse attenuation from the surface down to the 1%,
    10%, and penetration-depth (1/e) light levels (``kd_1pct``/``kd_10pct``/``kd_pd``, see
    :mod:`pycops.processing.attenuation`), one line per light level vs. wavelength, in the same
    side-by-side linear/log layout as Rrs (:func:`build_rrs_figure`) -- the single-cast
    counterpart of the station-wide comparison in
    :func:`build_station_kd_penetration_depth_figure`. Distinct from
    :func:`build_attenuation_figure`, which plots Kd vs. *depth* (colored by wavelength) rather
    than vs. wavelength. ``None`` when none of the three are present (no EdZ on this cast).
    """
    available = [
        (label, var)
        for label, var in (("1%", "kd_1pct"), ("10%", "kd_10pct"), ("penetration depth", "kd_pd"))
        if var in nc.data_vars
    ]
    if not available:
        return None
    waves = nc["wavelength"].values
    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax in (ax_linear, ax_log):
        for label, var in available:
            ax.plot(waves, nc[var].values, "-o", label=label, markersize=4)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Kd (m⁻¹)")
    ax_log.set_yscale("log")
    ax_linear.set_title("Spectral Kd, linear")
    ax_log.set_title("Spectral Kd, log")
    ax_linear.legend(fontsize="small")
    return fig


def build_penetration_depth_figure(nc: xr.Dataset) -> Figure | None:
    """Penetration depth (``pd_depth``, meters -- the 1/e-light-level crossing depth itself, not
    ``kd_pd``'s derived attenuation coefficient) vs. wavelength, depth 0 at the top. Simon's
    request: markers colored by each band's own approximate true display color
    (:func:`pycops.processing.color.wavelength_to_rgb`) rather than an arbitrary colormap, to
    directly visualize "what a satellite sees" -- how deep each visible color actually
    penetrates before being reduced to 1/e of its surface value.

    Deliberately plotted over the *full* wavelength grid (not pre-filtered to finite values) --
    a band that never reaches the penetration-depth light level within the cast's own measured
    range is a real gap, not something to silently connect across (see
    :func:`pycops.io.database_report._plot_mean_sd_band` for the same reasoning, found from a
    real report of exactly this kind of fabricated connecting line).
    """
    if "pd_depth" not in nc.data_vars:
        return None
    waves = nc["wavelength"].values
    depth = nc["pd_depth"].values
    if not np.isfinite(depth).any():
        return None

    fig, ax = _new_fig((8, 5))
    ax.plot(waves, depth, "-", color="lightgray", lw=1, zorder=1)
    ax.scatter(waves, depth, c=wavelength_to_rgb(waves), s=70, edgecolors="black", linewidths=0.5, zorder=2)
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Penetration depth (m)")
    ax.set_title("Penetration depth (1/e light level) by wavelength")
    return fig


def build_penetration_depth_comparison_figure(directory: Path) -> Figure | None:
    """:func:`build_penetration_depth_figure`, overlaid for every currently-kept cast in a
    station. Color is already used for wavelength (the whole point of the figure), so casts are
    distinguished by line style instead (solid/dashed/dash-dot/dotted, cycling), with the
    per-wavelength true-color markers layered on top of every cast's line."""
    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        return None
    kept_files = kept_nc_files(directory, nc_dir)
    applicable = []
    for nc_path in kept_files:
        with xr.open_dataset(nc_path) as opened:
            nc = opened.load()
        if "pd_depth" in nc.data_vars and np.isfinite(nc["pd_depth"].values).any():
            applicable.append((nc_path.stem, nc["wavelength"].values, nc["pd_depth"].values))
    if not applicable:
        return None

    linestyles = ("-", "--", "-.", ":")
    fig, ax = _new_fig((9, 5.5))
    for i, (label, waves, depth) in enumerate(applicable):
        ax.plot(waves, depth, linestyles[i % len(linestyles)], color="black", lw=1.2, label=label, zorder=1)
        ax.scatter(waves, depth, c=wavelength_to_rgb(waves), s=50, edgecolors="black", linewidths=0.4, zorder=2)
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Penetration depth (m)")
    ax.set_title("Penetration depth (1/e light level) by wavelength, by cast")
    ax.legend(fontsize="small")
    return fig


def build_par_kd_par_figures(nc: xr.Dataset) -> list[Figure]:
    depth = nc["EdZ_depth"].values
    par_d = nc["par_d_profile"].values
    par_0 = float(nc.attrs["par_0"])

    fig1, ax = _new_fig((7, 5))
    ax.plot(par_d, depth, color="tab:blue", lw=2, label="PAR_d (downwelling)")
    ax.axvline(par_0, color="gray", ls="--", lw=1.5, label="PAR_0 (surface reference)")
    for fraction in _PAR_LEVEL_FRACTIONS:
        # Treat PAR as a single "band" -- same reuse of the per-wavelength crossing-depth search
        # already used for kz_par/k0_par/kd_par_* elsewhere in this port (compute_K/
        # kd_at_light_fraction called with a (n_depth, 1)-shaped profile).
        z = depth_at_light_fraction(par_d[:, None], depth, np.array([par_0]), fraction)[0]
        label = f"{fraction * 100:g}%" if np.isnan(z) else f"{fraction * 100:g}% @ {z:.2f} m"
        ax.axvline(par_0 * fraction, color="gray", ls=":", lw=1)
        ax.text(
            par_0 * fraction, depth.max(), label, color="gray", fontsize="small",
            ha="center", va="bottom", rotation=90,
        )
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("PAR (µEin.m⁻².s⁻¹, log scale)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Vertical PAR profile")
    ax.legend(loc="best", fontsize="small")
    figures = [fig1]

    k0_depth = depth[1:]  # k0_par[0] is a leading-NaN pad, matching K0's own depth_grid[1:] alignment
    k0_values = nc["k0_par"].values[1:]
    deep_enough = k0_depth >= _KD_PAR_MIN_DEPTH_M
    k0_depth, k0_values = k0_depth[deep_enough], k0_values[deep_enough]
    if len(k0_depth) > 1:
        fig2, ax2 = _new_fig((7, 5))
        ax2.plot(k0_values, k0_depth, color="tab:blue", lw=2)
        ax2.invert_yaxis()
        ax2.set_xlabel("Kd(PAR) integrated from the surface to Z (m⁻¹)")
        ax2.set_ylabel("Depth Z (m)")
        ax2.set_title("Kd(PAR) vs depth")
        figures.append(fig2)

    return figures


def build_extrapolation_grid_figure(
    nc: xr.Dataset,
    raw_ds: xr.Dataset | None,
    depth_is_on: str | None,
    delta_capteur: float | None,
    instrument: str,
) -> Figure:
    """Small-multiples grid (one subplot per wavelength): LOESS vs. linear near-surface
    extrapolation, matching ``process.LuZ.R``'s own 4x5-subplot-grid convention for this
    diagnostic -- PDF-only, the interactive tab keeps its own single-wavelength dropdown
    (:func:`ui.analyze_app._render_extrapolation_comparison`) unchanged.

    The linear curve is reconstructed from already-stored fit parameters, not refit here, exactly
    like the interactive tab's own version: ``surface_linear.py``'s log-linear regression is
    anchored at the true surface (``z=0``), so ``value(z) = value_at_surface * exp(-k_surf * z)``
    over ``z`` in ``[0, z_interval]``.
    """
    waves = nc["wavelength"].values
    depth = nc[f"{instrument}_depth"].values
    fitted = nc[f"{instrument}_fitted"]
    value_at_surface = nc[f"{instrument}_surface_value_at_surface"].values
    k_surf = nc[f"{instrument}_surface_k_surf"].values
    z_interval = nc[f"{instrument}_surface_z_interval"].values

    n = len(waves)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.6 * nrows), squeeze=False)
    axes = axes.flatten()

    for wi, w in enumerate(waves):
        ax = axes[wi]
        raw = _raw_scan_values(raw_ds, nc, depth_is_on, delta_capteur, instrument, float(w))
        if raw is not None:
            raw_values, corrected_values, raw_depth = raw
            kept = _kept_mask(nc, raw_ds, instrument, raw_depth)
            near_surface = (
                raw_depth <= max(z_interval[wi] * 1.5, 1.0)
                if np.isfinite(z_interval[wi])
                else np.ones_like(raw_depth, dtype=bool)
            )
            show = kept & near_surface
            plot_values = corrected_values if corrected_values is not None else raw_values
            ax.plot(plot_values[show], raw_depth[show], ".", markersize=3, color="tab:blue")
        ax.plot(fitted.isel(wavelength=wi).values, depth, color="tab:red", lw=1.5)
        if np.isfinite(value_at_surface[wi]) and np.isfinite(z_interval[wi]):
            z_line = np.linspace(0, z_interval[wi], 30)
            linear_curve = value_at_surface[wi] * np.exp(-k_surf[wi] * z_line)
            ax.plot(linear_curve, z_line, color="tab:green", lw=1.5, ls="--")
        ax.set_xscale("log")
        ax.invert_yaxis()
        if np.isfinite(z_interval[wi]):
            ax.set_ylim(max(z_interval[wi] * 1.5, 1.0), 0)
        ax.set_title(f"{w:g} nm", fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{instrument}: LOESS (red) vs. linear (green) extrapolation, per wavelength")
    handles = [
        plt.Line2D([], [], color="tab:blue", marker=".", linestyle="", label="raw/corrected scans"),
        plt.Line2D([], [], color="tab:red", label="LOESS"),
        plt.Line2D([], [], color="tab:green", ls="--", label="linear"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def build_rrs_figure(nc: xr.Dataset) -> Figure | None:
    if "rrs_0p_loess" not in nc.data_vars and "rrs_0p_linear" not in nc.data_vars:
        return None
    waves = nc["wavelength"].values
    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax in (ax_linear, ax_log):
        for method, style in (("loess", "-o"), ("linear", "--s"), ("recommended", ":^")):
            var = f"rrs_0p_{method}"
            if var in nc.data_vars:
                ax.plot(waves, nc[var].values, style, label=method, markersize=4)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Rrs(0+)")
    ax_log.set_yscale("log")
    ax_linear.set_title("Rrs (linear)")
    ax_log.set_title("Rrs (log)")
    ax_linear.legend(fontsize="small")
    fig.suptitle("Rrs")
    return fig


def build_qfactor_figure(nc: xr.Dataset) -> Figure | None:
    """Empirical Q-factor (``EuZ.0m / LuZ.0m``), one point per wavelength -- only present in ``nc``
    when the cast carried both LuZ and EuZ sensors (see
    :func:`pycops.processing.process_cast._empirical_q_factor`). A rarer diagnostic than Rrs/K0:
    Simon requested it specifically because it's "rarement rapporte dans la litterature."
    """
    if "q_factor_loess" not in nc.data_vars and "q_factor_linear" not in nc.data_vars:
        return None
    waves = nc["wavelength"].values
    fig, ax = _new_fig((7, 4.5))
    for method, style in (("loess", "-o"), ("linear", "--s")):
        var = f"q_factor_{method}"
        if var in nc.data_vars:
            ax.plot(waves, nc[var].values, style, label=method, markersize=4)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Q-factor (Eu(0-) / Lu(0-), sr)")
    ax.set_title("Empirical Q-factor (EuZ.0m / LuZ.0m)")
    ax.legend(fontsize="small")
    return fig


def build_shadow_correction_figure(nc: xr.Dataset, instrument: str) -> Figure:
    waves = nc["wavelength"].values
    fig, ax = _new_fig((9, 3.5))
    ax.plot(waves, nc[f"{instrument}_shadow_correction"].values, "-o", color="tab:purple")
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Shadow correction ({instrument})")
    source = nc.attrs.get(f"{instrument}_absorption_source")
    ax.set_title(f"{instrument} shadow correction" + (f" (absorption source: {source})" if source else ""))
    return fig


def build_qwip_figure(nc: xr.Dataset) -> Figure | None:
    labels = [label for label in ("loess", "linear") if f"qwip_{label}_avw" in nc.attrs]
    if not labels:
        return None
    avw_range = np.linspace(400, 600, 200)
    predicted = _qwip_polynomial(avw_range)
    fig, ax = _new_fig((7, 4))
    ax.plot(avw_range, predicted, color="black", label="QWIP reference")
    ax.fill_between(avw_range, predicted - 0.1, predicted + 0.1, color="gray", alpha=0.2)
    for label, marker in zip(labels, ("o", "s")):
        avw = nc.attrs[f"qwip_{label}_avw"]
        ndi = nc.attrs[f"qwip_{label}_ndi"]
        score = nc.attrs[f"qwip_{label}_score"]
        ax.plot(avw, ndi, marker, markersize=10, label=f"{label} (score {score:.3f})")
    ax.set_xlabel("AVW (nm)")
    ax.set_ylabel("NDI")
    ax.set_title("QWIP / water class")
    ax.legend(fontsize="small")
    return fig


def build_bottom_figure(nc: xr.Dataset, instrument: str) -> Figure:
    waves = nc["wavelength"].values
    rb = nc[f"{instrument}_rb"].values
    rb_extrapolated = nc[f"{instrument}_rb_extrapolated"].values
    bottom_depth = nc.attrs.get(f"{instrument}_bottom_depth")

    if bottom_depth is not None and "EdZ_fitted" in nc.data_vars:
        edz_at_bottom = np.array(
            [
                np.interp(bottom_depth, nc["EdZ_depth"].values, nc["EdZ_fitted"].isel(wavelength=i).values)
                for i in range(len(waves))
            ]
        )
        rb, rb_extrapolated = _mask_negligible_rb(rb, rb_extrapolated, edz_at_bottom, nc["EdZ_value_at_0"].values)

    fig, ax = _new_fig((9, 3.5))
    ax.plot(waves, rb, "-o", label="Rb (~0.3 m above bottom)")
    ax.plot(waves, rb_extrapolated, "--s", label="Rb (extrapolated to bottom)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Bottom reflectance ({instrument})")
    ax.legend(fontsize="small")
    title = f"{instrument} bottom reflectance"
    if bottom_depth is not None:
        title += f" -- bottom depth: {bottom_depth:.2f} m"
    ax.set_title(title)

    ylim = _visible_band_ylim(rb, rb_extrapolated, waves)
    if ylim is not None:
        ax.set_ylim(0, ylim)

    if bottom_depth is not None and "EdZ_fitted" in nc.data_vars:
        pct_par = percent_par_at_depth(waves, nc["EdZ_fitted"].values, nc["EdZ_depth"].values, bottom_depth)
        if pct_par is not None:
            fig.text(0.5, 0.01, f"Benthic PAR available: {pct_par:.2f}% of surface", ha="center", fontsize=8)
    return fig


def build_station_comparison_figures(directory: Path) -> tuple[Figure | None, Figure | None]:
    """Every currently-kept cast in ``directory`` (``select.cops.dat`` flag != 0), overlaid --
    port of ``plot.Rrs.Kd.for.station.R``, matching ``ui/analyze_app.py``'s own interactive
    "Station comparison" mode exactly. Returns ``(rrs_figure, k0_figure)``, either ``None`` if no
    kept cast has that data."""
    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        return None, None
    kept_files = kept_nc_files(directory, nc_dir)
    if not kept_files:
        return None, None

    colors = plt.cm.tab20(np.linspace(0, 1, len(kept_files)))
    fig_rrs, (ax_rrs_linear, ax_rrs_log) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig_kd, ax_kd = _new_fig((9, 4.5))
    any_rrs = any_kd = False

    for color, nc_path in zip(colors, kept_files):
        with xr.open_dataset(nc_path) as opened:
            nc = opened.load()
        waves = nc["wavelength"].values
        label = nc_path.stem

        if "rrs_0p_loess" in nc.data_vars or "rrs_0p_linear" in nc.data_vars:
            any_rrs = True
            for ax_rrs in (ax_rrs_linear, ax_rrs_log):
                if "rrs_0p_loess" in nc.data_vars:
                    ax_rrs.plot(waves, nc["rrs_0p_loess"].values, "-", color=color, label=label)
                if "rrs_0p_linear" in nc.data_vars:
                    ax_rrs.plot(waves, nc["rrs_0p_linear"].values, "--", color=color)

        if "EdZ_K0" in nc.data_vars and "EdZ_surface_z_interval" in nc.data_vars:
            any_kd = True
            k0_adaptive = _k0_at_adaptive_depth(
                nc["EdZ_K0"].values, nc["EdZ_depth"].values, nc["EdZ_surface_z_interval"].values
            )
            ax_kd.plot(waves, k0_adaptive, "-", color=color, label=label)
            ax_kd.plot(waves, nc["EdZ_surface_k_surf"].values, "--", color=color)

    rrs_figure: Figure | None = None
    if any_rrs:
        for ax_rrs in (ax_rrs_linear, ax_rrs_log):
            ax_rrs.set_xlabel("Wavelength (nm)")
            ax_rrs.set_ylabel("Rrs(0+)")
        ax_rrs_log.set_yscale("log")
        ax_rrs_linear.set_title("Rrs, linear (solid: LOESS, dashed: linear fit)")
        ax_rrs_log.set_title("Rrs, log (solid: LOESS, dashed: linear fit)")
        ax_rrs_linear.legend(fontsize="small")
        rrs_figure = fig_rrs
    else:
        plt.close(fig_rrs)

    k0_figure: Figure | None = None
    if any_kd:
        ax_kd.set_xlabel("Wavelength (nm)")
        ax_kd.set_ylabel("K0 (m⁻¹)")
        ax_kd.set_title("K0(EdZ) (solid: adaptive depth, dashed: near-surface linear)")
        ax_kd.legend(fontsize="small")
        k0_figure = fig_kd
    else:
        plt.close(fig_kd)

    return rrs_figure, k0_figure


def build_station_par_depth_table(directory: Path) -> pd.DataFrame:
    """One row per kept cast (``select.cops.dat`` flag != 0), one column per PAR light-level
    fraction (:data:`_STATION_PAR_FRACTIONS`), giving the depth (m) at which the broadband PAR
    profile crosses that fraction of its own surface reference -- station-wide companion to the
    single-cast PAR profile's own dashed-line depth annotations
    (:func:`build_par_kd_par_figures`), for comparing light penetration across an entire station
    at a glance. A cast missing ``par_d_profile`` (no EdZ) is skipped; a fraction never reached
    within a cast's own fitted depth range shows as ``None`` (matches
    :func:`~pycops.processing.attenuation.depth_at_light_fraction`'s own NaN-if-unreachable
    convention).
    """
    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        return pd.DataFrame()
    kept_files = kept_nc_files(directory, nc_dir)

    rows = []
    for nc_path in kept_files:
        with xr.open_dataset(nc_path) as opened:
            nc = opened.load()
        if "par_d_profile" not in nc.data_vars or "par_0" not in nc.attrs:
            continue
        depth = nc["EdZ_depth"].values
        par_d = nc["par_d_profile"].values
        par_0 = float(nc.attrs["par_0"])
        row: dict[str, object] = {"cast": nc_path.stem}
        for fraction in _STATION_PAR_FRACTIONS:
            z = depth_at_light_fraction(par_d[:, None], depth, np.array([par_0]), fraction)[0]
            row[f"{fraction * 100:g}%"] = round(float(z), 2) if np.isfinite(z) else None
        rows.append(row)

    return pd.DataFrame(rows)


def build_station_par_depth_table_figure(directory: Path) -> Figure | None:
    """:func:`build_station_par_depth_table` rendered as a matplotlib table -- the PDF report's
    own equivalent of the interactive tab's ``st.dataframe`` (a real ``Figure`` is needed to put
    tabular data on a PDF page; the interactive tab instead calls
    :func:`build_station_par_depth_table` directly and hands it to ``st.dataframe``, since
    Streamlit doesn't need a matplotlib table for that)."""
    table = build_station_par_depth_table(directory)
    if table.empty:
        return None

    fig, ax = _new_fig((9, 0.5 + 0.35 * len(table)))
    ax.axis("off")
    cell_text = [["-" if pd.isna(v) else f"{v:g}" for v in row] for row in table.drop(columns="cast").values]
    mpl_table = ax.table(
        cellText=cell_text,
        rowLabels=table["cast"].tolist(),
        colLabels=list(table.columns[1:]),
        loc="center",
        cellLoc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8)
    ax.set_title("PAR penetration depth (m) by light level, per cast")
    return fig


def build_station_par_profile_figure(directory: Path) -> Figure | None:
    """Every currently-kept cast's PAR_d(z) profile, overlaid -- station-wide companion to
    :func:`build_par_kd_par_figures`'s single-cast vertical PAR profile, same styling convention
    as :func:`build_station_comparison_figures`'s Rrs/K0 overlays."""
    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        return None
    kept_files = kept_nc_files(directory, nc_dir)
    applicable = []
    for nc_path in kept_files:
        with xr.open_dataset(nc_path) as opened:
            nc = opened.load()
        if "par_d_profile" in nc.data_vars:
            applicable.append((nc_path.stem, nc["EdZ_depth"].values, nc["par_d_profile"].values))
    if not applicable:
        return None

    colors = plt.cm.tab20(np.linspace(0, 1, len(applicable)))
    fig, ax = _new_fig((9, 4.5))
    for color, (label, depth, par_d) in zip(colors, applicable):
        ax.plot(par_d, depth, color=color, label=label)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("PAR (µEin.m⁻².s⁻¹, log scale)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("PAR profile comparison")
    ax.legend(fontsize="small")
    return fig


def build_station_kd_penetration_depth_figure(directory: Path) -> Figure | None:
    """Spectral ``Kd`` at the penetration depth (``kd_pd``, mean diffuse attenuation per
    wavelength from the surface to the depth where 1/e of that band's own subsurface irradiance
    remains -- distinct from the existing K0(EdZ) comparison in
    :func:`build_station_comparison_figures`, which is near-surface only: its "adaptive depth" is
    each wavelength's own near-surface linear-fit window, and its dashed line is the near-surface
    linear ``K_surf`` -- neither reaches down to the actual penetration depth), overlaid one line
    per currently-kept cast, same styling convention as the Rrs/K0 comparisons.
    """
    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        return None
    kept_files = kept_nc_files(directory, nc_dir)
    applicable = []
    for nc_path in kept_files:
        with xr.open_dataset(nc_path) as opened:
            nc = opened.load()
        if "kd_pd" in nc.data_vars:
            applicable.append((nc_path.stem, nc["wavelength"].values, nc["kd_pd"].values))
    if not applicable:
        return None

    colors = plt.cm.tab20(np.linspace(0, 1, len(applicable)))
    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax in (ax_linear, ax_log):
        for color, (label, waves, kd_pd) in zip(colors, applicable):
            ax.plot(waves, kd_pd, "-o", color=color, label=label, markersize=4)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Kd at penetration depth (m⁻¹)")
    ax_log.set_yscale("log")
    ax_linear.set_title("Spectral Kd at penetration depth, linear, by cast")
    ax_log.set_title("Spectral Kd at penetration depth, log, by cast")
    ax_linear.legend(fontsize="small")
    return fig


def build_station_qfactor_figure(directory: Path) -> Figure | None:
    """Empirical Q-factor (``EuZ.0m / LuZ.0m``, see :func:`build_qfactor_figure`), overlaid one
    line per currently-kept cast that carried both LuZ and EuZ -- most stations have at most a few
    such casts (many deployments only carry one of the two instruments), so this figure is often
    ``None`` or thin on lines, unlike Rrs/K0 which every cast contributes to."""
    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        return None
    kept_files = kept_nc_files(directory, nc_dir)
    applicable = []
    for nc_path in kept_files:
        with xr.open_dataset(nc_path) as opened:
            nc = opened.load()
        if "q_factor_loess" in nc.data_vars or "q_factor_linear" in nc.data_vars:
            applicable.append((nc_path.stem, nc))
    if not applicable:
        return None

    colors = plt.cm.tab20(np.linspace(0, 1, len(applicable)))
    fig, ax = _new_fig((9, 4.5))
    for color, (label, nc) in zip(colors, applicable):
        waves = nc["wavelength"].values
        if "q_factor_loess" in nc.data_vars:
            ax.plot(waves, nc["q_factor_loess"].values, "-", color=color, label=label)
        if "q_factor_linear" in nc.data_vars:
            ax.plot(waves, nc["q_factor_linear"].values, "--", color=color)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Q-factor (Eu(0-) / Lu(0-), sr)")
    ax.set_title("Empirical Q-factor, by cast (solid: LOESS, dashed: linear fit)")
    ax.legend(fontsize="small")
    return fig


def build_cast_report_figures(
    nc: xr.Dataset,
    raw_ds: xr.Dataset | None,
    init: dict[str, object] | None,
    info: CastInfo | None,
    cast_file: str,
    instruments: tuple[str, ...],
    depth_is_on: str | None,
    delta_capteur_optics: dict[str, float],
    time_window: tuple[float, float] | None,
) -> list[Figure]:
    """Every applicable section for one cast, in the same order
    :func:`ui.analyze_app._render_single_cast` renders them, with the same skip-if-absent gates
    that function already has."""
    figures = [build_cover_page(nc, init, info, cast_file)]

    fig = build_ed0_stability_figure(nc, time_window)
    if fig is not None:
        figures.append(fig)

    if raw_ds is not None and depth_is_on is not None:
        figures.append(build_depth_vs_time_figure(raw_ds, depth_is_on, time_window))

    if raw_ds is not None and depth_is_on is not None and init is not None:
        fig = build_tilt_figures_combined(
            raw_ds, depth_is_on, delta_capteur_optics, init, info, ("Ed0", *instruments), time_window
        )
        if fig is not None:
            figures.append(fig)

    for instrument in instruments:
        figures.append(
            build_depth_profile_figure(nc, raw_ds, depth_is_on, delta_capteur_optics.get(instrument), instrument)
        )
        figures.append(build_attenuation_figure(nc, instrument))

    fig = build_spectral_kd_figure(nc)
    if fig is not None:
        figures.append(fig)

    fig = build_penetration_depth_figure(nc)
    if fig is not None:
        figures.append(fig)

    if "par_d_profile" in nc.data_vars:
        figures.extend(build_par_kd_par_figures(nc))

    for instrument in _SHADOW_INSTRUMENTS:
        if f"{instrument}_surface_value_at_surface" in nc.data_vars:
            figures.append(
                build_extrapolation_grid_figure(
                    nc, raw_ds, depth_is_on, delta_capteur_optics.get(instrument), instrument
                )
            )

    fig = build_rrs_figure(nc)
    if fig is not None:
        figures.append(fig)

    fig = build_qfactor_figure(nc)
    if fig is not None:
        figures.append(fig)

    for instrument in _SHADOW_INSTRUMENTS:
        if f"{instrument}_shadow_correction" in nc.data_vars:
            figures.append(build_shadow_correction_figure(nc, instrument))

    fig = build_qwip_figure(nc)
    if fig is not None:
        figures.append(fig)

    if nc.attrs.get("shallow"):
        for instrument in _SHADOW_INSTRUMENTS:
            if f"{instrument}_rb" in nc.data_vars:
                figures.append(build_bottom_figure(nc, instrument))

    return figures


def write_cast_pdf_report(
    nc: xr.Dataset,
    raw_ds: xr.Dataset | None,
    init: dict[str, object] | None,
    info: CastInfo | None,
    cast_file: str,
    instruments: tuple[str, ...],
    depth_is_on: str | None,
    delta_capteur_optics: dict[str, float],
    time_window: tuple[float, float] | None,
    path: str | Path,
) -> None:
    """Write one cast's full diagnostic report to a multi-page PDF at ``path`` (parent folder
    created if missing, matching R's own ``dirpdf`` convention of sitting alongside the cast
    files/``nc/`` in the deployment folder)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figures = build_cast_report_figures(
        nc, raw_ds, init, info, cast_file, instruments, depth_is_on, delta_capteur_optics, time_window
    )
    with PdfPages(path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)


def write_station_summary_pdf(directory: Path, path: str | Path) -> int:
    """Write the station-wide comparison (Rrs/K0, PAR penetration-depth table, PAR profile
    overlay, spectral Kd-at-penetration-depth comparison, penetration-depth-by-wavelength
    comparison, and empirical Q-factor comparison) to a PDF at ``path``. Returns the number of
    pages actually written (0 if every section came back ``None`` -- e.g. no kept cast has
    Rrs/K0/PAR/Kd/Q-factor data -- in which case ``path`` still exists but is a valid, page-less
    PDF; callers that want to flag this rather than silently produce an empty-looking file should
    check the return value)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figures = [
        *build_station_comparison_figures(directory),
        build_station_par_depth_table_figure(directory),
        build_station_par_profile_figure(directory),
        build_station_kd_penetration_depth_figure(directory),
        build_penetration_depth_comparison_figure(directory),
        build_station_qfactor_figure(directory),
    ]
    n_pages = 0
    with PdfPages(path) as pdf:
        for fig in figures:
            if fig is None:
                continue
            pdf.savefig(fig)
            plt.close(fig)
            n_pages += 1
    return n_pages


def find_raw_cast_for_stem(directory: Path, nc_stem: str) -> Path | None:
    """The raw cast file (in ``directory``) whose own stem matches ``nc_stem`` (an already-written
    ``.nc`` file's stem always matches its source cast file's stem 1:1) -- reuses
    :func:`pycops.io.scaffold.discover_l1_casts`'s own glob/parse, since an already-cleaned ``L2``
    deployment folder has the exact same raw-file layout an ``L1`` day folder does, just alongside
    ``init.cops.dat``/etc. rather than bare."""
    for path in discover_l1_casts(directory):
        if path.stem == nc_stem:
            return path
    return None


def write_station_pdf_reports(
    directory: Path,
    include_station_summary: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[int, list[str]]:
    """Write one PDF report per currently-kept cast in ``directory`` (see
    :func:`write_cast_pdf_report`), plus (when ``include_station_summary``) one station-summary
    PDF (:func:`write_station_summary_pdf`).

    The pure, Streamlit-free core of "regenerate every PDF for this station" -- shared by tab 3's
    batch/single processing (``ui/clean_app.py``, which can now call this instead of requiring a
    full :func:`pycops.processing.deployment.process_deployment` re-run just to pick up a PDF
    rendering change) and tab 4's own per-station PDF buttons (``ui/analyze_app.py``), so there's
    exactly one implementation of "loop every kept cast, reopen its raw file/``info.cops.dat``
    row, write its PDF" rather than two.

    Returns ``(written_count, failures)`` -- one bad cast's failure is isolated and reported
    rather than aborting the rest, matching ``process_deployment``'s own per-cast isolation.
    ``progress_callback(i, total)``, if given, is called after each cast (1-indexed ``i``) so a
    caller can drive its own progress bar. Needs ``directory / "nc"`` to already hold the kept
    casts' ``.nc`` output (i.e. the station must already have been processed at least once) --
    returns ``(0, [<reason>])`` immediately otherwise, same message shape as any other failure.
    """
    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        return 0, [f"no nc/ subfolder in {directory} -- process this station first"]
    kept_files = kept_nc_files(directory, nc_dir)
    if not kept_files:
        return 0, [f"no kept casts with .nc output found in {nc_dir}"]

    init: dict[str, object] | None = None
    depth_is_on = None
    init_path = directory / "init.cops.dat"
    if init_path.exists():
        try:
            init = read_init_cops(init_path)
            depth_is_on = init["depth.is.on"]
        except Exception:  # noqa: BLE001 -- degrade to fitted-curves-only rather than fail outright
            init = None
            depth_is_on = None
    delta_capteur_optics = init["delta.capteur.optics"] if init is not None else {}

    written = 0
    failures: list[str] = []
    for i, nc_path in enumerate(kept_files):
        stem = nc_path.stem
        try:
            with xr.open_dataset(nc_path) as opened:
                nc = opened.load()
            instruments = _instruments_present(nc)
            raw_path = find_raw_cast_for_stem(directory, stem)
            raw_ds = None
            if raw_path is not None:
                try:
                    raw_ds = read_cast(raw_path, instruments=("Ed0", *instruments))
                except Exception:  # noqa: BLE001 -- degrade to fitted-curves-only
                    raw_ds = None
            if raw_ds is not None and depth_is_on is not None and f"{depth_is_on}_Depth" not in raw_ds:
                raw_ds = None

            cast_file = raw_path.name if raw_path is not None else stem
            info = None
            if raw_path is not None:
                info_path = directory / "info.cops.dat"
                if info_path.exists():
                    info = next((e for e in read_info_cops(info_path) if e.file == cast_file), None)
            time_window = _effective_time_window(init, info) if init is not None else None

            write_cast_pdf_report(
                nc, raw_ds, init, info, cast_file, instruments, depth_is_on, delta_capteur_optics,
                time_window, directory / "pdf" / f"{stem}.pdf",
            )
            written += 1
        except Exception as exc:  # noqa: BLE001 -- isolate one bad cast from the rest of the batch
            failures.append(f"{stem}: {exc}")
        if progress_callback is not None:
            progress_callback(i + 1, len(kept_files))

    if include_station_summary:
        try:
            n_pages = write_station_summary_pdf(
                directory, directory / "pdf" / f"{directory.name}_station_summary.pdf"
            )
            if n_pages == 0:
                failures.append(
                    "station summary: no comparison data available (no Rrs/K0/PAR/Kd/Q-factor "
                    "data across the currently-kept casts)"
                )
        except Exception as exc:  # noqa: BLE001 -- one bad section shouldn't hide the per-cast results above
            failures.append(f"station summary: {type(exc).__name__}: {exc}")

    return written, failures
