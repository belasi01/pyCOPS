"""Cross-station Rrs/Kd comparison figures.

Simon's request: a way to pick several stations (already aggregated for the mission database,
see :mod:`pycops.processing.database`) and overlay their mean Rrs/Kd spectra, with each station's
own sample standard deviation across its kept casts shown as a shaded band -- one level up from
tab 4's own per-station "every cast in this one station" comparison
(:func:`pycops.io.pdf_report.build_station_comparison_figures`), and reusing the exact same
mean/sd aggregation :func:`pycops.processing.database.aggregate_station` already computes for the
mission-database export, rather than a separate statistics pass.

Pure figure builders (no Streamlit dependency), same convention as :mod:`pycops.io.pdf_report`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pycops.processing.database import STANDARD_WAVELENGTHS, MeanSd, StationAggregate

KD_METRIC_LABELS = {
    "kd_pd": "Kd at penetration depth (1/e)",
    "kd_1pct": "Kd at 1% light level",
    "kd_10pct": "Kd at 10% light level",
}


def _station_label(station: StationAggregate) -> str:
    # station_id alone can collide (two sibling deployment folders at the same physical station,
    # e.g. different instrument operators) -- the folder name disambiguates, same convention as
    # the SeaBASS per-station filename in ui/database_app.py.
    return f"{station.station_id} ({station.directory.name})"


def _plot_mean_sd_band(ax: Axes, waves: np.ndarray, mean_sd: MeanSd, color, label: str) -> bool:
    """Mean line + shaded mean +/- 1 SD band for one station.

    Returns whether anything was actually plotted (all-NaN stations are silently skipped by the
    caller, matching every other station-comparison figure in this port).

    Deliberately does **not** pre-filter to only-finite wavelengths before plotting: a station
    often has a whole band (e.g. green, where Kd is small enough that the 1%/10% light level
    isn't reached within the cast's own measured depth range -- "souvent le cas" per Simon) where
    a metric is genuinely NaN, while it's defined on either side (blue and red). Pre-filtering
    would silently drop those missing points *before* handing the array to matplotlib, so
    ``ax.plot``/``fill_between`` would draw a straight line directly connecting the last blue
    point to the first red point -- a real, misleading line across the gap, not present in the
    real data at all. Plotting over the *full* ``waves`` grid with NaN left in place instead lets
    matplotlib's own NaN-gap handling break the line/shading exactly where the data is missing.

    Where the sample SD exceeds the mean (real, not rare, at the near-detection-limit edges of
    the spectrum, e.g. UV/NIR -- confirmed on a real station, UMQ3, whose 875 nm Rrs across 3
    kept casts ranged ~5.8e-6 to 4.8e-5, an SD bigger than the mean itself), ``mean - sd`` goes
    negative -- not representable on a log-scale axis. An earlier version clamped that to an
    arbitrary fixed floor (1e-12), which rendered as a physically meaningless multi-order-of-
    magnitude plunge with no relationship to the real data. Leaving it as NaN instead makes
    ``fill_between`` skip shading at that wavelength (a real gap in the band, not a fabricated
    lower bound) -- the mean line itself is unaffected either way.
    """
    if not np.isfinite(mean_sd.mean).any():
        return False
    mean = mean_sd.mean
    sd = mean_sd.sd
    ax.plot(waves, mean, "-o", color=color, label=label, markersize=4)
    lower = mean - sd
    lower_for_fill = np.where(lower > 0, lower, np.nan)
    ax.fill_between(waves, lower_for_fill, mean + sd, color=color, alpha=0.2)
    return True


def build_rrs_comparison_figure(stations: list[StationAggregate], waves: np.ndarray | None = None) -> Figure | None:
    """Rrs(0+) mean +/- 1 SD across each station's kept casts, one color per station, side by
    side linear/log -- same layout convention as the per-cast Rrs figures
    (:func:`pycops.io.pdf_report.build_rrs_figure`/``build_station_comparison_figures``).

    ``waves`` must match each station's own ``MeanSd`` arrays band-for-band -- pass
    :func:`pycops.processing.database.trim_unused_wavelengths`'s own returned grid when calling
    with already-trimmed stations (the caller's job, matching how the comparison UI does it);
    defaults to the full, untrimmed :data:`STANDARD_WAVELENGTHS` otherwise.
    """
    if waves is None:
        waves = np.asarray(STANDARD_WAVELENGTHS, dtype=float)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(stations), 1)))
    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(13, 4.5))
    any_plotted = False
    for ax in (ax_linear, ax_log):
        for color, station in zip(colors, stations):
            if _plot_mean_sd_band(ax, waves, station.rrs, color, _station_label(station)):
                any_plotted = True
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Rrs(0+)")
    if not any_plotted:
        plt.close(fig)
        return None
    ax_log.set_yscale("log")
    ax_linear.set_title("Rrs, linear (mean +/- 1 SD)")
    ax_log.set_title("Rrs, log (mean +/- 1 SD)")
    ax_linear.legend(fontsize="small")
    return fig


def build_kd_comparison_figure(
    stations: list[StationAggregate], metric: str = "kd_pd", waves: np.ndarray | None = None
) -> Figure | None:
    """Spectral Kd (``metric`` -- one of ``kd_1pct``/``kd_10pct``/``kd_pd``, see
    :class:`pycops.processing.database.StationAggregate`) mean +/- 1 SD across each station's
    kept casts, same layout as :func:`build_rrs_comparison_figure` (see its own docstring for
    the ``waves`` parameter)."""
    if metric not in KD_METRIC_LABELS:
        raise ValueError(f"metric must be one of {sorted(KD_METRIC_LABELS)}, got {metric!r}")
    if waves is None:
        waves = np.asarray(STANDARD_WAVELENGTHS, dtype=float)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(stations), 1)))
    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(13, 4.5))
    any_plotted = False
    for ax in (ax_linear, ax_log):
        for color, station in zip(colors, stations):
            mean_sd: MeanSd = getattr(station, metric)
            if _plot_mean_sd_band(ax, waves, mean_sd, color, _station_label(station)):
                any_plotted = True
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Kd (m⁻¹)")
    if not any_plotted:
        plt.close(fig)
        return None
    ax_log.set_yscale("log")
    label = KD_METRIC_LABELS[metric]
    ax_linear.set_title(f"{label}, linear (mean +/- 1 SD)")
    ax_log.set_title(f"{label}, log (mean +/- 1 SD)")
    ax_linear.legend(fontsize="small")
    return fig


def build_pd_depth_comparison_figure(
    stations: list[StationAggregate], waves: np.ndarray | None = None
) -> Figure | None:
    """Penetration depth (``pd_depth`` -- the 1/e-light-level crossing depth itself, in meters,
    distinct from ``kd_pd``'s derived attenuation coefficient) mean +/- 1 SD across each
    station's kept casts, one color per station, depth 0 at the top.

    Unlike :func:`pycops.io.pdf_report.build_penetration_depth_figure`/
    ``build_penetration_depth_comparison_figure`` (tab 4's true-color-by-wavelength
    visualization of "what a satellite sees"), this is a *statistical* cross-station comparison
    -- color is used for station identity instead, matching :func:`build_rrs_comparison_figure`/
    :func:`build_kd_comparison_figure`'s own convention (mean line + shaded SD band, one color
    per station). A single panel (not linear/log side by side): depth is naturally linear, and
    the inverted axis already carries the "surface at the top" convention.
    """
    if waves is None:
        waves = np.asarray(STANDARD_WAVELENGTHS, dtype=float)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(stations), 1)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_plotted = False
    for color, station in zip(colors, stations):
        if _plot_mean_sd_band(ax, waves, station.pd_depth, color, _station_label(station)):
            any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Penetration depth (m)")
    ax.set_title("Penetration depth (1/e light level), mean +/- 1 SD")
    ax.legend(fontsize="small")
    return fig
