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
    """Mean line + shaded mean +/- 1 SD band for one station, on already-finite wavelengths only.
    Returns whether anything was actually plotted (all-NaN stations are silently skipped by the
    caller, matching every other station-comparison figure in this port)."""
    finite = np.isfinite(mean_sd.mean)
    if not finite.any():
        return False
    w = waves[finite]
    mean = mean_sd.mean[finite]
    sd = np.nan_to_num(mean_sd.sd[finite], nan=0.0)
    ax.plot(w, mean, "-o", color=color, label=label, markersize=4)
    # clipped to a small positive floor (not just >=0) so the shaded band stays well-behaved on
    # the log-scale axis too, without needing to special-case which axis is currently log.
    lower = np.clip(mean - sd, a_min=1e-12, a_max=None)
    ax.fill_between(w, lower, mean + sd, color=color, alpha=0.2)
    return True


def build_rrs_comparison_figure(stations: list[StationAggregate]) -> Figure | None:
    """Rrs(0+) mean +/- 1 SD across each station's kept casts, one color per station, side by
    side linear/log -- same layout convention as the per-cast Rrs figures
    (:func:`pycops.io.pdf_report.build_rrs_figure`/``build_station_comparison_figures``)."""
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


def build_kd_comparison_figure(stations: list[StationAggregate], metric: str = "kd_pd") -> Figure | None:
    """Spectral Kd (``metric`` -- one of ``kd_1pct``/``kd_10pct``/``kd_pd``, see
    :class:`pycops.processing.database.StationAggregate`) mean +/- 1 SD across each station's
    kept casts, same layout as :func:`build_rrs_comparison_figure`."""
    if metric not in KD_METRIC_LABELS:
        raise ValueError(f"metric must be one of {sorted(KD_METRIC_LABELS)}, got {metric!r}")
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
