"""Interactive results-analysis tab (section 4).

Replaces Simon's R workflow's per-cast, multi-page PDF diagnostic report (``process.cops.R``'s
``pdf(...)``/``dev.off()`` block driving ``process.LuZ.R``/``process.EdZ.R``/``process.EuZ.R``/
``compute.aops.R``'s ``plot()``/``matplot()`` calls) with an interactive equivalent, browsing the
results tab 3 already wrote to ``.nc`` files (:func:`pycops.io.netcdf.write_deployment_result`)
one cast at a time. This is a read-only viewer over already-computed data -- no new processing
happens here.

Depth-profile/attenuation plots also reopen the original raw cast file (still sitting next to
``nc/`` in the same deployment folder, per tabs 2/3's convention) to overlay raw per-scan points
on top of the ``.nc``'s fitted curve, matching the R PDF's ``matplot(..., type="p")`` + fitted
line -- the ``.nc`` alone only has the fitted curve (``{instrument}_fitted``), not raw scans.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from pycops.io.config import CastInfo, read_init_cops, update_cast_info
from pycops.io.discovery import FLAG_NORMAL, FLAG_REJECTED, kept_nc_files, update_cast_selection
from pycops.io.exclusions import read_wavelength_exclusions, update_wavelength_exclusions
from pycops.io.netcdf import write_cast_result
from pycops.io.raw import read_cast
from pycops.io.scaffold import discover_l1_casts
from pycops.processing.deployment import reprocess_single_cast
from pycops.processing.depth import time_window_mask
from pycops.processing.par import percent_par_at_depth
from pycops.processing.qwip import _qwip_polynomial
from pycops.processing.tilt import add_tilt
from pycops.ui._common import (
    OVERRIDE_FIELDS,
    _directory_input,
    existing_info,
    existing_selection,
    format_override,
    parsed_override_fields,
    render_override_fields_editor,
    render_time_window_editor,
)

_DEPTH_INSTRUMENTS = ("EdZ", "LuZ", "EuZ")
_SHADOW_INSTRUMENTS = ("LuZ", "EuZ")
_VISIBLE_MAX_NM = 700.0  # visible-band cutoff, matching QWIP's own 400-700 nm convention
_RB_NEGLIGIBLE_EDZ_FRACTION = 0.01  # Simon's own starting suggestion ("e.g. inferieur a 1%?")
# Matches discovery.py's own _DEFAULT_METHOD (private there, so not imported directly) --
# clean_app.py already duplicates this same constant for the same reason.
_DEFAULT_METHOD = "Rrs.0p.linear"
_METHOD_OPTIONS = ("Rrs.0p", "Rrs.0p.linear")
_METHOD_LABELS = {"Rrs.0p": "LOESS", "Rrs.0p.linear": "Linear"}


def _instruments_present(nc: xr.Dataset) -> tuple[str, ...]:
    return tuple(instr for instr in _DEPTH_INSTRUMENTS if f"{instr}_fitted" in nc.data_vars)


def _find_raw_cast(directory: Path, nc_stem: str) -> Path | None:
    for path in discover_l1_casts(directory):
        if path.stem == nc_stem:
            return path
    return None


def _wavelength_dim(ds: xr.Dataset, instrument: str) -> str:
    return "wavelength" if "wavelength" in ds[instrument].dims else f"wavelength_{instrument}"


def _wavelength_colors(waves: np.ndarray) -> np.ndarray:
    return plt.cm.viridis(np.linspace(0, 1, len(waves)))


def _new_fig(figsize: tuple[float, float] = (9, 4.5)):
    return plt.subplots(figsize=figsize)


def _show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def _render_overview(nc: xr.Dataset) -> None:
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    lon, lat = nc.attrs.get("longitude"), nc.attrs.get("latitude")
    with col1:
        st.metric("Longitude", "NA" if lon is None or np.isnan(lon) else f"{lon:.5f}")
        st.metric("Latitude", "NA" if lat is None or np.isnan(lat) else f"{lat:.5f}")
    with col2:
        st.write(f"**Rrs method**: {nc.attrs.get('rrs_method') or '-'}")
        st.write(f"**Rrs source**: {nc.attrs.get('rrs_source') or '-'}")
        st.write(f"**Shallow**: {'yes' if nc.attrs.get('shallow') else 'no'}")
    with col3:
        chl_flag = nc.attrs.get("chl_flag")
        st.write(f"**chl flag**: {chl_flag if chl_flag is not None else 'NA'}")
        qc_flag = nc.attrs.get("qc_flag")
        st.write(f"**QC flag**: {qc_flag if qc_flag is not None else '-'}")

    if nc.attrs.get("shadow_correction_note"):
        st.info(nc.attrs["shadow_correction_note"])
    if nc.attrs.get("bottom_note"):
        st.info(nc.attrs["bottom_note"])
    if nc.attrs.get("excluded_wavelengths"):
        st.warning(
            f"Wavelength(s) manually excluded from the final Rrs (set to NaN): "
            f"{nc.attrs['excluded_wavelengths']} nm."
        )


def _render_ed0_stability(nc: xr.Dataset) -> None:
    if "ed0_correction" not in nc.data_vars:
        return
    st.subheader("Ed0 stability")
    st.caption(
        "Ratio of the smoothed surface reference to each raw scan -- flags illumination changes "
        "(e.g. clouds) during the cast. Red lines mark the +/-5% acceptable range."
    )
    correction = nc["ed0_correction"].mean(dim="wavelength").values

    fig, ax = _new_fig((9, 3))
    ax.plot(nc["time"].values, correction, color="tab:orange")
    ax.axhline(1.05, color="red", ls="--", lw=1)
    ax.axhline(0.95, color="red", ls="--", lw=1)
    # +/-10% by default, widened to fit the data if it's more variable than that.
    ax.set_ylim(min(0.9, float(np.nanmin(correction))), max(1.1, float(np.nanmax(correction))))
    ax.set_xlabel("Time")
    ax.set_ylabel("Ed0 correction (mean over wavelengths)")
    fig.autofmt_xdate()
    _show(fig)

    outside = int(np.sum((correction < 0.95) | (correction > 1.05)))
    if outside:
        st.warning(
            f"Ed0 correction is outside the +/-5% acceptable range for {outside} scan(s) -- "
            "possible illumination instability (e.g. passing clouds) during this cast."
        )


def _raw_scan_values(
    raw_ds: xr.Dataset | None,
    depth_is_on: str | None,
    delta_capteur: float | None,
    instrument: str,
    w: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Raw per-scan (value, depth) for ``instrument`` at the wavelength nearest ``w``.

    Depth comes from ``depth_is_on``'s own column (``init.cops.dat``'s single reference
    depth/pressure sensor) -- *not* ``f"{instrument}_Depth"``, which only exists for whichever
    instrument physically carries that sensor (typically LuZ or EuZ). EdZ has no depth column of
    its own and shares the reference sensor's, exactly like the real fitting pipeline does.

    ``delta_capteur`` (``init.cops.dat``'s ``delta.capteur.optics`` for *this* instrument) is
    then added, matching ``cast_fit.py``'s own ``depth = depth_ref + delta_capteur_optics``:
    without it, the raw points are still in the reference sensor's own depth frame, offset from
    the fitted curve (which *is* in this instrument's corrected frame) by that sensor-to-sensor
    distance -- confirmed by Simon on a real cast (EdZ's fit sat above its raw points, LuZ's
    below, consistent with EdZ/LuZ's opposite-signed real-world offsets, e.g. -0.05 m / +0.238 m).
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
    return values, depth


def _render_depth_profile(
    nc: xr.Dataset,
    raw_ds: xr.Dataset | None,
    depth_is_on: str | None,
    delta_capteur: float | None,
    instrument: str,
) -> None:
    depth_dim = f"{instrument}_depth"
    waves = nc["wavelength"].values
    fitted = nc[f"{instrument}_fitted"]
    depth = nc[depth_dim].values

    wave_options = ["All"] + [f"{w:g}" for w in waves]
    wavelength_choice = st.selectbox(
        "Wavelength", wave_options, key=f"analyze_{instrument}_depth_wave"
    )

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
        raw = _raw_scan_values(raw_ds, depth_is_on, delta_capteur, instrument, w)
        if raw is not None:
            raw_values, raw_depth = raw
            kept_var = f"{instrument}_kept"
            if kept_var in nc.data_vars and nc.sizes["time"] == raw_ds.sizes["time"]:
                kept = nc[kept_var].values.astype(bool)
            else:
                kept = np.ones(raw_depth.shape, dtype=bool)
            ax.plot(raw_values[kept], raw_depth[kept], ".", markersize=3, color="tab:blue", label="kept scans")
            ax.plot(raw_values[~kept], raw_depth[~kept], ".", markersize=3, color="lightgray", label="excluded scans")
        ax.plot(fitted.isel(wavelength=wi).values, depth, color="tab:red", lw=2, label="fitted")
        detection_limit = nc[f"{instrument}_detection_limit"].values[wi]
        if np.isfinite(detection_limit):
            ax.axvline(detection_limit, color="black", ls="--", lw=1, label="detection limit")
        ax.legend(loc="best", fontsize="small")

    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel(f"{instrument} (log scale)")
    ax.set_ylabel("Depth (m)")
    _show(fig)


def _render_attenuation(nc: xr.Dataset, instrument: str) -> None:
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
    _show(fig)


def _render_par_and_kd_par(nc: xr.Dataset, selected: str) -> None:
    """Vertical PAR profile + Kd(PAR) -- port of ``compute.PAR.fitted.R``'s own plot, one of the
    richer diagnostics from Simon's R PDF report that pycops didn't have until now. ``PAR.0`` is a
    single scalar here (a vertical reference line), not a depth profile like R's: pycops fits Ed0
    at one point only (see ``par_0``'s own docstring in ``process_cast.py``), and the per-scan
    illumination-change diagnostic R's own ``PAR.0(z)`` plot doubles as is already covered by the
    "Ed0 stability" section above.
    """
    depth = nc["EdZ_depth"].values
    par_d = nc["par_d_profile"].values
    par_0 = float(nc.attrs["par_0"])

    fig, ax = _new_fig((7, 5))
    ax.plot(par_d, depth, color="tab:blue", lw=2, label="PAR_d (downwelling)")
    if "par_u_profile" in nc.data_vars:
        ax.plot(nc["par_u_profile"].values, depth, color="tab:orange", lw=2, label="PAR_u (upwelling)")
    ax.axvline(par_0, color="gray", ls="--", lw=1.5, label="PAR_0 (surface reference)")
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("PAR (µEin.m⁻².s⁻¹, log scale)")
    ax.set_ylabel("Depth (m)")
    ax.legend(loc="best", fontsize="small")
    _show(fig)

    st.caption("Kd(PAR): mean diffuse attenuation of broadband PAR from the surface to a given depth.")
    fraction_table = {
        "light level": ["1%", "10%", "penetration depth (1/e)"],
        "Kd(PAR) (m⁻¹)": [nc.attrs["kd_par_1pct"], nc.attrs["kd_par_10pct"], nc.attrs["kd_par_pd"]],
    }
    st.dataframe(fraction_table, hide_index=True)

    if "kd_1pct" in nc.data_vars:
        st.caption("Spectral Kd (per wavelength), for comparison:")
        spectral_table = {
            "wavelength (nm)": nc["wavelength"].values,
            "Kd 1% (m⁻¹)": nc["kd_1pct"].values,
            "Kd 10% (m⁻¹)": nc["kd_10pct"].values,
            "Kd penetration depth (m⁻¹)": nc["kd_pd"].values,
        }
        st.dataframe(spectral_table, hide_index=True)

    k0_depth = depth[1:]  # k0_par[0] is a leading-NaN pad, matching K0's own depth_grid[1:] alignment
    k0_values = nc["k0_par"].values[1:]
    if len(k0_depth) > 1:
        chosen_depth = st.slider(
            "Depth for Kd(PAR) integrated from the surface",
            min_value=float(k0_depth.min()),
            max_value=float(k0_depth.max()),
            value=float(k0_depth.min()),
            key=f"analyze_kdpar_depth_slider::{selected}",
        )
        kd_at_depth = np.interp(chosen_depth, k0_depth, k0_values)
        st.metric(f"Kd(PAR) at {chosen_depth:.2f} m", f"{kd_at_depth:.4f} m⁻¹" if np.isfinite(kd_at_depth) else "NA")


def _render_extrapolation_comparison(
    nc: xr.Dataset,
    raw_ds: xr.Dataset | None,
    depth_is_on: str | None,
    delta_capteur: float | None,
    instrument: str,
) -> None:
    """LOESS vs. linear surface extrapolation, side by side -- both feed ``rrs_loess``/
    ``rrs_linear``, so this is the QC step for deciding which one to actually trust.

    The linear curve is reconstructed from already-stored fit parameters, not refit here:
    ``surface_linear.py``'s log-linear regression is anchored at the true surface (``z=0``), so
    ``value(z) = value_at_surface * exp(-k_surf * z)`` over ``z`` in ``[0, z_interval]``.
    """
    waves = nc["wavelength"].values
    depth = nc[f"{instrument}_depth"].values
    fitted = nc[f"{instrument}_fitted"]
    value_at_surface = nc[f"{instrument}_surface_value_at_surface"].values
    k_surf = nc[f"{instrument}_surface_k_surf"].values
    z_interval = nc[f"{instrument}_surface_z_interval"].values

    wave_options = [f"{w:g}" for w in waves]
    wavelength_choice = st.selectbox(
        "Wavelength", wave_options, key=f"analyze_{instrument}_extrap_wave"
    )
    w = float(wavelength_choice)
    wi = int(np.argmin(np.abs(waves - w)))

    fig, ax = _new_fig((9, 4.5))
    raw = _raw_scan_values(raw_ds, depth_is_on, delta_capteur, instrument, w)
    if raw is not None:
        raw_values, raw_depth = raw
        near_surface = raw_depth <= max(z_interval[wi] * 1.5, 1.0) if np.isfinite(z_interval[wi]) else np.ones_like(raw_depth, dtype=bool)
        ax.plot(raw_values[near_surface], raw_depth[near_surface], ".", markersize=4, color="tab:blue", label="raw scans")
    ax.plot(fitted.isel(wavelength=wi).values, depth, color="tab:red", lw=2, label="LOESS fit")
    if np.isfinite(value_at_surface[wi]) and np.isfinite(z_interval[wi]):
        z_line = np.linspace(0, z_interval[wi], 50)
        linear_curve = value_at_surface[wi] * np.exp(-k_surf[wi] * z_line)
        ax.plot(linear_curve, z_line, color="tab:green", lw=2, ls="--", label="linear fit")
    else:
        st.caption(f"No linear fit available at {w:g} nm (R² below threshold or too few points).")
    ax.set_xscale("log")
    ax.invert_yaxis()
    if np.isfinite(z_interval[wi]):
        ax.set_ylim(max(z_interval[wi] * 1.5, 1.0), 0)
    ax.set_xlabel(f"{instrument} (log scale)")
    ax.set_ylabel("Depth (m)")
    ax.legend(loc="best", fontsize="small")
    _show(fig)

    table = {
        "wavelength": waves,
        "LOESS value_at_0": nc[f"{instrument}_value_at_0"].values,
        "linear value_at_surface": value_at_surface,
        "linear k_surf": k_surf,
        "linear z_interval": z_interval,
        "linear r2": nc[f"{instrument}_surface_r2"].values,
        "linear ks_pvalue": nc[f"{instrument}_surface_ks_pvalue"].values,
    }
    st.dataframe(table, width="stretch", hide_index=True)


def _render_rrs_spectra(nc: xr.Dataset) -> None:
    if "rrs_0p_loess" not in nc.data_vars and "rrs_0p_linear" not in nc.data_vars:
        return
    st.subheader("Rrs")
    rrs_method = nc.attrs.get("rrs_method")
    if rrs_method:
        st.caption(
            f"Recommended: **{_METHOD_LABELS.get(rrs_method, rrs_method)}** (from "
            f"select.cops.dat's method column -- not QWIP-based; QWIP below is an independent "
            f"quality check, not the criterion used to pick loess vs. linear)."
        )
    waves = nc["wavelength"].values
    fig, ax = _new_fig((7, 4.5))
    for method, style in (("loess", "-o"), ("linear", "--s"), ("recommended", ":^")):
        var = f"rrs_0p_{method}"
        if var in nc.data_vars:
            ax.plot(waves, nc[var].values, style, label=method, markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Rrs(0+)")
    ax.legend(fontsize="small")
    _show(fig)


def _render_shadow_correction(nc: xr.Dataset, instrument: str) -> None:
    waves = nc["wavelength"].values
    fig, ax = _new_fig((9, 3.5))
    ax.plot(waves, nc[f"{instrument}_shadow_correction"].values, "-o", color="tab:purple")
    ax.set_ylim(0.2, 1.05)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Shadow correction ({instrument})")
    source = nc.attrs.get(f"{instrument}_absorption_source")
    if source:
        ax.set_title(f"Absorption source: {source}")
    _show(fig)


def _render_qwip(nc: xr.Dataset) -> None:
    labels = [label for label in ("loess", "linear") if f"qwip_{label}_avw" in nc.attrs]
    if not labels:
        return
    st.subheader("QWIP / water class")
    shallow = bool(nc.attrs.get("shallow"))
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        with col:
            st.write(f"**{label}**")
            score = nc.attrs[f"qwip_{label}_score"]
            st.metric("AVW (nm)", f"{nc.attrs[f'qwip_{label}_avw']:.1f}")
            st.metric("Score", f"{score:.3f}")
            passed = bool(nc.attrs[f"qwip_{label}_passed"])
            if not passed and shallow and score < 0:
                # Bottom-reflected light distorts the spectral shape QWIP was calibrated on
                # (open-ocean waters) -- a negative score here is expected, not a quality problem.
                st.write("ℹ️ Negative QWIP score expected (shallow/bottom-influenced water)")
            else:
                st.write("✅ Passed" if passed else "⚠️ Failed")
            st.write(f"Water class: {nc.attrs[f'qwip_{label}_water_class']} (FU {nc.attrs[f'qwip_{label}_fu']})")

    avw_range = np.linspace(400, 600, 200)
    predicted = _qwip_polynomial(avw_range)
    fig, ax = _new_fig((7, 4))
    ax.plot(avw_range, predicted, color="black", label="QWIP reference")
    ax.fill_between(avw_range, predicted - 0.1, predicted + 0.1, color="gray", alpha=0.2)
    for label, marker in zip(labels, ("o", "s")):
        ax.plot(nc.attrs[f"qwip_{label}_avw"], nc.attrs[f"qwip_{label}_ndi"], marker, markersize=10, label=label)
    ax.set_xlabel("AVW (nm)")
    ax.set_ylabel("NDI")
    ax.legend()
    _show(fig)


def _mask_negligible_rb(
    rb: np.ndarray,
    rb_extrapolated: np.ndarray,
    edz_at_bottom: np.ndarray,
    edz_at_surface: np.ndarray,
    threshold: float = _RB_NEGLIGIBLE_EDZ_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """NaN out wavelengths where EdZ at the bottom is negligible relative to the surface.

    Rb's denominator there is dominated by noise/near-zero division, not a real reflectance
    signal (a known numerical fragility of ``compute.bottom.R``'s own ratio, see CLAUDE.md).
    ``threshold`` (default 1% of surface EdZ) is Simon's own starting suggestion, not yet tuned
    against real data. A NaN ``edz_at_bottom`` (e.g. the fitted profile has no valid points that
    deep at a fast-attenuating wavelength) also counts as negligible.
    """
    rb = rb.copy()
    rb_extrapolated = rb_extrapolated.copy()
    with np.errstate(invalid="ignore"):
        negligible = ~(edz_at_bottom > threshold * edz_at_surface)
    rb[negligible] = np.nan
    rb_extrapolated[negligible] = np.nan
    return rb, rb_extrapolated


def _visible_band_ylim(rb: np.ndarray, rb_extrapolated: np.ndarray, waves: np.ndarray) -> float | None:
    """Y-axis max from visible-band (<=700 nm) values only.

    Near-infrared fluorescence can push Rb over 100% there (Simon), which would otherwise blow
    out the scale for the whole plot even though the visible bands (what Rb is actually meant to
    characterize) look normal. Returns ``None`` if there's nothing finite to scale from.
    """
    visible = waves <= _VISIBLE_MAX_NM
    values = np.concatenate([rb[visible], rb_extrapolated[visible]])
    values = values[np.isfinite(values)]
    return float(np.max(values)) * 1.15 if len(values) else None


def _render_bottom(nc: xr.Dataset, instrument: str) -> None:
    waves = nc["wavelength"].values
    rb = nc[f"{instrument}_rb"].values
    rb_extrapolated = nc[f"{instrument}_rb_extrapolated"].values
    bottom_depth = nc.attrs.get(f"{instrument}_bottom_depth")

    if bottom_depth is not None and "EdZ_fitted" in nc.data_vars:
        edz_at_bottom = np.array(
            [np.interp(bottom_depth, nc["EdZ_depth"].values, nc["EdZ_fitted"].isel(wavelength=i).values) for i in range(len(waves))]
        )
        rb, rb_extrapolated = _mask_negligible_rb(rb, rb_extrapolated, edz_at_bottom, nc["EdZ_value_at_0"].values)

    fig, ax = _new_fig((9, 3.5))
    ax.plot(waves, rb, "-o", label="Rb (~0.3 m above bottom)")
    ax.plot(waves, rb_extrapolated, "--s", label="Rb (extrapolated to bottom)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Bottom reflectance ({instrument})")
    ax.legend(fontsize="small")
    if bottom_depth is not None:
        ax.set_title(f"Bottom depth: {bottom_depth:.2f} m")

    ylim = _visible_band_ylim(rb, rb_extrapolated, waves)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    _show(fig)

    if bottom_depth is not None and "EdZ_fitted" in nc.data_vars:
        pct_par = percent_par_at_depth(waves, nc["EdZ_fitted"].values, nc["EdZ_depth"].values, bottom_depth)
        if pct_par is not None:
            st.metric("Benthic PAR available (% of surface)", f"{pct_par:.2f}%")


def _effective_time_window(
    init: dict[str, object], info: CastInfo | None
) -> tuple[float, float] | None:
    """The ``time.window`` actually used by processing: this cast's ``info.cops.dat`` override if
    present, else ``init.cops.dat``'s deployment-wide default -- mirrors
    ``process_cast.py``'s own resolution (``ds.attrs.get("time_window") or init["time.window"]``),
    for display purposes only."""
    if info is not None and info.time_window is not None:
        return info.time_window
    time_window = init.get("time.window")
    return tuple(time_window) if time_window is not None else None


def _render_depth_vs_time(
    raw_ds: xr.Dataset,
    depth_is_on: str,
    time_window: tuple[float, float] | None,
) -> None:
    """Read-only ``depth_is_on`` depth vs. elapsed time, shading any region excluded by the
    cast's currently-saved ``time.window`` -- the diagnostic Simon looks at to spot where tilt/
    depth goes bad near the end of a cast, kept visible without opening "Adjust & reprocess"."""
    st.subheader(f"{depth_is_on} depth vs time")
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
    _show(fig)


def _effective_tiltmax(init: dict[str, object], info: CastInfo | None, instrument: str) -> float:
    """``tiltmax.optics`` for ``instrument``, the deployment default unless ``info.cops.dat``'s
    per-cast ``tiltmax`` override (a value per ``instruments.optics`` entry, same shape ``init``
    itself uses) has one -- mirrors ``process_cast.py``'s own ``_apply_info_overrides()`` merge,
    for display purposes only (this doesn't itself change what gets processed)."""
    if info is not None and info.tiltmax is not None:
        instruments_list = list(init["instruments.optics"])
        if instrument in instruments_list:
            idx = instruments_list.index(instrument)
            if idx < len(info.tiltmax):
                return info.tiltmax[idx]
    return init["tiltmax.optics"][instrument]


def _render_tilt(
    raw_ds: xr.Dataset,
    depth_is_on: str,
    delta_capteur: float | None,
    init: dict[str, object],
    info: CastInfo | None,
    instrument: str,
    time_window: tuple[float, float] | None = None,
) -> None:
    """Tilt-vs-depth for ``instrument``, with a red threshold line at the effective ``tiltmax``
    (deployment default or this cast's own override, whichever applies) and scans exceeding it
    colored red -- exactly the scans ``tilt_mask()`` would exclude during real processing.
    Scans already excluded by the cast's currently-saved ``time.window`` are grayed out instead,
    regardless of their tilt, so a trim actually shows up on the figure Simon uses to decide it
    (previously this plot only ever reflected the ``tiltmax`` threshold, so time.window edits --
    even ones that measurably changed Rrs -- looked like they'd done nothing here).

    Computed from the raw cast (already reopened for the other raw-scan overlays) rather than
    stored in the ``.nc``: tilt is a per-scan raw-data diagnostic like the other overlays, and
    keeping the ``.nc`` schema free of raw scans avoids bloating file size or needing to
    reprocess already-processed stations for a schema change.
    """
    try:
        tilt = add_tilt(raw_ds, instrument)[f"{instrument}_Tilt"].values
    except KeyError as exc:
        st.warning(f"No Roll/Pitch available for {instrument}: {exc}")
        return

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

    fig, ax = _new_fig((9, 4))
    ax.plot(tilt[excluded], depth[excluded], ".", markersize=3, color="gray", label="excluded by time.window")
    ax.plot(tilt[within], depth[within], ".", markersize=3, color="tab:blue", label="within limit")
    ax.plot(tilt[exceeds], depth[exceeds], ".", markersize=4, color="red", label="exceeds limit")
    if np.isfinite(tiltmax):
        ax.axvline(tiltmax, color="red", ls="--", lw=1)
    ax.invert_yaxis()
    ax.set_xlabel(f"{instrument} tilt (degrees)")
    ax.set_ylabel("Depth (m)")
    ax.legend(fontsize="small")
    _show(fig)


def _wavelength_exclusion_table(nc: xr.Dataset, existing: list[float]) -> pd.DataFrame:
    """Build the checkbox table's initial contents (Simon: "j'imagine un tableau ou on peut
    cocher les longueurs d'onde a 'Set to NaN'") -- one row per band, showing both methods'
    current Rrs for context, pre-checked from ``existing``. Split out from
    :func:`_wavelength_exclusion_editor` so the row-building logic is unit-testable without a
    live Streamlit runtime (``st.data_editor`` itself needs one)."""
    waves = nc["wavelength"].values
    loess = nc["rrs_0p_loess"].values if "rrs_0p_loess" in nc.data_vars else np.full(waves.shape, np.nan)
    linear = nc["rrs_0p_linear"].values if "rrs_0p_linear" in nc.data_vars else np.full(waves.shape, np.nan)
    return pd.DataFrame(
        {
            "Wavelength (nm)": waves,
            "Rrs (loess)": loess,
            "Rrs (linear)": linear,
            "Exclude (set to NaN)": [
                any(abs(w - existing_w) < 1e-6 for existing_w in existing) for w in waves
            ],
        }
    )


def _wavelength_exclusion_editor(nc: xr.Dataset, existing: list[float], *, key: str) -> list[float]:
    edited = st.data_editor(
        _wavelength_exclusion_table(nc, existing),
        key=key,
        hide_index=True,
        disabled=("Wavelength (nm)", "Rrs (loess)", "Rrs (linear)"),
        use_container_width=True,
    )
    return [float(w) for w in edited.loc[edited["Exclude (set to NaN)"], "Wavelength (nm)"]]


def _render_qc_actions(
    directory: Path,
    nc_dir: Path,
    nc: xr.Dataset,
    raw_path: Path | None,
    raw_ds: xr.Dataset | None,
    depth_is_on: str | None,
    info: CastInfo | None,
    instruments: tuple[str, ...],
    labels: list[str],
    selected: str,
) -> None:
    """Adjust this cast's processing parameters and reprocess it in place, discard it, or
    validate it and move to the next -- the QC loop Simon wants after spotting a problem in the
    diagnostics above (e.g. a linear-fit failure that needs a tighter sub-surface-removed-layer,
    or LuZ data near the bottom that needs the time.window trimmed earlier)."""
    st.divider()
    st.subheader("Adjust & reprocess")

    if raw_path is None or raw_ds is None:
        st.info("Needs the original raw cast file (not found) to adjust and reprocess this cast.")
        return
    if depth_is_on is None:
        st.info("Needs a readable init.cops.dat (not found) to adjust and reprocess this cast.")
        return

    info_path = directory / "info.cops.dat"
    select_path = directory / "select.cops.dat"
    exclusions_path = directory / "rrs_wavelength_exclusions.cops.dat"
    cast_file = raw_path.name

    current_selection = existing_selection(select_path, cast_file)
    default_method = (
        current_selection.method
        if current_selection and current_selection.method in _METHOD_OPTIONS
        else _DEFAULT_METHOD
    )
    shallow = current_selection.shallow if current_selection else False
    flag = current_selection.flag if current_selection else FLAG_NORMAL

    with st.expander("Adjust processing parameters"):
        elapsed = (raw_ds["time"].values - raw_ds["time"].values.min()) / np.timedelta64(1, "s")
        depth = raw_ds[f"{depth_is_on}_Depth"].values
        start, end = render_time_window_editor(
            elapsed,
            depth,
            f"{depth_is_on} depth (m)",
            cast_file,
            info.time_window if info else None,
            key=f"analyze_time_window::{selected}",
        )
        # select.cops.dat's method column, not info.cops.dat -- kept there deliberately (not
        # moved to pycops-only storage) since it's the R package's own 15-years-established
        # location for this exact field, and update_cast_selection()'s surgical per-row edit
        # already tolerates short/old-format rows -- moving it would be a new divergence, not a
        # simplification, for reprocessing older stations.
        method = st.selectbox(
            "Subsurface extrapolation of upwelling Lu/Eu (select.cops.dat's method)",
            _METHOD_OPTIONS,
            index=_METHOD_OPTIONS.index(default_method),
            format_func=lambda k: _METHOD_LABELS[k],
            key=f"analyze_method::{selected}",
        )
        st.caption("Final Rrs wavelength exclusions -- checked bands are set to NaN regardless of method.")
        existing_exclusions = read_wavelength_exclusions(exclusions_path).get(cast_file, [])
        excluded_waves = _wavelength_exclusion_editor(
            nc, existing_exclusions, key=f"analyze_exclude_waves::{selected}"
        )
        override_texts = render_override_fields_editor(
            info, instruments, key_prefix=f"analyze_override::{selected}"
        )

    # Whether any widget above still differs from what's actually saved (and therefore from what
    # the .nc/diagnostics currently reflect) -- found via a real bug report (Simon: picked LOESS,
    # excluded 340 nm, clicked "Validate and next" directly, and the Rrs at 340 nm was still not
    # NaN): "Discard"/"Validate and next" only ever saved `method` -- silently, straight from this
    # still-unreprocessed widget -- while wavelength exclusions/time.window/overrides were dropped
    # entirely and the .nc was never regenerated, so select.cops.dat ended up claiming a method the
    # actual fitted result didn't reflect. Fixed at the root: both buttons below now always reuse
    # the *already-saved* method/shallow (`default_method`/`shallow`), never the live widgets, so
    # neither button can silently commit an unreprocessed change -- this warning is what tells the
    # researcher *why* their edits didn't show up, and that "Reprocess" is the only save mechanism
    # (there's no separate "save without reprocessing": the .nc's numbers depend on these
    # parameters, so any save has to re-fit to stay consistent).
    total_duration = float(elapsed.max())
    saved_start, saved_end = info.time_window if (info and info.time_window is not None) else (0.0, total_duration)
    saved_start = max(0.0, min(saved_start, total_duration))
    saved_end = max(saved_start, min(saved_end, total_duration))
    saved_overrides = {
        attr: format_override(getattr(info, attr) if info else None) for attr, _ in OVERRIDE_FIELDS
    }
    has_unsaved_changes = (
        method != default_method
        or sorted(excluded_waves) != sorted(existing_exclusions)
        or (round(start, 6), round(end, 6)) != (round(saved_start, 6), round(saved_end, 6))
        or override_texts != saved_overrides
    )
    if has_unsaved_changes:
        st.warning(
            "You've changed processing parameters above that haven't been saved yet -- the "
            "diagnostics on this page still reflect the *previous* run. Click 'Reprocess with "
            "adjusted parameters' to save and apply them before validating this cast, or your "
            "edits will be silently dropped."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Reprocess with adjusted parameters", key="analyze_reprocess"):
            try:
                update_cast_info(
                    info_path,
                    cast_file,
                    time_window=(start, end),
                    **parsed_override_fields(override_texts),
                )
            except ValueError as exc:
                st.error(f"Couldn't parse a field: {exc}")
                return
            update_cast_selection(select_path, cast_file, flag, method, shallow=shallow)
            update_wavelength_exclusions(exclusions_path, cast_file, excluded_waves)
            try:
                reprocessed = reprocess_single_cast(directory, cast_file)
            except Exception as exc:  # noqa: BLE001 -- surface any processing failure in the UI
                st.error(f"Reprocessing failed: {exc}")
                return
            write_cast_result(reprocessed.result, nc_dir / f"{selected}.nc", ds=reprocessed.ds)
            st.session_state["analyze_action_message"] = (
                f"Reprocessed {cast_file} -- select.cops.dat, info.cops.dat, and the .nc file "
                "have all been saved with your new parameters (the diagnostics below already "
                "reflect them). Click 'Validate and next' whenever you're satisfied with this cast."
            )
            # st.data_editor keeps its own cached grid state across reruns under the same key --
            # without this, the wavelength-exclusion table's Rrs (loess)/Rrs (linear) preview
            # columns would keep showing pre-reprocess values even though the underlying .nc (and
            # every other plot on the page) is already correctly updated.
            st.session_state.pop(f"analyze_exclude_waves::{selected}", None)
            st.toast(f"Reprocessed {cast_file}")
            st.rerun()
    with col2:
        if st.button("Discard this cast", key="analyze_discard"):
            # Always the already-saved method/shallow, never the live widgets above -- this button
            # only ever touches the QC flag (see the has_unsaved_changes note above).
            update_cast_selection(select_path, cast_file, FLAG_REJECTED, default_method, shallow=shallow)
            st.session_state["analyze_action_message"] = (
                f"Discarded {cast_file} -- select.cops.dat has been updated (QC flag set to rejected)."
            )
            st.toast(f"Discarded {cast_file}")
            st.rerun()
    with col3:
        if st.button(
            "Validate and next", key="analyze_validate_next", disabled=has_unsaved_changes
        ):
            update_cast_selection(select_path, cast_file, FLAG_NORMAL, default_method, shallow=shallow)
            next_idx = min(labels.index(selected) + 1, len(labels) - 1)
            st.session_state["analyze_cast_select_pending"] = labels[next_idx]
            st.session_state["analyze_action_message"] = (
                f"Validated {cast_file} -- select.cops.dat has been updated (QC flag set to normal)."
            )
            st.toast(f"Validated {cast_file}")
            st.rerun()


def _k0_at_adaptive_depth(k0: np.ndarray, depth_grid: np.ndarray, z_interval: np.ndarray) -> np.ndarray:
    """K0(EdZ) sampled at each wavelength's own near-surface linear-fit ``z_interval`` depth, or
    at the depth nearest 2 m for *every* wavelength if any wavelength's ``z_interval`` is NaN --
    port of ``plot.Rrs.Kd.for.station.R``'s exact adaptive-depth/all-or-nothing-fallback logic."""
    if np.any(np.isnan(z_interval)):
        ix = int(np.argmin(np.abs(depth_grid - 2.0)))
        return k0[ix, :]
    values = np.empty(len(z_interval))
    for w in range(len(z_interval)):
        ix = int(np.argmin(np.abs(depth_grid - z_interval[w])))
        values[w] = k0[ix, w]
    return values


def _render_station_comparison(directory: Path) -> None:
    st.caption(
        "Every currently-kept cast (select.cops.dat flag != 0) in this station, overlaid -- "
        "spot an outlier before deciding what to include in the database."
    )

    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        st.error(f"No nc/ subfolder in {directory} -- process this station in tab 3 first.")
        return

    kept_files = kept_nc_files(directory, nc_dir)
    if not kept_files:
        st.warning(f"No kept casts with .nc output found in {nc_dir}.")
        return

    colors = plt.cm.tab20(np.linspace(0, 1, len(kept_files)))

    fig_rrs, ax_rrs = _new_fig((9, 4.5))
    fig_kd, ax_kd = _new_fig((9, 4.5))
    any_rrs = any_kd = False

    for color, nc_path in zip(colors, kept_files):
        with xr.open_dataset(nc_path) as opened:
            nc = opened.load()
        waves = nc["wavelength"].values
        label = nc_path.stem

        if "rrs_0p_loess" in nc.data_vars or "rrs_0p_linear" in nc.data_vars:
            any_rrs = True
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

    if any_rrs:
        st.subheader("Rrs (solid: LOESS, dashed: linear)")
        ax_rrs.set_yscale("log")
        ax_rrs.set_xlabel("Wavelength (nm)")
        ax_rrs.set_ylabel("Rrs(0+)")
        ax_rrs.legend(fontsize="small")
        _show(fig_rrs)
    else:
        plt.close(fig_rrs)

    if any_kd:
        st.subheader("K0(EdZ) (solid: adaptive depth, dashed: near-surface linear)")
        ax_kd.set_xlabel("Wavelength (nm)")
        ax_kd.set_ylabel("K0 (m⁻¹)")
        ax_kd.legend(fontsize="small")
        _show(fig_kd)
    else:
        plt.close(fig_kd)


def render_analyze_tab() -> None:
    directory_input = _directory_input(
        "Deployment folder (must contain an nc/ subfolder)", key="analyze_dir"
    )
    if not directory_input:
        st.info("Enter a deployment folder to begin.")
        return

    directory = Path(directory_input)
    if not directory.is_dir():
        st.error(f"{directory} is not a directory.")
        return

    mode = st.radio(
        "Mode", ("Single cast", "Station comparison (Rrs & Kd)"), key="analyze_mode"
    )
    if mode == "Single cast":
        _render_single_cast(directory)
    else:
        _render_station_comparison(directory)


def _render_single_cast(directory: Path) -> None:
    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        st.error(f"No nc/ subfolder in {directory} -- process this station in tab 3 first.")
        return

    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        st.warning(f"No .nc files found in {nc_dir}.")
        return

    labels = [p.stem for p in nc_files]
    # A widget's own session_state key can't be reassigned after it's been instantiated in the
    # same run -- "Validate and next" (below) stashes its target cast in this separate "_pending"
    # key instead, applied here before the selectbox is created (same trick clean_app.py's own
    # cast selector already uses).
    if "analyze_cast_select_pending" in st.session_state:
        st.session_state["analyze_cast_select"] = st.session_state.pop("analyze_cast_select_pending")
    if st.session_state.get("analyze_cast_select") not in labels:
        st.session_state.pop("analyze_cast_select", None)
    selected = st.selectbox(f"Cast ({len(nc_files)} found)", labels, key="analyze_cast_select")
    nc_path = nc_dir / f"{selected}.nc"

    action_message = st.session_state.pop("analyze_action_message", None)
    if action_message:
        st.success(action_message)

    with xr.open_dataset(nc_path) as opened:
        nc = opened.load()

    init: dict[str, object] | None = None
    depth_is_on = None
    delta_capteur_optics: dict[str, float] = {}
    init_path = directory / "init.cops.dat"
    if init_path.exists():
        try:
            init = read_init_cops(init_path)
            depth_is_on = init["depth.is.on"]
            delta_capteur_optics = init["delta.capteur.optics"]
        except Exception:  # noqa: BLE001 -- only needed for the raw-scan overlay, not fatal
            init = None
            depth_is_on = None

    instruments = _instruments_present(nc)
    raw_path = _find_raw_cast(directory, selected)
    raw_ds = None
    if raw_path is None:
        st.caption("⚠️ Original raw cast file not found -- showing fitted curves only, no raw scan overlay.")
    else:
        try:
            raw_ds = read_cast(raw_path, instruments=("Ed0", *instruments))
        except Exception as exc:  # noqa: BLE001 -- degrade gracefully, .nc content still renders
            st.warning(f"Couldn't reload the raw cast for raw-scan overlay: {exc}")
    if raw_ds is not None and depth_is_on is not None and f"{depth_is_on}_Depth" not in raw_ds:
        raw_ds = None  # can't align raw scans to depth without the reference depth column

    info = existing_info(directory / "info.cops.dat", raw_path.name) if raw_path is not None else None
    time_window = _effective_time_window(init, info) if init is not None else None

    _render_overview(nc)
    _render_ed0_stability(nc)

    if raw_ds is not None and depth_is_on is not None:
        _render_depth_vs_time(raw_ds, depth_is_on, time_window)

    if raw_ds is not None and depth_is_on is not None and init is not None:
        for instrument in ("Ed0", *instruments):
            with st.expander(f"{instrument} tilt"):
                _render_tilt(
                    raw_ds, depth_is_on, delta_capteur_optics.get(instrument), init, info, instrument, time_window
                )

    for instrument in instruments:
        with st.expander(f"{instrument} depth profile", expanded=(instrument == instruments[0])):
            _render_depth_profile(nc, raw_ds, depth_is_on, delta_capteur_optics.get(instrument), instrument)
        with st.expander(f"{instrument} attenuation (K)"):
            _render_attenuation(nc, instrument)

    if "par_d_profile" in nc.data_vars:
        with st.expander("PAR & Kd(PAR)"):
            _render_par_and_kd_par(nc, selected)

    for instrument in _SHADOW_INSTRUMENTS:
        if f"{instrument}_surface_value_at_surface" in nc.data_vars:
            with st.expander(f"{instrument} extrapolation methods (LOESS vs. linear)"):
                _render_extrapolation_comparison(
                    nc, raw_ds, depth_is_on, delta_capteur_optics.get(instrument), instrument
                )

    _render_rrs_spectra(nc)

    for instrument in _SHADOW_INSTRUMENTS:
        if f"{instrument}_shadow_correction" in nc.data_vars:
            with st.expander(f"{instrument} shadow correction"):
                _render_shadow_correction(nc, instrument)

    _render_qwip(nc)

    if nc.attrs.get("shallow"):
        for instrument in _SHADOW_INSTRUMENTS:
            if f"{instrument}_rb" in nc.data_vars:
                with st.expander(f"{instrument} bottom reflectance"):
                    _render_bottom(nc, instrument)

    _render_qc_actions(directory, nc_dir, nc, raw_path, raw_ds, depth_is_on, info, instruments, labels, selected)
