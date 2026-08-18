"""Interactive results-analysis tab (section 4).

Originally built to replace Simon's R workflow's per-cast, multi-page PDF diagnostic report
(``process.cops.R``'s ``pdf(...)``/``dev.off()`` block driving ``process.LuZ.R``/
``process.EdZ.R``/``process.EuZ.R``/``compute.aops.R``'s ``plot()``/``matplot()`` calls) with an
interactive equivalent, browsing the results tab 3 already wrote to ``.nc`` files
(:func:`pycops.io.netcdf.write_deployment_result`) one cast at a time. This is a read-only viewer
over already-computed data -- no new processing happens here (except reprocessing one cast on
demand, see ``_render_qc_actions``).

Every figure here is built by :mod:`pycops.io.pdf_report` (no Streamlit dependency there) and
just displayed with :func:`_show`; this module also now offers a static PDF export of the same
diagnostics (per cast, and a whole-station batch/summary) via that module's
``write_cast_pdf_report``/``write_station_summary_pdf`` -- a portable artifact for records/
collaborators who don't run the app, matching R's own per-cast PDF convention more directly than
this interactive tab alone did.

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
from pycops.io.ed0_correction import read_ed0_correction_methods, update_ed0_correction_method
from pycops.io.exclusions import read_wavelength_exclusions, update_wavelength_exclusions
from pycops.io.netcdf import write_cast_result
from pycops.io.pdf_report import (
    _KD_PAR_MIN_DEPTH_M,
    _effective_time_window,
    _instruments_present,
    _kept_mask,
    _raw_scan_values,
    build_attenuation_figure,
    build_bottom_figure,
    build_depth_profile_figure,
    build_depth_vs_time_figure,
    build_ed0_stability_figure,
    build_par_kd_par_figures,
    build_penetration_depth_comparison_figure,
    build_penetration_depth_figure,
    build_qfactor_figure,
    build_qwip_figure,
    build_rrs_figure,
    build_shadow_correction_figure,
    build_spectral_kd_figure,
    build_station_kd_penetration_depth_figure,
    build_station_par_depth_table,
    build_station_par_profile_figure,
    build_station_comparison_figures,
    build_station_qfactor_figure,
    build_tilt_figure,
    find_raw_cast_for_stem,
    write_cast_pdf_report,
    write_station_pdf_reports,
    write_station_summary_pdf,
)
from pycops.io.raw import read_cast
from pycops.processing.deployment import reprocess_single_cast
from pycops.processing.par import percent_par_at_depth
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

_SHADOW_INSTRUMENTS = ("LuZ", "EuZ")
# Matches discovery.py's own _DEFAULT_METHOD (private there, so not imported directly) --
# clean_app.py already duplicates this same constant for the same reason.
_DEFAULT_METHOD = "Rrs.0p.linear"
_METHOD_OPTIONS = ("Rrs.0p", "Rrs.0p.linear")
_METHOD_LABELS = {"Rrs.0p": "LOESS", "Rrs.0p.linear": "Linear"}
_DEFAULT_ED0_CORRECTION_METHOD = "raw"
_ED0_CORRECTION_METHOD_OPTIONS = ("raw", "smoothed")
_ED0_CORRECTION_METHOD_LABELS = {"raw": "Raw (matches R)", "smoothed": "Smoothed (pycops-only)"}


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


def _render_ed0_stability(nc: xr.Dataset, time_window: tuple[float, float] | None = None) -> None:
    if "ed0_correction" not in nc.data_vars:
        return
    st.subheader("Ed0 stability")
    active_method = nc.attrs.get("ed0_correction_method") or _DEFAULT_ED0_CORRECTION_METHOD
    st.caption(
        "Ratio of the smoothed surface reference to each scan -- flags illumination changes (e.g. "
        "clouds) during the cast. **Raw** (matches the R package) divides by each raw scan; "
        "**smoothed** (pycops-only) divides by the LOESS-smoothed Ed0 at that scan's own time "
        f"instead, for less noise. Currently active for this cast: **"
        f"{_ED0_CORRECTION_METHOD_LABELS.get(active_method, active_method)}** (see 'Adjust & "
        "reprocess' below to change it). Red lines mark the +/-5% acceptable range. Gray shading "
        "(if any) marks time excluded by this cast's time.window."
    )
    _show(build_ed0_stability_figure(nc, time_window))

    active_var = f"ed0_correction_{active_method}"
    active_correction = nc[active_var] if active_var in nc.data_vars else nc["ed0_correction"]
    active_values = active_correction.mean(dim="wavelength").values
    outside = int(np.sum((active_values < 0.95) | (active_values > 1.05)))
    if outside:
        st.warning(
            f"Ed0 correction is outside the +/-5% acceptable range for {outside} scan(s) -- "
            "possible illumination instability (e.g. passing clouds) during this cast."
        )


def _render_depth_profile(
    nc: xr.Dataset,
    raw_ds: xr.Dataset | None,
    depth_is_on: str | None,
    delta_capteur: float | None,
    instrument: str,
) -> None:
    waves = nc["wavelength"].values
    wave_options = ["All"] + [f"{w:g}" for w in waves]
    wavelength_choice = st.selectbox(
        "Wavelength", wave_options, key=f"analyze_{instrument}_depth_wave"
    )
    _show(build_depth_profile_figure(nc, raw_ds, depth_is_on, delta_capteur, instrument, wavelength_choice))


def _render_attenuation(nc: xr.Dataset, instrument: str) -> None:
    _show(build_attenuation_figure(nc, instrument))


def _render_spectral_kd(nc: xr.Dataset) -> None:
    fig = build_spectral_kd_figure(nc)
    if fig is None:
        return
    st.subheader("Spectral Kd")
    st.caption(
        "Mean diffuse attenuation from the surface down to the 1%, 10%, and penetration-depth "
        "(1/e) light levels, one line per level vs. wavelength -- distinct from the K attenuation "
        "panels above, which plot Kd vs. depth for one instrument at a time."
    )
    _show(fig)


def _render_penetration_depth(nc: xr.Dataset) -> None:
    fig = build_penetration_depth_figure(nc)
    if fig is None:
        return
    st.subheader("Penetration depth by wavelength")
    st.caption(
        "Depth (1/e light level) at which each band's own surface signal has attenuated to 1/e -- "
        "0 at the top, markers colored by each wavelength's own approximate true color, to "
        "visualize what a satellite actually \"sees\" per color."
    )
    _show(fig)


def _render_par_and_kd_par(nc: xr.Dataset, selected: str) -> None:
    """Vertical PAR profile + Kd(PAR) -- port of ``compute.PAR.fitted.R``'s own plot, one of the
    richer diagnostics from Simon's R PDF report that pycops didn't have until now. ``PAR.0`` is a
    single scalar here (a vertical reference line), not a depth profile like R's: pycops fits Ed0
    at one point only (see ``par_0``'s own docstring in ``process_cast.py``), and the per-scan
    illumination-change diagnostic R's own ``PAR.0(z)`` plot doubles as is already covered by the
    "Ed0 stability" section above. ``PAR_u`` is computed (see ``.nc``'s ``par_u_profile``) but not
    plotted here -- Simon: "pas vraiment pertinent sur le graph".
    """
    figures = build_par_kd_par_figures(nc)
    _show(figures[0])

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

    if len(figures) > 1:
        _show(figures[1])
    else:
        st.caption(f"Not enough fitted depths below {_KD_PAR_MIN_DEPTH_M:g} m to plot Kd(PAR) vs. depth.")


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

    This is the interactive, single-wavelength drill-down; the PDF report
    (:mod:`pycops.io.pdf_report`) has a separate small-multiples grid version of this same
    diagnostic (:func:`~pycops.io.pdf_report.build_extrapolation_grid_figure`, all wavelengths on
    one page) since per-band detail matters more in a static document than a one-at-a-time picker.
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

    fig, ax = plt.subplots(figsize=(9, 4.5))
    raw = _raw_scan_values(raw_ds, nc, depth_is_on, delta_capteur, instrument, w)
    if raw is not None:
        raw_values, corrected_values, raw_depth = raw
        kept = _kept_mask(nc, raw_ds, instrument, raw_depth)
        near_surface = raw_depth <= max(z_interval[wi] * 1.5, 1.0) if np.isfinite(z_interval[wi]) else np.ones_like(raw_depth, dtype=bool)
        show = kept & near_surface
        plot_values = corrected_values if corrected_values is not None else raw_values
        label = "kept scans (Ed0-corrected)" if corrected_values is not None else "kept scans"
        ax.plot(plot_values[show], raw_depth[show], ".", markersize=4, color="tab:blue", label=label)
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
    _show(build_rrs_figure(nc))


def _render_qfactor(nc: xr.Dataset) -> None:
    if "q_factor_loess" not in nc.data_vars and "q_factor_linear" not in nc.data_vars:
        return
    st.subheader("Empirical Q-factor (Eu(0-) / Lu(0-))")
    st.caption(
        "Measured ratio of EuZ to LuZ surface-extrapolated values (shadow-corrected when "
        "available) -- only available for casts carrying both instruments. A rarely-reported "
        "diagnostic, distinct from the theoretical Q.sun.nadir constant used elsewhere to convert "
        "between the two when only one is present."
    )
    _show(build_qfactor_figure(nc))


def _render_shadow_correction(nc: xr.Dataset, instrument: str) -> None:
    _show(build_shadow_correction_figure(nc, instrument))


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

    _show(build_qwip_figure(nc))


def _render_bottom(nc: xr.Dataset, instrument: str) -> None:
    _show(build_bottom_figure(nc, instrument))

    bottom_depth = nc.attrs.get(f"{instrument}_bottom_depth")
    if bottom_depth is not None and "EdZ_fitted" in nc.data_vars:
        waves = nc["wavelength"].values
        pct_par = percent_par_at_depth(waves, nc["EdZ_fitted"].values, nc["EdZ_depth"].values, bottom_depth)
        if pct_par is not None:
            st.metric("Benthic PAR available (% of surface)", f"{pct_par:.2f}%")


def _render_depth_vs_time(
    raw_ds: xr.Dataset,
    depth_is_on: str,
    time_window: tuple[float, float] | None,
) -> None:
    """Read-only ``depth_is_on`` depth vs. elapsed time, shading any region excluded by the
    cast's currently-saved ``time.window`` -- the diagnostic Simon looks at to spot where tilt/
    depth goes bad near the end of a cast, kept visible without opening "Adjust & reprocess"."""
    st.subheader(f"{depth_is_on} depth vs time")
    _show(build_depth_vs_time_figure(raw_ds, depth_is_on, time_window))


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
    fig = build_tilt_figure(raw_ds, depth_is_on, delta_capteur, init, info, instrument, time_window)
    if fig is None:
        st.warning(f"No Roll/Pitch available for {instrument}.")
        return
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
    init: dict[str, object] | None,
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
    ed0_method_path = directory / "ed0_correction_method.cops.dat"
    cast_file = raw_path.name

    current_selection = existing_selection(select_path, cast_file)
    default_method = (
        current_selection.method
        if current_selection and current_selection.method in _METHOD_OPTIONS
        else _DEFAULT_METHOD
    )
    shallow = current_selection.shallow if current_selection else False
    flag = current_selection.flag if current_selection else FLAG_NORMAL

    station_ed0_method = (init or {}).get("ed0.correction.method", _DEFAULT_ED0_CORRECTION_METHOD)
    if station_ed0_method not in _ED0_CORRECTION_METHOD_OPTIONS:
        station_ed0_method = _DEFAULT_ED0_CORRECTION_METHOD
    existing_ed0_override = read_ed0_correction_methods(ed0_method_path).get(cast_file)
    default_ed0_method = (
        existing_ed0_override if existing_ed0_override in _ED0_CORRECTION_METHOD_OPTIONS else station_ed0_method
    )

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
        ed0_method = st.selectbox(
            "Ed0 illumination correction method (this cast -- 'x' below to use the station "
            f"default, currently {_ED0_CORRECTION_METHOD_LABELS[station_ed0_method]})",
            _ED0_CORRECTION_METHOD_OPTIONS,
            index=_ED0_CORRECTION_METHOD_OPTIONS.index(default_ed0_method),
            format_func=lambda k: _ED0_CORRECTION_METHOD_LABELS[k],
            key=f"analyze_ed0_method::{selected}",
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
        or ed0_method != default_ed0_method
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
            update_ed0_correction_method(ed0_method_path, cast_file, ed0_method)
            try:
                reprocessed = reprocess_single_cast(directory, cast_file)
            except Exception as exc:  # noqa: BLE001 -- surface any processing failure in the UI
                st.error(f"Reprocessing failed: {exc}")
                return
            write_cast_result(reprocessed.result, nc_dir / f"{selected}.nc", ds=reprocessed.ds)
            st.session_state["analyze_action_message"] = (
                f"Reprocessed {cast_file} -- select.cops.dat, info.cops.dat, "
                "ed0_correction_method.cops.dat, and the .nc file have all been saved with your "
                "new parameters (the diagnostics below already reflect them). Click 'Validate and "
                "next' whenever you're satisfied with this cast."
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


def _generate_batch_pdf_reports(directory: Path, kept_files: list[Path]) -> None:
    """Write one PDF report per currently-kept cast -- thin Streamlit wrapper (progress bar +
    success/warning banner) over the pure :func:`~pycops.io.pdf_report.write_station_pdf_reports`,
    which also backs tab 3's own "regenerate PDF reports" action so there's exactly one
    implementation of this loop."""
    progress = st.progress(0.0)
    written, failures = write_station_pdf_reports(
        directory,
        include_station_summary=False,  # this button is per-cast only; the summary has its own button
        progress_callback=lambda i, total: progress.progress(i / total),
    )

    if failures:
        st.warning(f"{written}/{len(kept_files)} PDF report(s) written; failed: " + "; ".join(failures))
    else:
        st.success(f"{written} PDF report(s) written to {directory / 'pdf'}")


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

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Generate station summary PDF", key="analyze_station_summary_pdf"):
            pdf_path = directory / "pdf" / f"{directory.name}_station_summary.pdf"
            write_station_summary_pdf(directory, pdf_path)
            st.success(f"Station summary PDF written to {pdf_path}")
            st.download_button(
                "Download station summary PDF",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                key="analyze_station_pdf_download",
            )
    with col2:
        if st.button("📄 Generate PDF reports for every kept cast", key="analyze_batch_pdf"):
            _generate_batch_pdf_reports(directory, kept_files)

    rrs_figure, k0_figure = build_station_comparison_figures(directory)
    if rrs_figure is not None:
        st.subheader("Rrs (solid: LOESS, dashed: linear)")
        _show(rrs_figure)
    if k0_figure is not None:
        st.subheader("K0(EdZ) (solid: adaptive depth, dashed: near-surface linear)")
        _show(k0_figure)

    par_table = build_station_par_depth_table(directory)
    if not par_table.empty:
        st.subheader("PAR penetration depth (m) by light level, per cast")
        st.dataframe(par_table, hide_index=True)

    par_profile_figure = build_station_par_profile_figure(directory)
    if par_profile_figure is not None:
        st.subheader("PAR profile comparison")
        _show(par_profile_figure)

    kd_penetration_depth_figure = build_station_kd_penetration_depth_figure(directory)
    if kd_penetration_depth_figure is not None:
        st.subheader("Spectral Kd at penetration depth, by cast")
        _show(kd_penetration_depth_figure)

    penetration_depth_figure = build_penetration_depth_comparison_figure(directory)
    if penetration_depth_figure is not None:
        st.subheader("Penetration depth by wavelength, by cast")
        st.caption(
            "Markers colored by each wavelength's own approximate true color; line style "
            "distinguishes casts (color is already used for wavelength)."
        )
        _show(penetration_depth_figure)

    qfactor_figure = build_station_qfactor_figure(directory)
    if qfactor_figure is not None:
        st.subheader("Empirical Q-factor, by cast (Eu(0-) / Lu(0-))")
        _show(qfactor_figure)


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
    raw_path = find_raw_cast_for_stem(directory, selected)
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

    cast_file = raw_path.name if raw_path is not None else selected
    if st.button("📄 Generate PDF report for this cast", key="analyze_pdf_report"):
        pdf_path = directory / "pdf" / f"{selected}.pdf"
        write_cast_pdf_report(
            nc, raw_ds, init, info, cast_file, instruments, depth_is_on, delta_capteur_optics, time_window, pdf_path
        )
        st.session_state["analyze_pdf_report_message"] = f"PDF report written to {pdf_path}"
        st.rerun()
    pdf_report_message = st.session_state.pop("analyze_pdf_report_message", None)
    if pdf_report_message:
        st.success(pdf_report_message)
        pdf_report_path = directory / "pdf" / f"{selected}.pdf"
        if pdf_report_path.exists():
            st.download_button(
                "Download PDF report",
                data=pdf_report_path.read_bytes(),
                file_name=pdf_report_path.name,
                mime="application/pdf",
                key="analyze_pdf_report_download",
            )

    _render_overview(nc)
    _render_ed0_stability(nc, time_window)

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

    _render_spectral_kd(nc)
    _render_penetration_depth(nc)

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
    _render_qfactor(nc)

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

    _render_qc_actions(
        directory, nc_dir, nc, raw_path, raw_ds, depth_is_on, info, instruments, labels, selected, init
    )
