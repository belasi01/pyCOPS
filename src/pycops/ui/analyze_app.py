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
import streamlit as st
import xarray as xr

from pycops.io.config import read_init_cops
from pycops.io.raw import read_cast
from pycops.io.scaffold import discover_l1_casts
from pycops.processing.qwip import _qwip_polynomial
from pycops.ui._common import _directory_input

_DEPTH_INSTRUMENTS = ("EdZ", "LuZ", "EuZ")
_SHADOW_INSTRUMENTS = ("LuZ", "EuZ")


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


def _render_ed0_stability(nc: xr.Dataset) -> None:
    if "ed0_correction" not in nc.data_vars:
        return
    st.subheader("Ed0 stability")
    st.caption("Ratio of the smoothed surface reference to each raw scan -- flags illumination changes (e.g. clouds) during the cast.")
    correction = nc["ed0_correction"]
    fig, ax = _new_fig((9, 3))
    ax.plot(nc["time"].values, correction.mean(dim="wavelength").values, color="tab:orange")
    ax.set_xlabel("Time")
    ax.set_ylabel("Ed0 correction (mean over wavelengths)")
    fig.autofmt_xdate()
    _show(fig)


def _render_depth_profile(nc: xr.Dataset, raw_ds: xr.Dataset | None, instrument: str) -> None:
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
        if raw_ds is not None and instrument in raw_ds and f"{instrument}_Depth" in raw_ds:
            wdim = _wavelength_dim(raw_ds, instrument)
            raw_waves = raw_ds[wdim].values
            nearest_raw_wave = raw_waves[int(np.argmin(np.abs(raw_waves - w)))]
            raw_values = raw_ds[instrument].sel({wdim: nearest_raw_wave}).values
            raw_depth = raw_ds[f"{instrument}_Depth"].values
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


def _render_rrs_spectra(nc: xr.Dataset) -> None:
    if "rrs_0p_loess" not in nc.data_vars and "rrs_0p_linear" not in nc.data_vars:
        return
    st.subheader("Rrs / Lw / nLw spectra")
    waves = nc["wavelength"].values
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, base, label in zip(axes, ("lw_0p", "nlw_0p", "rrs_0p"), ("Lw(0+)", "nLw(0+)", "Rrs(0+)")):
        for method, style in (("loess", "-o"), ("linear", "--s"), ("recommended", ":^")):
            var = f"{base}_{method}"
            if var in nc.data_vars:
                ax.plot(waves, nc[var].values, style, label=method, markersize=4)
        ax.set_yscale("log")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(label)
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
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        with col:
            st.write(f"**{label}**")
            st.metric("AVW (nm)", f"{nc.attrs[f'qwip_{label}_avw']:.1f}")
            st.metric("Score", f"{nc.attrs[f'qwip_{label}_score']:.3f}")
            passed = bool(nc.attrs[f"qwip_{label}_passed"])
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


def _render_bottom(nc: xr.Dataset, instrument: str) -> None:
    waves = nc["wavelength"].values
    fig, ax = _new_fig((9, 3.5))
    ax.plot(waves, nc[f"{instrument}_rb"].values, "-o", label="Rb (~0.3 m above bottom)")
    ax.plot(waves, nc[f"{instrument}_rb_extrapolated"].values, "--s", label="Rb (extrapolated to bottom)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Bottom reflectance ({instrument})")
    ax.legend(fontsize="small")
    bottom_depth = nc.attrs.get(f"{instrument}_bottom_depth")
    if bottom_depth is not None:
        ax.set_title(f"Bottom depth: {bottom_depth:.2f} m")
    _show(fig)


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

    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        st.error(f"No nc/ subfolder in {directory} -- process this station in tab 3 first.")
        return

    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        st.warning(f"No .nc files found in {nc_dir}.")
        return

    labels = [p.stem for p in nc_files]
    selected = st.selectbox(f"Cast ({len(nc_files)} found)", labels, key="analyze_cast_select")
    nc_path = nc_dir / f"{selected}.nc"

    with xr.open_dataset(nc_path) as opened:
        nc = opened.load()

    depth_is_on = None
    init_path = directory / "init.cops.dat"
    if init_path.exists():
        try:
            depth_is_on = read_init_cops(init_path)["depth.is.on"]
        except Exception:  # noqa: BLE001 -- only needed for the raw-scan overlay, not fatal
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

    _render_overview(nc)
    _render_ed0_stability(nc)

    for instrument in instruments:
        with st.expander(f"{instrument} depth profile", expanded=(instrument == instruments[0])):
            _render_depth_profile(nc, raw_ds, instrument)
        with st.expander(f"{instrument} attenuation (K)"):
            _render_attenuation(nc, instrument)

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
