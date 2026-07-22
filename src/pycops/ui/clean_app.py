"""Interactive Streamlit tool covering two steps of Simon's real per-station workflow:

1. **Scaffold a new L2 station folder from L1 raw casts** (see :mod:`pycops.io.scaffold`) --
   ``L1`` is read-only-protected raw data; this only ever reads from it, never modifies it.
2. **Edit each cast's ``info.cops.dat`` row** (position, absorption flag, ``time.window`` trim,
   and per-instrument overrides) **and ``select.cops.dat`` row** (QC flag, Rrs method, SHALLOW) --
   non-destructively, without hand-editing either file as text.

For step 2: Simon's R workflow ends with ``cops.go(clean.files=TRUE)``: plot Depth vs. scan index,
click the start of the good downcast and the end (or, for a shallow cast, where the instrument hit
bottom), and R **overwrites the raw cast file** with just the trimmed rows. This tool reproduces
the same researcher-facing task -- pick a start/end for each cast -- but non-destructively: it
writes the choice into ``info.cops.dat``'s existing ``time.window`` field instead (see
:func:`pycops.io.config.update_cast_info`), which :func:`pycops.processing.process_cast.process_cast`
already applies as an early QC mask across every instrument (matching ``derived.data.R``'s
``Depth.good <- Depth.good & dates.good``) -- ``L2`` cast files themselves are never rewritten.

**Deliberate reinterpretation**: R's ``clean.cops.file()`` plots against raw scan index; this
plots against elapsed time in seconds, matching ``time.window``'s actual units. This is a
documented design choice, not a divergence bug.

Run with ``pycops-clean [deployment folder]`` (after ``uv sync --extra ui`` /
``pip install pycops[ui]``) -- the optional argument pre-fills step 2's deployment-folder field;
step 1 always starts blank. Or directly via ``streamlit run src/pycops/ui/clean_app.py --
[deployment folder]``.
"""

from __future__ import annotations

import html
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pycops.io.config import (
    INIT_COPS_DAT_HELP,
    CastInfo,
    default_init_cops_params,
    parse_optional_float,
    parse_override_field,
    read_info_cops,
    read_init_cops,
    update_cast_info,
    write_init_cops,
)
from pycops.io.discovery import (
    FLAG_BIOSHADE,
    FLAG_NORMAL,
    FLAG_REJECTED,
    FLAG_UNDER_ICE,
    CastSelection,
    discover_deployment,
    find_deployment_folders,
    read_deployment_casts,
    read_select_cops,
    update_cast_selection,
)
from pycops.io.netcdf import write_deployment_result
from pycops.io.raw import parse_cast_filename, read_cast
from pycops.io.scaffold import discover_l1_casts, scaffold_station, validate_station_id
from pycops.processing.deployment import DeploymentProcessingResult, process_deployment
from pycops.processing.position import find_gps_file, position_from_gps, read_gps_file
from pycops.ui._common import _directory_input
from pycops.ui.analyze_app import render_analyze_tab

# Per-instrument info.cops.dat override fields, edited as free-text comma-separated lists (one
# value per instruments.optics entry, or "x" for "use init.cops.dat's default") since they're
# rarely touched cast-by-cast and this matches the format researchers already know from hand-
# editing the file -- see the "Advanced per-instrument overrides" expander in _render_clean_tab().
_OVERRIDE_FIELDS = (
    ("sub_surface_removed_layer", "sub.surface.removed.layer"),
    ("tiltmax", "tiltmax"),
    ("depth_interval_for_smoothing", "depth.interval.for.smoothing"),
    ("linear_r2_threshold", "linear.fit.Rsquared.threshold"),
    ("linear_max_delta_depth", "linear.fit.max.delta.depth"),
)

# select.cops.dat's flag values (see pycops.io.discovery), labeled for the UI.
_FLAG_LABELS = {
    FLAG_REJECTED: "Invalid (0)",
    FLAG_NORMAL: "Valid Light profile (1)",
    FLAG_BIOSHADE: "Bioshade (2)",
    FLAG_UNDER_ICE: "UnderIce profile (3)",
}
# Matches discovery.py's own _DEFAULT_METHOD (private there, so not imported directly). The file
# still stores these R-package-native values ("Rrs.0p"/"Rrs.0p.linear") -- only the UI label
# changes (see _METHOD_LABELS).
_METHOD_OPTIONS = ("Rrs.0p", "Rrs.0p.linear")
_DEFAULT_METHOD = "Rrs.0p.linear"
_METHOD_LABELS = {
    "Rrs.0p": "LOESS (non-linear)",
    "Rrs.0p.linear": "Linear",
}

_TAB_SCAFFOLD = "1. Create a station (L1 -> L2)"
_TAB_CLEAN = "2. Clean casts"
_TAB_PROCESS = "3. Process casts"
_TAB_ANALYZE = "4. Analyze results"

# info.cops.dat's "chl" field selects the absorption model shadow correction uses (see
# pycops.processing.shadow.resolve_absorption): 999 = derived from this cast's own fitted Kd
# (Morel & Maritorena 2001 eq. 8'), >0 = Morel & Maritorena case-1-waters model from a manually
# entered chlorophyll concentration, 0 = a per-wavelength table read from absorption.cops.dat, NA
# = shadow correction skipped entirely. Presented as a dropdown instead of a raw numeric-sentinel
# text field, matching how flag/method are already dropdowns below.
_CHL_MODE_KD = "Kd-derived (default)"
_CHL_MODE_MOREL = "Morel & Maritorena (chlorophyll)"
_CHL_MODE_FILE = "From a file (absorption.cops.dat)"
_CHL_MODE_NONE = "No shadow correction"
_CHL_MODES = (_CHL_MODE_KD, _CHL_MODE_MOREL, _CHL_MODE_FILE, _CHL_MODE_NONE)


def _chl_mode_for_flag(chl_flag: float | None) -> str:
    """Maps an *existing* info.cops.dat row's chl value to its dropdown mode.

    Must faithfully reflect the file's actual content -- the Save button resends whichever mode
    is shown, so this is never allowed to silently reinterpret a value already on disk (e.g.
    turning a real chl=0 into a saved chl=999). The "default for a brand-new cast with no
    info.cops.dat row at all" case (999/Kd-derived, per Simon) is handled separately by the
    caller (`_render_clean_tab`), which only falls back to this function when `info` exists.
    """
    if chl_flag is None:
        return _CHL_MODE_NONE
    if chl_flag == 999:
        return _CHL_MODE_KD
    if chl_flag > 0:
        return _CHL_MODE_MOREL
    # chl_flag == 0: the file's own "read absorption.cops.dat" sentinel.
    return _CHL_MODE_FILE


def _existing_info(info_path: Path, filename: str) -> CastInfo | None:
    if not info_path.exists():
        return None
    for entry in read_info_cops(info_path):
        if entry.file == filename:
            return entry
    return None


def _format_optional_float(value: float | None) -> str:
    # ".10g", not the default 6-sig-fig "g": a plain "g" truncates a real coordinate like
    # -68.11626 down to -68.1163 in the text box, silently losing precision before the user
    # even hits Save.
    return "NA" if value is None else f"{value:.10g}"


def _format_override(value: list[float] | None) -> str:
    if value is None:
        return "x"
    # Matches the file's own "NA" sentinel for a per-value override that doesn't apply
    # (e.g. a linear-fit threshold at the surface Ed0 instrument) -- not Python's "nan" repr.
    return ",".join("NA" if np.isnan(v) else f"{v:.10g}" for v in value)


def _existing_selection(select_path: Path, filename: str) -> CastSelection | None:
    if not select_path.exists():
        return None
    for entry in read_select_cops(select_path):
        if entry.file == filename:
            return entry
    return None


def _cast_position_resolved_approx(cast_path: Path, info: CastInfo | None, gps_table) -> bool:
    """Best-effort check of whether a cast's position can be resolved automatically -- used to
    gate the "Next -> Process casts" button so a researcher can't move on to processing while a
    cast still has no usable position (it would just silently skip shadow correction otherwise).

    An explicit ``info.cops.dat`` lon/lat always counts. Otherwise, a GPS fix within +/-2 minutes
    of the cast's filename-derived start time counts as resolvable -- a cheap stand-in for the
    precise per-scan check (:func:`~pycops.processing.position.position_from_gps` against the
    cast's own ``time`` coordinate) that real processing does, since checking every cast that
    precisely here would mean reading every raw cast file just for this gate.
    """
    if info is not None and info.longitude is not None and info.latitude is not None:
        return True
    if gps_table is None:
        return False
    cast_dt = np.datetime64(parse_cast_filename(cast_path).date)
    margin = np.timedelta64(2, "m")
    return position_from_gps(gps_table, np.array([cast_dt - margin, cast_dt + margin])) is not None


_INIT_GEN_INSTRUMENT_PARAMS = (
    ("tiltmax.optics", "Max. tilt (°)"),
    ("depth.interval.for.smoothing.optics", "Smoothing interval (m)"),
    ("sub.surface.removed.layer.optics", "Sub-surface layer excluded (m)"),
    ("delta.capteur.optics", "Depth offset (m)"),
    ("radius.instrument.optics", "Housing radius (m)"),
    ("linear.fit.Rsquared.threshold.optics", "Min. R² (linear fit)"),
    ("linear.fit.max.delta.depth.optics", "Max. linear-fit window (m)"),
)


def _render_init_cops_dat_generator_form() -> dict[str, object]:
    """Form for a brand-new ``init.cops.dat``, pre-filled with :func:`default_init_cops_params`'s
    typical starting values and one help caption per field (:data:`INIT_COPS_DAT_HELP`)."""
    st.caption(
        "Typical starting values, taken from real deployments -- verify these, especially the "
        "depth offset and housing radius (physical properties of each sensor, see its "
        "calibration sheet)."
    )

    st.write("Optical sensors present")
    st.caption(INIT_COPS_DAT_HELP["instruments.optics"])
    cols = st.columns(4)
    instrument_checked = {"Ed0": True}
    with cols[0]:
        st.checkbox("Ed0", value=True, disabled=True, key="scaffold_init_instr_Ed0")
    for col, instr, default in zip(cols[1:], ("EdZ", "LuZ", "EuZ"), (True, True, False)):
        with col:
            instrument_checked[instr] = st.checkbox(
                instr, value=default, key=f"scaffold_init_instr_{instr}"
            )
    instruments = tuple(instr for instr in ("Ed0", "EdZ", "LuZ", "EuZ") if instrument_checked[instr])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        depth_is_on = st.selectbox(
            "Reference depth sensor",
            instruments,
            index=instruments.index("LuZ") if "LuZ" in instruments else 0,
            key="scaffold_init_depth_is_on",
            help=INIT_COPS_DAT_HELP["depth.is.on"],
        )
    with col_b:
        number_of_fields_before_date = st.number_input(
            "Segments before the date in the file name",
            min_value=1,
            value=3,
            step=1,
            key="scaffold_init_n_fields",
            help=INIT_COPS_DAT_HELP["number.of.fields.before.date"],
        )
    with col_c:
        windspeed_ms = st.number_input(
            "Default wind speed (m/s)",
            min_value=0.0,
            value=4.0,
            key="scaffold_init_windspeed",
            help=INIT_COPS_DAT_HELP["windspeed_ms"],
        )

    params = default_init_cops_params(
        instruments,
        depth_is_on=depth_is_on,
        number_of_fields_before_date=int(number_of_fields_before_date),
        windspeed_ms=windspeed_ms,
    )

    with st.expander("Per-instrument default values (verify these)"):
        _render_instrument_params_table(params, instruments, key_prefix="scaffold_init")

    return params


def _render_instrument_params_table(
    params: dict[str, object], instruments: tuple[str, ...], *, key_prefix: str
) -> None:
    """One editable row per entry in :data:`_INIT_GEN_INSTRUMENT_PARAMS`, one column per
    instrument -- mutates ``params[name][instr]`` in place from each widget's value. Shared by
    the scaffold tab's brand-new-file generator and the clean tab's existing-file editor (see
    ``_render_init_cops_dat_editor``) so both present the same table/help text."""
    header_cols = st.columns([2, *([1] * len(instruments))])
    header_cols[0].write("")
    for col, instr in zip(header_cols[1:], instruments):
        col.write(f"**{instr}**")
    for name, label in _INIT_GEN_INSTRUMENT_PARAMS:
        row_cols = st.columns([2, *([1] * len(instruments))])
        row_cols[0].caption(f"{label}  \n{INIT_COPS_DAT_HELP[name]}")
        for col, instr in zip(row_cols[1:], instruments):
            current_value = params[name][instr]
            with col:
                if np.isnan(current_value):
                    st.text_input(
                        instr,
                        value="NA",
                        key=f"{key_prefix}_{name}_{instr}",
                        label_visibility="collapsed",
                        disabled=True,
                        help="Doesn't apply to the surface sensor (Ed0).",
                    )
                else:
                    params[name][instr] = st.number_input(
                        instr,
                        value=float(current_value),
                        key=f"{key_prefix}_{name}_{instr}",
                        label_visibility="collapsed",
                    )


def _render_init_cops_dat_editor(init: dict[str, object], *, key_prefix: str) -> dict[str, object]:
    """Edit an *existing* ``init.cops.dat``'s values (clean tab), pre-filled from the file
    already on disk rather than from :func:`default_init_cops_params`'s generic defaults.

    ``instruments.optics``/``depth.is.on`` are shown read-only: changing which instruments are
    present reshapes every per-instrument field and the rest of the clean tab (which reads them
    once per run to decide what to plot/edit) -- regenerating the file from tab 1 is the
    supported way to do that, not an in-place edit here.
    """
    instruments = tuple(init["instruments.optics"])
    st.caption(
        f"Instruments: {', '.join(instruments)} -- depth reference: {init['depth.is.on']}. "
        "To change which instruments are present, regenerate this file from tab 1."
    )

    params = dict(init)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        params["number.of.fields.before.date"] = st.number_input(
            "Segments before the date in the file name",
            min_value=1,
            value=int(init["number.of.fields.before.date"]),
            step=1,
            key=f"{key_prefix}_n_fields",
            help=INIT_COPS_DAT_HELP["number.of.fields.before.date"],
        )
    with col_b:
        params["windspeed_ms"] = st.number_input(
            "Default wind speed (m/s)",
            min_value=0.0,
            value=float(init["windspeed_ms"]),
            key=f"{key_prefix}_windspeed",
            help=INIT_COPS_DAT_HELP["windspeed_ms"],
        )
    with col_c:
        params["bandwidth"] = st.number_input(
            "Bandwidth (nm)",
            min_value=0.0,
            value=float(init["bandwidth"]),
            key=f"{key_prefix}_bandwidth",
            help=INIT_COPS_DAT_HELP["bandwidth"],
        )

    _render_instrument_params_table(params, instruments, key_prefix=key_prefix)
    return params


def _render_scaffold_tab() -> None:
    st.caption(
        "Copies the chosen casts from an L1 folder (read-only, never modified) into a new "
        "L2/YYYYMMDD_StationXXX/cops/ station folder."
    )

    l1_input = _directory_input("Source L1 folder", key="scaffold_l1")
    if not l1_input:
        st.info("Enter an L1 folder to begin.")
        return

    l1_dir = Path(l1_input)
    if not l1_dir.is_dir():
        st.error(f"{l1_dir} is not a directory.")
        return

    casts = discover_l1_casts(l1_dir)
    if not casts:
        st.warning(f"No *_URC.tsv/.csv file found in {l1_dir}.")
        return

    st.write(f"{len(casts)} cast(s) found -- check the ones that belong to this station:")
    selected: list[str] = []
    for cast_path in casts:
        info = parse_cast_filename(cast_path)
        label = f"{cast_path.name} ({info.date.strftime('%Y-%m-%d %H:%M')})"
        if st.checkbox(label, value=False, key=f"scaffold_cast_{cast_path.name}"):
            selected.append(cast_path.name)

    station_id_input = st.text_input("Station ID (e.g. MAN-F05)", key="scaffold_station_id")
    l2_parent_input = _directory_input(
        "Parent L2 folder (the new station folder will be created inside it)", key="scaffold_l2_parent"
    )
    st.markdown("**init.cops.dat** (processing parameters, one per instrument system)")
    init_mode = st.radio(
        "How should init.cops.dat be obtained for this station?",
        ("Do nothing", "Copy from an existing station", "Generate a new file"),
        key="scaffold_init_mode",
        help="This file rarely changes from one deployment to the next for the same instrument "
        "system -- copying an existing file is usually faster than re-entering everything. Only "
        "choose 'Generate' for a brand-new instrument system.",
    )
    init_template_input = ""
    init_params: dict[str, object] | None = None
    if init_mode == "Copy from an existing station":
        init_template_input = st.text_input(
            "Path to an existing init.cops.dat", key="scaffold_init_template"
        )
    elif init_mode == "Generate a new file":
        init_params = _render_init_cops_dat_generator_form()

    overwrite = st.checkbox("Overwrite files already present at the destination", key="scaffold_overwrite")

    if selected and l2_parent_input:
        try:
            preview_date = parse_cast_filename(l1_dir / selected[0]).date.date()
            preview_id = validate_station_id(station_id_input) if station_id_input.strip() else "???"
            st.caption(
                f"Destination: {Path(l2_parent_input) / f'{preview_date:%Y%m%d}_Station{preview_id}' / 'cops'}"
            )
        except ValueError:
            pass

    if st.button("Create station", key="scaffold_create"):
        if not selected:
            st.error("Select at least one cast.")
            return
        if not l2_parent_input:
            st.error("Enter the parent L2 folder.")
            return
        try:
            result = scaffold_station(
                l1_dir,
                l2_parent_input,
                station_id_input,
                selected,
                init_cops_dat_template=init_template_input or None,
                overwrite=overwrite,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        has_warning = bool(result.skipped_existing)
        if init_params is not None:
            init_dest = result.destination / "init.cops.dat"
            if init_dest.exists() and not overwrite:
                st.warning(
                    f"init.cops.dat already exists at {init_dest} -- not overwritten (check "
                    "'Overwrite' to force it)."
                )
                has_warning = True
            else:
                write_init_cops(init_dest, init_params, overwrite=True)
                st.write("init.cops.dat generated from the form.")

        st.success(f"Station created: {result.destination}")
        st.write(f"Casts copied ({len(result.copied_casts)}): {result.copied_casts}")
        if result.copied_gps_files:
            st.write(f"GPS file(s) copied: {result.copied_gps_files}")
        if result.copied_init:
            st.write("init.cops.dat copied from the supplied template.")
        if result.skipped_existing:
            st.warning(
                f"Already present at the destination, not overwritten (check 'Overwrite' to "
                f"force it): {result.skipped_existing}"
            )

        if not has_warning:
            # Nothing needs a second look -- jump straight to the cleaning tab instead of
            # making the researcher go find the folder they just created (st.toast survives
            # this rerun, unlike st.success/st.write, so the confirmation isn't lost).
            st.session_state["clean_dir"] = str(result.destination)
            st.session_state.pop("clean_cast_select", None)
            st.session_state["active_tab_pending"] = _TAB_CLEAN
            st.toast(f"Station created: {result.destination}", icon="✅")
            st.rerun()


def _render_clean_tab() -> None:
    default_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    directory_input = _directory_input("Deployment folder (L2/.../COPS*)", key="clean_dir", default=default_dir)
    if not directory_input:
        st.info("Enter a deployment folder to begin.")
        return

    directory = Path(directory_input)
    if not directory.is_dir():
        st.error(f"{directory} is not a directory.")
        return

    init_path = directory / "init.cops.dat"
    if not init_path.exists():
        st.error(
            f"No init.cops.dat in {directory}. This deployment's instrument config must already "
            "exist (it's set up once per instrument system) -- create it before cleaning casts."
        )
        return
    init = read_init_cops(init_path)
    depth_is_on = init["depth.is.on"]
    instruments = tuple(init["instruments.optics"])

    with st.expander("init.cops.dat"):
        edited_init = _render_init_cops_dat_editor(init, key_prefix="clean_init")
        if st.button("Save", key="clean_init_save"):
            write_init_cops(init_path, edited_init, overwrite=True)
            st.toast("Saved init.cops.dat")
            st.rerun()

    casts = discover_l1_casts(directory)
    if not casts:
        st.warning(f"No *_URC.tsv/.csv cast files found in {directory}.")
        return

    info_path = directory / "info.cops.dat"
    select_path = directory / "select.cops.dat"

    gps_path = find_gps_file(directory)
    gps_table = None
    if gps_path is not None:
        try:
            gps_table = read_gps_file(gps_path)
        except Exception:  # noqa: BLE001 -- a bad GPS file just means "no GPS fallback" here
            gps_table = None

    labels = [p.name for p in casts]
    # A widget's own session_state key can't be reassigned after it's been instantiated in the
    # same run (Streamlit raises) -- so "advance to the next cast" (the Save button,
    # below) stashes its target in this separate "_pending" key instead, applied here, *before*
    # the selectbox is created. Same trick _directory_input() already uses for its own browser.
    if "clean_cast_select_pending" in st.session_state:
        st.session_state["clean_cast_select"] = st.session_state.pop("clean_cast_select_pending")
    st.session_state.setdefault("clean_cast_select", labels[0])
    if st.session_state["clean_cast_select"] not in labels:
        st.session_state["clean_cast_select"] = labels[0]

    selected_label = st.selectbox(f"Cast ({len(casts)} found)", labels, key="clean_cast_select")
    cast_idx = labels.index(selected_label)
    cast_path = casts[cast_idx]
    # Every per-cast widget below is keyed with this suffix so switching casts (via this
    # selectbox, or the "Save" button below) always shows *that* cast's own saved
    # values -- a fixed key shared across casts would otherwise stick to whatever was last typed
    # (Streamlit only honors a widget's `value=` the first time a given key appears).
    ck = lambda base: f"{base}::{cast_path.name}"  # noqa: E731

    try:
        ds = read_cast(cast_path, instruments=instruments)
    except Exception as exc:  # noqa: BLE001 -- surface any read failure in the UI, don't crash the app
        st.error(f"Failed to read {cast_path.name}: {exc}")
        return

    depth_var = f"{depth_is_on}_Depth"
    if depth_var not in ds:
        st.error(f"{cast_path.name} has no {depth_var!r} (depth.is.on={depth_is_on!r}).")
        return

    time = ds["time"].values
    elapsed = (time - time.min()) / np.timedelta64(1, "s")
    depth = ds[depth_var].values
    total_duration = float(elapsed.max())

    info = _existing_info(info_path, cast_path.name)

    st.subheader("Position & absorption")
    col_lon, col_lat, col_chl = st.columns(3)
    with col_lon:
        lon_text = st.text_input(
            "Longitude (deg, or NA)",
            value=_format_optional_float(info.longitude if info else None),
            key=ck("clean_lon"),
        )
    with col_lat:
        lat_text = st.text_input(
            "Latitude (deg, or NA)",
            value=_format_optional_float(info.latitude if info else None),
            key=ck("clean_lat"),
        )

    if parse_optional_float(lon_text) is None or parse_optional_float(lat_text) is None:
        # Actually resolve against the GPS file's real fixes (not just "a GPS file exists") --
        # a GPS file that doesn't cover this cast's time (e.g. the logger was off, or this cast
        # is a same-day revisit after the logger stopped) must not be reported as if position
        # were handled automatically; Simon: "on ne doit pas être en mesure de passer au
        # processing sans une position lat/lon."
        resolved_via_gps = gps_table is not None and position_from_gps(gps_table, ds["time"].values) is not None
        if resolved_via_gps:
            st.caption(
                f"📍 Position not entered above -- resolved automatically from the GPS file "
                f"found in this folder ({gps_path.name}), at the cast's time."
            )
        elif gps_path is not None:
            st.markdown(
                f"<span style='color:red; font-size:1.15rem;'>⚠️ Position not entered, and the "
                f"GPS file found in this folder ({html.escape(gps_path.name)}) has no fix "
                f"covering this cast's time -- shadow correction will be skipped unless a "
                f"position is entered manually.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span style='color:red; font-size:1.15rem;'>⚠️ Position not entered and no "
                "GPS_*.tsv/.csv file found in this folder -- shadow correction will be skipped "
                "unless a position is entered manually.</span>",
                unsafe_allow_html=True,
            )
    elif info is None or info.longitude != parse_optional_float(lon_text) or info.latitude != parse_optional_float(
        lat_text
    ):
        # The warning above reacts to what's typed, not what's on disk -- without this, a
        # researcher who types a valid position but never clicks Save sees the warning vanish
        # and (reasonably) assumes it's already handled, even though "Next -> Process casts"
        # below still won't appear until it's actually persisted.
        st.caption("📍 Position entered above but not saved yet -- click Save below to record it.")

    with col_chl:
        default_chl_mode = _chl_mode_for_flag(info.chl_flag if info else None) if info else _CHL_MODE_KD
        if default_chl_mode not in _CHL_MODES:
            default_chl_mode = _CHL_MODE_KD
        chl_mode = st.selectbox(
            "Shadow correction (absorption)",
            _CHL_MODES,
            index=_CHL_MODES.index(default_chl_mode),
            key=ck("clean_chl_mode"),
            help="Absorption model used for the LuZ/EuZ shadow correction.",
        )

    if chl_mode == _CHL_MODE_NONE:
        chl_text = "NA"
    elif chl_mode == _CHL_MODE_KD:
        chl_text = "999"
    elif chl_mode == _CHL_MODE_FILE:
        chl_text = "0"
        absorption_path = directory / "absorption.cops.dat"
        if absorption_path.exists():
            st.caption(f"Uses {absorption_path.name} (already present in this folder).")
        else:
            st.caption(
                "⚠️ No absorption.cops.dat found in this folder. File selection is coming later -- "
                "for now, place one manually in the station's folder."
            )
    else:  # _CHL_MODE_MOREL
        default_conc = info.chl_flag if (info and info.chl_flag and info.chl_flag > 0) else 1.0
        chl_text = st.text_input(
            "Chlorophyll concentration (mg/m³)",
            value=_format_optional_float(default_conc),
            key=ck("clean_chl_conc"),
        )

    existing_time_window = info.time_window if info else None
    default_start, default_end = existing_time_window if existing_time_window is not None else (0.0, total_duration)
    default_start = max(0.0, min(default_start, total_duration))
    default_end = max(default_start, min(default_end, total_duration))

    start, end = st.slider(
        "Time window kept (seconds elapsed from first scan)",
        min_value=0.0,
        max_value=total_duration,
        value=(default_start, default_end),
        step=max(total_duration / 500, 0.05),
        key=ck("clean_time_window"),
    )

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(elapsed, depth, ".", markersize=2, color="tab:blue")
    ax.axvspan(0, start, color="gray", alpha=0.3)
    ax.axvspan(end, total_duration, color="gray", alpha=0.3)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(f"{depth_is_on} depth (m)")
    ax.invert_yaxis()
    ax.set_title(cast_path.name)
    st.pyplot(fig)
    plt.close(fig)

    existing_selection = _existing_selection(select_path, cast_path.name)
    default_flag = existing_selection.flag if existing_selection else FLAG_NORMAL
    if default_flag not in _FLAG_LABELS:
        default_flag = FLAG_NORMAL
    default_method = existing_selection.method if existing_selection else _DEFAULT_METHOD
    if default_method not in _METHOD_OPTIONS:
        default_method = _DEFAULT_METHOD
    default_shallow = existing_selection.shallow if existing_selection else False

    col1, col2, col3 = st.columns(3)
    with col1:
        flag_keys = list(_FLAG_LABELS)
        flag = st.selectbox(
            "QC flag",
            flag_keys,
            index=flag_keys.index(default_flag),
            format_func=lambda k: _FLAG_LABELS[k],
            key=ck("clean_flag"),
        )
    with col2:
        method = st.selectbox(
            "Subsurface extrapolation of upwelling Lu/Eu",
            _METHOD_OPTIONS,
            index=_METHOD_OPTIONS.index(default_method),
            format_func=lambda k: _METHOD_LABELS[k],
            key=ck("clean_method"),
        )
    with col3:
        shallow = st.checkbox(
            "Shallow (profile ends near bottom)", value=default_shallow, key=ck("clean_shallow")
        )

    override_texts: dict[str, str] = {}
    with st.expander("Advanced per-instrument overrides (rarely changed)"):
        st.caption(
            f"Comma-separated values, one per instrument ({', '.join(instruments)}), or 'x' to "
            "keep init.cops.dat's default value."
        )
        for attr, label in _OVERRIDE_FIELDS:
            override_texts[attr] = st.text_input(
                label,
                value=_format_override(getattr(info, attr) if info else None),
                key=ck(f"clean_override_{attr}"),
            )

    if st.button("Save", key="clean_save"):
        try:
            update_cast_info(
                info_path,
                cast_path.name,
                longitude=parse_optional_float(lon_text),
                latitude=parse_optional_float(lat_text),
                chl_flag=parse_optional_float(chl_text),
                time_window=(start, end),
                **{attr: parse_override_field(override_texts[attr]) for attr, _ in _OVERRIDE_FIELDS},
            )
        except ValueError as exc:
            st.error(f"Couldn't parse a field: {exc}")
            return
        update_cast_selection(select_path, cast_path.name, flag, method, shallow=shallow)
        # A plain st.success() here would be discarded by the st.rerun() below before the
        # browser ever renders it (this was previously read as the button "not doing anything");
        # st.toast() is explicitly designed to survive one rerun.
        st.toast(f"Saved info.cops.dat and select.cops.dat rows for {cast_path.name}")
        # Advance by writing the selectbox's own key (not a separate index variable) -- that's
        # the only way Streamlit actually honors a programmatic change to a stateful widget.
        next_idx = min(cast_idx + 1, len(casts) - 1)
        st.session_state["clean_cast_select_pending"] = labels[next_idx]
        st.rerun()

    # Bottom of the tab, not next to the per-cast Save button above: this jumps between
    # *stations*, not casts, and Simon may want to prepare several stations' casts before
    # processing any of them, so it deliberately doesn't nudge the researcher forward until every
    # cast in this folder already has both a saved info.cops.dat and select.cops.dat row.
    all_cleaned = all(
        _existing_info(info_path, c.name) is not None and _existing_selection(select_path, c.name) is not None
        for c in casts
    )
    if all_cleaned:
        st.divider()
        # Hard gate, not just an advisory note: a cast with no resolvable position would
        # silently process with shadow correction skipped -- Simon wants that caught here,
        # before processing, not discovered later in a shadow_correction_note.
        missing_position = [
            c.name for c in casts if not _cast_position_resolved_approx(c, _existing_info(info_path, c.name), gps_table)
        ]
        if missing_position:
            st.error(
                "Can't move to processing yet -- position unresolved for: "
                f"{', '.join(missing_position)}. Enter a longitude/latitude for each above."
            )
        else:
            st.success("Every cast in this folder has been cleaned, and every position resolves.")
            col_next, col_another = st.columns(2)
            with col_next:
                if st.button("Next -> Process casts", key="clean_next_to_process", use_container_width=True):
                    st.session_state["process_dir"] = str(directory)
                    st.session_state["active_tab_pending"] = _TAB_PROCESS
                    st.rerun()
            with col_another:
                # Supports preparing several stations before processing any of them (batch mode
                # in tab 3) -- goes back to tab 1 rather than tab 3, folder fields left blank
                # there since scaffolding a new station starts from a fresh L1 selection.
                if st.button(
                    "Save and prepare another station", key="clean_prepare_another", use_container_width=True
                ):
                    st.session_state["active_tab_pending"] = _TAB_SCAFFOLD
                    st.rerun()


@dataclass
class _ProcessSummary:
    """Outcome of running :func:`process_deployment` + :func:`write_deployment_result` on one
    deployment folder, plus whatever :mod:`warnings` were raised along the way -- enough to
    render one row/expander in the process tab for either a single folder or one entry of a
    batch."""

    directory: Path
    result: DeploymentProcessingResult | None  # None if a deployment-level error occurred
    output_dir: Path
    n_written: int
    captured_warnings: list[str]
    error: str | None  # deployment-level failure (e.g. missing/malformed config), if any


def _process_one_deployment(directory: Path) -> _ProcessSummary:
    """Read, process, and write NetCDF output for one deployment folder.

    A failure before/outside ``process_deployment``'s own per-cast isolation (e.g.
    ``info.cops.dat`` missing entirely, so ``discover_deployment`` itself raises) is caught here
    and recorded as ``_ProcessSummary.error`` instead of propagating -- so one broken folder in a
    batch can't abort the rest, mirroring the per-cast failure isolation already built into
    ``process_deployment``/``read_deployment_casts`` one level down.
    """
    output_dir = directory / "nc"
    captured_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deployment = discover_deployment(directory)
            read_result = read_deployment_casts(deployment)
            result = process_deployment(directory)
            written = write_deployment_result(result, output_dir, datasets=read_result.datasets)
            captured_warnings = [str(w.message) for w in caught]
    except Exception as exc:  # noqa: BLE001 -- isolate one bad deployment from the rest of a batch
        return _ProcessSummary(
            directory=directory,
            result=None,
            output_dir=output_dir,
            n_written=0,
            captured_warnings=captured_warnings,
            error=f"{type(exc).__name__}: {exc}",
        )
    return _ProcessSummary(
        directory=directory,
        result=result,
        output_dir=output_dir,
        n_written=len(written),
        captured_warnings=captured_warnings,
        error=None,
    )


def _render_process_summary(label: str, summary: _ProcessSummary, *, expanded: bool) -> None:
    if summary.error is not None:
        icon = "❌"
    elif summary.result and (
        summary.result.read_failures or summary.result.processing_failures or summary.captured_warnings
    ):
        icon = "⚠️"
    else:
        icon = "✅"

    with st.expander(f"{icon} {label}", expanded=expanded):
        if summary.error is not None:
            st.error(summary.error)
            return

        result = summary.result
        assert result is not None
        st.write(
            f"{len(result.cast_results)} cast(s) processed -> `{summary.output_dir}` "
            f"({summary.n_written} file(s) written)"
        )
        if result.read_failures:
            for failure in result.read_failures:
                st.warning(f"Couldn't read {failure.file}: {failure.error}")
        if result.processing_failures:
            for failure in result.processing_failures:
                st.warning(f"Couldn't process {failure.file}: {failure.error}")
        if result.cast_results:
            st.dataframe(
                {
                    "cast": list(result.cast_results.keys()),
                    "Rrs source": [r.rrs_source or "-" for r in result.cast_results.values()],
                    "shadow correction note": [
                        r.shadow_correction_note or "" for r in result.cast_results.values()
                    ],
                },
                width="stretch",
                hide_index=True,
            )
        if summary.captured_warnings:
            with st.expander("Warnings", expanded=False):
                for message in summary.captured_warnings:
                    st.caption(message)


def _render_process_tab() -> None:
    mode = st.radio(
        "Mode", ("Single deployment", "Batch (multiple deployments)"), key="process_mode"
    )

    if mode == "Single deployment":
        directory_input = _directory_input(
            "Deployment folder (L2/.../COPS*)", key="process_dir"
        )
        if not directory_input:
            st.info("Enter a deployment folder to begin.")
            return
        directory = Path(directory_input)
        if not directory.is_dir():
            st.error(f"{directory} is not a directory.")
            return
        if not (directory / "init.cops.dat").exists():
            st.error(
                f"No init.cops.dat in {directory}. This deployment's instrument config must "
                "already exist (it's set up once per instrument system) -- create it before "
                "processing casts."
            )
            return

        if st.button("Process", key="process_single_run"):
            with st.spinner(f"Processing {directory.name}..."):
                summary = _process_one_deployment(directory)
            _render_process_summary(directory.name, summary, expanded=True)
        return

    # Batch mode.
    parent_input = _directory_input("Parent folder", key="process_parent")
    if not parent_input:
        st.info("Enter a parent folder to begin (searched recursively for init.cops.dat).")
        return
    parent = Path(parent_input)
    if not parent.is_dir():
        st.error(f"{parent} is not a directory.")
        return

    folders = find_deployment_folders(parent)
    if not folders:
        st.warning(f"No deployment folder (with init.cops.dat) found under {parent}.")
        return

    st.write(f"{len(folders)} deployment folder(s) found -- uncheck any to exclude them:")
    checked: list[Path] = []
    for folder in folders:
        rel = folder.relative_to(parent)
        if st.checkbox(str(rel), value=True, key=f"process_batch_{rel}"):
            checked.append(folder)

    if st.button(f"Process checked deployments ({len(checked)})", key="process_batch_run"):
        if not checked:
            st.error("Select at least one deployment.")
            return
        progress = st.progress(0.0)
        status = st.empty()
        n_ok = n_warn = n_failed = 0
        for i, folder in enumerate(checked):
            rel = folder.relative_to(parent)
            status.write(f"Processing {rel} ({i + 1}/{len(checked)})...")
            summary = _process_one_deployment(folder)
            _render_process_summary(str(rel), summary, expanded=False)
            if summary.error is not None:
                n_failed += 1
            elif summary.result and (summary.result.read_failures or summary.result.processing_failures):
                n_warn += 1
            else:
                n_ok += 1
            progress.progress((i + 1) / len(checked))
        status.write(
            f"Done: {n_ok} ok, {n_warn} with warnings, {n_failed} failed "
            f"(of {len(checked)} processed)."
        )


def run_app() -> None:
    st.set_page_config(page_title="pycops -- cast cleaning", layout="wide")
    st.title("pycops -- interactive station tools")

    # key + on_change="rerun" makes the active tab a real, programmatically settable piece of
    # state (st.session_state["active_tab"]) -- used by _render_scaffold_tab() to jump straight to
    # the cleaning tab (folder pre-filled) right after a station is created, and by
    # _render_clean_tab() to jump to the process tab (folder pre-filled) once every cast in the
    # folder has been cleaned. Applied via a "_pending" key (see clean_cast_select_pending above)
    # since active_tab itself can't be reassigned after st.tabs() below has already instantiated
    # it for this run.
    if "active_tab_pending" in st.session_state:
        st.session_state["active_tab"] = st.session_state.pop("active_tab_pending")
    tab_scaffold, tab_clean, tab_process, tab_analyze = st.tabs(
        [_TAB_SCAFFOLD, _TAB_CLEAN, _TAB_PROCESS, _TAB_ANALYZE], key="active_tab", on_change="rerun"
    )
    with tab_scaffold:
        _render_scaffold_tab()
    with tab_clean:
        st.caption(
            "Non-destructive: writes info.cops.dat's time.window field and a select.cops.dat "
            "row, never rewrites the cast file."
        )
        _render_clean_tab()
    with tab_process:
        st.caption(
            "Runs the full processing pipeline and writes one NetCDF file per cast into an "
            "nc/ subfolder of each deployment folder -- overwriting any nc/ output already "
            "there from a previous run."
        )
        _render_process_tab()
    with tab_analyze:
        st.caption(
            "Read-only: browse a cast's already-processed results (nc/), one cast at a time."
        )
        render_analyze_tab()


def main() -> None:
    """Console-script entry point: shells out to ``streamlit run`` on this file."""
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__)), "--", *sys.argv[1:]],
        check=False,
    )


if __name__ == "__main__":
    run_app()
