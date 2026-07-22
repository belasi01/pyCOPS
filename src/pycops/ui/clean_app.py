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
    read_select_cops,
    update_cast_selection,
)
from pycops.io.raw import parse_cast_filename, read_cast
from pycops.io.scaffold import discover_l1_casts, scaffold_station, validate_station_id
from pycops.processing.position import find_gps_file

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


def _list_subdirs(directory: Path) -> list[str]:
    try:
        return sorted(p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))
    except (PermissionError, OSError):
        return []


def _directory_browser(key: str, chosen_key: str) -> None:
    """An in-app directory browser, rendered entirely with Streamlit widgets.

    A native OS dialog (``tkinter``) was tried first but crashes the whole app on macOS: Streamlit
    runs script logic on a worker thread, while Tk/AppKit requires windows to be created on the
    main thread (confirmed by a real crash -- ``NSInternalInconsistencyException: NSWindow should
    only be instantiated on the main thread!``). This sidesteps that entirely -- no native GUI
    calls, so no threading constraint to violate.

    ``chosen_key`` (not ``key`` itself) is where the final pick is written: Streamlit forbids
    writing to a widget's own ``session_state`` key after that widget has already been
    instantiated earlier in the same run (which ``key``'s own text input always has been, by the
    time this browser renders inside the popover) -- see :func:`_directory_input`, which applies
    ``chosen_key`` to ``key`` at the very start of the *next* run, before the widget exists yet.
    """
    browse_key = f"{key}_browse_path"
    current = st.session_state.get(browse_key) or st.session_state.get(key) or str(Path.home())
    current_path = Path(current) if current else Path.home()
    if not current_path.is_dir():
        current_path = Path.home()

    st.caption(f"📂 {current_path}")

    if st.button("⬆️ Parent folder", key=f"{key}_up", disabled=current_path.parent == current_path):
        st.session_state[browse_key] = str(current_path.parent)
        st.rerun()

    subdirs = _list_subdirs(current_path)
    if not subdirs:
        st.caption("(no subfolders)")
    for sub in subdirs:
        if st.button(f"📁 {sub}", key=f"{key}_sub_{sub}", use_container_width=True):
            st.session_state[browse_key] = str(current_path / sub)
            st.rerun()

    st.divider()
    if st.button("✅ Choose this folder", key=f"{key}_choose", type="primary"):
        st.session_state[chosen_key] = str(current_path)
        st.session_state.pop(browse_key, None)
        st.rerun()


def _directory_input(label: str, key: str, default: str = "") -> str:
    """A text input paired with a popover-based in-app directory browser (see :func:`_directory_browser`)."""
    chosen_key = f"{key}_chosen"
    if chosen_key in st.session_state:
        st.session_state[key] = st.session_state.pop(chosen_key)

    col_text, col_button = st.columns([5, 1])
    with col_text:
        value = st.text_input(label, value=default, key=key)
    with col_button:
        st.write("")  # vertical spacer, roughly aligns the popover button with the text field
        with st.popover("📁 Browse"):
            _directory_browser(key, chosen_key)
    return value


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
        header_cols = st.columns([2, *([1] * len(instruments))])
        header_cols[0].write("")
        for col, instr in zip(header_cols[1:], instruments):
            col.write(f"**{instr}**")
        for name, label in _INIT_GEN_INSTRUMENT_PARAMS:
            row_cols = st.columns([2, *([1] * len(instruments))])
            row_cols[0].caption(f"{label}  \n{INIT_COPS_DAT_HELP[name]}")
            for col, instr in zip(row_cols[1:], instruments):
                default_value = params[name][instr]
                with col:
                    if np.isnan(default_value):
                        st.text_input(
                            instr,
                            value="NA",
                            key=f"scaffold_init_{name}_{instr}",
                            label_visibility="collapsed",
                            disabled=True,
                            help="Doesn't apply to the surface sensor (Ed0).",
                        )
                    else:
                        params[name][instr] = st.number_input(
                            instr,
                            value=float(default_value),
                            key=f"scaffold_init_{name}_{instr}",
                            label_visibility="collapsed",
                        )

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

    casts = discover_l1_casts(directory)
    if not casts:
        st.warning(f"No *_URC.tsv/.csv cast files found in {directory}.")
        return

    labels = [p.name for p in casts]
    # A widget's own session_state key can't be reassigned after it's been instantiated in the
    # same run (Streamlit raises) -- so "advance to the next cast" (the Save && next button,
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
    # selectbox, or the "Save && next" button below) always shows *that* cast's own saved
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

    info_path = directory / "info.cops.dat"
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
        gps_file = find_gps_file(directory)
        if gps_file is not None:
            st.markdown(
                f"<span style='color:red; font-size:1.15rem;'>📍 Position not entered above -- "
                f"will be taken automatically from the GPS file found in this folder "
                f"({html.escape(gps_file.name)}), at the cast's time.</span>",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "⚠️ Position not entered and no GPS_*.tsv/.csv file found in this folder -- shadow "
                "correction will be disabled for this cast until a position is available."
            )

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

    select_path = directory / "select.cops.dat"
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

    if st.button("Save && next", key="clean_save"):
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


def run_app() -> None:
    st.set_page_config(page_title="pycops -- cast cleaning", layout="wide")
    st.title("pycops -- interactive station tools")

    # key + on_change="rerun" makes the active tab a real, programmatically settable piece of
    # state (st.session_state["active_tab"]) -- used by _render_scaffold_tab() to jump straight
    # to the cleaning tab, with its folder pre-filled, right after a station is created. Applied
    # via a "_pending" key (see clean_cast_select_pending above) since active_tab itself can't be
    # reassigned after st.tabs() below has already instantiated it for this run.
    if "active_tab_pending" in st.session_state:
        st.session_state["active_tab"] = st.session_state.pop("active_tab_pending")
    tab_scaffold, tab_clean = st.tabs(
        [_TAB_SCAFFOLD, _TAB_CLEAN], key="active_tab", on_change="rerun"
    )
    with tab_scaffold:
        _render_scaffold_tab()
    with tab_clean:
        st.caption(
            "Non-destructive: writes info.cops.dat's time.window field and a select.cops.dat "
            "row, never rewrites the cast file."
        )
        _render_clean_tab()


def main() -> None:
    """Console-script entry point: shells out to ``streamlit run`` on this file."""
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__)), "--", *sys.argv[1:]],
        check=False,
    )


if __name__ == "__main__":
    run_app()
