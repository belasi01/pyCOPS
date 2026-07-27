"""Widgets shared across the ``pycops-clean`` app's tabs.

Moved out of :mod:`pycops.ui.clean_app` (where tabs 1/2/3 originally defined them) so
:mod:`pycops.ui.analyze_app` (tab 4) can reuse the same "pick a folder" UX, the ``time.window``
trim editor, and the per-instrument override fields without duplicating them or reaching into a
sibling module's private names.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pycops.io.config import CastInfo, parse_override_field, read_info_cops
from pycops.io.discovery import CastSelection, read_select_cops


def existing_info(info_path: Path, filename: str) -> CastInfo | None:
    if not info_path.exists():
        return None
    for entry in read_info_cops(info_path):
        if entry.file == filename:
            return entry
    return None


def existing_selection(select_path: Path, filename: str) -> CastSelection | None:
    if not select_path.exists():
        return None
    for entry in read_select_cops(select_path):
        if entry.file == filename:
            return entry
    return None

# Per-instrument info.cops.dat override fields, edited as free-text comma-separated lists (one
# value per instruments.optics entry, or "x" for "use init.cops.dat's default") since they're
# rarely touched cast-by-cast and this matches the format researchers already know from hand-
# editing the file -- shared by the clean tab's own editor and the analyze tab's
# "Adjust processing parameters" section.
OVERRIDE_FIELDS = (
    ("sub_surface_removed_layer", "sub.surface.removed.layer"),
    ("tiltmax", "tiltmax"),
    ("depth_interval_for_smoothing", "depth.interval.for.smoothing"),
    ("linear_r2_threshold", "linear.fit.Rsquared.threshold"),
    ("linear_max_delta_depth", "linear.fit.max.delta.depth"),
)


def format_override(value: list[float] | None) -> str:
    if value is None:
        return "x"
    # Matches the file's own "NA" sentinel for a per-value override that doesn't apply
    # (e.g. a linear-fit threshold at the surface Ed0 instrument) -- not Python's "nan" repr.
    return ",".join("NA" if np.isnan(v) else f"{v:.10g}" for v in value)


def render_override_fields_editor(
    info: CastInfo | None, instruments: tuple[str, ...], *, key_prefix: str
) -> dict[str, str]:
    """The 5 per-instrument override text fields, pre-filled from ``info`` (or blank/"x" if
    ``info`` is ``None``). Returns the *raw* text per field -- parse with
    :func:`pycops.io.config.parse_override_field` at save time, matching the existing pattern of
    resending every field on save rather than tracking a diff."""
    st.caption(
        f"Comma-separated values, one per instrument ({', '.join(instruments)}), or 'x' to "
        "keep init.cops.dat's default value."
    )
    override_texts: dict[str, str] = {}
    for attr, label in OVERRIDE_FIELDS:
        override_texts[attr] = st.text_input(
            label,
            value=format_override(getattr(info, attr) if info else None),
            key=f"{key_prefix}_{attr}",
        )
    return override_texts


def parsed_override_fields(override_texts: dict[str, str]) -> dict[str, list[float] | None]:
    """Parse every field :func:`render_override_fields_editor` returned, ready to splat into
    :func:`pycops.io.config.update_cast_info`."""
    return {attr: parse_override_field(override_texts[attr]) for attr, _ in OVERRIDE_FIELDS}


def render_time_window_editor(
    elapsed: np.ndarray,
    depth: np.ndarray,
    depth_label: str,
    title: str,
    existing_time_window: tuple[float, float] | None,
    *,
    key: str,
) -> tuple[float, float]:
    """Depth-vs-elapsed-time plot + a ``time.window`` range slider, shading excluded regions.

    ``existing_time_window`` is the cast's current ``info.cops.dat`` override, if any (``None``
    defaults to keeping the whole cast). Returns the ``(start, end)`` the researcher chose.
    """
    total_duration = float(elapsed.max())
    default_start, default_end = existing_time_window if existing_time_window is not None else (0.0, total_duration)
    default_start = max(0.0, min(default_start, total_duration))
    default_end = max(default_start, min(default_end, total_duration))

    start, end = st.slider(
        "Time window kept (seconds elapsed from first scan)",
        min_value=0.0,
        max_value=total_duration,
        value=(default_start, default_end),
        step=max(total_duration / 500, 0.05),
        key=key,
    )

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(elapsed, depth, ".", markersize=2, color="tab:blue")
    ax.axvspan(0, start, color="gray", alpha=0.3)
    ax.axvspan(end, total_duration, color="gray", alpha=0.3)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(depth_label)
    ax.invert_yaxis()
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

    return start, end


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
    # Falls back to the project-wide parent folder (set once, in tab 1, via _directory_input's own
    # "project_root_dir" key) before finally falling back to the home directory -- Simon: landing
    # in $HOME every time was "ennuyant" ("annoying") when every project folder lives elsewhere.
    current = (
        st.session_state.get(browse_key)
        or st.session_state.get(key)
        or st.session_state.get("project_root_dir")
        or str(Path.home())
    )
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
