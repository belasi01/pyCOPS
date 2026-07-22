"""Widgets shared across the ``pycops-clean`` app's tabs.

Moved out of :mod:`pycops.ui.clean_app` (where tabs 1/2/3 originally defined them) so
:mod:`pycops.ui.analyze_app` (tab 4) can reuse the same "pick a folder" UX without duplicating
it or reaching into a sibling module's private names.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


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
