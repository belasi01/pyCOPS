"""Interactive Streamlit tool to set each cast's ``time.window`` (non-destructive trim).

Simon's R workflow ends with ``cops.go(clean.files=TRUE)``: plot Depth vs. scan index, click the
start of the good downcast and the end (or, for a shallow cast, where the instrument hit bottom),
and R **overwrites the raw cast file** with just the trimmed rows. This tool reproduces the same
researcher-facing task -- pick a start/end for each cast -- but non-destructively: it writes the
choice into ``info.cops.dat``'s existing ``time.window`` field instead (see
:func:`pycops.io.config.update_time_window`), which :func:`pycops.processing.process_cast.process_cast`
already applies as an early QC mask across every instrument (matching ``derived.data.R``'s
``Depth.good <- Depth.good & dates.good``) -- ``L2`` cast files themselves are never rewritten.

**Deliberate reinterpretation**: R's ``clean.cops.file()`` plots against raw scan index; this
plots against elapsed time in seconds, matching ``time.window``'s actual units. This is a
documented design choice, not a divergence bug.

Run with ``pycops-clean <deployment folder>`` (after ``uv sync --extra ui`` /
``pip install pycops[ui]``), or directly via ``streamlit run src/pycops/ui/clean_app.py --
<deployment folder>``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pycops.io.config import read_info_cops, read_init_cops, update_time_window
from pycops.io.raw import parse_cast_filename, read_cast

_CAST_GLOBS = ("*_URC.tsv", "*_URC.csv")


def _discover_casts(directory: Path) -> list[Path]:
    """Cast files in ``directory``, sorted by their own recorded date/time.

    Globs the established ``*_URC.tsv``/``.csv`` naming convention directly --
    deliberately does not require ``info.cops.dat`` to exist yet, since this tool may be the one
    that creates it. Files that don't parse (e.g. a stray non-cast file) are skipped with a
    warning rather than aborting discovery, matching
    :func:`pycops.io.discovery.read_deployment_casts`'s existing resilience pattern.
    """
    candidates = [p for pattern in _CAST_GLOBS for p in directory.glob(pattern)]
    parsed: list[tuple[Path, object]] = []
    for path in sorted(candidates):
        try:
            info = parse_cast_filename(path)
        except ValueError as exc:
            warnings.warn(f"skipping {path.name}: {exc}", stacklevel=2)
            continue
        parsed.append((path, info.date))
    parsed.sort(key=lambda item: item[1])
    return [path for path, _ in parsed]


def _existing_time_window(info_path: Path, filename: str) -> tuple[float, float] | None:
    if not info_path.exists():
        return None
    for entry in read_info_cops(info_path):
        if entry.file == filename:
            return entry.time_window
    return None


def run_app() -> None:
    st.set_page_config(page_title="pycops -- cast cleaning", layout="wide")
    st.title("pycops -- interactive cast cleaning")
    st.caption(
        "Non-destructive: writes info.cops.dat's time.window field, never rewrites the cast file."
    )

    default_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    directory_input = st.text_input("Deployment folder (L2/.../COPS*)", value=default_dir)
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

    casts = _discover_casts(directory)
    if not casts:
        st.warning(f"No *_URC.tsv/.csv cast files found in {directory}.")
        return

    st.session_state.setdefault("cast_idx", 0)
    st.session_state["cast_idx"] = min(st.session_state["cast_idx"], len(casts) - 1)

    labels = [p.name for p in casts]
    selected_label = st.selectbox(
        f"Cast ({len(casts)} found)", labels, index=st.session_state["cast_idx"]
    )
    st.session_state["cast_idx"] = labels.index(selected_label)
    cast_path = casts[st.session_state["cast_idx"]]

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
    existing = _existing_time_window(info_path, cast_path.name)
    default_start, default_end = existing if existing is not None else (0.0, total_duration)
    default_start = max(0.0, min(default_start, total_duration))
    default_end = max(default_start, min(default_end, total_duration))

    start, end = st.slider(
        "Time window kept (seconds elapsed from first scan)",
        min_value=0.0,
        max_value=total_duration,
        value=(default_start, default_end),
        step=max(total_duration / 500, 0.05),
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(elapsed, depth, ".", markersize=2, color="tab:blue")
    ax.axvspan(0, start, color="gray", alpha=0.3)
    ax.axvspan(end, total_duration, color="gray", alpha=0.3)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(f"{depth_is_on} depth (m)")
    ax.invert_yaxis()
    ax.set_title(cast_path.name)
    st.pyplot(fig)
    plt.close(fig)

    if st.button("Save && next"):
        update_time_window(info_path, cast_path.name, (start, end))
        st.success(f"Saved time.window = {start:g},{end:g} for {cast_path.name}")
        st.session_state["cast_idx"] = min(st.session_state["cast_idx"] + 1, len(casts) - 1)
        st.rerun()


def main() -> None:
    """Console-script entry point: shells out to ``streamlit run`` on this file."""
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__)), "--", *sys.argv[1:]],
        check=False,
    )


if __name__ == "__main__":
    run_app()
