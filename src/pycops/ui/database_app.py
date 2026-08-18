"""Tab 5: generate a mission/campaign-wide AOP database + SeaBASS export.

Mirrors tab 3's own discovery pattern (recursive scan via
:func:`~pycops.io.discovery.find_deployment_folders`, one checkbox per discovered station,
default all checked) rather than requiring a hand-maintained ``directories.for.cops.dat`` list
(R's own approach) -- confirmed with Simon as the preferred discovery mechanism for this feature.
Reads each checked station's already-written ``nc/`` folder (from tab 3); doesn't reprocess
anything itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from pycops.io.database import write_mission_database_csv, write_mission_database_netcdf
from pycops.io.database_report import (
    KD_METRIC_LABELS,
    build_kd_comparison_figure,
    build_pd_depth_comparison_figure,
    build_rrs_comparison_figure,
)
from pycops.io.discovery import FLAG_BIOSHADE, FLAG_NORMAL, FLAG_UNDER_ICE, find_deployment_folders, read_select_cops
from pycops.io.seabass import SeaBASSHeaderFields, write_seabass_station_file
from pycops.processing.database import (
    StationAggregate,
    aggregate_station,
    assemble_mission_database,
    trim_unused_wavelengths,
)
from pycops.ui._common import _directory_input

_KEPT_FLAGS = (FLAG_NORMAL, FLAG_BIOSHADE, FLAG_UNDER_ICE)


def _kept_cast_count(directory: Path) -> int | None:
    """Number of ``select.cops.dat``-kept casts for ``directory``, or ``None`` if that file
    doesn't exist yet -- shown per discovered station so a researcher can see, before generating,
    which stations will actually contribute data (and which still need tab 3's processing)."""
    select_path = directory / "select.cops.dat"
    if not select_path.exists():
        return None
    return sum(1 for s in read_select_cops(select_path) if s.flag in _KEPT_FLAGS)


def _show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def _aggregate_checked_stations(checked: list[Path]) -> tuple[list[StationAggregate], list[str]]:
    stations: list[StationAggregate] = []
    failures: list[str] = []
    progress = st.progress(0.0)
    for i, folder in enumerate(checked):
        try:
            stations.append(aggregate_station(folder))
        except Exception as exc:  # noqa: BLE001 -- isolate one bad station from the rest
            failures.append(f"{folder}: {type(exc).__name__}: {exc}")
        progress.progress((i + 1) / len(checked))
    return stations, failures


def _render_station_comparison(checked: list[Path]) -> None:
    """Overlay several stations' mean Rrs/Kd spectra (mean +/- 1 SD shaded band per station,
    across each station's own kept casts) -- reuses the exact same aggregation
    (:func:`~pycops.processing.database.aggregate_station`) the mission-database export computes,
    just for plotting rather than writing files. One level up from tab 4's own "every cast in
    this one station" comparison."""
    st.caption(
        "Mean Rrs/Kd across each station's kept casts, with +/- 1 SD as a shaded band -- pick a "
        "handful of stations above to spot-check consistency or find outliers."
    )
    if not checked:
        st.info("Check at least one station above to compare.")
        return

    if st.button(f"Compare {len(checked)} station(s)", key="database_compare_run"):
        stations, failures = _aggregate_checked_stations(checked)
        # Different COPS instrument systems carry genuinely different fixed wavelength sets --
        # drop standard-grid bands none of the *selected* stations ever populate (matching R's
        # own ix.to.remove step, generate.cops.DB.R), rather than showing every comparison figure
        # with spurious permanently-empty slots. A real gap where some, but not all, selected
        # stations lack a band still shows as a break in that station's own line -- honest, not
        # a bug -- see _plot_mean_sd_band.
        waves, stations = trim_unused_wavelengths(stations)
        st.session_state["database_compare_stations"] = stations
        st.session_state["database_compare_waves"] = waves
        st.session_state["database_compare_failures"] = failures

    stations = st.session_state.get("database_compare_stations")
    waves = st.session_state.get("database_compare_waves")
    if stations is None:
        return

    for failure in st.session_state.get("database_compare_failures", []):
        st.warning(f"Couldn't aggregate {failure}")
    if not stations:
        st.error("No station could be aggregated -- nothing to compare.")
        return

    st.dataframe(
        {
            "station": [s.station_id for s in stations],
            "folder": [s.directory.name for s in stations],
            "n_casts": [s.n_casts for s in stations],
        },
        width="stretch",
        hide_index=True,
    )

    rrs_fig = build_rrs_comparison_figure(stations, waves=waves)
    if rrs_fig is not None:
        st.subheader("Rrs comparison")
        _show(rrs_fig)
    else:
        st.info("No Rrs data available across the selected stations.")

    # Always show all three Kd metrics rather than letting the researcher pick just one --
    # Simon's request, so a station's near-surface vs. deeper-reaching attenuation is never
    # hidden behind a selector.
    for metric, label in KD_METRIC_LABELS.items():
        kd_fig = build_kd_comparison_figure(stations, metric=metric, waves=waves)
        if kd_fig is not None:
            st.subheader(f"{label} comparison")
            _show(kd_fig)
        else:
            st.info(f"No {label.lower()} data available across the selected stations.")

    pd_depth_fig = build_pd_depth_comparison_figure(stations, waves=waves)
    if pd_depth_fig is not None:
        st.subheader("Penetration depth comparison")
        _show(pd_depth_fig)
    else:
        st.info("No penetration depth data available across the selected stations.")


def render_database_tab() -> None:
    tool = st.radio(
        "Tool", ("Generate database", "Compare stations (Rrs & Kd)"), key="database_tool"
    )
    parent_input = _directory_input(
        "Parent folder (searched recursively for init.cops.dat)", key="database_parent"
    )
    mission = st.text_input("Mission name", key="database_mission") if tool == "Generate database" else None
    if not parent_input:
        st.info("Enter a parent folder to begin.")
        return
    parent = Path(parent_input)
    if not parent.is_dir():
        st.error(f"{parent} is not a directory.")
        return

    folders = find_deployment_folders(parent)
    if not folders:
        st.warning(f"No deployment folder (with init.cops.dat) found under {parent}.")
        return

    st.write(f"{len(folders)} station(s) found -- uncheck any to exclude them:")
    checked: list[Path] = []
    for folder in folders:
        rel = folder.relative_to(parent)
        n_kept = _kept_cast_count(folder)
        has_nc = (folder / "nc").is_dir()
        label = f"{rel} ({n_kept if n_kept is not None else '?'} kept cast(s)"
        label += ")" if has_nc else ", not yet processed in tab 3)"
        if st.checkbox(label, value=True, key=f"database_station_{rel}"):
            checked.append(folder)

    if tool == "Compare stations (Rrs & Kd)":
        _render_station_comparison(checked)
        return

    st.divider()
    st.subheader("SeaBASS metadata")
    st.caption("Applied to every station's .sb file -- pycops has no other source for these.")
    col1, col2 = st.columns(2)
    with col1:
        investigators = st.text_input("Investigators", key="database_investigators")
        affiliations = st.text_input("Affiliations", key="database_affiliations")
        contact = st.text_input("Contact email", key="database_contact")
    with col2:
        experiment = st.text_input("Experiment", value=mission, key="database_experiment")
        cruise = st.text_input("Cruise", key="database_cruise")

    if st.button(f"Generate database ({len(checked)} station(s))", key="database_generate"):
        if not mission:
            st.error("Enter a mission name.")
            return
        if not checked:
            st.error("Select at least one station.")
            return

        stations = []
        skipped: list[tuple[Path, str]] = []
        progress = st.progress(0.0)
        for i, folder in enumerate(checked):
            try:
                stations.append(aggregate_station(folder))
            except Exception as exc:  # noqa: BLE001 -- isolate one bad station from the rest
                skipped.append((folder, f"{type(exc).__name__}: {exc}"))
            progress.progress((i + 1) / len(checked))

        db = assemble_mission_database(mission, stations, skipped)
        if not db.stations:
            st.error("No station could be aggregated -- nothing to write.")
            return

        write_mission_database_netcdf(db, parent / f"{mission}.nc")
        write_mission_database_csv(db, parent / f"{mission}.csv")
        seabass_dir = parent / "seabass"
        seabass_dir.mkdir(exist_ok=True)
        header = SeaBASSHeaderFields(
            investigators=investigators,
            affiliations=affiliations,
            contact=contact,
            experiment=experiment,
            cruise=cruise,
        )
        for station in db.stations:
            # station_id alone can collide -- two sibling deployment folders at the same physical
            # station (e.g. different instrument operators, COPS_FJSaucier vs. COPS_Kildir) share
            # one station_id but must not silently overwrite each other's .sb file.
            filename = f"{station.station_id}_{station.directory.name}.sb"
            write_seabass_station_file(station, header, db.waves, seabass_dir / filename)

        st.success(
            f"Wrote {mission}.nc / {mission}.csv to {parent}, and {len(db.stations)} SeaBASS "
            f".sb file(s) to {seabass_dir}."
        )
        st.dataframe(
            {
                "station": [s.station_id for s in db.stations],
                "n_casts": [s.n_casts for s in db.stations],
                "date": [str(s.date_mean) if s.date_mean is not None else "-" for s in db.stations],
            },
            width="stretch",
            hide_index=True,
        )
        if db.skipped:
            with st.expander(f"Skipped {len(db.skipped)} station(s)"):
                for folder, reason in db.skipped:
                    st.warning(f"{folder}: {reason}")
