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

import streamlit as st

from pycops.io.database import write_mission_database_csv, write_mission_database_netcdf
from pycops.io.discovery import FLAG_BIOSHADE, FLAG_NORMAL, FLAG_UNDER_ICE, find_deployment_folders, read_select_cops
from pycops.io.seabass import SeaBASSHeaderFields, write_seabass_station_file
from pycops.processing.database import aggregate_station, assemble_mission_database
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


def render_database_tab() -> None:
    parent_input = _directory_input(
        "Parent folder (searched recursively for init.cops.dat)", key="database_parent"
    )
    mission = st.text_input("Mission name", key="database_mission")
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
