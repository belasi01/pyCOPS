# pycops

A Python port of the LOV/UQAR `Cops` R package (Bernard Gentili / Simon Belanger) for processing raw
data from the **C-OPS** (Compact Optical Profiling System, Biospherical Instruments) underwater
radiometer. It reads raw cast files, fits depth profiles, derives Rrs/Kd/nLw and other apparent
optical properties, and aggregates a whole mission/campaign into a database — including
SeaBASS-compliant export.

The port is organized around an interactive Streamlit UI (`pycops-clean`) that walks through the
whole workflow, from raw L1 data to a mission-wide database.

## Requirements

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management (recommended)

## Installation

```bash
git clone https://github.com/belasi01/pyCOPS.git
cd pyCOPS
uv sync --extra ui
```

The `ui` extra installs Streamlit and matplotlib, needed for the interactive tool. If you only
need the Python library (no UI), `uv sync` alone is enough.

## Launching the UI

```bash
uv run pycops-clean [deployment folder]
```

The optional argument pre-fills the "Clean casts" tab's deployment-folder field; it's otherwise
fine to leave it out and just navigate to the right folder from within the app. This opens the
app in your browser (Streamlit's default: http://localhost:8501).

You can also run it directly with Streamlit:

```bash
uv run streamlit run src/pycops/ui/clean_app.py
```

## Using the UI

The app has five tabs, meant to be used roughly in order for a new station, though each is also
usable on its own once earlier steps are done.

### 1. Create a station (L1 → L2)

Copies the cast files you select from a read-only **L1** folder (the straight-from-the-instrument
export) into a new, organized `L2/YYYYMMDD_StationXXX/cops/` folder — the layout the rest of the
tool expects. L1 is never modified. This step also lets you generate or copy in an
`init.cops.dat` (the per-instrument-system processing parameters file, which rarely changes
between deployments of the same instrument).

### 2. Clean casts

For each cast in a station folder, edit:

- **`info.cops.dat`**: position (longitude/latitude), the chlorophyll/absorption flag, a
  `time.window` trim (depth-vs-time plot with a range slider — replaces the R workflow's
  destructive raw-file trimming with a **non-destructive** override), and five optional
  per-instrument overrides.
- **`select.cops.dat`**: the QC flag (kept/rejected/BioShade/under-ice), the Rrs extrapolation
  method (LOESS vs. linear), and the SHALLOW flag.

Nothing here ever rewrites the raw cast file itself — only `info.cops.dat`/`select.cops.dat` are
touched. The tab won't let you move on to processing until every cast has a resolvable position.

### 3. Process casts

Runs the full processing pipeline (Ed0/EdZ/LuZ/EuZ fitting, shadow correction, Rrs/Kd/QWIP, bottom
reflectance for shallow casts, etc.) and writes one NetCDF file per cast into an `nc/` subfolder —
overwriting any previous output there. Two modes:

- **Single deployment**: process one station folder.
- **Batch**: point at a parent folder and it recursively finds and processes every station folder
  under it (anywhere containing an `init.cops.dat`).

### 4. Analyze results

A read-only, per-cast diagnostic viewer over the `.nc` files tab 3 produced — the interactive
equivalent of the R package's per-cast PDF report. For one selected cast, it shows: an overview
(position, flags, notes), Ed0 illumination stability, per-instrument tilt and depth profiles (with
the raw scans overlaid), attenuation (Kd), the LOESS-vs-linear surface extrapolation comparison,
the Rrs spectrum, shadow correction, QWIP/water-class, and bottom reflectance (for SHALLOW casts).

An **"Adjust & reprocess"** section at the bottom lets you tweak a cast's `time.window`, its five
numeric overrides, and its Rrs method or wavelength exclusions, then reprocess just that cast in
place — followed by **Discard this cast** or **Validate and next** to move through a station's
casts one by one.

Switching to **"Station comparison (Rrs & Kd)"** mode overlays every currently-kept cast's Rrs and
Kd spectra for one station, to spot an outlier before finalizing which casts to keep.

### 5. Generate database

Once a mission/campaign's stations have all been processed and QC'd, this tab aggregates them into
one database:

1. Point at a parent folder; it recursively discovers every station folder under it and shows how
   many kept casts each one has.
2. Uncheck any station you don't want included, and fill in the SeaBASS metadata fields
   (investigators, affiliations, contact, experiment, cruise) once — they apply to every station's
   export.
3. **Generate database** writes:
   - one mission-wide NetCDF and CSV (mean + standard deviation of Rrs, nLw, Ed0, bottom
     reflectance, and Kd at the 1%/10%/penetration-depth light levels, across every kept cast per
     station), and
   - one SeaBASS-compliant `.sb` file per station, in a `seabass/` subfolder.

## Development

```bash
uv sync --extra ui        # install with dev/test/UI dependencies
uv run pytest -q           # run the test suite
uv run ruff check src tests  # lint
```
