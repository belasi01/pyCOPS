"""Per-cast final-Rrs wavelength exclusions: a pycops-only QC step with no R equivalent.

Simon's request: after picking a Rrs extrapolation method (LOESS/linear), a specific band can
still be bad for a given cast (e.g. 380 nm on a UV-noisy profile) and should be set to NaN in the
final Rrs regardless of which method produced it. Kept in its own file
(``rrs_wavelength_exclusions.cops.dat``), not added as a new field to ``select.cops.dat``/
``info.cops.dat``: those are the R package's own long-established formats, and this feature has
no R-side counterpart to stay compatible with, so adding it there would be a pure divergence risk
for zero compatibility benefit. A deployment that never uses this feature simply has no such file
-- every reader here treats "missing file" as "no exclusions", so it can't break any existing
station's reprocessing.
"""

from __future__ import annotations

from pathlib import Path


def read_wavelength_exclusions(path: str | Path) -> dict[str, list[float]]:
    """Parse ``rrs_wavelength_exclusions.cops.dat`` into ``{cast file name: [excluded nm, ...]}``.

    Returns an empty dict if the file doesn't exist -- the "no exclusions anywhere" default.
    """
    path = Path(path)
    if not path.exists():
        return {}

    exclusions: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            file, _, waves = stripped.partition(";")
            exclusions[file] = [float(w) for w in waves.split(",") if w.strip()]
    return exclusions


def update_wavelength_exclusions(path: str | Path, file: str, wavelengths: list[float]) -> None:
    """Write ``file``'s excluded-wavelength row, replacing it in place -- same surgical,
    line-level edit as :func:`pycops.io.discovery.update_cast_selection` (every other row, and
    its own terminator, is left byte-for-byte untouched). An empty ``wavelengths`` removes the
    row entirely rather than writing a dangling ``file;``, keeping "no exclusions" and "row
    absent" the same state.
    """
    path = Path(path)
    new_row = f"{file};{','.join(f'{float(w):.10g}' for w in wavelengths)}" if wavelengths else None

    if not path.exists():
        if new_row is not None:
            path.write_text(new_row + "\n")
        return

    with path.open(newline="") as f:
        text = f.read()
    lines = text.splitlines(keepends=True)

    found = False
    new_lines = []
    for line in lines:
        content = line.splitlines()[0] if line else line
        terminator = line[len(content) :]
        stripped = content.strip()
        if not stripped or stripped.startswith("#") or content.split(";", 1)[0].strip() != file:
            new_lines.append(line)
            continue
        found = True
        if new_row is not None:
            new_lines.append(new_row + terminator)
        # else: drop this row (wavelengths cleared back to "no exclusions")

    if not found and new_row is not None:
        terminator = "\n"
        for line in reversed(lines):
            if line.endswith("\r\n"):
                terminator = "\r\n"
                break
            if line.endswith("\n"):
                terminator = "\n"
                break
        if new_lines and not new_lines[-1].endswith(("\n", "\r\n", "\r")):
            new_lines[-1] += terminator
        new_lines.append(new_row + terminator)

    with path.open("w", newline="") as f:
        f.write("".join(new_lines))
