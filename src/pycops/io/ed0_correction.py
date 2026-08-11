"""Per-cast Ed0 illumination-correction method override: a pycops-only QC step with no R
equivalent.

``init.cops.dat``'s ``ed0.correction.method`` sets a station-wide default ("raw", matching the R
package, or "smoothed", pycops-only -- see :mod:`pycops.processing.ed0`); this sidecar file
(``ed0_correction_method.cops.dat``) lets a researcher override that choice for one specific cast,
mirroring :mod:`pycops.io.exclusions`'s ``rrs_wavelength_exclusions.cops.dat`` exactly and for the
same reason: ``info.cops.dat`` is a fixed-position format with no spare field, and this feature has
no R-side counterpart to stay compatible with, so adding it there would be a pure divergence risk
for zero compatibility benefit. A deployment that never uses this feature simply has no such file
-- every reader here treats "missing file"/"missing row" as "use the station default".
"""

from __future__ import annotations

from pathlib import Path


def read_ed0_correction_methods(path: str | Path) -> dict[str, str]:
    """Parse ``ed0_correction_method.cops.dat`` into ``{cast file name: "raw" | "smoothed"}``.

    Returns an empty dict if the file doesn't exist -- the "no overrides anywhere" default.
    """
    path = Path(path)
    if not path.exists():
        return {}

    methods: dict[str, str] = {}
    with path.open(newline="") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            file, _, method = stripped.partition(";")
            methods[file] = method.strip()
    return methods


def update_ed0_correction_method(path: str | Path, file: str, method: str | None) -> None:
    """Write ``file``'s Ed0 correction method override, replacing it in place -- same surgical,
    line-level edit as :func:`pycops.io.exclusions.update_wavelength_exclusions` (every other row,
    and its own terminator, is left byte-for-byte untouched). ``method=None`` removes the row
    entirely, reverting to the station-wide ``init.cops.dat`` default.
    """
    path = Path(path)
    new_row = f"{file};{method}" if method else None

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
        # else: drop this row (reverted back to "use the station default")

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
