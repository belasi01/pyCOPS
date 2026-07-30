"""Mission-wide AOP database: aggregate every station in a mission/campaign into one dataset.

Python port of the core idea in Simon's R workflow's ``generate.cops.DB()``
(``~/RPackages/Cops/R/generate.cops.DB.R``): for every ``select.cops.dat``-kept cast in a
station, average (mean + sd) Rrs, nLw, Ed0.0p, bottom reflectance (``Rb``), and Kd at the
1%/10%/penetration-depth light levels (see :mod:`pycops.processing.attenuation`) onto a fixed
standard wavelength grid, plus scalar per-station fields (date, sun zenith, position, Forel-Ule
class, shadow-correction method, bottom depth). Rebuilt on top of pycops's own ``.nc``-per-cast
pipeline (:mod:`pycops.io.netcdf`) instead of R's ``.RData``-per-cast + ``BIN/`` convention, and
stations are discovered with a recursive scan (:func:`pycops.io.discovery.find_deployment_folders`)
rather than a hand-maintained ``directories.for.cops.dat`` list.

**Not included** (confirmed with Simon): the real Q-factor/f BRDF tables (``Q.Factor``) and the
bottom Q-factor (``Rb.Q``) -- both need the un-ported ``popt.f.Q``/``popt.f.f`` 5-/3-D lookup
tables (see ``CLAUDE.md``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pycops.io.discovery import find_deployment_folders, kept_nc_files

# pycops's own reference grid -- matches generate.cops.DB.R's waves.DB exactly, so a mission
# combining stations from different instrument systems/years still lands on one shared grid.
STANDARD_WAVELENGTHS = (
    305, 313, 320, 330, 340, 380, 395, 412, 443, 465, 490, 510, 532, 555,
    560, 565, 589, 625, 665, 683, 694, 710, 765, 780, 875,
)
# A real cast's own calibrated wavelengths rarely land exactly on a standard band (unlike R's
# brittle exact-equality match()), so each is matched to the nearest standard band within this
# tolerance instead.
WAVELENGTH_TOLERANCE_NM = 2.0

# Rb/bottom-depth source-instrument preference, matching R's "Rb.EuZ when both instruments and
# Rb.Q are available, else Rb.LuZ" -- minus the Rb.Q gate, since that isn't ported.
_SHADOW_INSTRUMENTS = ("EuZ", "LuZ")


@dataclass(frozen=True)
class MeanSd:
    """Per-wavelength mean and sample standard deviation across a station's kept casts."""

    mean: np.ndarray  # (n_waves,)
    sd: np.ndarray  # (n_waves,) -- NaN where fewer than 2 casts contributed a finite value


@dataclass(frozen=True)
class ScalarMeanSd:
    """Mean and sample standard deviation of a single (non-wavelength-indexed) broadband metric
    across a station's kept casts -- e.g. PAR/Kd(PAR), which unlike Rrs/nLw/Kd/etc. have no
    wavelength dimension at all (PAR is already integrated over the whole 400-700 nm band)."""

    mean: float
    sd: float  # NaN when fewer than 2 casts contributed a finite value


@dataclass(frozen=True)
class StationAggregate:
    """One station's worth of ``generate.cops.DB()``-style aggregated AOPs."""

    station_id: str  # from the deployment path's own "..._Station<ID>" component
    directory: Path
    n_casts: int
    date_mean: pd.Timestamp | None
    sun_zenith_mean: float | None
    longitude_mean: float | None
    latitude_mean: float | None
    forel_ule_mean: float | None  # arithmetic mean of the Forel-Ule class, matching R's mean(FU)
    shadow_correction_method: str | None  # from the last kept cast's chl_flag, matching R
    bottom_depth_mean: float | None
    ed0_diffuse_fraction: np.ndarray | None  # per standard wavelength, mean across kept casts
    rrs: MeanSd
    nlw: MeanSd
    rb: MeanSd
    kd_1pct: MeanSd
    kd_10pct: MeanSd
    kd_pd: MeanSd
    ed0_0p: MeanSd
    par_0: ScalarMeanSd  # broadband PAR of Ed0's surface reference (see pycops.processing.par)
    kd_par_1pct: ScalarMeanSd
    kd_par_10pct: ScalarMeanSd
    kd_par_pd: ScalarMeanSd


@dataclass(frozen=True)
class MissionDatabase:
    """Every station in a mission, aggregated -- the pycops equivalent of R's ``COPS.DB``."""

    mission: str
    waves: np.ndarray
    stations: list[StationAggregate]
    skipped: list[tuple[Path, str]] = field(default_factory=list)


def _station_id_from_path(directory: Path) -> str:
    """Extract a station ID from whichever path component contains ``"_Station"`` (e.g.
    ``20190818_StationMAN-F05``), matching ``generate.cops.DB.R``'s own path-parsing loop.
    Falls back to the deployment folder's own name if the convention isn't followed."""
    for part in directory.parts:
        if "_Station" in part:
            return part.split("_Station", 1)[1]
    return directory.name


def _shadow_correction_method_label(chl_flag: float | None) -> str:
    """Port of ``generate.cops.DB.R``'s ``chl``-based label (``is.na(cops$chl)`` / ``== 999`` /
    ``== 0`` / else a positive chlorophyll value)."""
    if chl_flag is None or np.isnan(chl_flag):
        return "No correction"
    if chl_flag == 0:
        return "abs.measured"
    if chl_flag == 999:
        return "abs.Kd.method"
    return "abs.Case1.model"


def _match_to_standard_grid(waves_cast: np.ndarray | None, values_cast: np.ndarray | None) -> np.ndarray:
    """One value per :data:`STANDARD_WAVELENGTHS` band, nearest-matched from ``values_cast``
    (indexed by ``waves_cast``) within :data:`WAVELENGTH_TOLERANCE_NM`, ``NaN`` elsewhere --
    including when the cast doesn't have this metric at all (``values_cast is None``)."""
    standard = np.asarray(STANDARD_WAVELENGTHS, dtype=float)
    out = np.full(len(standard), np.nan)
    if values_cast is None or waves_cast is None or len(waves_cast) == 0:
        return out
    waves_cast = np.asarray(waves_cast, dtype=float)
    for i, w in enumerate(standard):
        diffs = np.abs(waves_cast - w)
        j = int(np.argmin(diffs))
        if diffs[j] <= WAVELENGTH_TOLERANCE_NM:
            out[i] = values_cast[j]
    return out


def _mean_of(values: list[float | None]) -> float | None:
    finite = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(finite)) if finite else None


def _scalar_mean_sd(values: list[float | None]) -> ScalarMeanSd:
    finite = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not finite:
        return ScalarMeanSd(mean=np.nan, sd=np.nan)
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan
    return ScalarMeanSd(mean=mean, sd=sd)


def aggregate_station(directory: str | Path) -> StationAggregate:
    """Aggregate one station folder's kept casts (per ``select.cops.dat``, via
    :func:`pycops.io.discovery.kept_nc_files`) from its already-written ``nc/`` folder.

    Raises ``ValueError`` (caught by :func:`build_mission_database`, which isolates one bad
    station from the rest of the mission) if there's no ``nc/`` subfolder or zero kept casts.
    """
    directory = Path(directory)
    nc_dir = directory / "nc"
    if not nc_dir.is_dir():
        raise ValueError(f"no nc/ subfolder in {directory} -- process this station in tab 3 first")

    nc_paths = kept_nc_files(directory, nc_dir)
    if not nc_paths:
        raise ValueError(f"no kept casts with .nc output in {nc_dir}")

    rrs_rows: list[np.ndarray] = []
    nlw_rows: list[np.ndarray] = []
    rb_rows: list[np.ndarray] = []
    kd1_rows: list[np.ndarray] = []
    kd10_rows: list[np.ndarray] = []
    kdpd_rows: list[np.ndarray] = []
    ed0_rows: list[np.ndarray] = []
    ed0_diffuse_rows: list[np.ndarray] = []
    dates: list[pd.Timestamp | None] = []
    sunzens: list[float | None] = []
    lons: list[float | None] = []
    lats: list[float | None] = []
    fus: list[float | None] = []
    bottom_depths: list[float | None] = []
    chl_flags: list[float | None] = []
    par_0_values: list[float | None] = []
    kd_par_1pct_values: list[float | None] = []
    kd_par_10pct_values: list[float | None] = []
    kd_par_pd_values: list[float | None] = []

    for nc_path in nc_paths:
        with xr.open_dataset(nc_path) as nc:
            waves_cast = nc["wavelength"].values

            rrs_rows.append(
                _match_to_standard_grid(waves_cast, nc["rrs_0p_recommended"].values if "rrs_0p_recommended" in nc else None)
            )
            nlw_rows.append(
                _match_to_standard_grid(waves_cast, nc["nlw_0p_recommended"].values if "nlw_0p_recommended" in nc else None)
            )
            ed0_rows.append(
                _match_to_standard_grid(waves_cast, nc["ed0_value_at_0"].values if "ed0_value_at_0" in nc else None)
            )
            kd1_rows.append(_match_to_standard_grid(waves_cast, nc["kd_1pct"].values if "kd_1pct" in nc else None))
            kd10_rows.append(_match_to_standard_grid(waves_cast, nc["kd_10pct"].values if "kd_10pct" in nc else None))
            kdpd_rows.append(_match_to_standard_grid(waves_cast, nc["kd_pd"].values if "kd_pd" in nc else None))

            rb_values = None
            bottom_depth = None
            for instrument in _SHADOW_INSTRUMENTS:
                if f"{instrument}_rb" in nc:
                    rb_values = nc[f"{instrument}_rb"].values
                    bottom_depth = nc.attrs.get(f"{instrument}_bottom_depth")
                    break
            rb_rows.append(_match_to_standard_grid(waves_cast, rb_values))
            bottom_depths.append(bottom_depth)

            edif = edir = None
            for instrument in _SHADOW_INSTRUMENTS:
                if f"{instrument}_shadow_edif" in nc:
                    edif = nc[f"{instrument}_shadow_edif"].values
                    edir = nc[f"{instrument}_shadow_edir"].values
                    break
            diffuse_fraction = edif / (edif + edir) if edif is not None else None
            ed0_diffuse_rows.append(_match_to_standard_grid(waves_cast, diffuse_fraction))

            method = nc.attrs.get("rrs_method")
            fu_label = "linear" if method == "Rrs.0p.linear" else "loess"
            fus.append(nc.attrs.get(f"qwip_{fu_label}_fu"))

            if "time" in nc.coords and nc.sizes.get("time", 0):
                dates.append(pd.DatetimeIndex(nc["time"].values).mean())
            else:
                dates.append(None)
            sunzen = nc.attrs.get("sun_zenith_deg")
            sunzens.append(sunzen if sunzen is not None and not np.isnan(sunzen) else None)
            lon = nc.attrs.get("longitude")
            lons.append(lon if lon is not None and not np.isnan(lon) else None)
            lat = nc.attrs.get("latitude")
            lats.append(lat if lat is not None and not np.isnan(lat) else None)
            chl_flags.append(nc.attrs.get("chl_flag"))
            par_0_values.append(nc.attrs.get("par_0"))
            kd_par_1pct_values.append(nc.attrs.get("kd_par_1pct"))
            kd_par_10pct_values.append(nc.attrs.get("kd_par_10pct"))
            kd_par_pd_values.append(nc.attrs.get("kd_par_pd"))

    def _mean_sd(rows: list[np.ndarray]) -> MeanSd:
        stacked = np.asarray(rows)
        return MeanSd(mean=np.nanmean(stacked, axis=0), sd=np.nanstd(stacked, axis=0, ddof=1))

    with warnings.catch_warnings():
        # A wavelength/metric only one cast actually has produces a single-sample sd -- an
        # expected, not exceptional, "Degrees of freedom <= 0"/"Mean of empty slice" warning.
        warnings.simplefilter("ignore", category=RuntimeWarning)

        ed0_diffuse_fraction = np.nanmean(np.asarray(ed0_diffuse_rows), axis=0)
        valid_dates = [d.value for d in dates if d is not None]
        date_mean = pd.Timestamp(int(np.mean(valid_dates))) if valid_dates else None
        bottom_depth_mean = _mean_of(bottom_depths)
        rrs = _mean_sd(rrs_rows)
        nlw = _mean_sd(nlw_rows)
        rb = _mean_sd(rb_rows)
        kd_1pct = _mean_sd(kd1_rows)
        kd_10pct = _mean_sd(kd10_rows)
        kd_pd = _mean_sd(kdpd_rows)
        ed0_0p = _mean_sd(ed0_rows)
        par_0 = _scalar_mean_sd(par_0_values)
        kd_par_1pct = _scalar_mean_sd(kd_par_1pct_values)
        kd_par_10pct = _scalar_mean_sd(kd_par_10pct_values)
        kd_par_pd = _scalar_mean_sd(kd_par_pd_values)

    return StationAggregate(
        station_id=_station_id_from_path(directory),
        directory=directory,
        n_casts=len(nc_paths),
        date_mean=date_mean,
        sun_zenith_mean=_mean_of(sunzens),
        longitude_mean=_mean_of(lons),
        latitude_mean=_mean_of(lats),
        forel_ule_mean=_mean_of(fus),
        shadow_correction_method=_shadow_correction_method_label(chl_flags[-1] if chl_flags else None),
        bottom_depth_mean=bottom_depth_mean,
        ed0_diffuse_fraction=ed0_diffuse_fraction,
        rrs=rrs,
        nlw=nlw,
        rb=rb,
        kd_1pct=kd_1pct,
        kd_10pct=kd_10pct,
        kd_pd=kd_pd,
        ed0_0p=ed0_0p,
        par_0=par_0,
        kd_par_1pct=kd_par_1pct,
        kd_par_10pct=kd_par_10pct,
        kd_par_pd=kd_par_pd,
    )


def _trim_station_waves(station: StationAggregate, keep_mask: np.ndarray) -> StationAggregate:
    def _trim(m: MeanSd) -> MeanSd:
        return MeanSd(mean=m.mean[keep_mask], sd=m.sd[keep_mask])

    return replace(
        station,
        rrs=_trim(station.rrs),
        nlw=_trim(station.nlw),
        rb=_trim(station.rb),
        kd_1pct=_trim(station.kd_1pct),
        kd_10pct=_trim(station.kd_10pct),
        kd_pd=_trim(station.kd_pd),
        ed0_0p=_trim(station.ed0_0p),
        ed0_diffuse_fraction=(
            station.ed0_diffuse_fraction[keep_mask] if station.ed0_diffuse_fraction is not None else None
        ),
    )


def assemble_mission_database(
    mission: str, stations: list[StationAggregate], skipped: list[tuple[Path, str]] | None = None
) -> MissionDatabase:
    """Build a :class:`MissionDatabase` from already-aggregated stations, dropping wavelengths
    where every station's ``Ed0.0p`` is entirely missing (matching ``generate.cops.DB.R``'s own
    ``ix.to.remove`` step). Shared by :func:`build_mission_database` and the "Generate database"
    UI tab, which aggregates only the researcher-checked subset of discovered stations rather
    than every station :func:`~pycops.io.discovery.find_deployment_folders` finds.
    """
    waves = np.asarray(STANDARD_WAVELENGTHS, dtype=float)
    if stations:
        all_ed0 = np.asarray([s.ed0_0p.mean for s in stations])
        keep_mask = ~np.all(np.isnan(all_ed0), axis=0)
        if not keep_mask.all():
            waves = waves[keep_mask]
            stations = [_trim_station_waves(s, keep_mask) for s in stations]

    return MissionDatabase(mission=mission, waves=waves, stations=stations, skipped=skipped or [])


def build_mission_database(parent: str | Path, mission: str) -> MissionDatabase:
    """Discover every deployment folder under ``parent`` (recursively, via
    :func:`~pycops.io.discovery.find_deployment_folders`) and aggregate each one.

    A station that fails to aggregate (no ``nc/`` folder, zero kept casts, or any other read
    error) is recorded in ``MissionDatabase.skipped`` with a ``warnings.warn`` rather than
    aborting the whole mission -- matches ``process_deployment()``'s own per-item failure
    isolation used everywhere else in this port.
    """
    parent = Path(parent)
    stations: list[StationAggregate] = []
    skipped: list[tuple[Path, str]] = []

    for directory in find_deployment_folders(parent):
        try:
            stations.append(aggregate_station(directory))
        except Exception as exc:  # noqa: BLE001 -- isolate one bad station from the rest
            warnings.warn(f"{directory}: skipped ({exc})", stacklevel=2)
            skipped.append((directory, str(exc)))

    return assemble_mission_database(mission, stations, skipped)
