"""Chlorophyll-based (case-1 waters) absorption model, port of ``popt.R``.

Only the piece consumed by ``shadow.correction.R`` for a genuine positive
``info.cops.dat`` ``chl`` (see :func:`pycops.processing.shadow.resolve_absorption`):
``popt.f.a``, a Morel & Maritorena-style absorption estimate from an
iterative backscatter/reflectance solve, plus its lookup-table dependencies
(``popt.f.e``/``popt.f.chi``/``popt.f.aw``/``popt.f.mud``) and closed-form
helpers. The Q/f bidirectional-reflectance tables (``popt.f.Q``/``popt.f.f``,
``Q.and.f.factors.R``) are a separate, much larger port (5-/3-dimensional
lookup tables shipped as ``.RData``, not plain-text data in the R source) and
are not included here -- see :func:`pycops.processing.qfactor.compute_q_factor`,
which still raises for a genuine ``chl > 0``.

``chlorophyll_absorption()`` reproduces ``shadow.correction.R``'s inline
wavelength-domain handling around the ``popt.fv.a``/``popt.f.a`` calls (not a
separate named R function): the chlorophyll-backscatter model is only valid
350-700 nm, so wavelengths above 700 nm get pure-water absorption only (no
chlorophyll term) and wavelengths below 350 nm all reuse a single value
computed at the shortest usable wavelength (350.001 nm) -- both exactly as
``shadow.correction.R`` does it. ``chl`` itself is clamped to
``[0.03001, 9.99999]`` before any of this, again matching the R source
(the raw ``popt.f.mud`` table only covers chlorophyll 0.03-10 mg/m^3 and
returns ``NaN``, which would break the iterative solve, outside that range).

Validated against the real R functions (`popt.R`, run directly): for waves
``[320, 340, 380, 443, 555, 665, 780, 875]`` and a 0.035 m instrument radius,
``chl in {0, 0.03}`` -> clamped to 0.03001, chl in {10, 15} -> clamped to
9.99999 (confirmed identical output for each clamped pair), and the >700 nm
band matches ``popt.f.aw`` exactly regardless of ``chl``.
"""

from __future__ import annotations

import numpy as np

_CHL_MIN = 0.03001
_CHL_MAX = 9.99999

_D_E = np.array(
    [
        [350.0, 0.778], [355.0, 0.767],
        [360.0, 0.756], [365.0, 0.737], [370.0, 0.720], [375.0, 0.700],
        [380.0, 0.685], [385.0, 0.670], [390.0, 0.655], [395.0, 0.645],
        [400.0, 0.64358], [405.0, 0.64776], [410.0, 0.65175], [415.0, 0.65555],
        [420.0, 0.65917], [425.0, 0.66259], [430.0, 0.66583], [435.0, 0.66889],
        [440.0, 0.67175], [445.0, 0.67443], [450.0, 0.67692], [455.0, 0.67923],
        [460.0, 0.68134], [465.0, 0.68327], [470.0, 0.68501], [475.0, 0.68657],
        [480.0, 0.68794], [485.0, 0.68903], [490.0, 0.68955], [495.0, 0.68947],
        [500.0, 0.68880], [505.0, 0.68753], [510.0, 0.68567], [515.0, 0.68320],
        [520.0, 0.68015], [525.0, 0.67649], [530.0, 0.67224], [535.0, 0.66739],
        [540.0, 0.66195], [545.0, 0.65591], [550.0, 0.64927], [555.0, 0.64204],
        [560.0, 0.6400], [565.0, 0.6300], [570.0, 0.6230], [575.0, 0.6150],
        [580.0, 0.6100], [585.0, 0.6140], [590.0, 0.6180], [595.0, 0.6220],
        [600.0, 0.6260], [605.0, 0.6300], [610.0, 0.6340], [615.0, 0.6380],
        [620.0, 0.6420], [625.0, 0.6470], [630.0, 0.6530], [635.0, 0.6580],
        [640.0, 0.6630], [645.0, 0.6670], [650.0, 0.6720], [655.0, 0.6770],
        [660.0, 0.6820], [665.0, 0.6870], [670.0, 0.6950], [675.0, 0.6970],
        [680.0, 0.6930], [685.0, 0.6650], [690.0, 0.6400], [695.0, 0.6200],
        [700.0, 0.6000],
    ]
)

_D_CHI = np.array(
    [
        [350.0, 0.153], [355.0, 0.149],
        [360.0, 0.144], [365.0, 0.140], [370.0, 0.136], [375.0, 0.131],
        [380.0, 0.127], [385.0, 0.123], [390.0, 0.119], [395.0, 0.118],
        [400.0, 0.11748], [405.0, 0.12066], [410.0, 0.12259], [415.0, 0.12326],
        [420.0, 0.12269], [425.0, 0.12086], [430.0, 0.11779], [435.0, 0.11372],
        [440.0, 0.10963], [445.0, 0.10560], [450.0, 0.10165], [455.0, 0.09776],
        [460.0, 0.09393], [465.0, 0.09018], [470.0, 0.08649], [475.0, 0.08287],
        [480.0, 0.07932], [485.0, 0.07584], [490.0, 0.07242], [495.0, 0.06907],
        [500.0, 0.06579], [505.0, 0.06257], [510.0, 0.05943], [515.0, 0.05635],
        [520.0, 0.05341], [525.0, 0.05072], [530.0, 0.04829], [535.0, 0.04611],
        [540.0, 0.04419], [545.0, 0.04253], [550.0, 0.04111], [555.0, 0.03996],
        [560.0, 0.0390], [565.0, 0.0375], [570.0, 0.0360], [575.0, 0.0340],
        [580.0, 0.0330], [585.0, 0.0328], [590.0, 0.0325], [595.0, 0.0330],
        [600.0, 0.0340], [605.0, 0.0350], [610.0, 0.0360], [615.0, 0.0375],
        [620.0, 0.0385], [625.0, 0.0400], [630.0, 0.0420], [635.0, 0.0430],
        [640.0, 0.0440], [645.0, 0.0445], [650.0, 0.0450], [655.0, 0.0460],
        [660.0, 0.0475], [665.0, 0.0490], [670.0, 0.0515], [675.0, 0.0520],
        [680.0, 0.0505], [685.0, 0.0440], [690.0, 0.0390], [695.0, 0.0340],
        [700.0, 0.0300],
    ]
)

# Pope & Fry from 385 nm; Segelstein & Fry 350-375 nm; 380 nm is a splice point.
_D_AW = np.array(
    [
        [350.0, 0.02037], [355.0, 0.01819], [360.0, 0.01565], [365.0, 0.01319],
        [370.0, 0.01241], [375.0, 0.01096],
        [380.0, 0.01040], [382.5, 0.01044], [385.0, 0.00941], [387.5, 0.00917],
        [390.0, 0.00851], [392.5, 0.00829],
        [395.0, 0.00813], [397.5, 0.00775], [400.0, 0.00663], [402.5, 0.00579],
        [405.0, 0.00530], [407.5, 0.00503],
        [410.0, 0.00473], [412.5, 0.00452], [415.0, 0.00444], [417.5, 0.00442],
        [420.0, 0.00454], [422.5, 0.00474],
        [425.0, 0.00478], [427.5, 0.00482], [430.0, 0.00495], [432.5, 0.00504],
        [435.0, 0.00530], [437.5, 0.00580],
        [440.0, 0.00635], [442.5, 0.00696], [445.0, 0.00751], [447.5, 0.00830],
        [450.0, 0.00922], [452.5, 0.00969],
        [455.0, 0.00962], [457.5, 0.00957], [460.0, 0.00979], [462.5, 0.01005],
        [465.0, 0.01011], [467.5, 0.0102],
        [470.0, 0.0106], [472.5, 0.0109], [475.0, 0.0114], [477.5, 0.0121],
        [480.0, 0.0127], [482.5, 0.0131],
        [485.0, 0.0136], [487.5, 0.0144], [490.0, 0.0150], [492.5, 0.0162],
        [495.0, 0.0173], [497.5, 0.0191],
        [500.0, 0.0204], [502.5, 0.0228], [505.0, 0.0256], [507.5, 0.0280],
        [510.0, 0.0325], [512.5, 0.0372],
        [515.0, 0.0396], [517.5, 0.0399], [520.0, 0.0409], [522.5, 0.0416],
        [525.0, 0.0417], [527.5, 0.0428],
        [530.0, 0.0434], [532.5, 0.0447], [535.0, 0.0452], [537.5, 0.0466],
        [540.0, 0.0474], [542.5, 0.0489],
        [545.0, 0.0511], [547.5, 0.0537], [550.0, 0.0565], [552.5, 0.0593],
        [555.0, 0.0596], [557.5, 0.0606],
        [560.0, 0.0619], [562.5, 0.0640], [565.0, 0.0642], [567.5, 0.0672],
        [570.0, 0.0695], [572.5, 0.0733],
        [575.0, 0.0772], [577.5, 0.0836], [580.0, 0.0896], [582.5, 0.0989],
        [585.0, 0.1100], [587.5, 0.1220],
        [590.0, 0.1351], [592.5, 0.1516], [595.0, 0.1672], [597.5, 0.1925],
        [600.0, 0.2224], [602.5, 0.2470],
        [605.0, 0.2577], [607.5, 0.2629], [610.0, 0.2644], [612.5, 0.2665],
        [615.0, 0.2678], [617.5, 0.2707],
        [620.0, 0.2755], [622.5, 0.2810], [625.0, 0.2834], [627.5, 0.2904],
        [630.0, 0.2916], [632.5, 0.2995],
        [635.0, 0.3012], [637.5, 0.3077], [640.0, 0.3108], [642.5, 0.322],
        [645.0, 0.325], [647.5, 0.335],
        [650.0, 0.340], [652.5, 0.358], [655.0, 0.371], [657.5, 0.393],
        [660.0, 0.410], [662.5, 0.424],
        [665.0, 0.429], [667.5, 0.436], [670.0, 0.439], [672.5, 0.448],
        [675.0, 0.448], [677.5, 0.461],
        [680.0, 0.465], [682.5, 0.478], [685.0, 0.486], [687.5, 0.502],
        [690.0, 0.516], [692.5, 0.538],
        [695.0, 0.559], [697.5, 0.592], [700.0, 0.624], [702.5, 0.663],
        [705.0, 0.704], [707.5, 0.756],
        [710.0, 0.827], [712.5, 0.914], [715.0, 1.007], [717.5, 1.119],
        [720.0, 1.231], [722.5, 1.356],
        [725.0, 1.489], [727.5, 1.678], [775.0, 2.400], [865.0, 5.550],
    ]
)

_MUD_CHLOR = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
_MUD_W = np.array([350, 400, 412, 443, 490, 510, 555, 620, 670, 700], dtype=float)
_MUD_VAL = np.array(
    [
        [0.770, 0.769, 0.766, 0.767, 0.767, 0.767],
        [0.770, 0.769, 0.766, 0.767, 0.767, 0.767],
        [0.765, 0.770, 0.774, 0.779, 0.782, 0.782],
        [0.800, 0.797, 0.796, 0.797, 0.799, 0.799],
        [0.841, 0.824, 0.808, 0.797, 0.791, 0.791],
        [0.872, 0.855, 0.834, 0.811, 0.796, 0.796],
        [0.892, 0.879, 0.858, 0.827, 0.795, 0.795],
        [0.911, 0.908, 0.902, 0.890, 0.871, 0.871],
        [0.914, 0.912, 0.909, 0.901, 0.890, 0.890],
        [0.914, 0.912, 0.909, 0.901, 0.890, 0.890],
    ]
)  # val[w_index, chlor_index], 0-based (R's popt.d.mud$val is 1-based)


def _interp_rule1(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation, NaN outside ``[xp[0], xp[-1]]`` -- ``approxfun(..., rule=1)``."""
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    return np.where((x < xp[0]) | (x > xp[-1]), np.nan, y)


def _e(wl):
    return _interp_rule1(wl, _D_E[:, 0], _D_E[:, 1])


def _chi(wl):
    return _interp_rule1(wl, _D_CHI[:, 0], _D_CHI[:, 1])


def _aw(wl):
    return _interp_rule1(wl, _D_AW[:, 0], _D_AW[:, 1])


def _mud(wl: float, chl: float) -> float:
    """Port of ``popt.f.mud``: bilinear-ish lookup, ``NaN`` outside its tabulated domain."""
    if wl < _MUD_W[0] or wl > _MUD_W[-1] or chl < _MUD_CHLOR[0] or chl > _MUD_CHLOR[-1]:
        return float("nan")

    i = min(int(np.searchsorted(_MUD_W, wl, side="right")) - 1, len(_MUD_W) - 2)
    j = min(int(np.searchsorted(_MUD_CHLOR, chl, side="right")) - 1, len(_MUD_CHLOR) - 2)

    mud1 = np.interp(wl, _MUD_W[i : i + 2], _MUD_VAL[i : i + 2, j])
    mud2 = np.interp(wl, _MUD_W[i : i + 2], _MUD_VAL[i : i + 2, j + 1])
    return float(np.interp(chl, _MUD_CHLOR[j : j + 2], [mud1, mud2]))


def _b550(chl: float) -> float:
    return 0.416 * chl**0.767


def _bw(wl) -> float:
    return 0.00288 / (wl / 500.0) ** 4.32


def _kw(wl) -> float:
    return _aw(wl) + 0.5 * _bw(wl)


def _sdobp(wl: float, chl: float) -> float:
    expo = -0.5 * np.log10(0.5 * chl) if chl < 2 else 0.0
    return (550.0 / wl) ** expo


def _a_single(wl: float, chl: float) -> float:
    """Port of ``popt.f.a`` for one wavelength/chlorophyll pair."""
    chi = float(_chi(wl))
    e = float(_e(wl))
    mud = _mud(wl, chl)

    bbw = 0.5 * _bw(wl)
    if abs(chl) < 0.0001:
        bbp = 0.0
        kd = float(_kw(wl))
    else:
        bp = (_b550(chl) - _bw(550.0)) * _sdobp(wl, chl)
        bbpt = 0.002 + 0.01 * (0.5 - 0.25 * np.log10(chl))
        bbp = bbpt * bp
        kd = float(_kw(wl)) + chi * chl**e
    bb = bbw + bbp

    u1 = 0.75
    r1 = 0.33 * bb / u1 / kd
    err = 1.0
    u2 = r1
    while err >= 0.0001:
        u2 = mud * (1.0 - r1) / (1.0 + mud * r1 / 0.42)
        r2 = 0.33 * bb / u2 / kd
        err = abs((r2 - r1) / r2)
        r1 = r2
    return u2 * kd


def chlorophyll_absorption(waves: np.ndarray, chl: float) -> np.ndarray:
    """Per-wavelength absorption ``a(lambda)`` for a genuine positive ``chl``.

    Port of the inline wavelength-domain handling in ``shadow.correction.R``'s
    ``!is.na(chl) && chl != 999`` branch: 350-700 nm uses the full Morel &
    Maritorena-style backscatter/reflectance solve (:func:`popt.f.a`-equivalent
    -- ``_a_single``); above 700 nm (up to the ``popt.f.aw`` table's 865 nm
    limit) uses pure-water absorption only; below 350 nm every band reuses
    one value computed at 350.001 nm. ``chl`` is clamped to
    ``[0.03001, 9.99999]`` first, matching the R source.
    """
    waves = np.asarray(waves, dtype=float)
    chl_clamped = min(max(chl, _CHL_MIN), _CHL_MAX)

    values = np.empty(waves.shape, dtype=float)

    is_ir = waves > 700.0
    is_uv = waves < 350.0
    is_main = ~is_ir & ~is_uv

    main_waves = np.clip(waves[is_main], 350.001, 699.999)
    values[is_main] = [_a_single(wl, chl_clamped) for wl in main_waves]

    ir_waves = np.clip(waves[is_ir], 699.999, 864.999)
    values[is_ir] = _aw(ir_waves)

    if np.any(is_uv):
        values[is_uv] = _a_single(350.001, chl_clamped)

    return values
