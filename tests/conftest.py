from __future__ import annotations

HEADER_ROW = (
    "GeneralExcelTime,DateTime,DateTimeUTC,Millisecond (ms),"
    "LuZDepth (m),EuZDepth (m),"
    "Ed0Roll (deg),Ed0Pitch (deg),EdZRoll (deg),EdZPitch (deg),"
    "Ed0320 (uW/(cm2 nm)),Ed0340 (uW/(cm2 nm)),Ed0443 (uW/(cm2 nm)),"
    "EdZ320 (uW/(cm2 nm)),EdZ340 (uW/(cm2 nm)),EdZ443 (uW/(cm2 nm)),"
    "LuZ320 (uW/(sr cm2 nm)),LuZ340 (uW/(sr cm2 nm)),LuZ443 (uW/(sr cm2 nm)),LuZTemp (degC),"
    "EuZ320 (uW/(cm2 nm)),EuZ340 (uW/(cm2 nm)),EuZ443 (uW/(cm2 nm)),EuZTemp (degC),"
    "BioGPSPosition (position),BioShadePosition (position)"
)

DATA_ROWS = (
    "43694.9229143056,08/17/2019 22:08:59,08/17/2019 10:08:59.796 PM,796,"
    "0.098,0.099,2.79,2.23,9.97,-5.88,"
    "2.25,6.04,19.29,"
    "2.98,7.49,25.59,"
    "4.22e-05,8.79e-05,0.0114,9.39,"
    "-0.0001,0.0005,0.0245,9.98,"
    "-998,-38400",
    "43694.922915081,08/17/2019 22:08:59,08/17/2019 10:08:59.863 PM,863,"
    "0.092,0.091,2.72,1.81,9.76,-5.74,"
    "2.25,6.04,19.25,"
    "3.35,8.40,28.23,"
    "3.58e-05,7.41e-05,0.0117,9.46,"
    "0.0004,0.0007,0.0566,10.00,"
    "-998,-38400",
    "43694.9229158565,08/17/2019 22:08:59,08/17/2019 10:08:59.929 PM,929,"
    "0.093,0.094,2.65,1.90,9.80,-5.70,"
    "2.26,6.05,19.20,"
    "3.40,8.50,28.50,"
    "3.60e-05,7.50e-05,0.0118,9.44,"
    "0.0005,0.0008,0.0570,10.01,"
    "-998,-38400",
)


def _write_cast_file(tmp_path, filename, with_header_block=False):
    lines = list(DATA_ROWS)
    content = "\n".join([HEADER_ROW, *lines]) + "\n"
    if with_header_block:
        content = "Start of Header\ninstrument config dump\nEnd of Header\n" + content
    path = tmp_path / filename
    path.write_text(content)
    return path


def write_urc_cast(tmp_path, with_header_block=False):
    path = _write_cast_file(tmp_path, "WISE_CAST_001_190817_220856_URC.csv", with_header_block)
    (tmp_path / "GPS_190817.tsv").write_text("dummy gps content\n")
    return path


INIT_COPS_DAT = """\
###############################################################################
# lines beginning with # are comments; blank lines are skipped
###############################################################################

verbose;logical;TRUE

indice.water;numeric; 1.34
rau.Fresnel;numeric; 0.043

instruments.optics;character;Ed0,EdZ,LuZ,EuZ
tiltmax.optics;numeric; 10,5,5,5
depth.interval.for.smoothing.optics;numeric; 10, 4,4,4
sub.surface.removed.layer.optics;numeric; 0, 0.1, 0,0
delta.capteur.optics;numeric; 0, -0.05, 0.238,0.238
radius.instrument.optics;numeric; 0.035, 0.035, 0.035,0.035

format.date;character;%m/%d/%Y %H:%M:%S
instruments.others;character;NA
depth.is.on;character;LuZ

number.of.fields.before.date;numeric; 3
time.window;numeric;0, 10000
bandwidth;numeric;10
"""

INFO_COPS_DAT = """\
#######################################################################
# file;lon;lat;chl_or_flag;time.window;sub_surface;tiltmax;depth_interval;darks...
#######################################################################
WISE_CAST_001_190817_220856_URC.csv;-68.11626;49.24872;0;x;x;x;x;;;;
WISE_CAST_002_190817_221224_URC.csv;-68.11626;49.24872;NA;x;x;x;x;;;;
WISE_CAST_003_190817_221636_URC.csv;-68.11626;49.24872;0.2;0,90;0.1,0.05,0.1,0;10,10,5,5;40,60,80,80;dark_001.csv
"""

# Only covers casts 001 and 002 on purpose, to exercise the "missing row
# defaults to kept/normal" fallback in discover_deployment() for cast 003.
SELECT_COPS_DAT = """\
WISE_CAST_001_190817_220856_URC.csv;1;Rrs.0p;NA
WISE_CAST_002_190817_221224_URC.csv;0;Rrs.0p.linear;NA
"""


def write_deployment(tmp_path, with_select=True):
    """Write a small synthetic COPS deployment folder: init/info/select + 3 casts."""
    (tmp_path / "init.cops.dat").write_text(INIT_COPS_DAT)
    (tmp_path / "info.cops.dat").write_text(INFO_COPS_DAT)
    if with_select:
        (tmp_path / "select.cops.dat").write_text(SELECT_COPS_DAT)

    for filename in (
        "WISE_CAST_001_190817_220856_URC.csv",
        "WISE_CAST_002_190817_221224_URC.csv",
        "WISE_CAST_003_190817_221636_URC.csv",
    ):
        _write_cast_file(tmp_path, filename)

    return tmp_path


BAD_CAST_FILENAME = "WISE_CAST_002_notadate_notatime_URC.csv"


def write_deployment_with_bad_cast(tmp_path):
    """Like :func:`write_deployment`, but cast 002 has an unparseable file name.

    Exercises ``read_deployment_casts()``'s per-cast failure isolation: casts
    001/003 should still be read even though 002 can't be.
    """
    (tmp_path / "init.cops.dat").write_text(INIT_COPS_DAT)
    (tmp_path / "info.cops.dat").write_text(
        INFO_COPS_DAT.replace("WISE_CAST_002_190817_221224_URC.csv", BAD_CAST_FILENAME)
    )
    (tmp_path / "select.cops.dat").write_text(
        SELECT_COPS_DAT.replace("WISE_CAST_002_190817_221224_URC.csv", BAD_CAST_FILENAME)
    )

    for filename in (
        "WISE_CAST_001_190817_220856_URC.csv",
        BAD_CAST_FILENAME,
        "WISE_CAST_003_190817_221636_URC.csv",
    ):
        _write_cast_file(tmp_path, filename)

    return tmp_path
