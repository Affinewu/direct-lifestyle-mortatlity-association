# -*- coding: utf-8 -*-
"""
lsm_core.py -- library for "How much of a lifestyle-mortality association is
direct?  An information-theoretic analysis of six habits in 29 371 US adults"
(NHANES 2007-2018 with the 2019 public-use Linked Mortality Files).

Three files make up the whole package (plus an optional cross-check of the
Cox models against statsmodels, crosscheck_statsmodels.py):

    lsm_core.py       this file: configuration, data acquisition, harmonisation,
                      the regularised direct-association measures, the survival
                      and regression models, and a self-test suite
    lsm_analysis.py   every analysis, writing output/*.csv
    lsm_report.py     tables, numbers.json, figures, Markdown -> PDF rendering,
                      and the checks of the manuscript against the analysis

Command line
------------
    python lsm_core.py test          validate the estimators (run this first)
    python lsm_core.py fetch         download the NHANES and linked mortality files
    python lsm_core.py verify-data   check data/ against MANIFEST.csv (SHA-256)
    python lsm_core.py build         pool the cycles, harmonise, and extract the
                                     survey design variables (no download)
    python lsm_core.py all           fetch, build, test

Neither ``lifelines`` nor ``statsmodels`` is required: the Cox model and the
IRLS fits are implemented here and validated in ``selftest()`` against data with
a known hazard ratio, including staggered administrative censoring and
exposure-dependent left truncation.

Requires: numpy, pandas (and matplotlib for lsm_report).
"""
from __future__ import annotations

import itertools
import os
import ssl
import sys
import time
import urllib.request

# One BLAS thread per process, set before numpy is imported: the resampling
# loops already run one process per core, and a multithreaded BLAS in each of
# them oversubscribes the machine (the cross-fitted IRLS fits ran about 18
# times slower with default threading under load).  Override by setting the
# variables before starting Python.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Optional (LSM_LOW_PRIORITY=1): run below normal priority, so that a long run
# leaves a machine in other use responsive; worker processes inherit it.
if os.environ.get("LSM_LOW_PRIORITY"):
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            _k = ctypes.windll.kernel32
            _k.GetCurrentProcess.restype = wintypes.HANDLE
            _k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            _k.SetPriorityClass(_k.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL
        else:
            os.nice(5)
    except Exception:                                               # noqa: BLE001
        pass

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

# =====================================================================
# A.  Configuration
# =====================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
#: data/ next to this file (the reader package), or, in the authors' project
#: folder, shared/data; LSM_DATA overrides both
_DEFAULT_DATA = os.path.join(HERE, "data")
if (not os.path.isdir(_DEFAULT_DATA)
        and os.path.isdir(os.path.join(HERE, "shared", "data"))):
    _DEFAULT_DATA = os.path.join(HERE, "shared", "data")
DATA_DIR = os.environ.get("LSM_DATA", _DEFAULT_DATA)
OUT_DIR = os.environ.get("LSM_OUT", os.path.join(HERE, "output"))
FIG_DIR = os.path.join(HERE, "figures")
SUB_DIR = os.path.join(HERE, "submission")
for _d in (OUT_DIR, FIG_DIR, SUB_DIR):
    os.makedirs(_d, exist_ok=True)

RAW_PKL = os.path.join(OUT_DIR, "nhanes_raw.pkl")
CLEAN_PKL = os.path.join(OUT_DIR, "nhanes_clean.pkl")
DESIGN_CSV = os.path.join(OUT_DIR, "design_vars.csv")

#: master seed (the reproduction seed of Wu & Wu, arXiv:2604.18653v2)
SEED = 20260419

#: horizon in months for the standardised cumulative-risk estimand
HORIZON = 120.0

CYCLES = [("1999-2000", "", "1999"), ("2001-2002", "_B", "2001"),
          ("2003-2004", "_C", "2003"), ("2005-2006", "_D", "2005"),
          ("2007-2008", "_E", "2007"), ("2009-2010", "_F", "2009"),
          ("2011-2012", "_G", "2011"), ("2013-2014", "_H", "2013"),
          ("2015-2016", "_I", "2015"), ("2017-2018", "_J", "2017")]

COMPONENTS = ["DEMO", "BMX", "SMQ", "ALQ", "PAQ", "SLQ", "MCQ", "DIQ", "HSQ"]
#: GPAQ (physical activity and sedentary time) begins in 2007-2008.  This is the
#: binding constraint on the primary analytic sample.
PA_CYCLES = {"2007-2008", "2009-2010", "2011-2012",
             "2013-2014", "2015-2016", "2017-2018"}
#: the sleep questionnaire begins in 2005-2006; requesting it earlier is a 404.
#: The medical-conditions, diabetes and health-status questionnaires are used
#: only for the prevalent-disease analyses (SA17, and the healthy/ill panel of SA19), so only the
#: cycles of the primary sample are downloaded.
COMPONENT_CYCLES = {"SLQ": {"2005-2006", "2007-2008", "2009-2010", "2011-2012",
                            "2013-2014", "2015-2016", "2017-2018"},
                    "MCQ": PA_CYCLES, "DIQ": PA_CYCLES, "HSQ": PA_CYCLES}

XPT_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
LMF_URL = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/"
           "linked_mortality/{name}")
#: byte sizes from the official NCHS listing, used as an integrity check
LMF_SIZE = {"1999_2000": 487666, "2001_2002": 540362, "2003_2004": 495722,
            "2005_2006": 506774, "2007_2008": 498376, "2009_2010": 517751,
            "2011_2012": 475574, "2013_2014": 494268, "2015_2016": 484288,
            "2017_2018": 449623}

HABITS = ["smoke", "alc", "bmi", "pa", "sleep", "sed"]
LABEL = {"smoke": "Smoking", "alc": "Alcohol use", "bmi": "Body mass index",
         "pa": "Physical activity", "sleep": "Sleep duration",
         "sed": "Sitting time"}
LEVELS = {
    "smoke": ["never", "former", "current"],
    "alc": ["lifetime abstainer", "former drinker", "light (>0–3.5 drinks/week)",
            "moderate (>3.5–14 drinks/week)", "heavy (>14 drinks/week)"],
    "bmi": ["<18.5", "18.5–24.9", "25–29.9", "30–34.9", "≥35"],
    "pa": ["none (0 MET-min/week)", "insufficient (>0–<600)",
           "meeting (600–1799)", "active (≥1800)"],
    "sleep": ["<6 h", "6–<7 h", "7–<9 h", "≥9 h"],
    "sed": ["<240 min/day", "240–419", "420–599", "≥600"],
}
#: reference (lowest adjusted-risk) level for each exposure
REF_LEVEL = {"smoke": 0, "alc": 2, "bmi": 1, "pa": 3, "sleep": 2, "sed": 0}

#: covariates required complete for the analytic sample, and adjusted for in
#: the Cox models: age band, sex, race/ethnicity (5), education (3),
#: income-to-poverty ratio (4, the fourth being "missing"), marital status (3)
#: and four-year survey period
COVARIATES = ["age_b", "sex_b", "race5", "educ", "pir", "marital", "cyc5"]
#: the prognostic score uses 5-year rather than 10-year age groups, so that
#: its within-band tertiles also order participants by age
SCORE_COVARIATES = ["age5", "sex_b", "race5", "educ", "pir", "marital", "cyc5"]
#: exact demographic stratification (marital status omitted to limit sparsity)
RAW_STRATA = ["age_b", "sex_b", "race", "educ", "pir", "cyc5"]
#: 20-34, 35-44, 45-54, 55-64, 65-74, >=75 years (left-closed)
AGE_BINS = [20, 35, 45, 55, 65, 75, np.inf]
#: 20-24, ..., 75-79, and 80 (RIDAGEYR is top-coded at 80 in 2007-2018)
AGE5_BINS = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, np.inf]


def component_cycles(component):
    return COMPONENT_CYCLES.get(component, {c[0] for c in CYCLES})


# =====================================================================
# B.  Data acquisition
# =====================================================================
def _fetch(url, path, expected=None, insecure=False, log=None):
    if log is None:
        log = []
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        log.append("SKIP  %s" % os.path.basename(path))
        return True
    ctx = ssl._create_unverified_context() if insecure else None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180, context=ctx) as resp:
                data = resp.read()
            if expected is not None and len(data) != expected:
                log.append("BAD   %s: %d bytes, expected %d"
                                   % (os.path.basename(path), len(data), expected))
                continue
            with open(path, "wb") as fh:
                fh.write(data)
            log.append("OK    %s (%d bytes)" % (os.path.basename(path), len(data)))
            return True
        except Exception as exc:                                    # noqa: BLE001
            log.append("ERR   %s: %s (attempt %d/3)"
                               % (os.path.basename(path), str(exc)[:70], attempt))
            time.sleep(2 * attempt)
    return False


def fetch(verbose=True):
    """Download every public source file that is not already in DATA_DIR, then
    check all of them against the SHA-256 manifest.  To download afresh, point
    LSM_DATA at an empty directory: the manifest shipped with the code
    (data/MANIFEST.csv next to this file) is then used for the check."""
    log, ok = [], True
    os.makedirs(DATA_DIR, exist_ok=True)
    for cycle, suffix, year in CYCLES:
        for comp in COMPONENTS:
            if cycle not in component_cycles(comp):
                log.append("N/A   %s%s (%s)" % (
                    comp, suffix, "not fielded in " + cycle if comp == "SLQ"
                    else "not needed: only the 2007-2018 cycles are used"))
                continue
            name = comp + suffix
            ok &= _fetch(XPT_URL.format(year=year, name=name),
                         os.path.join(DATA_DIR, name + ".xpt"), log=log)
    for cycle, suffix, year in CYCLES:
        key = "%s_%02d" % (year, int(year) + 1)
        fname = "NHANES_%s_MORT_2019_PUBLIC.dat" % key
        # ftp.cdc.gov presents a chain some minimal environments cannot verify;
        # the byte-size check below is what actually guarantees integrity
        ok &= _fetch(LMF_URL.format(name=fname), os.path.join(DATA_DIR, fname),
                     expected=LMF_SIZE[key], insecure=True, log=log)
    if verbose:
        print("\n".join(log))
    if not ok:
        raise SystemExit("at least one file failed to download")
    if verify_data(verbose):
        raise SystemExit("downloaded files do not match the manifest")
    return log


def manifest_path():
    """MANIFEST.csv in DATA_DIR, or else the one shipped with the code (data/
    next to this file, or shared/data in the authors' project folder), which a
    fresh download into an empty LSM_DATA directory is checked against."""
    cands = [os.path.join(p, "MANIFEST.csv") for p in
             (DATA_DIR, os.path.join(HERE, "data"), os.path.join(HERE, "shared", "data"))]
    return next((m for m in cands if os.path.exists(m)), cands[0])


def verify_data(verbose=True):
    """Check every file in DATA_DIR against MANIFEST.csv (byte size and SHA-256).
    Returns the list of files that are missing or differ."""
    import csv
    import hashlib
    man = manifest_path()
    if not os.path.exists(man):
        if verbose:
            print("no MANIFEST.csv found -- data NOT verified")
        return ["MANIFEST.csv"]
    bad, n = [], 0
    for r in csv.DictReader(open(man, encoding="utf-8")):
        n += 1
        p = os.path.join(DATA_DIR, r["file"])
        if (not os.path.exists(p) or os.path.getsize(p) != int(r["bytes"])
                or hashlib.sha256(open(p, "rb").read()).hexdigest() != r["sha256"]):
            bad.append(r["file"])
    if verbose:
        print("data: %d files checked against MANIFEST.csv, %d missing or different%s"
              % (n, len(bad), (": %s" % bad) if bad else ""))
    return bad


def build_design_vars(verbose=True):
    """Masked stratum and variance unit and the examination weight, for the
    complex-survey sensitivity analysis, extracted from the DEMO files already
    in DATA_DIR (no download).  Strata and PSUs are prefixed with the cycle so
    that they are unique across cycles.  The weight is left as the 2-year
    WTMEC2YR; the analysis divides it by the number of cycles it pools."""
    frames = []
    for cycle, suffix, year in CYCLES:
        path = os.path.join(DATA_DIR, "DEMO%s.xpt" % suffix)
        if not os.path.exists(path):
            continue
        d = pd.read_sas(path, format="xport")
        keep = [c for c in ["SEQN", "SDMVSTRA", "SDMVPSU", "WTMEC2YR", "WTINT2YR"]
                if c in d.columns]
        d = d[keep].copy()
        for c in d.columns:
            d[c] = _denormal_to_zero(pd.to_numeric(d[c], errors="coerce"))
        d["cycle"] = cycle
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["stratum"] = out["cycle"].astype(str) + "_" + out["SDMVSTRA"].astype("Int64").astype(str)
    out["psu"] = out["stratum"] + "_" + out["SDMVPSU"].astype("Int64").astype(str)
    out.to_csv(DESIGN_CSV, index=False)
    if verbose:
        print("design variables: %d rows, %d strata, %d variance units"
              % (len(out), out.stratum.nunique(), out.psu.nunique()))


# =====================================================================
# C.  Cohort assembly
# =====================================================================
#: source component -> candidate variable names, in order of preference.
#: RIDRETH1 is preferred over RIDRETH3 deliberately: it is the five-category
#: variable issued unchanged in all ten cycles, whereas RIDRETH3 exists only from
#: 2011 and splits category 5, leaving those codes unmapped when pooled -- which
#: silently drops ~3.7k adults from every race-adjusted model.  RIDRETH1's five
#: groups are used in the scores and Cox models, three in the exact strata.
DEMO_MAP = [(["RIAGENDR"], "sex"), (["RIDAGEYR"], "age"),
            (["RIDRETH1", "RIDRETH3"], "race"), (["DMDEDUC2"], "educ"),
            (["INDFMPIR"], "pir"), (["DMDMARTL"], "marital"),
            (["WTMEC2YR", "WTINT2YR"], "wt")]
SMQ_VARS = ["SMQ020", "SMQ040", "SMQ030", "SMQ050Q", "SMD030", "SMD055", "SMD057"]
ALQ_VARS = ["ALQ100", "ALQ101", "ALQ110", "ALQ111", "ALQ120Q", "ALQ120U",
            "ALQ121", "ALQ130"]
PAQ_VARS = ["PAQ605", "PAQ610", "PAD615", "PAQ620", "PAQ625", "PAD630",
            "PAQ635", "PAQ640", "PAD645", "PAQ650", "PAQ655", "PAD660",
            "PAQ665", "PAQ670", "PAD675", "PAD680"]
SLQ_VARS = ["SLD010H", "SLD012", "SLQ050", "SLQ060"]
#: prevalent disease (sensitivity analyses SA17 and the SA19 healthy/ill panel only): ever told by a doctor of
#: congestive heart failure, coronary heart disease, angina, heart attack,
#: stroke, emphysema, chronic bronchitis, COPD or cancer; diabetes; and
#: self-rated general health
MCQ_VARS = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ160G",
            "MCQ160K", "MCQ160O", "MCQ220"]
DIQ_VARS = ["DIQ010"]
HSQ_VARS = ["HSD010"]

#: public-use 2019 Linked Mortality File, fixed-width layout
#:   cols 1-6 SEQN | col 15 ELIGSTAT | col 16 MORTSTAT | 17-19 UCOD | 43-45 PERMTH_INT
#:   | 46-48 PERMTH_EXM (months from the examination; sensitivity analysis SA21)
LMF_FIELDS = [(0, 6, "SEQN"), (14, 15, "ELIGSTAT"), (15, 16, "MORTSTAT"),
              (16, 19, "UCOD"), (42, 45, "PERMTH_INT"), (45, 48, "PERMTH_EXM")]


def _read_xpt(name):
    path = os.path.join(DATA_DIR, name + ".xpt")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_sas(path, format="xport", encoding="utf-8")
    except Exception as exc:                                        # noqa: BLE001
        raise RuntimeError("cannot read %s" % path) from exc


def _read_lmf(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None

    def num(t):
        t = t.strip()
        if t == "" or set(t) <= set(". "):
            return np.nan
        try:
            return float(t)
        except ValueError:
            return np.nan

    rows = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if len(line) < 45:
                continue
            rows.append(tuple(num(line[a:b]) for a, b, _ in LMF_FIELDS))
    return pd.DataFrame(rows, columns=[n for _, _, n in LMF_FIELDS])


def _pick(df, names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def build_raw(verbose=True):
    """Pool the ten cycles and join the linked mortality files."""
    frames, log = [], []
    for cycle, suffix, year in CYCLES:
        demo = _read_xpt("DEMO" + suffix)
        if demo is None:
            log.append("SKIPPED %s (no DEMO)" % cycle)
            continue
        base = demo[["SEQN"]].copy()
        base["cycle"] = cycle
        for names, target in DEMO_MAP:
            s = _pick(demo, names)
            if s is not None:
                base[target] = s.values
        for comp, variables in (("BMX", ["BMXBMI"]), ("SMQ", SMQ_VARS),
                                ("ALQ", ALQ_VARS), ("PAQ", PAQ_VARS),
                                ("SLQ", SLQ_VARS), ("MCQ", MCQ_VARS),
                                ("DIQ", DIQ_VARS), ("HSQ", HSQ_VARS)):
            if cycle not in component_cycles(comp):
                continue
            frame = _read_xpt(comp + suffix)
            if frame is None:
                continue
            keep = ["SEQN"] + [c for c in variables if c in frame.columns]
            if len(keep) > 1:
                base = base.merge(frame[keep], on="SEQN", how="left")
        lmf = _read_lmf("NHANES_%s_%02d_MORT_2019_PUBLIC.dat" % (year, int(year) + 1))
        if lmf is not None:
            lmf["SEQN"] = lmf["SEQN"].astype(np.int64)
            base["SEQN"] = base["SEQN"].astype(np.int64)
            base = base.merge(lmf, on="SEQN", how="left")
        frames.append(base)
        log.append("BUILT %s: n=%d" % (cycle, len(base)))
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw.to_pickle(RAW_PKL)
    log.append("TOTAL pooled records: n=%d" % len(raw))
    if verbose:
        print("\n".join(log))
    return raw


# =====================================================================
# D.  Harmonisation
# =====================================================================
def _num(s):
    return None if s is None else pd.to_numeric(s, errors="coerce")


def _denormal_to_zero(s):
    """pandas.read_sas converts the IBM-format exact zero of SAS transport
    files to 2**-260 (about 5.4e-79) rather than 0, so an ``== 0`` test
    silently misses every true zero (e.g. ALQ120Q = 0, no drinking in the past
    year).  Map all magnitudes <= 1e-30 back to 0."""
    return s.mask(s.notna() & (s.abs() <= 1e-30), 0.0)


def harmonise(verbose=True):
    """Discretise the pooled file into the analysis table."""
    raw = pd.read_pickle(RAW_PKL)
    df = pd.DataFrame(index=raw.index)
    df["SEQN"] = raw["SEQN"]
    df["cycle"] = raw["cycle"]
    df["wt"] = _denormal_to_zero(_num(raw.get("wt")))
    df["mortstat"] = _num(raw["MORTSTAT"])
    df["permth"] = _num(raw["PERMTH_INT"])
    df["permth_exm"] = _num(raw["PERMTH_EXM"])
    df["eligstat"] = _num(raw["ELIGSTAT"])
    age = _num(raw["age"])
    df["age"] = age
    df["sex"] = _num(raw["sex"])

    # ---- race (3): White / Black / other
    race = _num(raw["race"])
    r3 = pd.Series(np.nan, index=race.index)
    r3[race == 3] = 0
    r3[race == 4] = 1
    r3[race.isin([1, 2, 5, 6, 7])] = 2
    unmapped = sorted(set(race.dropna().unique()) - {1, 2, 3, 4, 5, 6, 7})
    if unmapped:
        raise ValueError("unmapped RIDRETH code(s) %s" % unmapped)
    df["race"] = r3
    # ---- race (5), RIDRETH1: White / Black / Mexican American / other Hispanic
    # / other or multiracial.  Used in the prognostic scores and the Cox models,
    # where the finer coding costs nothing; the exact strata keep three groups.
    r5 = pd.Series(np.nan, index=race.index)
    for code, lv in ((3, 0), (4, 1), (1, 2), (2, 3)):
        r5[race == code] = lv
    r5[race.isin([5, 6, 7])] = 4
    df["race5"] = r5

    # ---- education (3)
    edu = _num(raw["educ"])
    e3 = pd.Series(np.nan, index=edu.index)
    e3[edu.isin([1, 2])] = 0
    e3[edu.isin([3, 4])] = 1
    e3[edu == 5] = 2
    e3[edu.isin([7, 9])] = np.nan
    df["educ"] = e3

    # ---- income-to-poverty (4): <1.3, 1.3-<3.5, >=3.5, and missing kept as
    # its own category (INDFMPIR is top-coded at 5)
    pir = _num(raw["pir"])
    p4 = pd.Series(3.0, index=pir.index)
    p4[pir < 1.3] = 0
    p4[(pir >= 1.3) & (pir < 3.5)] = 1
    p4[pir >= 3.5] = 2
    df["pir"] = p4

    # ---- marital status (3)
    mar = _num(raw["marital"])
    m3 = pd.Series(np.nan, index=mar.index)
    m3[mar.isin([1, 6])] = 0
    m3[mar.isin([2, 3, 4])] = 1
    m3[mar == 5] = 2
    df["marital"] = m3

    # ---- BMI (5)
    bmi = _num(raw["BMXBMI"])
    df["bmi_val"] = bmi
    b5 = pd.Series(np.nan, index=bmi.index)
    for code, (lo, hi) in enumerate([(-np.inf, 18.5), (18.5, 25), (25, 30),
                                     (30, 35), (35, np.inf)]):
        b5[(bmi >= lo) & (bmi < hi)] = code
    df["bmi"] = b5

    # ---- smoking (3)
    smq020, smq040 = _num(raw.get("SMQ020")), _num(raw.get("SMQ040"))
    s3 = pd.Series(np.nan, index=smq020.index)
    s3[smq020 == 2] = 0
    s3[(smq020 == 1) & (smq040 == 3)] = 1
    s3[(smq020 == 1) & (smq040.isin([1, 2]))] = 2
    df["smoke"] = s3

    # ---- alcohol (5), with the lifetime-abstention gate
    alq130 = _denormal_to_zero(_num(raw.get("ALQ130")))
    # 777/999 (refused / don't know) are missing; larger valid answers are
    # winsorised at 15, the top code of the 2015-2018 questionnaires
    alq130 = alq130.where(alq130 < 777).clip(upper=15)
    freq = pd.Series(np.nan, index=df.index)
    q = _denormal_to_zero(_num(raw.get("ALQ120Q")))
    u = _denormal_to_zero(_num(raw.get("ALQ120U")))
    if q is not None:
        f = pd.Series(np.nan, index=q.index)
        f[u == 1] = q[u == 1]                            # per week
        f[u == 2] = q[u == 2] * 12 / 52                  # per month
        f[u == 3] = q[u == 3] / 52                       # per year
        f[q == 0] = 0.0
        f[q.isin([777, 999])] = np.nan
        freq = freq.combine_first(f)
    q2 = _denormal_to_zero(_num(raw.get("ALQ121")))
    if q2 is not None:
        map121 = {0.0: 0.0, 1.0: 7.0, 2.0: 6.0, 3.0: 3.5, 4.0: 2.0, 5.0: 1.0,
                  6.0: 0.577, 7.0: 0.230, 8.0: 0.173, 9.0: 0.0865, 10.0: 0.029}
        freq = freq.combine_first(q2.map(map121))
    dpw = freq * alq130                                  # NaN if either is unknown
    dpw[freq == 0] = 0.0                                 # past-year abstainers
    # Respondents failing the lifetime gate (ALQ110 = 2, fewer than 12 drinks in
    # their life, 2007-2016; ALQ111 = 2, never had a drink, 2017-2018) are not
    # asked the past-year items: they are non-drinkers, not missing.
    never = pd.Series(False, index=df.index, dtype=bool)
    for gate in ("ALQ110", "ALQ111"):
        if gate in raw.columns:
            g = _denormal_to_zero(_num(raw[gate]))
            never = never | g.eq(2).reindex(df.index).fillna(False).astype(bool)
    dpw[never] = 0.0
    # Everyone else without a usable answer (no ALQ record, refused / don't
    # know at the gate or on frequency or quantity) is MISSING -- not level 0.
    # Lifetime abstainers (the gate) and former drinkers (past the gate, none
    # in the past 12 months) are kept apart: former drinkers include those who
    # stopped because they became ill.
    alc = pd.Series(np.nan, index=df.index)
    alc[never] = 0
    alc[(~never) & (freq == 0)] = 1
    alc[(~never) & (freq > 0) & (dpw <= 3.5)] = 2
    alc[(~never) & (dpw > 3.5) & (dpw <= 14)] = 3
    alc[(~never) & (dpw > 14)] = 4
    df["alc"] = alc
    df["dpw"] = dpw
    df["alc_never"] = never.astype(float)

    # ---- physical activity and sedentary time (GPAQ, 2007-2008 onwards)
    def bounded(variable, valid_max):
        v = _num(raw.get(variable))
        if v is None:
            return None
        v = v.where(~v.isin([7777, 9999]))
        return v.where(v <= valid_max)

    def pa_component(response, days, minutes, met):
        r = _num(raw.get(response)) if response in raw.columns else None
        if r is None:
            return None
        d = bounded(days, 7)
        m = bounded(minutes, 1440)
        d = d.where(d.between(1, 7))
        # 1 = yes (days x minutes x MET), 2 = no (0); 7/9 (refused, don't
        # know) and missing stay missing
        out = pd.Series(np.nan, index=r.index)
        out[r == 1] = d[r == 1] * m[r == 1] * met
        out[r == 2] = 0.0
        return out

    pa_mask = raw["cycle"].isin(PA_CYCLES)
    met_wk, met_lt, missing = None, None, None
    for spec, work in [(("PAQ605", "PAQ610", "PAD615", 8.0), True),    # vigorous work
                       (("PAQ620", "PAQ625", "PAD630", 4.0), True),    # moderate work
                       (("PAQ635", "PAQ640", "PAD645", 4.0), False),   # transport
                       (("PAQ650", "PAQ655", "PAD660", 8.0), False),   # vigorous leisure
                       (("PAQ665", "PAQ670", "PAD675", 4.0), False)]:  # moderate leisure
        comp = pa_component(*spec)
        if comp is None:
            continue
        comp = comp.where(pa_mask)
        missing = comp.isna() if missing is None else (missing | comp.isna())
        met_wk = comp.fillna(0.0) if met_wk is None else met_wk + comp.fillna(0.0)
        if not work:
            met_lt = comp.fillna(0.0) if met_lt is None else met_lt + comp.fillna(0.0)

    def pa_level(met):
        p4v = pd.Series(np.nan, index=met.index)
        p4v[met == 0] = 0
        p4v[(met > 0) & (met < 600)] = 1
        p4v[(met >= 600) & (met < 1800)] = 2
        p4v[met >= 1800] = 3
        return p4v
    if met_wk is not None:
        met_wk = met_wk.where(pa_mask & ~missing)
        df["met_wk"] = met_wk
        df["pa"] = pa_level(met_wk)
        # leisure and transport only (sensitivity analysis SA19): the work
        # domains dominate the most active level, which then partly contrasts
        # manual workers with people not working
        met_lt = met_lt.where(pa_mask & ~missing)
        df["met_lt"] = met_lt
        df["pa_lt"] = pa_level(met_lt)
    sed = bounded("PAD680", 1440)
    if sed is not None:
        sed = sed.where(pa_mask)
        s4 = pd.Series(np.nan, index=sed.index)
        for code, (lo, hi) in enumerate([(-np.inf, 240), (240, 420),
                                         (420, 600), (600, np.inf)]):
            s4[(sed >= lo) & (sed < hi)] = code
        df["sed"] = s4

    # ---- sleep (4)
    sleep_h = pd.Series(np.nan, index=df.index)
    for var in ("SLD010H", "SLD012"):
        v = _num(raw.get(var))
        if v is not None:
            # SLD010H: whole hours 1-12 (2005-2014); SLD012: half hours
            # 2-14.5 (2015-2018); 77/99 are missing
            sleep_h = sleep_h.combine_first(v.where(v.between(1, 14.5)))
    df["sleep_h"] = sleep_h
    # half-open bins, so that the half-hour reports of 2015-2018 (6.5, 8.5 h)
    # are classified rather than silently lost
    s4s = pd.Series(np.nan, index=df.index)
    s4s[sleep_h < 6] = 0
    s4s[(sleep_h >= 6) & (sleep_h < 7)] = 1
    s4s[(sleep_h >= 7) & (sleep_h < 9)] = 2
    s4s[sleep_h >= 9] = 3
    df["sleep"] = s4s

    # ---- prevalent disease and self-rated health (SA17 and the SA19 healthy/ill panel only)
    # 1 = yes, 2 = no; 7/9 (refused / don't know) and not asked are missing.
    # chronic = 1 if any condition is reported, 0 if every condition that was
    # asked in the cycle is denied (MCQ160O exists only from 2013-2014).
    conds = [c for c in MCQ_VARS if c in raw.columns] + ["DIQ010"]
    yes = pd.Series(False, index=df.index)
    known = pd.Series(True, index=df.index)
    for c in conds:
        if c not in raw.columns:
            continue
        v = _num(raw[c])
        yes = yes | v.eq(1)
        asked = raw["cycle"].isin(
            [cy for cy in PA_CYCLES if c != "MCQ160O"
             or cy in ("2013-2014", "2015-2016", "2017-2018")])
        known = known & (~asked | v.isin([1, 2, 3]))      # DIQ010 3 = borderline
    chronic = pd.Series(np.nan, index=df.index)
    chronic[known] = 0.0
    chronic[yes] = 1.0
    df["chronic"] = chronic
    hs = _num(raw.get("HSD010"))
    poor = pd.Series(np.nan, index=df.index)
    if hs is not None:
        poor[hs.isin([1, 2, 3])] = 0.0
        poor[hs.isin([4, 5])] = 1.0
    df["poor_health"] = poor

    # ---- eligibility and outcome
    eligible = ((age >= 20) & df["mortstat"].isin([0, 1])
                & df["eligstat"].eq(1) & df["permth"].notna())
    df["adult_ok"] = eligible
    df["died"] = df["mortstat"].where(df["mortstat"].isin([0, 1]))
    assert int(eligible.sum()) > 0, "empty eligible cohort"
    if int(df.loc[eligible, "race"].isna().sum()):
        raise ValueError("eligible adults with unmapped race -- check DEMO_MAP")
    df.to_pickle(CLEAN_PKL)
    if verbose:
        sel = df["adult_ok"]
        print("rows pooled                   : %d" % len(df))
        print("adults >=20, linkage-eligible : %d" % int(sel.sum()))
        print("deaths                        : %d" % int((df.loc[sel, "died"] == 1).sum()))
        print("median follow-up (months)     : %.0f" % df.loc[sel, "permth"].median())
        e7 = sel & df["cycle"].isin(PA_CYCLES)
        a7 = load(common=True)
        print("2007-2018, eligible adults    : %d" % int(e7.sum()))
        print("2007-2018, complete cases     : %d" % len(a7))
        print("2007-2018, deaths             : %d" % int(a7.died.sum()))
        print("2007-2018, median follow-up   : %.0f" % a7.permth.median())
    return df


def load(common=True, complete_covariates=True):
    """The analysis table.

    ``common=True`` returns participants complete on all six habits -- the
    primary analytic sample, so that every exposure is estimated in the same
    people under the same adjustment set.  ``complete_covariates=False`` keeps
    adults with a missing covariate (education or marital status), whose
    covariates then stay float with NaN; only the completeness model of SA16
    uses them.
    """
    d = pd.read_pickle(CLEAN_PKL)
    d = d[d.adult_ok].copy()
    d["age_b"] = pd.cut(d["age"], AGE_BINS, right=False, labels=False)
    d["age5"] = pd.cut(d["age"], AGE5_BINS, right=False, labels=False)
    d["sex_b"] = (d["sex"] == 2).astype(float)
    order = [c for c, _, _ in CYCLES]
    d["cyc"] = d["cycle"].map({c: i for i, c in enumerate(order)}).astype(float)
    d["cyc5"] = d["cyc"] // 2
    d["py"] = d["permth"] / 12.0
    # exit age from whole months, so that deaths in the same month of age are
    # exactly tied (age + permth/12 in floating point split one such tie)
    d["age_exit"] = (d["age"] * 12.0 + d["permth"]) / 12.0
    need = (["died", "permth"] + (COVARIATES if complete_covariates else [])
            + (HABITS if common else []))
    d = d.dropna(subset=need).copy()
    cast = ((COVARIATES if complete_covariates else []) + ["race"]
            + (HABITS if common else []))
    d = d.astype({c: int for c in cast})
    d["died"] = d["died"].astype(int)
    return d.reset_index(drop=True)


# =====================================================================
# E.  Regularised direct-association measures (Wu & Wu, arXiv:2604.18653v2)
# =====================================================================
LOG2 = np.log(2.0)


def H(p):
    """Shannon entropy in bits, ignoring zero-probability entries."""
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return 0.0 if p.size == 0 else float(-(p * np.log(p)).sum() / LOG2)


def h2(u):
    u = float(np.clip(u, 0.0, 1.0))
    if u <= 0 or u >= 1:
        return 0.0
    return float(-(u * np.log(u) + (1 - u) * np.log(1 - u)) / LOG2)


def phi(c):
    return h2((1.0 + c) / 2.0) - 0.5 * h2(c)


def cmax(m):
    """Achievable maximum of rMI / rCMI / doMI given only min(dX, dY) = m.
    cmax(2) = 0.5579 is the ceiling for every value reported with a binary
    outcome."""
    return float(np.sqrt(max(phi(1.0 / m), 0.0)))


def js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return max(H(0.5 * (p + q)) - 0.5 * (H(p) + H(q)), 0.0)


def rjs(p, q):
    """sqrt of the Jensen-Shannon divergence: a metric bounded by 1."""
    return float(np.sqrt(js(p, q)))


class Table:
    """Contingency table over integer-coded (X, Y, Z)."""

    def __init__(self, x, y, Z=None, dx=None, dy=None):
        self.x = np.asarray(x, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)
        self.n = self.x.size
        if Z is None or len(Z) == 0:
            self.Z, self.dz = np.zeros((0, self.n), dtype=np.int64), ()
        else:
            self.Z = np.asarray(Z, dtype=np.int64)
            if self.Z.ndim == 1:
                self.Z = self.Z[None, :]
            self.dz = tuple(int(z.max()) + 1 for z in self.Z)
        self.dx = int(dx if dx is not None else self.x.max() + 1)
        self.dy = int(dy if dy is not None else self.y.max() + 1)
        self.dzt = int(np.prod(self.dz)) if self.dz else 1
        code = np.zeros(self.n, dtype=np.int64)
        for z, d in zip(self.Z, self.dz):
            code = code * d + z
        self.zc = code
        self.code = self.x * (self.dy * self.dzt) + self.y * self.dzt + self.zc
        self.D = self.dx * self.dy * self.dzt

    def counts(self, idx=None):
        c = np.bincount(self.code if idx is None else self.code[idx], minlength=self.D)
        return c.astype(float).reshape(self.dx, self.dy, self.dzt)


class _CountTable:
    """Adapter so the null calibration can consume a raw counts array."""

    def __init__(self, cnt):
        self._cnt = np.asarray(cnt, float)

    def counts(self, idx=None):
        return self._cnt


def _parts(cnt):
    p3 = cnt / cnt.sum()
    p_xz, p_yz, p_z = p3.sum(axis=1), p3.sum(axis=0), p3.sum(axis=(0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(p_z[None, :] > 0, p_z[None, :], 1)
        xgz = np.where(p_z[None, :] > 0, p_xz / den, 0.0)
        ygz = np.where(p_z[None, :] > 0, p_yz / den, 0.0)
    return p3, p_xz, p_yz, p_z, xgz, ygz


def rmi_of(cnt):
    """Total association: rMI = sqrt(D_JS(p(x,y) || p(x)p(y)))."""
    p3 = cnt / cnt.sum()
    p_xy = p3.sum(axis=2)
    return rjs(p_xy.ravel(), np.outer(p_xy.sum(axis=1), p_xy.sum(axis=0)).ravel())


def rcmi_of(cnt):
    """Direct association: rCMI = sqrt(D_JS(p(x,y,z) || p(x|z)p(y|z)p(z)))."""
    p3, _, _, p_z, xgz, ygz = _parts(cnt)
    q = xgz[:, None, :] * ygz[None, :, :] * p_z[None, None, :]
    return rjs(p3.ravel(), q.ravel())


def djs_levels(cnt):
    """The exposure levels' shares of rCMI^2 = D_JS(p(x,y,z) || p(x|z)p(y|z)p(z)).

    D_JS is a sum over cells of m log m-type terms, each of which is
    non-negative (for cell probabilities p and q with mean m the term is
    ((p + q) / 2) (1 - h2(p / (p + q))) in bits), so the sum over the cells of
    exposure level x is that level's contribution and the contributions add up
    to rCMI^2 exactly."""
    p3, _, _, p_z, xgz, ygz = _parts(cnt)
    q = xgz[:, None, :] * ygz[None, :, :] * p_z[None, None, :]
    m = 0.5 * (p3 + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (0.5 * np.where(p3 > 0, p3 * np.log(p3), 0.0)
             + 0.5 * np.where(q > 0, q * np.log(q), 0.0)
             - np.where(m > 0, m * np.log(m), 0.0))
    return np.maximum(t, 0.0).sum(axis=(1, 2)) / LOG2


def null_djs_levels(tab, n_null=500, seed=SEED):
    """Mean of djs_levels over the null draws of null_rcmi (the same draws for
    the same seed), so that the level contributions of the null floor add up to
    E_0[rCMI^2]."""
    rng = np.random.default_rng(seed)
    cnt = tab.counts()
    n = int(cnt.sum())
    p3, _, _, p_z, xgz, ygz = _parts(cnt)
    q = (xgz[:, None, :] * ygz[None, :, :] * p_z[None, None, :]).ravel()
    draws = rng.multinomial(n, q / q.sum(), size=n_null)
    return np.mean([djs_levels(c.reshape(p3.shape).astype(float)) for c in draws],
                   axis=0)


def do_dist(cnt):
    """p(y | do(x)); strata with an empty X-cell contribute the marginal p(y)
    rather than a singular conditional (strategy (b) of the source paper)."""
    p3, p_xz, p_yz, p_z, _, _ = _parts(cnt)
    p_y = p_yz.sum(axis=1)
    safe = np.where(p_xz > 0, p_xz, 1.0)
    coef = np.where(p_xz > 0, p_z[None, :] / safe, 0.0)
    pdo = np.einsum("ixk,ik->ix", p3, coef)
    extra = (p_z[None, :] * (p_xz <= 0)).sum(axis=1)
    return pdo + np.outer(extra, p_y)


def rmi_do_of(cnt):
    """Interventional counterpart of rMI."""
    p3, p_xz, _, _, _, _ = _parts(cnt)
    p_x = p_xz.sum(axis=1)
    pdo = do_dist(cnt)
    return rjs((pdo * p_x[:, None]).ravel(), np.outer(p_x, pdo.T @ p_x).ravel())


def race_of(cnt):
    """Largest JS distance between two interventional outcome distributions.
    Being a maximum over pairs, it is driven by rare extreme levels."""
    pdo = do_dist(cnt)
    return float(max((rjs(pdo[i], pdo[j])
                      for i, j in itertools.permutations(range(pdo.shape[0]), 2)),
                     default=0.0))


def achievable_upper_rcmi(cnt, max_enum=4096):
    """Maximum of rCMI over deterministic couplings Y = f_z(X) that preserve
    the observed p(x, z), allowing a different function in each stratum.

    For a coupling the reference distribution is p(x|z) times the coupling's
    OWN outcome margin p_f(y|z); rCMI**2 is then the p(z)-weighted mean of the
    within-stratum JS divergences, each of which is at most cmax(2)**2 when Y
    is binary, so the bound never exceeds cmax(2) = 0.558.  (An earlier
    version used the observed p(y|z) as the reference, which is not the rCMI of
    any distribution and returned 0.87-0.92 for every exposure.)"""
    p3, _, _, p_z, xgz, ygz = _parts(cnt)
    dx, dy, dzt = p3.shape
    if dy ** dx > max_enum:
        return float("nan")
    funcs = list(itertools.product(range(dy), repeat=dx))
    total = 0.0
    for k in range(dzt):
        if p_z[k] <= 0:
            continue
        px = xgz[:, k]
        best = 0.0
        for f in funcs:
            r = np.zeros((dx, dy))
            r[np.arange(dx), f] = px
            best = max(best, js(r.ravel(), np.outer(px, r.sum(axis=0)).ravel()))
        total += p_z[k] * best
    return float(np.sqrt(max(total, 0.0)))


def null_rcmi(tab, n_null=500, seed=SEED, levels=False):
    """Parametric-bootstrap draws of rCMI under H0: X _||_ Y | Z.

    rCMI is a plug-in functional and is strictly positive even when the
    conditional independence holds exactly; the mean of these draws is the
    finite-sample null floor.

    Under the null model the n individuals are independent draws of
    (z, x, y) from p(z) p(x|z) p(y|z), so the table itself is one multinomial
    draw of size n over its cells with those probabilities.  Drawing the table
    directly is exactly the same distribution as simulating every individual,
    and orders of magnitude faster."""
    rng = np.random.default_rng(seed)
    cnt = tab.counts()
    n = int(cnt.sum())
    p3, _, _, p_z, xgz, ygz = _parts(cnt)
    q = (xgz[:, None, :] * ygz[None, :, :] * p_z[None, None, :]).ravel()
    draws = rng.multinomial(n, q / q.sum(), size=n_null)
    tabs = [c.reshape(p3.shape).astype(float) for c in draws]
    out = np.array([rcmi_of(t) for t in tabs])
    if levels:
        # the same draws, split by exposure level (djs_levels); their mean
        # is each level's share of the null floor E_0[rCMI^2]
        return out, np.mean([djs_levels(t) for t in tabs], axis=0)
    return out


def calibrate(tab, n_null=500, seed=SEED):
    """Plug-in rCMI and its two null calibrations.

    ``excess`` subtracts the null expectation on the rCMI (square-root) scale:
    it is centred on zero under conditional independence, but because the
    square root is concave it under-states a real association (see the
    known-truth simulation).  ``calibrated`` subtracts the null expectation
    on the D_JS scale, where the plug-in bias is approximately additive, and
    then takes the square root: sqrt(max(rCMI^2 - E0[rCMI^2], 0)).  ``p_null``
    is the add-one Monte-Carlo upper-tail P value."""
    r = rcmi_of(tab.counts())
    nul = np.asarray(null_rcmi(tab, n_null=n_null, seed=seed), float)
    ms = float(np.mean(nul ** 2))
    return dict(rcmi=r, null_mean=float(nul.mean()), null_ms=ms,
                excess=r - float(nul.mean()),
                calibrated=float(np.sqrt(max(r * r - ms, 0.0))),
                p_null=float((1 + (nul >= r).sum()) / (1 + nul.size)))


def excess_rcmi(tab, n_null=500, seed=SEED):
    """(rCMI, null floor, excess, one-sided add-one P) -- the square-root-scale
    calibration; see ``calibrate`` for the D_JS-scale one."""
    c = calibrate(tab, n_null=n_null, seed=seed)
    return c["rcmi"], c["null_mean"], c["excess"], c["p_null"]


# =====================================================================
# F.  Models
# =====================================================================
def _norm_cdf(z):
    """Standard normal CDF via the error function (no scipy dependency)."""
    import math
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def chi2_sf(x, df):
    """Upper tail of the chi-square distribution, by series/continued fraction
    for the regularised incomplete gamma Q(df/2, x/2)."""
    import math
    a, xx = df / 2.0, x / 2.0
    if xx <= 0:
        return 1.0
    if xx < a + 1.0:                                  # series for P, then Q = 1-P
        term = 1.0 / a
        s = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= xx / n
            s += term
            if abs(term) < abs(s) * 1e-14:
                break
        return max(0.0, 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a)))
    b, c = xx + 1.0 - a, 1e300                        # continued fraction for Q
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return float(h * math.exp(-xx + a * math.log(xx) - math.lgamma(a)))


def onehot(v, cats):
    v = np.asarray(v)
    out = np.zeros((v.size, len(cats) - 1))
    for j, c in enumerate(cats[1:]):
        out[:, j] = (v == c)
    return out


def design(frame, columns, cats=None, intercept=True):
    cats = cats or {c: sorted(pd.unique(frame[c]).tolist()) for c in columns}
    blocks = [np.ones((len(frame), 1))] if intercept else []
    for c in columns:
        blocks.append(onehot(frame[c].values, cats[c]))
    return np.hstack(blocks), cats


def irls(X, y, offset=None, link="logit", w=None, maxit=100, tol=1e-10, ridge=1e-8):
    """Logistic (``logit``) or Poisson (``log``) fit by iteratively reweighted
    least squares."""
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    sw = np.ones(n) if w is None else np.asarray(w, float)
    b = np.zeros(p)
    for _ in range(maxit):
        eta = np.clip(X @ b + off, -30, 30)
        if link == "logit":
            mu = 1.0 / (1.0 + np.exp(-eta))
            wt = np.clip(mu * (1 - mu), 1e-10, None)
        else:
            mu = np.exp(eta)
            wt = np.clip(mu, 1e-10, None)
        z = (eta - off) + (y - mu) / wt
        XtW = X.T * (wt * sw)
        bn = np.linalg.solve(XtW @ X + ridge * np.eye(p), XtW @ z)
        if np.max(np.abs(bn - b)) < tol:
            b = bn
            break
        b = bn
    cov = np.linalg.inv((X.T * (wt * sw)) @ X + ridge * np.eye(p))
    return b, cov


class CoxPH:
    """Cox proportional hazards by Newton-Raphson on the Efron partial
    likelihood, evaluated with reverse cumulative sums.

    Supports left truncation (so attained age can be the time scale) and case
    weights (for the survey-weighted sensitivity analysis).  At an event time t
    the risk set is {i : entry_i < t <= exit_i}, so every risk-set sum is a
    difference of two reverse cumulative sums.
    """

    def __init__(self, ridge=1e-8):
        self.ridge = ridge

    def _efron_loglik(self, X, b, event, weight, di, dX, dw, d_cnt, d_wsum):
        """Efron log partial likelihood at ``b``; -inf where it is undefined.

        Used for step halving.  Plain Newton on this likelihood can overshoot
        badly -- with time on study and a strong age effect it ran away to
        |beta| ~ 1e11 within four iterations -- so every step is accepted only
        if it does not decrease the likelihood.
        """
        eta = np.clip(X @ b, -30, 30)
        s0, _, _ = self._sums(X, eta, weight, want_s2=False)
        t = self._etimes
        e_d = np.exp(eta[event == 1]) * dw
        t0 = np.bincount(di, weights=e_d, minlength=len(t))
        ll = float((dw * eta[event == 1]).sum())
        for m in range(int(d_cnt.max()) if len(d_cnt) else 0):
            act = d_cnt > m
            if not act.any():
                continue
            f = m / np.maximum(d_cnt[act], 1)
            a0 = s0[act] - f * t0[act]
            if np.any(a0 <= 0) or not np.all(np.isfinite(a0)):
                return -np.inf
            ll -= float((d_wsum[act] / d_cnt[act] * np.log(a0)).sum())
        return ll if np.isfinite(ll) else -np.inf

    @staticmethod
    def _rev_cumsum_at(sorted_times, values, query):
        rc = np.concatenate([np.cumsum(values[::-1], axis=0)[::-1],
                             np.zeros((1,) + values.shape[1:])])
        return rc[np.searchsorted(sorted_times, query, side="left")]

    def _prepare(self, X, entry, exit_, event, weight):
        self._oe, self._oi = (np.argsort(exit_, kind="mergesort"),
                              np.argsort(entry, kind="mergesort"))
        self._exit_s, self._entry_s = exit_[self._oe], entry[self._oi]
        self._Xe, self._Xi = X[self._oe], X[self._oi]
        self._we, self._wi = weight[self._oe], weight[self._oi]
        self._etimes = np.unique(exit_[event == 1])

    def _sums(self, X, eta, weight, want_s2=True, chunk=256):
        """S0, S1 and optionally S2 of the risk set at each event time.  S2 is an
        n x p x p intermediate that dominates cost, so it is built in column
        chunks and skipped when the caller supplies a fixed Hessian."""
        e = np.exp(np.clip(eta, -30, 30)) * weight
        ee, ei = e[self._oe], e[self._oi]
        p, t = X.shape[1], self._etimes
        s0 = (self._rev_cumsum_at(self._exit_s, ee[:, None], t)
              - self._rev_cumsum_at(self._entry_s, ei[:, None], t)).ravel()
        s1 = (self._rev_cumsum_at(self._exit_s, self._Xe * ee[:, None], t)
              - self._rev_cumsum_at(self._entry_s, self._Xi * ei[:, None], t))
        if not want_s2:
            return s0, s1, None
        s2 = np.empty((len(t), p, p))
        step = max(1, chunk // max(p, 1))
        for a in range(0, p, step):
            b = min(a + step, p)
            xe = self._Xe[:, a:b, None] * self._Xe[:, None, :] * ee[:, None, None]
            xi = self._Xi[:, a:b, None] * self._Xi[:, None, :] * ei[:, None, None]
            k = (b - a) * p
            s2[:, a:b, :] = (
                self._rev_cumsum_at(self._exit_s, xe.reshape(len(ee), k), t)
                - self._rev_cumsum_at(self._entry_s, xi.reshape(len(ei), k), t)
            ).reshape(len(t), b - a, p)
        return s0, s1, s2

    def fit_stratified(self, X, exit_, event, strata, entry=None, weight=None,
                       maxit=40, tol=1e-9):
        """Cox fit with a separate baseline hazard per stratum.

        Needed whenever a covariate determines who remains at risk.  With time
        on study as the time scale, survey cycle does exactly that -- the
        2017-2018 cycle is administratively censored at ~3 years -- so adjusting
        for cycle as a covariate makes the partial likelihood monotone and the
        coefficients diverge (we saw hazard ratios of 1e8 before stratifying).
        Stratifying lets each cycle have its own baseline and removes the
        problem.  Only coefficients are returned; the baseline is stratum
        specific and is not used downstream.
        """
        X = np.asarray(X, float)
        n, p = X.shape
        exit_, event = np.asarray(exit_, float), np.asarray(event, int)
        entry = np.zeros(n) if entry is None else np.asarray(entry, float)
        weight = np.ones(n) if weight is None else np.asarray(weight, float)
        strata = np.asarray(strata)
        blocks = []
        for s in np.unique(strata):
            m = strata == s
            if event[m].sum() == 0:
                continue                      # a stratum with no events is uninformative
            sub = CoxPH(ridge=self.ridge)
            sub._prepare(X[m], entry[m], exit_[m], event[m], weight[m])
            blocks.append((sub, m))

        b = np.zeros(p)
        for _ in range(maxit):
            grad, hess = np.zeros(p), np.zeros((p, p))
            for sub, m in blocks:
                Xs, ws, evs = X[m], weight[m], event[m]
                s0, s1, s2 = sub._sums(Xs, Xs @ b, ws)
                t = sub._etimes
                di = np.searchsorted(t, exit_[m][evs == 1])
                dw, dX = ws[evs == 1], Xs[evs == 1]
                d_cnt = np.bincount(di, minlength=len(t))
                d_wsum = np.bincount(di, weights=dw, minlength=len(t))
                for j in range(p):
                    grad[j] += np.bincount(di, weights=dw * dX[:, j],
                                           minlength=len(t)).sum()
                e_d = np.exp(np.clip((dX @ b), -30, 30)) * dw
                t0 = np.bincount(di, weights=e_d, minlength=len(t))
                t1 = np.zeros((len(t), p))
                np.add.at(t1, di, e_d[:, None] * dX)
                t2 = np.zeros((len(t), p, p))
                np.add.at(t2, di, e_d[:, None, None] * dX[:, :, None] * dX[:, None, :])
                for k in range(int(d_cnt.max()) if len(d_cnt) else 0):
                    act = d_cnt > k
                    if not act.any():
                        continue
                    f = k / np.maximum(d_cnt[act], 1)
                    a0 = s0[act] - f * t0[act]
                    a1 = s1[act] - f[:, None] * t1[act]
                    a2 = s2[act] - f[:, None, None] * t2[act]
                    wbar = d_wsum[act] / d_cnt[act]
                    r = a1 / a0[:, None]
                    grad -= (wbar[:, None] * r).sum(axis=0)
                    hess += np.einsum("i,ijk->jk", wbar, a2 / a0[:, None, None])
                    hess -= np.einsum("i,ij,ik->jk", wbar, r, r)
            def _ll(bv):
                tot = 0.0
                for sub, m in blocks:
                    Xs, ws, evs = X[m], weight[m], event[m]
                    ts = sub._etimes
                    dis = np.searchsorted(ts, exit_[m][evs == 1])
                    dws, dXs = ws[evs == 1], Xs[evs == 1]
                    dc = np.bincount(dis, minlength=len(ts))
                    dws_sum = np.bincount(dis, weights=dws, minlength=len(ts))
                    v = sub._efron_loglik(Xs, bv, evs, ws, dis, dXs, dws, dc, dws_sum)
                    if not np.isfinite(v):
                        return -np.inf
                    tot += v
                return tot

            step = np.linalg.solve(hess + self.ridge * np.eye(p), grad)
            ll0 = _ll(b)
            for _ in range(30):                       # step halving
                if _ll(b + step) >= ll0 - 1e-9:
                    break
                step = step * 0.5
            b = b + step
            if np.max(np.abs(step)) < tol:
                break
        self.beta_, self.hess_ = b, hess
        self.cov_ = np.linalg.inv(hess + self.ridge * np.eye(p))
        self.se_ = np.sqrt(np.diag(self.cov_))
        return self

    def fit(self, X, exit_, event, entry=None, weight=None, maxit=40, tol=1e-9,
            beta0=None, fixed_hess=None):
        """``beta0`` with ``fixed_hess`` runs a quasi-Newton iteration holding the
        Hessian at a caller-supplied value.  The bootstrap uses this: resample
        estimates sit close to the full-data estimate, so the full-data Hessian
        is an excellent preconditioner and the expensive S2 term is skipped
        entirely (verified identical to a full refit to 8e-10)."""
        X = np.asarray(X, float)
        n, p = X.shape
        exit_, event = np.asarray(exit_, float), np.asarray(event, int)
        entry = np.zeros(n) if entry is None else np.asarray(entry, float)
        weight = np.ones(n) if weight is None else np.asarray(weight, float)
        self._prepare(X, entry, exit_, event, weight)
        t = self._etimes

        di = np.searchsorted(t, exit_[event == 1])
        dw, dX = weight[event == 1], X[event == 1]
        d_cnt = np.bincount(di, minlength=len(t))
        d_wsum = np.bincount(di, weights=dw, minlength=len(t))
        d_Xsum = np.zeros((len(t), p))
        for j in range(p):
            d_Xsum[:, j] = np.bincount(di, weights=dw * dX[:, j], minlength=len(t))

        want_s2 = fixed_hess is None
        b = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()
        hess = fixed_hess
        for _ in range(maxit):
            eta = X @ b
            s0, s1, s2 = self._sums(X, eta, weight, want_s2=want_s2)
            e_d = np.exp(np.clip(eta[event == 1], -30, 30)) * dw
            t0 = np.bincount(di, weights=e_d, minlength=len(t))
            t1 = np.zeros((len(t), p))
            np.add.at(t1, di, e_d[:, None] * dX)
            if want_s2:
                t2 = np.zeros((len(t), p, p))
                np.add.at(t2, di, e_d[:, None, None] * dX[:, :, None] * dX[:, None, :])
            grad = d_Xsum.sum(axis=0).copy()
            Hm = np.zeros((p, p)) if want_s2 else None
            for m in range(int(d_cnt.max()) if len(d_cnt) else 0):
                act = d_cnt > m
                if not act.any():
                    continue
                f = m / np.maximum(d_cnt[act], 1)
                a0 = s0[act] - f * t0[act]
                a1 = s1[act] - f[:, None] * t1[act]
                wbar = d_wsum[act] / d_cnt[act]
                r = a1 / a0[:, None]
                grad -= (wbar[:, None] * r).sum(axis=0)
                if want_s2:
                    a2 = s2[act] - f[:, None, None] * t2[act]
                    Hm += np.einsum("i,ijk->jk", wbar, a2 / a0[:, None, None])
                    Hm -= np.einsum("i,ij,ik->jk", wbar, r, r)
            if want_s2:
                hess = Hm
            step = np.linalg.solve(hess + self.ridge * np.eye(p), grad)
            ll0 = self._efron_loglik(X, b, event, weight, di, dX, dw, d_cnt, d_wsum)
            for _ in range(30):                       # step halving
                cand = b + step
                if self._efron_loglik(X, cand, event, weight, di, dX, dw,
                                      d_cnt, d_wsum) >= ll0 - 1e-9:
                    break
                step = step * 0.5
            b = b + step
            if np.max(np.abs(step)) < tol:
                break

        self.beta_, self.hess_ = b, hess
        self.cov_ = np.linalg.inv(hess + self.ridge * np.eye(p))
        self.se_ = np.sqrt(np.diag(self.cov_))
        self._X, self._event = X, event
        self._exit, self._entry, self._w = exit_, entry, weight
        self._baseline()
        return self

    def _baseline(self):
        """Breslow cumulative baseline hazard on the event-time grid."""
        s0, _, _ = self._sums(self._X, self._X @ self.beta_, self._w, want_s2=False)
        t = self._etimes
        di = np.searchsorted(t, self._exit[self._event == 1])
        dw = np.bincount(di, weights=self._w[self._event == 1], minlength=len(t))
        self.baseline_t_ = t
        self.baseline_H_ = np.cumsum(dw / np.maximum(s0, 1e-300))

    def schoenfeld_test(self, cols=None):
        """Proportional-hazards score test (Grambsch & Therneau, 1994).

        Under the alternative the coefficient drifts with time,
        beta_j(t) = beta_j + theta_j g(t), with g the rank of the event time.
        The score for theta at (beta_hat, theta = 0) is the sum over deaths of
        g(t_i) times the Schoenfeld residual x_i - xbar(t_i), where xbar is the
        risk-set weighted mean already used by the score equation.  Its
        variance is I_gg - I_gb I_bb^-1 I_bg, with I_bb = sum V(t_i),
        I_gb = sum g_i V(t_i), I_gg = sum g_i^2 V(t_i) and V(t) the risk-set
        covariance matrix (Breslow form).  This accounts both for the
        estimation of beta and for the correlation between the contrasts of
        one categorical exposure, which a sum of per-contrast z^2 ignores.

        Returns the per-column results (z, P, and the descriptive Pearson
        correlation of the residual with the rank of the event time) and the
        joint chi-square over ``cols`` with its degrees of freedom.
        """
        X = self._X
        eta = X @ self.beta_
        s0, s1, s2 = self._sums(X, eta, self._w, want_s2=True)
        t = self._etimes
        ev = self._event == 1
        di = np.searchsorted(t, self._exit[ev])
        xbar = s1 / s0[:, None]
        V = s2 / s0[:, None, None] - xbar[:, :, None] * xbar[:, None, :]
        resid = X[ev] - xbar[di]                               # (n_events, p)
        wd = self._w[ev]
        g = pd.Series(self._exit[ev]).rank().values             # rank of event time
        Vd = V[di]
        U = (wd * g) @ resid
        I_bb = np.einsum("i,ijk->jk", wd, Vd)
        I_gb = np.einsum("i,ijk->jk", wd * g, Vd)
        I_gg = np.einsum("i,ijk->jk", wd * g * g, Vd)
        var = I_gg - I_gb @ np.linalg.solve(I_bb + self.ridge * np.eye(len(U)), I_gb)
        cols = list(range(X.shape[1])) if cols is None else list(cols)
        rt = g - g.mean()
        out = []
        for j in cols:
            z = float(U[j] / np.sqrt(var[j, j])) if var[j, j] > 0 else 0.0
            r = resid[:, j] - resid[:, j].mean()
            den = np.sqrt((r ** 2).sum() * (rt ** 2).sum())
            out.append(dict(column=int(j), z=z,
                            p=float(2 * (1 - _norm_cdf(abs(z)))),
                            rho=float((r * rt).sum() / den) if den > 0 else 0.0))
        vs = var[np.ix_(cols, cols)]
        chi2 = float(U[cols] @ np.linalg.solve(vs, U[cols]))
        return pd.DataFrame(out), chi2, len(cols)

    def cumhaz(self, t):
        idx = np.searchsorted(self.baseline_t_, t, side="right") - 1
        return np.where(idx < 0, 0.0, self.baseline_H_[np.clip(idx, 0, None)])

    def risk_at(self, X, t, entry=None):
        """P(event by time t), optionally conditional on survival to ``entry``."""
        h = np.exp(np.clip(np.asarray(X, float) @ self.beta_, -30, 30))
        H_ = self.cumhaz(t)
        if entry is None:
            return 1.0 - np.exp(-H_ * h)
        return 1.0 - np.exp(-(H_ - self.cumhaz(entry)) * h)


# =====================================================================
# G.  Adjustment-set construction
# =====================================================================
def deciles(v, q=10):
    """Equal-count quantile groups 0..q-1 of v (ties broken by position).

    The bin edges are computed with np.quantile on the ranks, not pd.qcut:
    with n = 31 003 the tertile edges are exact integers, which pandas 2.x
    evaluates as 10334.999... and pandas 3 as 10335, so pd.qcut put the two
    boundary ranks in different tertiles depending on the pandas version."""
    r = pd.Series(v).rank(method="first").values
    edges = np.quantile(r, np.linspace(0.0, 1.0, q + 1))
    return np.searchsorted(edges[1:-1], r, side="left").astype(int)


def crossfit_score(d, covariates=None, k=5, seed=SEED):
    """K-fold cross-fitted predicted probability of death from demographics.

    A disease risk score, not a propensity score.  Cross-fitting matters: fitting
    it on the full sample lets an individual's own outcome inform their own
    adjustment variable.
    """
    covariates = covariates or SCORE_COVARIATES
    fold = np.random.default_rng(seed).permutation(len(d)) % k
    y = d["died"].values.astype(float)
    X, _ = design(d, covariates)
    score = np.empty(len(d))
    for f in range(k):
        tr, te = fold != f, fold == f
        b, _ = irls(X[tr], y[tr], link="logit")
        score[te] = 1.0 / (1.0 + np.exp(-np.clip(X[te] @ b, -30, 30)))
    return score


def cohabit_score(d, habit, strata, k=5, seed=SEED, others=None):
    """Cross-fitted co-habit risk index for ``habit``.

    A logistic model for death is fitted on the demographic strata (as
    dummies) and the other habits; the index is the part of its linear
    predictor contributed by the other habits only.  Including the strata in
    the fit adjusts the habit coefficients for demographics; leaving them out
    of the index keeps it from re-encoding the demographic risk.  (The first
    version used the whole linear predictor, so its tertile was almost a
    coarsening of the risk decile: only 21-22 of 30 decile x tertile cells
    were occupied and the co-habits refined the adjustment in few deciles.)
    """
    others = others if others is not None else [g for g in HABITS if g != habit]
    f = d.copy()
    f["__st"] = np.unique(np.asarray(strata), return_inverse=True)[1]
    Xs, _ = design(f, ["__st"])                      # intercept + strata dummies
    Xh, _ = design(f, others, intercept=False)       # co-habit dummies
    X = np.hstack([Xs, Xh])
    ks = Xs.shape[1]
    y = d["died"].values.astype(float)
    fold = np.random.default_rng(seed + 7).permutation(len(d)) % k
    s_ = np.empty(len(d))
    for j in range(k):
        tr, te = fold != j, fold == j
        b, _ = irls(X[tr], y[tr], link="logit")
        s_[te] = Xh[te] @ b[ks:]
    return s_


def within_quantiles(v, groups, q):
    """Equal-count quantile groups of v formed separately within each group."""
    v, groups = np.asarray(v, float), np.asarray(groups)
    out = np.empty(len(v), dtype=int)
    for g in np.unique(groups):
        m = groups == g
        out[m] = deciles(v[m], q)
    return out


def lifestyle_index(d, habit, strata, k=5, seed=SEED, n_bins=3, within=True,
                    others=None):
    """Tertile (``n_bins``) of the co-habit index, formed within each
    demographic stratum when ``within`` (so that every stratum is split by the
    co-habits), or over the whole sample otherwise."""
    sc = cohabit_score(d, habit, strata, k=k, seed=seed, others=others)
    return within_quantiles(sc, strata, n_bins) if within else deciles(sc, n_bins)


def demographic_strata(d, score, n_bins=3):
    """Age band crossed with a tertile of the prognostic score formed within
    each band (6 x 3 = 18 strata).  Age is stratified explicitly because it is
    the dominant common cause of every habit and of death: a decile of the
    prognostic score alone left enough residual age within deciles to inflate
    the association of physical activity (which falls with age) and to deflate
    that of smoking (which also falls with age, while mortality rises)."""
    ab = d["age_b"].values
    return combine_codes(ab, within_quantiles(score, ab, n_bins))


def adjustment_set(d, habit, demo, n_bins=3):
    """The primary adjustment set for ``habit``: each demographic stratum split
    into tertiles of the cross-fitted index of the other five habits."""
    return combine_codes(demo, lifestyle_index(d, habit, demo, n_bins=n_bins))


#: covariates of the demographic prognostic score (5-year age groups, so that
#: tertiles within an age band still order participants by age)
DEMOG_COVARIATES = ["age5", "sex_b", "race5", "educ", "pir", "marital", "cyc5"]


def prognostic_score(d, habit, covariates, k=5, seed=SEED):
    """Cross-fitted prognostic (disease-risk) score for estimating the direct
    association of ``habit``.

    A logistic model for death is fitted on the habit's own levels and the
    ``covariates`` (as dummies) in k-1 folds and predicted in the k-th; the
    score is the part of the linear predictor contributed by the covariates.
    Fitting the habit alongside keeps its own effect out of the covariate
    coefficients (a co-habit correlated with the habit would otherwise absorb
    part of it), and leaving it out of the score keeps the stratification from
    being a proxy for the exposure.  Cross-fitting keeps an individual's own
    outcome out of their own score."""
    Xx, _ = design(d, [habit])                        # intercept + habit dummies
    Xc, _ = design(d, covariates, intercept=False)
    X = np.hstack([Xx, Xc])
    k0 = Xx.shape[1]
    y = d["died"].values.astype(float)
    if k <= 1:                                   # full-sample fit (SA5)
        b, _ = irls(X, y, link="logit")
        return Xc @ b[k0:]
    fold = np.random.default_rng(seed + 3).permutation(len(d)) % k
    out = np.empty(len(d))
    for j in range(k):
        tr, te = fold != j, fold == j
        b, _ = irls(X[tr], y[tr], link="logit")
        out[te] = Xc[te] @ b[k0:]
    return out


def score_strata(d, score, n_bins=3):
    """Age band x four-year survey period (6 x 3 cells), each split into
    equal-count quantile groups (tertiles by default) of ``score``: 54 strata.
    Age is stratified explicitly because it is the dominant common cause of
    every habit and of death; survey period because it fixes the length of
    follow-up (and so the meaning of 'died by the end of follow-up') and
    because the exposures changed between periods (notably the sleep
    instrument in 2015-2016)."""
    cell = combine_codes(d["age_b"].values, d["cyc5"].values)
    return combine_codes(cell, within_quantiles(score, cell, n_bins))


def direct_set(d, habit, n_bins=3, others=None, covariates=None, k=5):
    """The primary adjustment set for ``habit``: age band x survey period x
    tertile of a prognostic score built from the demographic covariates and
    the other five habits."""
    others = others if others is not None else [g for g in HABITS if g != habit]
    covs = list(covariates or DEMOG_COVARIATES) + list(others)
    return score_strata(d, prognostic_score(d, habit, covs, k=k), n_bins)


def demographic_set(d, habit, n_bins=3, covariates=None, k=5):
    """As direct_set, with a prognostic score built from the demographic
    covariates only: the direct association given demographics alone."""
    return score_strata(d, prognostic_score(d, habit, list(covariates or DEMOG_COVARIATES),
                                            k=k), n_bins)


def combine_codes(*cols):
    """Dense integer code for the cross-classification of the given columns."""
    arr = np.vstack([np.asarray(c).astype(np.int64) for c in cols])
    code = np.zeros(arr.shape[1], dtype=np.int64)
    for row in arr:
        _, r = np.unique(row, return_inverse=True)
        code = code * (r.max() + 1) + r
    return np.unique(code, return_inverse=True)[1]


def table_for(d, habit, z_codes):
    x, y = d[habit].values.astype(int), d["died"].values.astype(int)
    if z_codes is None:
        return Table(x, y)
    return Table(x, y, [np.unique(np.asarray(z_codes), return_inverse=True)[1]])


# =====================================================================
# H.  Self-tests
# =====================================================================
def selftest(verbose=True):
    """Validate every estimator.  Nothing downstream should be trusted until
    this prints ALL PASS."""
    rng = np.random.default_rng(SEED)
    fails, lines = [], []

    def check(name, got, want, tol):
        ok = abs(got - want) <= tol
        lines.append("  [%s] %-56s got=%9.4f want=%9.4f" %
                     ("PASS" if ok else "FAIL", name, got, want))
        if verbose:
            print(lines[-1])
        if not ok:
            fails.append(name)

    # ---- information measures against closed forms
    check("cmax(2) = 0.55792 (source paper, Sec. IV E)", cmax(2), 0.55792, 1e-5)
    check("cmax(1) = 0", cmax(1), 0.0, 1e-12)
    check("rjs(Bern(.5) || Bern(1)) = cmax(2)",
          rjs([0.5, 0.5], [1.0, 0.0]), cmax(2), 1e-12)
    # rCMI collapses to zero under an exactly confounding model
    n = 40000
    z = rng.integers(0, 4, n)
    x = (rng.random(n) < (0.2 + 0.15 * z)).astype(int)
    y = (rng.random(n) < (0.1 + 0.2 * z)).astype(int)       # y depends only on z
    check("rCMI ~ 0 under exact conditional independence",
          rcmi_of(Table(x, y, [z]).counts()), 0.0, 0.02)
    check("rMI > 0 for the same data (marginal association exists)",
          float(rmi_of(Table(x, y).counts()) > 0.01), 1.0, 1e-9)
    # invariance to an irrelevant Z
    zr = rng.integers(0, 3, n)
    check("rCMI | irrelevant Z ~ rMI",
          rcmi_of(Table(x, y, [zr]).counts()), rmi_of(Table(x, y).counts()), 0.02)
    # p(y|do(x)) is a proper distribution
    pdo = do_dist(Table(x, y, [z]).counts())
    check("p(y|do(x)) rows sum to 1", float(np.abs(pdo.sum(axis=1) - 1).max()), 0.0, 1e-10)
    # the null calibration brackets the observed value under H0
    tab = Table(x, y, [z])
    nv = null_rcmi(tab, n_null=120, seed=SEED)
    lo, hi = np.percentile(nv, [1, 99])
    check("observed rCMI inside its own null distribution under H0",
          float(lo <= rcmi_of(tab.counts()) <= hi), 1.0, 1e-9)

    # ---- survival model against a known hazard ratio
    def sim(n_, beta, admin=None, entry_dep=False, rate0=0.02):
        xs = rng.integers(0, 2, n_).astype(float)
        t = rng.exponential(1.0 / (rate0 * np.exp(beta * xs)))
        entry = rng.uniform(0, 20, n_) + 5.0 * xs if entry_dep else np.zeros(n_)
        cens = np.full(n_, np.inf) if admin is None else (
            admin if np.ndim(admin) else np.full(n_, admin, float))
        ex = np.minimum(entry + t, cens)
        ev = ((entry + t) <= cens).astype(int)
        keep = ex > entry
        return xs[keep], entry[keep], ex[keep], ev[keep]

    xs, en, ex, ev = sim(20000, np.log(2.0))
    check("Cox recovers log HR = log 2, no censoring",
          CoxPH().fit(xs[:, None], ex, ev).beta_[0], np.log(2.0), 0.05)

    n2 = 30000
    xx = rng.integers(0, 2, n2).astype(float)
    tt = rng.exponential(1.0 / (0.02 * np.exp(np.log(2.0) * xx)))
    admin = rng.choice([2.0, 5.0, 10.0, 20.0], n2)      # NHANES-like staggered entry
    exx, evv = np.minimum(tt, admin), (tt <= admin).astype(int)
    check("Cox under staggered administrative censoring",
          CoxPH().fit(xx[:, None], exx, evv).beta_[0], np.log(2.0), 0.06)

    x3, en3, ex3, ev3 = sim(40000, np.log(1.8), admin=60.0, entry_dep=True)
    check("Cox with exposure-dependent left truncation",
          CoxPH().fit(x3[:, None], ex3, ev3, entry=en3).beta_[0], np.log(1.8), 0.06)
    naive = CoxPH().fit(x3[:, None], ex3, ev3).beta_[0]
    check("ignoring left truncation IS biased (sanity check on the check)",
          float(abs(naive - np.log(1.8)) > 0.1), 1.0, 1e-9)

    x4, _, ex4, ev4 = sim(20000, np.log(2.0), admin=20.0)
    m4 = CoxPH().fit(x4[:, None], ex4, ev4)
    check("Efron ties: monthly-rounded times vs continuous",
          CoxPH().fit(x4[:, None], np.ceil(ex4 * 12) / 12, ev4).beta_[0],
          m4.beta_[0], 0.03)
    bp, _ = irls(np.hstack([np.ones((x4.size, 1)), x4[:, None]]), ev4.astype(float),
                 offset=np.log(np.maximum(ex4, 1e-6)), link="log")
    check("Cox vs piecewise-exponential Poisson", bp[1], m4.beta_[0], 0.05)

    hits, betas = 0, []
    for _ in range(200):
        xa, _, exa, eva = sim(3000, np.log(1.5), admin=25.0)
        ma = CoxPH().fit(xa[:, None], exa, eva)
        betas.append(ma.beta_[0])
        hits += (ma.beta_[0] - 1.96 * ma.se_[0] <= np.log(1.5)
                 <= ma.beta_[0] + 1.96 * ma.se_[0])
    check("95% CI coverage over 200 replicates", hits / 200.0, 0.95, 0.045)
    check("mean beta over replicates", float(np.mean(betas)), np.log(1.5), 0.02)

    # step halving: a strong effect with heavily tied event times and a second
    # strong covariate is exactly the configuration in which plain Newton
    # overshot to |beta| ~ 1e11 before step halving was added
    n9 = 20000
    x9 = rng.integers(0, 2, n9).astype(float)
    g9 = rng.normal(0, 1, n9)
    lam9 = 0.02 * np.exp(np.log(3.0) * x9 + 1.2 * g9)
    t9 = rng.exponential(1.0 / lam9)
    ex9 = np.minimum(np.ceil(t9 * 12) / 12, 12.0)          # monthly ties, censored
    ev9 = (t9 <= 12.0).astype(int)
    m9 = CoxPH().fit(np.column_stack([x9, g9]), ex9, ev9)
    check("step halving: strong effect with heavy ties recovers log 3",
          m9.beta_[0], np.log(3.0), 0.08)
    check("step halving: nuisance coefficient stays finite and sane",
          m9.beta_[1], 1.2, 0.08)

    x7, _, ex7, ev7 = sim(30000, 0.0, admin=15.0, rate0=0.05)
    m7 = CoxPH().fit(np.zeros((x7.size, 1)), ex7, ev7)
    check("Breslow H0(10) with true rate 0.05",
          float(m7.cumhaz(np.array([10.0]))[0]), 0.50, 0.04)

    x8, _, ex8, ev8 = sim(4000, np.log(2.0), admin=20.0)
    w8 = rng.choice([1.0, 3.0], x8.size)
    rep = np.repeat(np.arange(x8.size), w8.astype(int))
    # agree up to the Efron correction: replication manufactures tied deaths,
    # which Efron down-weights; under Breslow they would coincide exactly
    check("case weights vs row replication",
          CoxPH().fit(x8[:, None], ex8, ev8, weight=w8).beta_[0],
          CoxPH().fit(x8[rep][:, None], ex8[rep], ev8[rep]).beta_[0], 5e-3)

    # ---- the null table drawn as one multinomial equals simulating every
    # individual from p(z) p(x|z) p(y|z) (same distribution of rCMI)
    cnt0 = Table(x, y, [z]).counts()
    p3_, _, _, pz_, xgz_, ygz_ = _parts(cnt0)
    ind = []
    for _ in range(300):
        zz = rng.choice(pz_.size, size=n, p=pz_)
        xx = (rng.random(n)[:, None] > np.cumsum(xgz_, axis=0)[:, zz].T).sum(axis=1)
        yy = (rng.random(n)[:, None] > np.cumsum(ygz_, axis=0)[:, zz].T).sum(axis=1)
        ind.append(rcmi_of(Table(np.minimum(xx, p3_.shape[0] - 1),
                                 np.minimum(yy, p3_.shape[1] - 1), [zz],
                                 dx=p3_.shape[0], dy=p3_.shape[1]).counts()))
    tab_draws = null_rcmi(_CountTable(cnt0), n_null=300, seed=SEED + 3)
    se = np.sqrt(np.var(ind) / 300 + np.var(tab_draws) / 300)
    check("null table sampler matches individual-level simulation (z-score)",
          float((np.mean(ind) - np.mean(tab_draws)) / se), 0.0, 3.5)

    # ---- the bootstrap's warm-started fixed-Hessian fit equals a full refit
    full = CoxPH().fit(x4[:, None], ex4, ev4)
    rs = rng.integers(0, x4.size, x4.size)
    b_full = CoxPH().fit(x4[rs][:, None], ex4[rs], ev4[rs]).beta_[0]
    b_warm = CoxPH().fit(x4[rs][:, None], ex4[rs], ev4[rs], beta0=full.beta_,
                         fixed_hess=full.hess_).beta_[0]
    check("bootstrap fast path (fixed Hessian) = full refit", b_warm, b_full, 1e-6)

    # ---- the achievable bound never exceeds the alphabet ceiling
    bnd = max(achievable_upper_rcmi(Table(x, y, [z]).counts()),
              achievable_upper_rcmi(Table(rng.integers(0, 5, 5000),
                                          rng.integers(0, 2, 5000)).counts()))
    check("achievable bound <= cmax(2) with a binary outcome",
          float(bnd <= cmax(2) + 1e-12), 1.0, 1e-9)

    # ---- proportional-hazards score test: size under PH, power against it
    pv_null, pv_alt = [], []
    for _ in range(100):
        xa = rng.integers(0, 2, 2000).astype(float)
        ta = rng.exponential(1.0 / np.exp(0.5 * xa))
        ca = rng.exponential(2.0, 2000)
        _, c2, k = CoxPH().fit(xa[:, None], np.minimum(ta, ca),
                               (ta <= ca).astype(int)).schoenfeld_test()
        pv_null.append(chi2_sf(c2, k))
        u = rng.exponential(1.0, 2000)                  # HR 3 before t = 0.5, 1 after
        lam = np.exp(np.log(3.0) * xa)
        tb = np.where(u < 0.5 * lam, u / lam, 0.5 + (u - 0.5 * lam))
        _, c2, k = CoxPH().fit(xa[:, None], np.minimum(tb, ca),
                               (tb <= ca).astype(int)).schoenfeld_test()
        pv_alt.append(chi2_sf(c2, k))
    check("PH score test: rejection rate under PH (nominal 0.05)",
          float(np.mean(np.array(pv_null) < 0.05)), 0.05, 0.045)
    check("PH score test: power against a time-varying HR",
          float(np.mean(np.array(pv_alt) < 0.05)), 1.0, 0.05)

    # ---- quantile groups: equal counts, exact integer edges handled alike
    q3 = deciles(np.arange(31003.0), 3)
    check("tertiles of 31 003 ranks have sizes 10 335 / 10 334 / 10 334",
          float(np.array_equal(np.bincount(q3), [10335, 10334, 10334])), 1.0, 1e-9)

    # ---- the two distribution functions used in place of SciPy
    check("chi-square upper 5% points (1, 2, 5, 9 df)",
          max(abs(chi2_sf(3.841459, 1) - 0.05), abs(chi2_sf(5.991465, 2) - 0.05),
              abs(chi2_sf(11.070498, 5) - 0.05), abs(chi2_sf(16.918978, 9) - 0.05)),
          0.0, 1e-6)
    check("standard normal CDF at 1.96", _norm_cdf(1.96), 0.9750021, 1e-6)

    lines.append("\n" + ("ALL PASS" if not fails else "FAILURES: %s" % fails))
    if verbose:
        print(lines[-1])
    # kept with the outputs: Supplementary Methods S4 quotes these values, and
    # lsm_report.py checks the text against them
    open(os.path.join(OUT_DIR, "log_selftest.txt"), "w", encoding="utf-8",
         newline="\n").write("\n".join(lines) + "\n")
    return fails


# =====================================================================
if __name__ == "__main__":
    # one or more of: fetch, verify-data, build, test, all (in the order given)
    opts = [a for a in sys.argv[1:] if a.startswith("-")]
    if "-h" in opts or "--help" in opts:
        print(__doc__)
        sys.exit(0)
    bad_opts = sorted(set(opts) - {"--force"})
    if bad_opts:
        raise SystemExit("unknown option(s): %s; the only option is --force"
                         % ", ".join(bad_opts))
    cmds = [a for a in sys.argv[1:] if not a.startswith("-")] or ["test"]
    unknown = sorted(set(cmds) - {"fetch", "verify-data", "build", "test", "all"})
    if unknown:
        raise SystemExit("unknown command(s): %s; use fetch, verify-data, build, "
                         "test or all" % ", ".join(unknown))
    bad = False
    for cmd in cmds:
        if cmd in ("fetch", "all"):
            fetch()
        if cmd == "verify-data":
            bad |= bool(verify_data())
        if cmd in ("build", "all"):
            if verify_data() and "--force" not in sys.argv:
                raise SystemExit("data do not match MANIFEST.csv; not building "
                                 "(use --force to build anyway)")
            build_raw()
            harmonise()
            build_design_vars()
        if cmd in ("test", "all"):
            bad |= bool(selftest())
    sys.exit(1 if bad else 0)
