# -*- coding: utf-8 -*-
"""
lsm_analysis.py -- every analysis in the paper, writing to output/.

    python lsm_analysis.py all           run everything
    python lsm_analysis.py measures      direct-association measures + intervals
    python lsm_analysis.py survival      Cox models + standardised 10-year risks
    python lsm_analysis.py ph            proportional-hazards score tests
    python lsm_analysis.py sensitivity   the sensitivity analyses SA1-SA21
    python lsm_analysis.py simulation    operating characteristics, known truth
    python lsm_analysis.py calcheck      semi-parametric check on the real tables
    python lsm_analysis.py all --quick   few replicates, for a smoke test

Replicate counts can be set individually, e.g.
    B_BOOT=200 N_NULL=200 python lsm_analysis.py measures
and the resampling loops run in LSM_WORKERS processes (default: CPUs - 2, at
most 16; LSM_WORKERS=1 runs serially).  Every resample is seeded by its own
index, so the results do not depend on the number of workers.

Design decisions that differ from a conventional single-exposure workflow, and
why, are documented at each function.
"""
from __future__ import annotations

import json
import os
import sys
import time

# one BLAS thread per process (see lsm_core); must precede the numpy import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# a smoke test (--quick) writes to output_quick/, so that it never overwrites
# the full-run results; the built analysis table is copied there from output/
if "--quick" in sys.argv and "LSM_OUT" not in os.environ:
    import shutil
    _here = os.path.dirname(os.path.abspath(__file__))
    _q = os.path.join(_here, "output_quick")
    os.makedirs(_q, exist_ok=True)
    for _f in ("nhanes_raw.pkl", "nhanes_clean.pkl", "design_vars.csv"):
        _src, _dst = os.path.join(_here, "output", _f), os.path.join(_q, _f)
        if os.path.exists(_src) and (
                not os.path.exists(_dst)
                or os.path.getmtime(_src) > os.path.getmtime(_dst)
                or os.path.getsize(_src) != os.path.getsize(_dst)):
            shutil.copy2(_src, _dst)
    os.environ["LSM_OUT"] = _q
    print("--quick: writing to %s" % _q)

import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

import lsm_core as C                                                 # noqa: E402

QUICK = "--quick" in sys.argv


def _env(name, full, fast):
    return int(os.environ.get(name, fast if QUICK else full))


def _save_params(step, **kw):
    """Record the replicate counts a step actually used (output/params_*.json),
    so that the numbers quoted in the paper can be checked against them, and
    the platform the step ran on."""
    kw.update(_platform())
    json.dump(kw, open(os.path.join(C.OUT_DIR, "params_%s.json" % step), "w",
                       encoding="utf-8", newline="\n"), indent=1)


# =====================================================================
# 0.  Parallel map with deterministic per-task seeds
# =====================================================================
def _workers():
    """Worker processes for the resampling loops (LSM_WORKERS; 1 = serial).
    Every task is seeded by its own index, so results do not depend on the
    number of workers."""
    default = max(1, min(16, (os.cpu_count() or 2) - 2))
    return max(1, int(os.environ.get("LSM_WORKERS", default)))


def _pmap(fn, items, initializer=None, initargs=()):
    items = list(items)
    nw = min(_workers(), len(items)) if items else 1
    if nw <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [fn(i) for i in items]
    from concurrent.futures import ProcessPoolExecutor
    out, t0, step = [], time.time(), max(1, len(items) // 10)
    with ProcessPoolExecutor(nw, initializer=initializer, initargs=initargs) as ex:
        for k, r in enumerate(ex.map(fn, items,
                                     chunksize=max(1, len(items) // (8 * nw))), 1):
            out.append(r)
            if k % step == 0 or k == len(items):
                el = time.time() - t0
                print("    %d/%d done (%.0fs elapsed, ~%.0fs left)"
                      % (k, len(items), el, el / k * (len(items) - k)), flush=True)
    return out


# =====================================================================
# 1.  Regularised direct-association measures
# =====================================================================
#: specifications evaluated for every habit: no adjustment (the total
#: association); age band alone (point estimates only); demographics and survey
#: period (age band x period x tertile of a demographic prognostic score, 54
#: strata); the primary set (the same with a prognostic score that also
#: contains the other five habits, 54 strata); exact demographic stratification
SPECS = ["total", "AGE", "DEMO", "PRIMARY", "RAW"]
BOOT_SPECS = ["total", "DEMO", "PRIMARY"]

_BW = {}


def _outcome_design(d):
    """Design matrix of the smoothing model of the bootstrap: an additive
    logistic model for death on the demographic covariates of the prognostic
    score (5-year age group, sex, race/ethnicity, education, income-to-poverty
    ratio, marital status, survey period) and the six habits."""
    return C.design(d, C.DEMOG_COVARIATES + C.HABITS)[0]


def _fit_p(X, y):
    """Fitted probabilities of the smoothing model."""
    b, _ = C.irls(X, np.asarray(y, float), link="logit")
    return 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))


def _expected_table(x, z, p1):
    """(x, y, z) table of expected counts when individual i dies with
    probability p1[i]."""
    z = np.zeros(len(x), dtype=int) if z is None else z
    dx, dz = int(x.max() + 1), int(z.max() + 1)
    cell = x * dz + z
    n = np.bincount(cell, minlength=dx * dz).reshape(dx, dz).astype(float)
    n1 = np.bincount(cell, weights=p1, minlength=dx * dz).reshape(dx, dz)
    return np.stack([n - n1, n1], axis=1)


def _boot_init(p1, xs, zs, n_inner, psu=None, seed=1):
    _BW.update(p1=p1, xs=xs, zs=zs, n_inner=n_inner, psu=psu, seed=seed)


def _resample_rows(rng, n, psu):
    """Row indices of one resample: individuals with replacement, or (``psu``,
    a list per design stratum of the row-index arrays of its masked variance
    units) a Rao-Wu rescaled resample of variance units: in each stratum with
    m units, m - 1 are drawn with replacement and each drawn unit enters
    m / (m - 1) times (2 with the two units NHANES has in almost every
    stratum; a fractional multiple is rounded at random, which keeps its
    expectation).  Drawing m of m would understate the between-unit variance by
    the factor (m - 1)/m, a half with two units per stratum."""
    if psu is None:
        return rng.integers(0, n, n)
    out = []
    for units in psu:
        m = len(units)
        if m < 2:
            out.append(units[0])
            continue
        f = m / (m - 1.0)
        for j in rng.integers(0, m, m - 1):
            copies = int(f) + int(rng.random() < f - int(f))
            out.extend([units[j]] * copies)
    return np.concatenate(out)


def _boot_one(b):
    """One smoothed, paired resample.  Individuals (or masked variance units)
    are resampled with their six habits and adjustment strata, so that every
    habit is recomputed on the same people, and death is simulated from the
    fitted smoothing model rather than copied.  The resampled population then
    carries the model's association, close to the calibrated estimate, instead
    of the plug-in value inflated by the null floor; a plain resample of the
    observed deaths gives standard errors that are too small
    (calibration_check)."""
    p1, xs, zs, n_inner = _BW["p1"], _BW["xs"], _BW["zs"], _BW["n_inner"]
    rng = np.random.default_rng([C.SEED + _BW["seed"], b])
    idx = _resample_rows(rng, len(p1), _BW["psu"])
    ys = (rng.random(idx.size) < p1[idx]).astype(int)
    out = {}
    for key, z in zs.items():
        if z is None:
            tb = C.Table(xs[key[0]][idx], ys)
        else:
            tb = C.Table(xs[key[0]][idx], ys, [np.unique(z[idx], return_inverse=True)[1]])
        r = C.rcmi_of(tb.counts())
        if key[1] == "PRIMARY":
            # with each level's calibrated D_JS (plug-in minus null floor),
            # for the uncertainty of the level shares (Supplementary Table S1f)
            nv, nl = C.null_rcmi(tb, n_null=n_inner, seed=C.SEED + 1000 + b, levels=True)
            lev = C.djs_levels(tb.counts()) - nl
        else:
            nv, lev = C.null_rcmi(tb, n_null=n_inner, seed=C.SEED + 1000 + b), None
        u = r * r - float(np.mean(nv ** 2))
        out[key] = (r - float(nv.mean()), float(np.sqrt(max(u, 0.0))), u, lev)
    return out


def _plain_init(xs, y, zs, psu):
    _BW.update(pxs=xs, py=y, pzs=zs, ppsu=psu)


def _plain_one(b):
    """One plain resample (observed deaths kept) of the primary tables: the
    plug-in rCMI of every habit."""
    rng = np.random.default_rng([C.SEED + 17, b])
    y = _BW["py"]
    idx = _resample_rows(rng, len(y), _BW["ppsu"])
    return {h: C.rcmi_of(C.Table(_BW["pxs"][h][idx], y[idx],
                                 [np.unique(_BW["pzs"][h][idx], return_inverse=True)[1]]
                                 ).counts())
            for h in C.HABITS}


def _platform():
    """The platform a step ran on: CSV output is reproducible bit for bit on
    the same OS, CPU and numpy/BLAS build."""
    import platform
    return dict(platform=platform.platform(), processor=platform.processor(),
                python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__)


def adjustment_sets(d):
    """{spec: {habit: stratum codes or None}} for SPECS on analytic table d,
    and the demographic prognostic score (for the refinement curve)."""
    raw = C.combine_codes(*[d[c].values for c in C.RAW_STRATA])
    return {"total": {h: None for h in C.HABITS},
            "AGE": {h: d["age_b"].values for h in C.HABITS},
            "DEMO": {h: C.demographic_set(d, h) for h in C.HABITS},
            "PRIMARY": {h: C.direct_set(d, h) for h in C.HABITS},
            "RAW": {h: raw for h in C.HABITS}}, C.crossfit_score(d)


def _psu_groups(d):
    """Masked variance units of the analytic sample, grouped by design stratum
    (cycle-prefixed), as row-index arrays."""
    dv = pd.read_csv(C.DESIGN_CSV, usecols=["SEQN", "stratum", "psu"])
    m = d[["SEQN"]].reset_index(drop=True).merge(dv, on="SEQN", how="left")
    # m keeps the row order of d; indices() are positions within blk, mapped
    # back to rows of d through blk.index
    out = []
    for _, blk in m.groupby("stratum"):
        rows = blk.index.values
        out.append([rows[np.asarray(ix)] for ix in blk.groupby("psu").indices.values()])
    return out


def _summ(v, est, th):
    """Bootstrap summaries of one quantity: SE, mean, shift from the bootstrap
    population value th (in SEs), percentile interval, and the basic interval
    pivoted on th."""
    lo, hi = np.percentile(v, [2.5, 97.5])
    sd = float(v.std(ddof=1))
    return dict(se=sd, boot_mean=float(v.mean()),
                shift_in_sds=float((v.mean() - th) / sd) if sd > 0 else np.nan,
                pct_lo=lo, pct_hi=hi, basic_lo=est - (hi - th), basic_hi=est - (lo - th))


_NC = {}


def _nc_init(xs, zs, y0, n_null, n_perm):
    _NC.update(xs=xs, zs=zs, y0=y0, n_null=n_null, n_perm=n_perm)


def _nc_one(task):
    """One dataset of the null size check: death redrawn under conditional
    independence given the strata (each stratum's proportion of deaths), the
    exposures and strata fixed, then calibrated as the real data are."""
    design, h, r_ = task
    zc = _NC["zs"][(design, h)]
    y0 = _NC["y0"]
    py = (np.bincount(zc, weights=y0) / np.bincount(zc))[zc]
    rng = np.random.default_rng([C.SEED + 71, ("PRIMARY", "RAW").index(design),
                                 C.HABITS.index(h), r_])
    ys = (rng.random(zc.size) < py).astype(int)
    x = _NC["xs"][h]
    c = C.calibrate(C.Table(x, ys, [zc]), n_null=_NC["n_null"],
                    seed=[C.SEED + 73, C.HABITS.index(h), r_])
    pp = np.nan
    if design == "PRIMARY":
        pp = _wcal(x, ys, zc, np.ones(zc.size), _NC["n_perm"],
                   [C.SEED + 79, C.HABITS.index(h), r_])["p_null"]
    return design, h, c["p_null"], c["excess"], pp


def measures():
    """Total and direct association for all six habits.

    Every exposure is evaluated on the SAME analytic sample under the SAME kind
    of adjustment set: age band x survey period x tertile of a cross-fitted
    prognostic score built from the demographic covariates and the other five
    habits (54 strata).  Two null calibrations are reported (see C.calibrate);
    the D_JS-scale one is the magnitude estimate.  Standard errors come from a
    smoothed, paired bootstrap (individuals resampled, death simulated from an
    additive logistic model, null floor refitted in every resample; see
    _boot_one), repeated with masked variance units as the resampling unit.
    Intervals are Wald intervals from the bootstrap SD, for each habit and for
    the paired difference between habits; basic intervals pivoted on the
    association of the bootstrap population (the fitted model) are reported
    alongside, as are the bootstrap probabilities of each ranking position, the
    direct/total ratios on both scales, and a refinement diagnostic.
    """
    B = _env("B_BOOT", 1000, 20)
    B_PSU = _env("B_BOOT_PSU", 500, 10)
    N_NULL = _env("N_NULL", 500, 60)
    N_INNER = _env("N_NULL_INNER", 100, 20)
    N_SPLIT = _env("N_SPLIT", 100, 4)
    _save_params("measures", bootstrap_resamples=B, cluster_bootstrap_resamples=B_PSU,
                 null_draws=N_NULL, inner_null_draws=N_INNER, folds=5,
                 refinement_splits=N_SPLIT)
    t0 = time.time()

    d = C.load(common=True)
    n = len(d)
    print("primary analytic sample: n=%d, deaths=%d" % (n, int(d.died.sum())))
    sets, score = adjustment_sets(d)

    rows = []
    for habit in C.HABITS:
        for tag in SPECS:
            z = sets[tag][habit]
            tab = C.table_for(d, habit, z)
            cnt = tab.counts()
            c = C.calibrate(tab, n_null=N_NULL, seed=C.SEED)
            bach = C.achievable_upper_rcmi(cnt)
            y1 = cnt[:, 1, :].sum(axis=0)
            # deaths in every exposure-by-stratum cell: the mean hides a skewed
            # distribution (young and recent strata have almost no deaths)
            yc = cnt[:, 1, :].ravel()
            rows.append(dict(
                habit=habit, label=C.LABEL[habit], spec=tag, n=n,
                levels=int(d[habit].max() + 1),
                n_strata=1 if z is None else len(set(np.asarray(z))),
                deaths_per_cell=float(y1.sum() / (cnt.shape[0] * cnt.shape[2])),
                cell_deaths_median=float(np.median(yc)),
                cell_deaths_p10=float(np.percentile(yc, 10)),
                cell_deaths_p90=float(np.percentile(yc, 90)),
                cells_lt2=float((yc < 2).mean()), cells_lt5=float((yc < 5).mean()),
                rcmi=c["rcmi"], null_mean=c["null_mean"], null_ms=c["null_ms"],
                excess=c["excess"], calibrated=c["calibrated"], p_null=c["p_null"],
                djs=c["rcmi"] ** 2 - c["null_ms"],
                null_share=c["null_mean"] / c["rcmi"] if c["rcmi"] > 0 else np.nan,
                null_share_js=c["null_ms"] / c["rcmi"] ** 2 if c["rcmi"] > 0 else np.nan,
                domi=C.rmi_do_of(cnt), race=C.race_of(cnt),
                bound_alphabet=C.cmax(2), bound_achievable=bach))
            print("  %-6s %-8s K=%4d rCMI=%.4f floor=%.4f excess=%.4f calibrated=%.4f"
                  % (habit, tag, rows[-1]["n_strata"], c["rcmi"], c["null_mean"],
                     c["excess"], c["calibrated"]), flush=True)
    res = pd.DataFrame(rows)

    def est(h, s, col):
        return float(res[(res.habit == h) & (res.spec == s)][col].iloc[0])

    # ---- smoothed paired bootstrap ------------------------------------------
    zs = {(h, s): sets[s][h] for h in C.HABITS for s in BOOT_SPECS}
    p1 = _fit_p(_outcome_design(d), d["died"].values)
    xs = {h: d[h].values.astype(int) for h in C.HABITS}
    # the association in the bootstrap population: the rCMI of the table of
    # expected deaths under the fitted model (the pivot of the intervals)
    world = {}
    for (h, s), z in zs.items():
        zc = None if z is None else np.unique(z, return_inverse=True)[1]
        world[(h, s)] = C.rcmi_of(_expected_table(xs[h], zc, p1))
    draws = {}
    for unit, BB, psu, sd_ in (("individual", B, None, 1),
                               ("psu", B_PSU, _psu_groups(d), 2)):
        print("\nbootstrap (%s): B=%d, %d inner null draws, %d workers"
              % (unit, BB, N_INNER, _workers()), flush=True)
        keys = zs if unit == "individual" else {k: v for k, v in zs.items()
                                                if k[1] == "PRIMARY"}
        out = _pmap(_boot_one, range(BB), _boot_init, (p1, xs, keys, N_INNER, psu, sd_))
        for k in keys:
            for j, sc in enumerate(("excess", "calibrated", "djs")):
                draws[(unit, k, sc)] = np.array([o[k][j] for o in out])
            if unit == "individual" and k[1] == "PRIMARY":
                draws[(unit, k, "levels")] = np.array([o[k][3] for o in out])
        print("    done (%.0fs)" % (time.time() - t0), flush=True)

    ci = []
    for (h, s) in zs:
        row = dict(habit=h, spec=s, boot_world=world[(h, s)])
        for sc in ("excess", "calibrated"):
            v = draws[("individual", (h, s), sc)]
            e = est(h, s, sc)
            sm = _summ(v, e, world[(h, s)])
            row.update({sc + "_" + k: val for k, val in sm.items()})
            row[sc + "_pct_covers"] = bool(sm["pct_lo"] <= e <= sm["pct_hi"])
        if s == "PRIMARY":
            row["calibrated_se_psu"] = float(draws[("psu", (h, s), "calibrated")].std(ddof=1))
        ci.append(row)
    res = res.merge(pd.DataFrame(ci), on=["habit", "spec"], how="left")
    res.to_csv(os.path.join(C.OUT_DIR, "measures.csv"), index=False)

    # ---- paired differences --------------------------------------------------
    diffs = []
    for s in ("DEMO", "PRIMARY"):
        for sc in ("calibrated", "excess"):
            for i, a in enumerate(C.HABITS):
                for bb in C.HABITS[i + 1:]:
                    v = (draws[("individual", (a, s), sc)]
                         - draws[("individual", (bb, s), sc)])
                    e = est(a, s, sc) - est(bb, s, sc)
                    th = world[(a, s)] - world[(bb, s)]
                    sm = _summ(v, e, th)
                    sd = sm["se"]
                    # the dual of the basic interval: the resamples whose
                    # deviation from the model value is at least as large as
                    # the estimate, on its side
                    k = int((np.sign(e) * (v - th) >= abs(e)).sum())
                    zz = e / sd if sd > 0 else 0.0
                    row = dict(
                        spec=s, scale=sc, habit_a=a, habit_b=bb,
                        label_a=C.LABEL[a], label_b=C.LABEL[bb], diff=e, se=sd,
                        wald_lo=e - 1.96 * sd, wald_hi=e + 1.96 * sd,
                        p_wald=float(2 * (1 - C._norm_cdf(abs(zz)))),
                        pct_lo=sm["pct_lo"], pct_hi=sm["pct_hi"],
                        basic_lo=sm["basic_lo"], basic_hi=sm["basic_hi"],
                        p_boot=float(min(1.0, 2 * (k + 1) / (len(v) + 1))),
                        residual_shift=float(v.mean() - th),
                        sep_wald=bool(e - 1.96 * sd > 0 or e + 1.96 * sd < 0),
                        sep_pct=bool(sm["pct_lo"] > 0 or sm["pct_hi"] < 0),
                        sep_basic=bool(sm["basic_lo"] > 0 or sm["basic_hi"] < 0))
                    if s == "PRIMARY" and sc == "calibrated":
                        vp = (draws[("psu", (a, s), sc)] - draws[("psu", (bb, s), sc)])
                        sp = float(vp.std(ddof=1))
                        row.update(se_psu=sp, p_wald_psu=float(
                            2 * (1 - C._norm_cdf(abs(e / sp)))) if sp > 0 else np.nan)
                    diffs.append(row)

    # ---- ranking probabilities (pivoted draws: v - model value + estimate) ----
    def _rank_probs(dr, wd):
        piv = np.column_stack([dr[h] - wd[h] + est(h, "PRIMARY", "calibrated")
                               for h in C.HABITS])
        order = np.argsort(-piv, axis=1)
        out_ = {}
        for j, h in enumerate(C.HABITS):
            pos = np.argmax(order == j, axis=1)
            out_[h] = dict(p_first=float((pos == 0).mean()),
                           p_top2=float((pos <= 1).mean()),
                           p_last=float((pos == len(C.HABITS) - 1).mean()),
                           rank_lo=int(np.percentile(pos, 2.5)) + 1,
                           rank_hi=int(np.percentile(pos, 97.5)) + 1)
        return out_
    rp = _rank_probs({h: draws[("individual", (h, "PRIMARY"), "calibrated")]
                      for h in C.HABITS}, {h: world[(h, "PRIMARY")] for h in C.HABITS})
    rk = [dict(habit=h, label=C.LABEL[h], **rp[h]) for h in C.HABITS]

    # ---- direct / total, on the square-root and the D_JS scale ----------------
    # One interval, computed on the square-root scale and rescaled
    # multiplicatively: the bootstrap ratios divided by the ratio in the
    # bootstrap population, times the estimate.  It stays in [0, inf) and
    # respects the squaring map, so the D_JS-scale limits (the D_JS ratio is
    # the square of the square-root-scale ratio) are the squares of the
    # square-root-scale limits.
    ratios = []
    for h in C.HABITS:
        e = est(h, "PRIMARY", "calibrated") / est(h, "total", "calibrated")
        th = world[(h, "PRIMARY")] / world[(h, "total")]
        v = (draws[("individual", (h, "PRIMARY"), "calibrated")]
             / draws[("individual", (h, "total"), "calibrated")])
        lo, hi = np.percentile(v[np.isfinite(v)] / th, [2.5, 97.5])
        for scale, pw in (("sqrt", 1), ("djs", 2)):
            ratios.append(dict(habit=h, label=C.LABEL[h], scale=scale, ratio=e ** pw,
                               lo=(e * lo) ** pw, hi=(e * hi) ** pw,
                               model_ratio=th ** pw))
    pd.DataFrame(ratios).to_csv(os.path.join(C.OUT_DIR, "ratios.csv"), index=False)

    # ---- clustering of the outcome: plain resamples of the observed deaths ----
    # The smoothed bootstrap re-simulates death independently, so resampling
    # variance units there captures the clustering of exposures and strata but
    # not of death.  Resampling individuals and variance units with the observed
    # deaths kept, the ratio of the two standard errors of the plug-in estimate
    # measures the design effect that includes the outcome.
    B_PLAIN = _env("B_PLAIN", 500, 10)
    zp = {h: np.unique(sets["PRIMARY"][h], return_inverse=True)[1] for h in C.HABITS}
    plain = {}
    for unit, psu in (("individual", None), ("psu", _psu_groups(d))):
        out = _pmap(_plain_one, range(B_PLAIN), _plain_init,
                    (xs, d["died"].values.astype(int), zp, psu))
        plain[unit] = {h: np.array([o[h] for o in out]) for h in C.HABITS}
    deff = {h: float(plain["psu"][h].std(ddof=1) / plain["individual"][h].std(ddof=1))
            for h in C.HABITS}
    print("\ndesign effect on the SE of the plug-in, observed deaths kept "
          "(variance units / individuals): "
          + ", ".join("%s %.2f" % (h, v) for h, v in deff.items()), flush=True)

    # ---- the smoothing model: fit, and standard errors under a richer one -----
    # The additive smoothing model is tested against the model with habit x age
    # band and habit x period terms (the 'interaction' population of the
    # semi-parametric check), and the primary standard errors are recomputed
    # with that model as the smoother.
    Xa = _outcome_design(d)
    Xi = _interaction_design(d)
    yv = d["died"].values.astype(float)
    bi, _ = C.irls(Xi, yv, link="logit", ridge=1e-4)
    p1i = 1.0 / (1.0 + np.exp(-np.clip(Xi @ bi, -30, 30)))

    def _dev(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-2 * np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
    lr = _dev(p1) - _dev(p1i)
    df_lr = int(np.linalg.matrix_rank(Xi) - np.linalg.matrix_rank(Xa))
    p_lr = float(C.chi2_sf(lr, df_lr))
    print("smoothing model: additive vs interaction LR = %.1f on %d df, P = %.2g"
          % (lr, df_lr, p_lr), flush=True)
    # How much of the interaction model's larger association is overfitting:
    # deaths simulated from the fitted additive model (no interactions in the
    # truth), both models refitted, and the rCMI of their tables of expected
    # deaths compared on the D_JS scale.
    R_OVF = _env("R_OVERFIT", 30, 2)
    zc_p = {h: np.unique(sets["PRIMARY"][h], return_inverse=True)[1] for h in C.HABITS}
    ovf = {h: [] for h in C.HABITS}
    for rr in range(R_OVF):
        ys_o = (np.random.default_rng([C.SEED + 61, rr]).random(n) < p1).astype(float)
        pa_o = _fit_p(Xa, ys_o)
        bo, _ = C.irls(Xi, ys_o, link="logit", ridge=1e-4)
        pi_o = 1.0 / (1.0 + np.exp(-np.clip(Xi @ bo, -30, 30)))
        for h in C.HABITS:
            ovf[h].append(C.rcmi_of(_expected_table(xs[h], zc_p[h], pi_o)) ** 2
                          - C.rcmi_of(_expected_table(xs[h], zc_p[h], pa_o)) ** 2)
    print("overfitting of the interaction model (D_JS, %d datasets from the additive "
          "model): " % R_OVF + ", ".join("%s %.2e" % (h, np.mean(v)) for h, v in ovf.items()),
          flush=True)
    B_INT = _env("B_BOOT_INT", 500, 10)
    keys_p = {k: v for k, v in zs.items() if k[1] == "PRIMARY"}
    out = _pmap(_boot_one, range(B_INT), _boot_init, (p1i, xs, keys_p, N_INNER, None, 3))
    # the smoother with the two-year survey cycle in place of the four-year
    # period: the alcohol questionnaire changed between the two cycles of
    # 2015-2018 (lifetime abstainers 2015-2016 vs 2017-2018)
    dcy = d.copy()
    dcy["cyc5"] = dcy["cyc"].astype(int)
    p1c = _fit_p(_outcome_design(dcy), d["died"].values)
    world_int, draws_int = {}, {}
    smooth = []
    for (h, s), z in keys_p.items():
        zc = np.unique(z, return_inverse=True)[1]
        world_int[h] = C.rcmi_of(_expected_table(xs[h], zc, p1i))
        draws_int[h] = np.array([o[(h, s)][1] for o in out])
        smooth.append(dict(
            habit=h, label=C.LABEL[h], se_additive=est(h, s, "calibrated_se"),
            se_interaction=float(draws_int[h].std(ddof=1)),
            world_additive=world[(h, s)], world_interaction=world_int[h],
            world_cycle=C.rcmi_of(_expected_table(xs[h], zc, p1c)),
            overfit_djs=float(np.mean(ovf[h])),
            overfit_djs_mcse=float(np.std(ovf[h], ddof=1) / np.sqrt(len(ovf[h]))),
            deff_plain=deff[h], lr=lr, lr_df=df_lr, lr_p=p_lr,
            aic_diff=float(2 * df_lr - lr)))
    pd.DataFrame(smooth).to_csv(os.path.join(C.OUT_DIR, "smoothing.csv"), index=False)

    # paired differences and ranking probabilities with the interaction model
    # as the smoother (the additive one is rejected by the likelihood-ratio test)
    for row in diffs:
        if row["spec"] != "PRIMARY" or row["scale"] != "calibrated":
            continue
        a, bb, e = row["habit_a"], row["habit_b"], row["diff"]
        v = draws_int[a] - draws_int[bb]
        th = world_int[a] - world_int[bb]
        sd = float(v.std(ddof=1))
        k = int((np.sign(e) * (v - th) >= abs(e)).sum())
        # the SE ratio (variance units / individuals) of the plug-in
        # difference with the observed deaths kept, and the normal-reference
        # P with the SE inflated by it: the clustering of death that the
        # smoothed bootstrap cannot see
        fd = float((plain["psu"][a] - plain["psu"][bb]).std(ddof=1)
                   / (plain["individual"][a] - plain["individual"][bb]).std(ddof=1))
        row.update(se_int=sd,
                   p_wald_int=float(2 * (1 - C._norm_cdf(abs(e / sd)))) if sd > 0 else np.nan,
                   p_boot_int=float(min(1.0, 2 * (k + 1) / (len(v) + 1))),
                   deff_plain_diff=fd,
                   p_wald_deff=float(2 * (1 - C._norm_cdf(abs(e / (row["se"] * fd))))))
    pd.DataFrame(diffs).to_csv(os.path.join(C.OUT_DIR, "pairwise.csv"), index=False)
    rpi = _rank_probs(draws_int, world_int)
    for row in rk:
        row.update({kk + "_int": vv for kk, vv in rpi[row["habit"]].items()})
    pd.DataFrame(rk).to_csv(os.path.join(C.OUT_DIR, "ranks.csv"), index=False)
    # ---- size of the null calibration --------------------------------------
    # Datasets generated under conditional independence given the strata
    # (exposures and strata fixed, death drawn from each stratum's proportion
    # of deaths), calibrated like the real data: how often P <= 0.05 and 0.10,
    # and the mean excess, under the multinomial null (the primary one, drawn
    # from the fitted p(z) p(x|z) p(y|z)) and, for the primary strata, under
    # the within-stratum permutation null of SA15-SA16.  The exact
    # demographic strata show what happens at fewer than one death per cell.
    R_NC = _env("R_NULL_CHECK", 120, 4)
    R_NC_RAW = _env("R_NULL_CHECK_RAW", 20, 2)
    N_NC = _env("N_NULL_CHECK", 200, 30)
    zc_nc = {(s, h): np.unique(sets[s][h], return_inverse=True)[1]
             for s in ("PRIMARY", "RAW") for h in C.HABITS}
    tasks = ([("PRIMARY", h, r_) for h in C.HABITS for r_ in range(R_NC)]
             + [("RAW", h, r_) for h in C.HABITS for r_ in range(R_NC_RAW)])
    nc = _pmap(_nc_one, tasks, _nc_init,
               (xs, zc_nc, d["died"].values.astype(float), N_NC, N_NC // 2))
    ncd = pd.DataFrame(nc, columns=["design", "habit", "p_multinomial", "excess",
                                    "p_permutation"])
    out_nc = {}
    for s, g in ncd.groupby("design"):
        k = len(g)
        o = dict(datasets=int(k), per_habit=int(k // len(C.HABITS)), null_draws=N_NC,
                 reject_05=float((g.p_multinomial <= 0.05).mean()),
                 reject_10=float((g.p_multinomial <= 0.10).mean()),
                 reject_mcse_05=float(np.sqrt((g.p_multinomial <= 0.05).mean()
                                              * (1 - (g.p_multinomial <= 0.05).mean()) / k)),
                 reject_mcse_10=float(np.sqrt((g.p_multinomial <= 0.10).mean()
                                              * (1 - (g.p_multinomial <= 0.10).mean()) / k)),
                 mean_excess={h: float(gg.excess.mean()) for h, gg in g.groupby("habit")},
                 mean_excess_mcse={h: float(gg.excess.std(ddof=1) / np.sqrt(len(gg)))
                                   for h, gg in g.groupby("habit")},
                 sd_excess={h: float(gg.excess.std(ddof=1)) for h, gg in g.groupby("habit")})
        if g.p_permutation.notna().any():
            o.update(permutations=N_NC // 2,
                     perm_reject_05=float((g.p_permutation <= 0.05).mean()),
                     perm_reject_10=float((g.p_permutation <= 0.10).mean()))
        out_nc[s] = o
        print("null check, %s strata (%d datasets): multinomial null P<=0.05 in %.3f, "
              "P<=0.10 in %.3f%s; mean excess %s" % (
                  s, k, o["reject_05"], o["reject_10"],
                  "" if "perm_reject_05" not in o else
                  "; permutation null %.3f and %.3f" % (o["perm_reject_05"],
                                                        o["perm_reject_10"]),
                  ", ".join("%s %.5f" % (h, v) for h, v in o["mean_excess"].items())),
              flush=True)
    json.dump(out_nc, open(os.path.join(C.OUT_DIR, "null_check.json"), "w",
                           encoding="utf-8", newline=chr(10)), indent=1)
    _save_params("measures", bootstrap_resamples=B, cluster_bootstrap_resamples=B_PSU,
                 null_draws=N_NULL, inner_null_draws=N_INNER, folds=5,
                 refinement_splits=N_SPLIT, plain_resamples=B_PLAIN,
                 interaction_smoother_resamples=B_INT, overfit_datasets=R_OVF,
                 null_check_datasets_per_habit=R_NC,
                 null_check_datasets_per_habit_exact=R_NC_RAW,
                 null_check_null_draws=N_NC, null_check_permutations=N_NC // 2)

    # ---- each exposure level's share of the direct association -----------------
    # rCMI^2 (D_JS) is a sum of non-negative cell terms, so it splits exactly
    # over the exposure levels (C.djs_levels): for the plug-in table, for the
    # null floor (the same draws as the calibration, so that plug-in minus null
    # adds up to the calibrated D_JS before truncation) and for the tables of
    # expected deaths under the two smoothing models.
    lev = []
    yv_i = d["died"].values.astype(int)
    for h in C.HABITS:
        tab = C.table_for(d, h, sets["PRIMARY"][h])
        zc = np.unique(sets["PRIMARY"][h], return_inverse=True)[1]
        plug = C.djs_levels(tab.counts())
        nul = C.null_djs_levels(tab, n_null=N_NULL, seed=C.SEED)
        assert np.isclose(nul.sum(), est(h, "PRIMARY", "null_ms"), rtol=1e-9)
        add = C.djs_levels(_expected_table(xs[h], zc, p1))
        itr = C.djs_levels(_expected_table(xs[h], zc, p1i))
        cal = plug - nul
        # the bootstrap SE of each level's calibrated contribution (its share
        # is unstable when the total is near zero, as for sleep)
        lb = draws[("individual", (h, "PRIMARY"), "levels")]
        for lv in range(plug.size):
            m = xs[h] == lv
            lev.append(dict(
                habit=h, label=C.LABEL[h], level=lv, level_name=C.LEVELS[h][lv],
                reference=bool(lv == C.REF_LEVEL[h]), n=int(m.sum()),
                pct=float(100 * m.mean()), deaths=int(yv_i[m].sum()),
                mean_age=float(d["age"].values[m].mean()),
                plugin=float(plug[lv]), null=float(nul[lv]), calibrated=float(cal[lv]),
                model_additive=float(add[lv]), model_interaction=float(itr[lv]),
                share_plugin=float(plug[lv] / plug.sum()),
                share_calibrated=float(cal[lv] / cal.sum()),
                share_additive=float(add[lv] / add.sum()),
                share_interaction=float(itr[lv] / itr.sum()),
                calibrated_se=float(lb[:, lv].std(ddof=1))))
    lev = pd.DataFrame(lev)
    lev.to_csv(os.path.join(C.OUT_DIR, "levels.csv"), index=False)
    print("\nshare of D_JS by exposure level (additive smoothing model):")
    for h in C.HABITS:
        q_ = lev[lev.habit == h]
        print("  %-18s " % C.LABEL[h] + "; ".join(
            "%s %.0f%%" % (a, 100 * b) for a, b in zip(q_.level_name, q_.share_additive)))

    # ---- refinement diagnostic: every primary stratum split at random in two --
    ref = []
    for h in C.HABITS:
        base = sets["PRIMARY"][h]
        for r in range(N_SPLIT):
            rnd = np.random.default_rng([C.SEED + 31, r]).integers(0, 2, n)
            c = C.calibrate(C.table_for(d, h, C.combine_codes(base, rnd)),
                            n_null=N_NULL, seed=C.SEED)
            ref.append(dict(habit=h, label=C.LABEL[h], split=r,
                            calibrated=c["calibrated"],
                            djs=c["rcmi"] ** 2 - c["null_ms"]))
    ref = pd.DataFrame(ref)
    ref.to_csv(os.path.join(C.OUT_DIR, "refinement.csv"), index=False)
    print("\nrefinement (each primary stratum split at random in two, %d splits):"
          % N_SPLIT)
    for h in C.HABITS:
        r = ref[ref.habit == h]
        ch = r.calibrated.mean() - est(h, "PRIMARY", "calibrated")
        mc = r.calibrated.std(ddof=1) / np.sqrt(len(r))
        print("  %-18s primary %.4f  split mean %.4f (SD %.4f, MC SE %.4f; "
              "change %+.4f = %.1f MC SE)"
              % (C.LABEL[h], est(h, "PRIMARY", "calibrated"), r.calibrated.mean(),
                 r.calibrated.std(ddof=1), mc, ch, ch / mc))

    # ---- the null floor as a prognostic-score stratification is refined ------
    curve = []
    for k in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]:
        z = C.deciles(score, k)
        for habit in ["smoke", "pa", "bmi"]:
            c = C.calibrate(C.table_for(d, habit, z), n_null=N_NULL, seed=C.SEED)
            curve.append(dict(habit=habit, label=C.LABEL[habit], k_strata=k,
                              mean_per_stratum=n / k, rcmi=c["rcmi"],
                              null_mean=c["null_mean"], excess=c["excess"],
                              calibrated=c["calibrated"],
                              null_share=c["null_mean"] / c["rcmi"]))
        print("  k=%4d  n/stratum=%7.1f  null share (smoking)=%.1f%%"
              % (k, n / k, 100 * curve[-3]["null_share"]), flush=True)
    pd.DataFrame(curve).to_csv(os.path.join(C.OUT_DIR, "biasfloor.csv"), index=False)

    pri = res[res.spec == "PRIMARY"].sort_values("calibrated", ascending=False)
    print("\nPRIMARY RESULT (age band x period x prognostic-score tertile)")
    print(pri[["label", "n_strata", "calibrated", "calibrated_se", "calibrated_se_psu",
               "excess", "p_null"]].round(5).to_string(index=False))
    print(pd.DataFrame(rk).round(3).to_string(index=False))
    dd = pd.DataFrame(diffs)
    dd = dd[(dd.spec == "PRIMARY") & (dd.scale == "calibrated")]
    print("\nseparable pairs (Wald / percentile / basic): %d / %d / %d of %d"
          % (dd.sep_wald.sum(), dd.sep_pct.sum(), dd.sep_basic.sum(), len(dd)))
    print(dd[["label_a", "label_b", "diff", "se", "wald_lo", "wald_hi", "p_wald",
              "p_boot", "p_wald_psu"]].round(4).to_string(index=False))
    print("\n%.0fs" % (time.time() - t0))


# =====================================================================
# 2.  Time-to-event analysis
# =====================================================================
def _fit_habit(d, habit, risk_bin, timescale="age", weight=None,
               beta0=None, fixed_hess=None, extra=()):
    """Cox fit for one exposure.

    ``cyc5`` is the four-year survey period (2007-2010, 2011-2014, 2015-2018 in
    the primary sample).  With attained age as the time scale it is an ordinary
    covariate.  With time on study, period fixes administrative censoring (the
    2017-2018 cycle ends at about three years), so the time-on-study
    sensitivity analysis gives each period its own baseline hazard; adjusting
    for it as a covariate instead gives the same hazard ratios to two decimals
    once step halving is in place.
    """
    others = [g for g in C.HABITS if g != habit]
    adj = ["__rb", "race5", "educ", "pir", "marital", "cyc5", "sex_b"] + others + list(extra)
    f = d.copy()
    f["__rb"] = risk_bin
    levels = sorted(pd.unique(f[habit]).tolist())
    ref = C.REF_LEVEL[habit]
    ordered = [ref] + [l for l in levels if l != ref]
    Hm = C.onehot(f[habit].values, ordered)
    if timescale == "age":
        entry = f["age"].values.astype(float)
        exit_ = f["age_exit"].values.astype(float)
        Z, _ = C.design(f, adj, intercept=False)
        m = C.CoxPH().fit(np.hstack([Z, Hm]), exit_,
                          f["died"].values.astype(int), entry=entry,
                          weight=weight, beta0=beta0, fixed_hess=fixed_hess)
    else:
        entry = np.zeros(len(f))
        exit_ = np.maximum(f["py"].values.astype(float), 1e-6)
        adj = ["age_b"] + [c for c in adj if c != "cyc5"]
        Z, _ = C.design(f, adj, intercept=False)
        m = C.CoxPH().fit_stratified(np.hstack([Z, Hm]), exit_,
                                     f["died"].values.astype(int),
                                     strata=f["cyc5"].values, entry=entry,
                                     weight=weight)
    return m, Z, Hm, ordered, levels


def _standardised_risk(m, Z, Hm, ordered, levels, timescale="age", mask=None):
    """g-formula standardised cumulative risk at the fixed horizon, averaged
    over the cohort (or over the rows selected by ``mask``)."""
    out = {}
    sel = slice(None) if mask is None else np.asarray(mask, bool)
    for lv in levels:
        Hc = np.zeros_like(Hm)
        j = ordered.index(lv) - 1
        if j >= 0:
            Hc[:, j] = 1.0
        X = np.hstack([Z, Hc])[sel]
        if timescale == "age":
            entry = m._entry[sel]
            out[lv] = float(np.mean(m.risk_at(X, entry + C.HORIZON / 12.0,
                                              entry=entry)))
        else:
            out[lv] = float(np.mean(m.risk_at(X, np.full(len(X), C.HORIZON / 12.0))))
    return out


_SW = {}


def _surv_init(d, risk_bin, anchor):
    _SW.update(d=d, risk_bin=risk_bin, anchor=anchor)


def _surv_one(b):
    """One bootstrap resample of the Cox models and standardised risks, warm-
    started at the full-sample fit with its Hessian held fixed."""
    d, rbin, anchor = _SW["d"], _SW["risk_bin"], _SW["anchor"]
    n = len(d)
    idx = np.random.default_rng([C.SEED + 2, b]).integers(0, n, n)
    db, rb = d.iloc[idx].reset_index(drop=True), rbin[idx]
    out = {}
    for habit in C.HABITS:
        try:
            b0, h0 = anchor[habit]
            m, Z, Hm, ordered, levels = _fit_habit(db, habit, rb,
                                                   beta0=b0, fixed_hess=h0)
            p = Z.shape[1]
            risks = _standardised_risk(m, Z, Hm, ordered, levels)
            ref = ordered[0]
            for lv in levels:
                j = ordered.index(lv) - 1
                out[(habit, int(lv))] = (1.0 if j < 0 else float(np.exp(m.beta_[p + j])),
                                         risks[lv] / risks[ref],
                                         100 * (risks[lv] - risks[ref]))
        except Exception as exc:                                    # noqa: BLE001
            print("    resample %d, %s failed: %s" % (b, habit, exc))
    return out


def survival():
    """Cox models with attained age as the time scale, plus the standardised
    10-year risk.

    Follow-up ranges from a median of about 12 years in the 2007-2008 cycle to
    about 2 years in 2017-2018, and differs across levels of the same exposure, so a
    binary "died by end of follow-up" indicator largely measures observation
    time rather than risk.  Hazard-ratio intervals are Wald intervals; the
    risk-difference intervals come from a percentile bootstrap in which the
    Cox model is refitted in every resample (prognostic-score deciles held
    fixed).
    """
    B = _env("B_SURV", 400, 5)
    _save_params("survival", survival_bootstrap_resamples=B, horizon_years=10)
    t0 = time.time()
    d = C.load(common=True)
    n = len(d)
    print("n=%d deaths=%d person-years=%.0f" % (n, int(d.died.sum()), d.py.sum()))
    risk_bin = C.deciles(C.crossfit_score(d), 10)

    hr_rows, risk_rows, anchor = [], [], {}
    for habit in C.HABITS:
        m, Z, Hm, ordered, levels = _fit_habit(d, habit, risk_bin)
        anchor[habit] = (m.beta_.copy(), m.hess_.copy())
        p = Z.shape[1]
        beta_h, se_h = m.beta_[p:], m.se_[p:]
        risks = _standardised_risk(m, Z, Hm, ordered, levels)
        # the same, averaged only over participants below the top-coded age
        risks80 = _standardised_risk(m, Z, Hm, ordered, levels,
                                     mask=d["age"].values < 80)
        ref = ordered[0]
        for lv in levels:
            j = ordered.index(lv) - 1
            hr = 1.0 if j < 0 else float(np.exp(beta_h[j]))
            hr_rows.append(dict(
                habit=habit, label=C.LABEL[habit], level=int(lv),
                level_name=C.LEVELS[habit][int(lv)], reference=bool(lv == ref),
                hr=hr,
                hr_lo=1.0 if j < 0 else float(np.exp(beta_h[j] - 1.96 * se_h[j])),
                hr_hi=1.0 if j < 0 else float(np.exp(beta_h[j] + 1.96 * se_h[j])),
                n_level=int((d[habit] == lv).sum()),
                deaths_level=int(d.loc[d[habit] == lv, "died"].sum()),
                py_level=float(d.loc[d[habit] == lv, "py"].sum())))
            risk_rows.append(dict(
                habit=habit, label=C.LABEL[habit], level=int(lv),
                level_name=C.LEVELS[habit][int(lv)], reference=bool(lv == ref),
                risk10=100 * risks[lv], rr10=risks[lv] / risks[ref],
                rd10=100 * (risks[lv] - risks[ref]),
                risk10_lt80=100 * risks80[lv], rd10_lt80=100 * (risks80[lv] - risks80[ref])))
        print("  %-6s max HR=%.3f" % (habit, max(np.exp(beta_h))), flush=True)

    print("\nbootstrap (%d resamples, Cox refitted in each; %d workers)"
          % (B, _workers()), flush=True)
    out = _pmap(_surv_one, range(B), _surv_init, (d, risk_bin, anchor))
    keys = [(h, int(l)) for h in C.HABITS for l in sorted(pd.unique(d[h]).tolist())]
    bh = {k: [o[k][0] for o in out if k in o] for k in keys}
    br = {k: [o[k][1] for o in out if k in o] for k in keys}
    brd = {k: [o[k][2] for o in out if k in o] for k in keys}

    def qci(v):
        v = np.asarray(v, float)
        return ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                if v.size else (np.nan, np.nan))

    hr, rk = pd.DataFrame(hr_rows), pd.DataFrame(risk_rows)
    hr["hr_boot_lo"], hr["hr_boot_hi"] = zip(*[qci(bh[(r.habit, r.level)])
                                               for r in hr.itertuples()])
    rk["rr10_lo"], rk["rr10_hi"] = zip(*[qci(br[(r.habit, r.level)]) for r in rk.itertuples()])
    rk["rd10_lo"], rk["rd10_hi"] = zip(*[qci(brd[(r.habit, r.level)]) for r in rk.itertuples()])
    hr["n_boot"] = [len(bh[(r.habit, r.level)]) for r in hr.itertuples()]
    hr.to_csv(os.path.join(C.OUT_DIR, "cox.csv"), index=False)
    rk.to_csv(os.path.join(C.OUT_DIR, "standardised_risk.csv"), index=False)
    print(hr[["label", "level_name", "n_level", "deaths_level", "hr", "hr_lo", "hr_hi"]]
          .round(3).to_string(index=False))
    print("\n%.0fs" % (time.time() - t0))


def _age_split(d, cut=65.0):
    """Counting-process episodes split at attained age ``cut``: an episode
    before and one after, the death (if any) assigned to the episode in which
    it occurred, and ``older`` = 1 for the episode after the cut."""
    a0, a1 = d["age"].values.astype(float), d["age_exit"].values.astype(float)
    ev = d["died"].values.astype(int)
    rows_idx, entry, exit_, event, older = [], [], [], [], []
    for i in range(len(d)):
        if a1[i] <= cut:
            rows_idx.append(i); entry.append(a0[i]); exit_.append(a1[i])
            event.append(ev[i]); older.append(0)
        elif a0[i] >= cut:
            rows_idx.append(i); entry.append(a0[i]); exit_.append(a1[i])
            event.append(ev[i]); older.append(1)
        else:
            rows_idx += [i, i]
            entry += [a0[i], cut]
            exit_ += [cut, a1[i]]
            event += [0, ev[i]]
            older += [0, 1]
    ep = d.iloc[rows_idx].reset_index(drop=True).copy()
    ep["age"], ep["age_exit"] = np.asarray(entry), np.asarray(exit_)
    ep["died"], ep["older"] = np.asarray(event), np.asarray(older)
    return ep, np.asarray(rows_idx)


def hr_by_attained_age(d, risk_bin, cut=65.0):
    """Hazard ratio of every contrast below and above attained age ``cut``
    (one Cox model per habit with habit x [age >= cut] interaction terms), for
    the exposures whose relative hazard may change with age."""
    ep, rows = _age_split(d, cut)
    rb = risk_bin[rows]
    out = []
    for habit in C.HABITS:
        others = [g for g in C.HABITS if g != habit]
        adj = ["__rb", "race5", "educ", "pir", "marital", "cyc5", "sex_b"] + others
        f = ep.copy()
        f["__rb"] = rb
        ref = C.REF_LEVEL[habit]
        levels = sorted(pd.unique(f[habit]).tolist())
        ordered = [ref] + [l for l in levels if l != ref]
        Hm = C.onehot(f[habit].values, ordered)
        Z, _ = C.design(f, adj, intercept=False)
        old = f["older"].values[:, None].astype(float)
        X = np.hstack([Z, old, Hm, Hm * old])
        m = C.CoxPH().fit(X, f["age_exit"].values, f["died"].values, entry=f["age"].values)
        p, k = Z.shape[1] + 1, Hm.shape[1]
        for j, lv in enumerate(ordered[1:]):
            b1, b2 = m.beta_[p + j], m.beta_[p + k + j]
            v11, v22 = m.cov_[p + j, p + j], m.cov_[p + k + j, p + k + j]
            v12 = m.cov_[p + j, p + k + j]
            s_old = np.sqrt(v11 + v22 + 2 * v12)
            out.append(dict(
                habit=habit, label=C.LABEL[habit], level=C.LEVELS[habit][int(lv)],
                hr_young=np.exp(b1), lo_young=np.exp(b1 - 1.96 * np.sqrt(v11)),
                hi_young=np.exp(b1 + 1.96 * np.sqrt(v11)),
                hr_old=np.exp(b1 + b2), lo_old=np.exp(b1 + b2 - 1.96 * s_old),
                hi_old=np.exp(b1 + b2 + 1.96 * s_old),
                p_interaction=float(2 * (1 - C._norm_cdf(abs(b2 / np.sqrt(v22)))))))
    return pd.DataFrame(out)


def ph_check():
    """Proportional-hazards assessment for every exposure: the score test of
    Grambsch and Therneau for a coefficient that drifts with the rank of the
    event time, per contrast and jointly over each exposure's contrasts."""
    t0 = time.time()
    _save_params("ph", ph_test="Grambsch-Therneau score test, rank of event time",
                 age_split=65)
    d = C.load(common=True)
    risk_bin = C.deciles(C.crossfit_score(d), 10)
    rows = []
    for habit in C.HABITS:
        m, Z, Hm, ordered, levels = _fit_habit(d, habit, risk_bin)
        p = Z.shape[1]
        cols = list(range(p, p + Hm.shape[1]))
        per, chi2, k = m.schoenfeld_test(cols=cols)
        pg = C.chi2_sf(chi2, k)
        for row, lv in zip(per.itertuples(), ordered[1:]):
            rows.append(dict(habit=habit, label=C.LABEL[habit],
                             level=C.LEVELS[habit][int(lv)], z=row.z, rho=row.rho,
                             p=row.p))
        rows.append(dict(habit=habit, label=C.LABEL[habit], level="GLOBAL",
                         z=np.nan, rho=np.nan, p=pg, chi2=chi2, df=k))
        print("  %-18s chi2=%6.2f on %d df, P=%.3f" % (C.LABEL[habit], chi2, k, pg))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.OUT_DIR, "ph_check.csv"), index=False)
    by_age = hr_by_attained_age(d, risk_bin)
    by_age.to_csv(os.path.join(C.OUT_DIR, "cox_by_age.csv"), index=False)
    print(by_age.round(3).to_string(index=False))
    g = out[out.level == "GLOBAL"]
    print("\nexposures with P < 0.05 for non-proportionality: %s"
          % (list(g[g.p < 0.05].label) or "none"))
    print("\n%.0fs" % (time.time() - t0))
    return out


# =====================================================================
# 3.  Sensitivity analyses
# =====================================================================
def _excess_table(d, z_map, label, n_null=500, extra=None, habits=None, boot=0,
                  n_inner=50, n_perm=None):
    """Calibrated rCMI of every habit under the adjustment sets ``z_map``, with
    the deaths per exposure-by-stratum cell (mean, and the share of cells with
    fewer than two), the Monte-Carlo P against the multinomial null (p_null)
    and against the within-stratum permutation null (p_perm, ``n_perm``
    permutations; the multinomial null is liberal in sparse strata, see the
    size checks) and, if ``boot`` > 0, the smoothed-bootstrap standard error
    of the calibrated estimate (``boot`` resamples; the smoothing model refitted
    on ``d``, the strata held fixed, ``n_inner`` null draws per resample)."""
    rows = []
    hs = habits or C.HABITS
    for habit in hs:
        tab = C.table_for(d, habit, z_map[habit])
        c = C.calibrate(tab, n_null=n_null, seed=C.SEED)
        yc = tab.counts()[:, 1, :].ravel()
        zc = np.unique(np.asarray(z_map[habit]), return_inverse=True)[1]
        n_perm = n_perm or _env("N_PERM_SA", 500, 60)
        # the seeds of the unweighted SA16 baseline, so that the primary
        # analysis has one permutation P (Supplementary Tables S1 and S3)
        pp = _wcal(d[habit].values.astype(int), d["died"].values.astype(int), zc,
                   np.ones(len(d)), n_perm, [C.SEED + 53, C.HABITS.index(habit)])["p_null"]
        rows.append(dict(analysis=label, habit=habit, label=C.LABEL[habit],
                         n=len(d), n_deaths=int(d.died.sum()),
                         n_strata=len(np.unique(z_map[habit])),
                         deaths_per_cell=float(yc.mean()),
                         cells_lt2=float((yc < 2).mean()),
                         rcmi=c["rcmi"], null=c["null_mean"], excess=c["excess"],
                         calibrated=c["calibrated"], p_null=c["p_null"], p_perm=pp,
                         **(extra or {})))
    out = pd.DataFrame(rows)
    if boot:
        p1 = _fit_p(_outcome_design(d), d["died"].values)
        xs = {h: d[h].values.astype(int) for h in hs}
        keys = {(h, "sa"): np.unique(z_map[h], return_inverse=True)[1] for h in hs}
        res = _pmap(_boot_one, range(boot), _boot_init, (p1, xs, keys, n_inner, None, 4))
        out["calibrated_se"] = [float(np.std([o[(h, "sa")][1] for o in res], ddof=1))
                                for h in hs]
    return out


def _refine(d, z_map, n_split, n_null):
    """Refinement diagnostic of an adjustment set: every stratum split at
    random in two (population values unchanged); mean calibrated estimate over
    ``n_split`` splits, and its Monte-Carlo standard error."""
    out, mcse = {}, {}
    for h in C.HABITS:
        v = [C.calibrate(C.table_for(d, h, C.combine_codes(
                 z_map[h], np.random.default_rng([C.SEED + 31, r]).integers(0, 2, len(d)))),
                 n_null=n_null, seed=C.SEED)["calibrated"] for r in range(n_split)]
        out[h] = float(np.mean(v))
        mcse[h] = float(np.std(v, ddof=1) / np.sqrt(len(v)))
    return out, mcse


# ---- survey-weighted information measures (SA15) --------------------------
def _wtable(x, y, zc, w, dx, dz):
    return np.bincount((x * 2 + y) * dz + zc, weights=w,
                       minlength=dx * 2 * dz).reshape(dx, 2, dz)


def _wfun(wpar, y):
    """Completeness weights that depend on death (SA16): the inverse of the
    fitted probability of being complete, 1 + exp(-(eta + y * delta)), where
    eta is the linear predictor without the death terms and delta the effect
    of death at the participant's follow-up.  Their scale is irrelevant to
    the information measures."""
    eta, delta = wpar
    return 1.0 + np.exp(-np.clip(eta + y * delta, -30, 30))


def _wcal(x, y, zc, w, n_perm, seed, wpar=None):
    """Calibrated rCMI of the survey-weighted (x, y, z) table.  The null
    permutes death within each stratum, the weights and exposures staying with
    their rows, so it carries the weights' within-stratum variability (a single
    effective sample size for the whole table overstates it, because weights
    vary less within strata than overall).  When the weights depend on death
    (``wpar``, SA16) they are recomputed from each permuted death vector, so
    that the null tables up-weight deaths as the observed one does."""
    dx, dz = int(x.max() + 1), int(zc.max() + 1)
    wt = (lambda yy: w) if wpar is None else (lambda yy: _wfun(wpar, yy))  # noqa: E731
    r = C.rcmi_of(_wtable(x, y, zc, wt(y), dx, dz))
    rng = np.random.default_rng(seed)
    order = np.argsort(zc, kind="stable")
    nv = np.empty(n_perm)
    for k in range(n_perm):
        perm = np.lexsort((rng.random(len(y)), zc))   # grouped by stratum, random within
        yp = np.empty_like(y)
        yp[order] = y[perm]
        nv[k] = C.rcmi_of(_wtable(x, yp, zc, wt(yp), dx, dz))
    ms = float(np.mean(nv ** 2))
    return dict(rcmi=r, null_mean=float(nv.mean()), excess=r - float(nv.mean()),
                calibrated=float(np.sqrt(max(r * r - ms, 0.0))),
                p_null=float((1 + (nv >= r).sum()) / (n_perm + 1)))


_WW = {}


def _w_init(xs, zs, w, p1, y0, n_perm, wpar=None):
    _WW.update(xs=xs, zs=zs, w=w, p1=p1, y0=y0, n_perm=n_perm, wpar=wpar)


def _wboot_one(b):
    """One smoothed resample of the survey-weighted analysis: individuals
    resampled with their weights, death simulated from the smoothing model
    (weights that depend on death recomputed from the simulated deaths)."""
    rng = np.random.default_rng([C.SEED + 23, b])
    p1 = _WW["p1"]
    idx = rng.integers(0, p1.size, p1.size)
    ys = (rng.random(idx.size) < p1[idx]).astype(int)
    wp = None if _WW["wpar"] is None else (_WW["wpar"][0][idx], _WW["wpar"][1][idx])
    return {h: _wcal(_WW["xs"][h][idx], ys, np.unique(_WW["zs"][h][idx],
                                                      return_inverse=True)[1],
                     _WW["w"][idx], _WW["n_perm"], [C.SEED + 29, b], wpar=wp)["calibrated"]
            for h in C.HABITS}


def _wnull_one(task):
    """Calibration check of the weighted null: death redrawn under conditional
    independence (from the within-stratum death proportion), weights kept, or
    recomputed from the redrawn deaths when they depend on death (SA16)."""
    h, r = task
    rng = np.random.default_rng([C.SEED + 37, C.HABITS.index(h), r])
    zc = _WW["zs"][h]
    py = (np.bincount(zc, weights=_WW["y0"]) / np.bincount(zc))[zc]
    ys = (rng.random(zc.size) < py).astype(int)
    return h, _wcal(_WW["xs"][h], ys, zc, _WW["w"], _WW["n_perm"],
                    [C.SEED + 41, C.HABITS.index(h), r], wpar=_WW["wpar"])["p_null"]


# ---- completeness weights (SA16) ------------------------------------------
def _completeness_design(el):
    """Model for being a complete case among all eligible adults of 2007-2018:
    the covariates (education and marital status, missing for 71 adults, with
    a category for missing), the four habits observed for almost everyone
    (likewise), death, years of follow-up and their product.  Alcohol and BMI
    are what is mostly missing."""
    f = el[list(C.SCORE_COVARIATES)].copy()
    cols = list(C.SCORE_COVARIATES)
    for c in ("educ", "marital"):
        f[c] = f[c].fillna(-1).astype(int) + 1
    for h in ("smoke", "pa", "sleep", "sed"):
        f[h] = el[h].fillna(-1).astype(int) + 1
        cols.append(h)
    f["died"] = el["died"].astype(int)
    cols.append("died")
    X, _ = C.design(f, cols)
    py = el["py"].values.astype(float)
    return np.hstack([X, py[:, None], (py * f["died"].values)[:, None]])


def _ipw_init(el, comp, X, d16, rb16, anchor, fixed):
    _CW.update(el=el, comp=comp, X16=X, d16=d16, rb16=rb16, anchor16=anchor,
               fixed16=fixed, pos=np.cumsum(comp) - 1)


def _ipw_one(b):
    """One bootstrap resample of SA16: eligible adults resampled, the
    completeness model refitted, the weighted Cox models refitted (the
    risk-score decile held at its full-sample value)."""
    comp, X = _CW["comp"], _CW["X16"]
    rng = np.random.default_rng([C.SEED + 43, b])
    idx = rng.integers(0, comp.size, comp.size)
    ci = comp[idx]
    pc = _fit_p(X[idx], ci.astype(float))
    rows = _CW["pos"][idx[ci]]
    w = 1.0 / pc[ci]
    w = w / w.mean()
    try:
        hh = _headline_hr(_CW["d16"].iloc[rows].reset_index(drop=True), _CW["rb16"][rows],
                          weight=w, fixed_level=_CW["fixed16"], anchor=_CW["anchor16"])
        return {h: hh[h]["hr"] for h in C.HABITS}
    except Exception as exc:                                        # noqa: BLE001
        print("    SA16 resample %d failed: %s" % (b, exc))
        return {}


def _group_baseline(beta, X, entry, exit_, event):
    """A Cox model with fixed coefficients and the Breslow baseline of one
    stratum of a stratified fit, for standardised risks within that stratum."""
    ms = C.CoxPH()
    w = np.ones(len(X))
    ms._prepare(X, entry, exit_, event, w)
    ms._X, ms._event = X, event
    ms._exit, ms._entry, ms._w = exit_, entry, w
    ms.beta_ = np.asarray(beta, float)
    ms._baseline()
    return ms


#: the coarser codings of SA20 (level codes as in lsm_core.LEVELS)
COARSE = {"smoke": {0: 0, 1: 0, 2: 1},
          "alc": {0: 0, 1: 0, 2: 1, 3: 1, 4: 2},
          "bmi": {0: 0, 1: 0, 2: 1, 3: 2, 4: 2},
          "pa": {0: 0, 1: 1, 2: 1, 3: 1},
          "sleep": {0: 0, 1: 0, 2: 1, 3: 2},
          "sed": {0: 0, 1: 0, 2: 1, 3: 1}}


def _interaction_hr(d, habit, risk_bin, mod, level, stratified=True):
    """The Cox model of Table 3 (attained age, delayed entry, the demographic
    covariates, the prognostic-score decile and the other five habits) with the
    habit's levels crossed with a binary modifier ``mod``.  With ``stratified``
    the two groups have separate baseline hazards (illness does not act
    proportionally over attained age); otherwise ``mod`` enters as a
    proportional covariate, and the Grambsch-Therneau test of that assumption
    is returned.  Returns the hazard ratio of ``level`` against the reference
    when mod = 0 and when mod = 1, their ratio, the Wald test of that ratio
    (1 df), the joint Wald test of all habit x modifier terms and every
    contrast of the habit (by_level)."""
    others = [g for g in C.HABITS if g != habit]
    adj = (["__rb", "race5", "educ", "pir", "marital", "cyc5", "sex_b"] + others
           + ([] if stratified else [mod]))
    f = d.copy()
    f["__rb"] = risk_bin
    ref = C.REF_LEVEL[habit]
    ordered = [ref] + [l for l in sorted(pd.unique(f[habit]).tolist()) if l != ref]
    Hm = C.onehot(f[habit].values, ordered)
    g = f[mod].values.astype(float)[:, None]
    Z, _ = C.design(f, adj, intercept=False)
    X_ = np.hstack([Z, Hm, Hm * g])
    args_ = (f["age_exit"].values.astype(float), f["died"].values.astype(int))
    ph_mod = {}
    if stratified:
        m = C.CoxPH().fit_stratified(X_, *args_, strata=f[mod].values.astype(int),
                                     entry=f["age"].values.astype(float))
        # g-formula 10-year standardised risk of the fixed contrast and of the
        # reference level within each group, from its own Breslow baseline and
        # the shared coefficients (the additive-scale counterpart of the ratio)
        k_ = Hm.shape[1]
        ent_, ext_ = f["age"].values.astype(float), f["age_exit"].values.astype(float)
        gi = f[mod].values.astype(int)
        for gv in (0, 1):
            rows = gi == gv
            ms = _group_baseline(m.beta_, X_[rows], ent_[rows], ext_[rows],
                                 f["died"].values[rows].astype(int))
            rk_ = {}
            for lv in (ordered[0], level):
                Hc = np.zeros((int(rows.sum()), k_))
                if ordered.index(lv) > 0:
                    Hc[:, ordered.index(lv) - 1] = 1.0
                Xc = np.hstack([Z[rows], Hc, Hc * gv])
                rk_[lv] = float(np.mean(ms.risk_at(Xc, ent_[rows] + C.HORIZON / 12.0,
                                                   entry=ent_[rows])))
            ph_mod["risk_ref%d" % gv] = rk_[ordered[0]]
            ph_mod["risk_level%d" % gv] = rk_[level]
            ph_mod["rd%d" % gv] = rk_[level] - rk_[ordered[0]]
    else:
        m = C.CoxPH().fit(X_, *args_, entry=f["age"].values.astype(float))
        # 'mod' is the last column of Z
        per_, _, _ = m.schoenfeld_test(cols=[Z.shape[1] - 1])
        ph_mod = dict(ph_mod_z=float(per_.z.iloc[0]), ph_mod_p=float(per_.p.iloc[0]))
    p, k = Z.shape[1], Hm.shape[1]
    j = ordered.index(level) - 1
    i0, i1 = p + j, p + k + j
    V = m.cov_
    b0, b1 = m.beta_[i0], m.beta_[i1]
    s0, s1 = np.sqrt(V[i0, i0]), np.sqrt(V[i1, i1])
    s01 = np.sqrt(V[i0, i0] + V[i1, i1] + 2 * V[i0, i1])
    bi = m.beta_[p + k:p + 2 * k]
    chi = float(bi @ np.linalg.solve(V[p + k:p + 2 * k, p + k:p + 2 * k], bi))
    ex = lambda v: float(np.exp(v))                                 # noqa: E731

    def contrast(jj):
        a0, a1 = p + jj, p + k + jj
        c0 = m.beta_[a0]
        c1 = m.beta_[a1]
        t0 = np.sqrt(V[a0, a0])
        t01 = np.sqrt(V[a0, a0] + V[a1, a1] + 2 * V[a0, a1])
        t1 = np.sqrt(V[a1, a1])
        return dict(level_name=C.LEVELS[habit][int(ordered[jj + 1])],
                    hr0=ex(c0), hr0_lo=ex(c0 - 1.96 * t0), hr0_hi=ex(c0 + 1.96 * t0),
                    hr1=ex(c0 + c1), hr1_lo=ex(c0 + c1 - 1.96 * t01),
                    hr1_hi=ex(c0 + c1 + 1.96 * t01),
                    ratio=ex(c1), ratio_lo=ex(c1 - 1.96 * t1), ratio_hi=ex(c1 + 1.96 * t1),
                    p=float(2 * (1 - C._norm_cdf(abs(c1 / t1)))))
    return dict(
        model="stratified" if stratified else "proportional", **ph_mod,
        by_level=[contrast(jj) for jj in range(k)],
        level_name=C.LEVELS[habit][int(level)],
        n0=int((g[:, 0] == 0).sum()), n1=int((g[:, 0] == 1).sum()),
        deaths0=int(f["died"].values[g[:, 0] == 0].sum()),
        deaths1=int(f["died"].values[g[:, 0] == 1].sum()),
        hr0=ex(b0), hr0_lo=ex(b0 - 1.96 * s0), hr0_hi=ex(b0 + 1.96 * s0),
        hr1=ex(b0 + b1), hr1_lo=ex(b0 + b1 - 1.96 * s01), hr1_hi=ex(b0 + b1 + 1.96 * s01),
        ratio=ex(b1), ratio_lo=ex(b1 - 1.96 * s1), ratio_hi=ex(b1 + 1.96 * s1),
        p_level=float(2 * (1 - C._norm_cdf(abs(b1 / s1)))),
        chi2=chi, df=int(k), p_joint=float(C.chi2_sf(chi, k)))


def _prevalence_factor(p1, target, xs, zs):
    """How much a different proportion of deaths alone changes each habit's
    rCMI: the smoothing model with its intercept shifted (by bisection) so
    that the expected proportion of deaths is ``target``, the odds ratios
    unchanged; the ratio of the rCMI of its table of expected deaths to that
    under the unshifted model."""
    eta = np.log(p1 / (1 - p1))
    lo_, hi_ = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo_ + hi_)
        if np.mean(1 / (1 + np.exp(-(eta + mid)))) > target:
            hi_ = mid
        else:
            lo_ = mid
    ps = 1 / (1 + np.exp(-(eta + 0.5 * (lo_ + hi_))))
    return {h: float(C.rcmi_of(_expected_table(xs[h], zs[h], ps))
                     / C.rcmi_of(_expected_table(xs[h], zs[h], p1))) for h in C.HABITS}


def _stratified_hr(d, habit, z, level):
    """Cox model with attained age as the time scale and a separate baseline
    hazard for each stratum of the primary adjustment set, the habit's levels
    the only covariates: the hazard ratio of ``level`` against the reference."""
    ref = C.REF_LEVEL[habit]
    ordered = [ref] + [l for l in sorted(pd.unique(d[habit]).tolist()) if l != ref]
    Hm = C.onehot(d[habit].values, ordered)
    m = C.CoxPH().fit_stratified(Hm, d["age_exit"].values.astype(float),
                                 d["died"].values.astype(int), strata=z,
                                 entry=d["age"].values.astype(float))
    j = ordered.index(level) - 1
    return dict(hr=float(np.exp(m.beta_[j])),
                hr_lo=float(np.exp(m.beta_[j] - 1.96 * m.se_[j])),
                hr_hi=float(np.exp(m.beta_[j] + 1.96 * m.se_[j])),
                level_name=C.LEVELS[habit][int(level)])


def _landmark(d, months):
    """Participants still under follow-up ``months`` after the interview, with
    entry (attained age), exit and person-time counted from that landmark."""
    out = d[d.permth > months].copy()
    out["age"] = out["age"] + months / 12.0
    out["permth"] = out["permth"] - months
    out["py"] = out["permth"] / 12.0
    return out.reset_index(drop=True)


def _build_adj(d, covariates=None, n_bins=3, k=5):
    """Prognostic-score decile (for the Cox models) and the primary adjustment
    set of every habit, rebuilt on the analytic table ``d``."""
    covs = list(covariates or C.DEMOG_COVARIATES)
    score = C.crossfit_score(d, covariates=covs)
    return C.deciles(score, 10), {h: C.direct_set(d, h, n_bins=n_bins, covariates=covs,
                                                  k=k) for h in C.HABITS}


_HW = {}


def _hr_init(d, risk_bin, timescale, weight, anchor, fixed_level, extra=()):
    _HW.update(d=d, risk_bin=risk_bin, timescale=timescale, weight=weight,
               anchor=anchor, fixed_level=fixed_level, extra=extra)


def _hr_one(habit):
    d, risk_bin, anchor, fixed_level = (_HW["d"], _HW["risk_bin"], _HW["anchor"],
                                        _HW["fixed_level"])
    kw = {}
    if anchor and habit in anchor:
        kw = dict(beta0=anchor[habit][0], fixed_hess=anchor[habit][1])
    m, Z, Hm, ordered, levels = _fit_habit(d, habit, risk_bin,
                                           timescale=_HW["timescale"],
                                           weight=_HW["weight"],
                                           extra=_HW.get("extra", ()), **kw)
    p = Z.shape[1]
    hr = np.exp(m.beta_[p:])
    if fixed_level and habit in fixed_level:
        j = ordered.index(fixed_level[habit]) - 1
    else:
        j = int(np.argmax(hr))
    # model-based Wald interval (not valid for weighted fits, whose intervals
    # come from the survey bootstrap)
    se = float(m.se_[p + j]) if getattr(m, "se_", None) is not None else np.nan
    return habit, dict(hr=float(hr[j]), level=int(ordered[j + 1]),
                       level_name=C.LEVELS[habit][int(ordered[j + 1])],
                       hr_lo=float(np.exp(m.beta_[p + j] - 1.96 * se)),
                       hr_hi=float(np.exp(m.beta_[p + j] + 1.96 * se)),
                       beta=m.beta_.copy(), hess=m.hess_.copy())


def _headline_hr(d, risk_bin, timescale="age", weight=None, anchor=None,
                 fixed_level=None, parallel=False, extra=()):
    """Hazard ratio of each exposure's highest-HR level against its reference.

    ``fixed_level`` ({habit: level}) pins the contrast instead of re-choosing
    the argmax, which a resampling interval must do: an interval for the
    maximum over levels is not an interval for any named contrast.
    ``parallel`` fits the six habits in separate processes (top level only:
    never inside a resampling worker)."""
    args = (d, risk_bin, timescale, weight, anchor, fixed_level, tuple(extra))
    if parallel:
        res = _pmap(_hr_one, C.HABITS, _hr_init, args)
    else:
        _hr_init(*args)
        res = [_hr_one(h) for h in C.HABITS]
    return dict(res)


_CW = {}


def _cluster_init(dws, rbw, w, psu_rows, anchor, fixed):
    _CW.update(dws=dws, rbw=rbw, w=w, psu_rows=psu_rows, anchor=anchor, fixed=fixed)


def _cluster_one(b):
    """One Rao-Wu rescaling resample: in each stratum draw n_h - 1 of its n_h
    masked PSUs with replacement and scale their weights by n_h/(n_h - 1)
    times their multiplicity.  (Drawing n_h of n_h understates the between-PSU
    variance by (n_h - 1)/n_h -- a factor of two with the two PSUs per stratum
    that NHANES has almost everywhere.)"""
    dws, rbw, w = _CW["dws"], _CW["rbw"], _CW["w"]
    rng = np.random.default_rng([C.SEED + 5, b])
    mult = np.zeros(len(dws))
    for rows in _CW["psu_rows"]:                 # one list of PSU row-arrays per stratum
        nh = len(rows)
        for j in rng.choice(nh, size=nh - 1, replace=True):
            mult[rows[j]] += nh / (nh - 1.0)
    keep = mult > 0
    try:
        hh = _headline_hr(dws[keep].reset_index(drop=True), rbw[keep],
                          weight=w[keep] * mult[keep], fixed_level=_CW["fixed"],
                          anchor=_CW["anchor"])
        return {h: hh[h]["hr"] for h in C.HABITS}
    except Exception as exc:                                        # noqa: BLE001
        print("    cluster resample %d failed: %s" % (b, exc))
        return {}


def sensitivity():
    """The sensitivity analyses SA1-SA21 (SA3 at two and four years; SA4 in
    never-smokers overall and from a 2-year landmark; SA17 excluding (a, c) or
    adjusting for (b) prevalent disease).  Every information-measure analysis
    except SA1 carries a smoothed-bootstrap standard error."""
    B_CLUSTER = _env("B_CLUSTER", 500, 5)
    N_NULL = _env("N_NULL", 500, 60)
    B_SA = _env("B_BOOT_SA", 200, 10)
    B_IPW = _env("B_IPW", 200, 5)
    N_PERM = _env("N_PERM_SA15", 500, 30)
    R_WNULL = _env("R_SA15_CHECK", 100, 5)
    N_SPLIT = _env("N_SPLIT", 100, 4)
    _save_params("sensitivity", cluster_bootstrap_resamples=B_CLUSTER,
                 null_draws=N_NULL, bootstrap_resamples=B_SA, ipw_resamples=B_IPW,
                 sa15_permutations=N_PERM,
                 permutations_all=_env("N_PERM_SA", 500, 60),
                 sa17a_null_check_datasets_per_habit=_env("R_NULL_CHECK_SA17A", 80, 4),
                 sa15_null_check_datasets=R_WNULL * len(C.HABITS),
                 sa15_null_check_datasets_per_habit=R_WNULL,
                 sa15_null_check_permutations=N_PERM // 5,
                 sa16_null_check_datasets=R_WNULL * len(C.HABITS),
                 refinement_splits=N_SPLIT, landmark_years=[2, 4], censor_years=10,
                 age_topcode=80)
    t0 = time.time()
    rcmi_parts, surv_rows, extra_out = [], [], {}

    d = C.load(common=True)
    rb, zmap = _build_adj(d)
    primary = _excess_table(d, zmap, "primary", n_null=N_NULL)
    rcmi_parts.append(primary)

    def hr_rows(tag, dd, hh, **extra):
        for h in C.HABITS:
            row = dict(analysis=tag, habit=h, label=C.LABEL[h], n=len(dd),
                       n_deaths=int(dd.died.sum()), contrast=hh[h]["level_name"],
                       hr=hh[h]["hr"], hr_lo=hh[h]["hr_lo"], hr_hi=hh[h]["hr_hi"],
                       ci="Wald")
            row.update(extra.get(h, {}))
            surv_rows.append(row)

    # every analysis reports the same contrast for a habit: the highest-hazard
    # level of the primary fit against the reference level
    prim = _headline_hr(d, rb, parallel=True)
    fixed = {h: prim[h]["level"] for h in C.HABITS}
    hr_rows("primary", d, prim)

    # ---- SA1: the conventional single-exposure workflow ---------------------
    # Each habit in the largest sample complete on it and on those co-habits
    # whose completeness there exceeds 0.85, adjusted with the same construction
    # as the primary analysis for those co-habits only.  For smoking, alcohol
    # and BMI (all ten cycles) this drops the 2007-onward instruments and gives
    # one shared sample complete on the three; for physical activity, sleep and
    # sitting it is the primary sample.
    print("SA1  conventional single-exposure workflow", flush=True)
    full = C.load(common=False)
    for habit in C.HABITS:
        sub = full.dropna(subset=[habit]).copy()
        others = [g for g in C.HABITS if g != habit and sub[g].notna().mean() > 0.85]
        sub = sub.dropna(subset=others).copy()
        sub = sub.astype({c: int for c in [habit] + others})
        sub = sub.reset_index(drop=True)
        z = C.direct_set(sub, habit, others=others)
        rcmi_parts.append(_excess_table(
            sub, {habit: z}, "SA1 conventional single-exposure workflow",
            n_null=N_NULL, habits=[habit]).assign(n_cohabits=len(others)))
        r = rcmi_parts[-1].iloc[0]
        print("  %-6s n=%6d co-habits=%d calibrated=%.4f"
              % (habit, len(sub), len(others), r.calibrated), flush=True)

    # ---- SA2: complex survey design ----------------------------------------
    print("\nSA2  MEC-weighted Cox, Rao-Wu rescaling bootstrap over masked PSUs",
          flush=True)
    dv = pd.read_csv(C.DESIGN_CSV)
    dw = d.merge(dv[["SEQN", "stratum", "psu", "WTMEC2YR"]], on="SEQN", how="left")
    dws = dw[dw["WTMEC2YR"].notna() & (dw["WTMEC2YR"] > 0)].reset_index(drop=True)
    # pooled weight: the 2-year weight divided by the number of cycles pooled
    # (a constant, which does not change a weighted hazard ratio)
    w = dws["WTMEC2YR"].values / dws["cycle"].nunique()
    rbw = C.deciles(C.crossfit_score(dws), 10)
    anchor_w = _headline_hr(dws, rbw, weight=w, fixed_level=fixed, parallel=True)
    by_psu = dws.groupby("psu").indices
    psu_rows = [[by_psu[p] for p in dws.loc[ix, "psu"].unique()]
                for _, ix in dws.groupby("stratum").indices.items()]
    out = _pmap(_cluster_one, range(B_CLUSTER), _cluster_init,
                (dws, rbw, w, psu_rows,
                 {h: (anchor_w[h]["beta"], anchor_w[h]["hess"]) for h in C.HABITS},
                 fixed))
    # Replicate weights estimate a VARIANCE: with two PSUs per stratum each
    # Rao-Wu replicate is a reweighted half-sample, so its percentiles inherit
    # the half-sample bias of the Cox estimator.  The interval is therefore
    # log HR +/- 1.96 x the replicate standard error of log HR.
    ci = {}
    for h in C.HABITS:
        v = np.log(np.asarray([o[h] for o in out if h in o], float))
        se = float(v.std(ddof=1)) if v.size > 1 else np.nan
        lhr = np.log(anchor_w[h]["hr"])
        ci[h] = dict(hr_lo=float(np.exp(lhr - 1.96 * se)),
                     hr_hi=float(np.exp(lhr + 1.96 * se)),
                     se_log_hr=se, n_boot=int(v.size), ci="Rao-Wu")
        print("  %-6s weighted HR=%.3f (%.3f-%.3f)" % (h, anchor_w[h]["hr"],
                                                       ci[h]["hr_lo"], ci[h]["hr_hi"]))
    hr_rows("SA2 MEC-weighted", dws, anchor_w, **ci)
    print("  %d strata, %d PSUs" % (len(psu_rows), sum(len(r) for r in psu_rows)))

    # ---- SA3: reverse causation, 2-year landmark ---------------------------
    # Everyone still under follow-up at two years enters the risk set two years
    # after the interview; deaths (and censoring) before that are excluded.
    # A 4-year landmark as well: illness curtails activity (and changes weight)
    # long before death, so two years may be too short a lag.
    for months, tag in ((24, "SA3 2-year landmark"), (48, "SA3b 4-year landmark")):
        print("\n%s" % tag, flush=True)
        d3 = _landmark(d, months)
        rb3, z3 = _build_adj(d3)
        rcmi_parts.append(_excess_table(d3, z3, tag, n_null=N_NULL, boot=B_SA))
        hr_rows(tag, d3, _headline_hr(d3, rb3, fixed_level=fixed, parallel=True))

    # ---- SA4: BMI among never-smokers --------------------------------------
    print("\nSA4  BMI restricted to never-smokers", flush=True)
    for tag, dd in [("SA4a never-smokers", d[d.smoke == 0]),
                    ("SA4b never-smokers, 2-year landmark",
                     _landmark(d[d.smoke == 0], 24))]:
        dd = dd.reset_index(drop=True)
        rbx = C.deciles(C.crossfit_score(dd), 10)
        m, Z, Hm, ordered, levels = _fit_habit(dd, "bmi", rbx)
        p = Z.shape[1]
        hr, se = np.exp(m.beta_[p:]), m.se_[p:]
        for j, lv in enumerate(ordered[1:]):
            surv_rows.append(dict(analysis=tag, habit="bmi", label="Body mass index",
                                  n=len(dd), n_deaths=int(dd.died.sum()),
                                  contrast=C.LEVELS["bmi"][int(lv)],
                                  hr=float(hr[j]),
                                  hr_lo=float(np.exp(m.beta_[p + j] - 1.96 * se[j])),
                                  hr_hi=float(np.exp(m.beta_[p + j] + 1.96 * se[j])),
                                  ci="Wald"))
        print("  %-36s n=%5d  HR <18.5 = %.3f" % (tag, len(dd), hr[ordered.index(0) - 1]))

    # ---- SA5-SA11 ------------------------------------------------------------
    print("\nSA5  full-sample rather than cross-fitted prognostic score", flush=True)
    _, z5 = _build_adj(d, k=1)
    rcmi_parts.append(_excess_table(d, z5, "SA5 full-sample score", n_null=N_NULL,
                                    boot=B_SA))

    print("SA6  prognostic score without marital status", flush=True)
    _, z6 = _build_adj(d, covariates=[c for c in C.DEMOG_COVARIATES if c != "marital"])
    rcmi_parts.append(_excess_table(d, z6, "SA6 no marital status", n_null=N_NULL,
                                    boot=B_SA))

    print("SA7  time on study instead of attained age", flush=True)
    hr_rows("SA7 time-on-study scale", d,
            _headline_hr(d, rb, timescale="study", fixed_level=fixed, parallel=True))

    print("SA8  administrative censoring at 10 years", flush=True)
    d8 = d.copy()
    d8["died"] = np.where(d8["permth"] > 120, 0, d8["died"])
    d8["permth"] = np.minimum(d8["permth"], 120)
    d8["py"] = d8["permth"] / 12.0
    d8["age_exit"] = (d8["age"] * 12.0 + d8["permth"]) / 12.0
    rb8, z8 = _build_adj(d8)
    rcmi_parts.append(_excess_table(d8, z8, "SA8 censored at 10 years", n_null=N_NULL,
                                    boot=B_SA))
    hr_rows("SA8 censored at 10 years", d8,
            _headline_hr(d8, rb8, fixed_level=fixed, parallel=True))

    # RIDAGEYR is top-coded at 80 in 2007-2018, so everyone aged 80 or over
    # enters the attained-age risk sets at 80 whatever their true age.
    print("SA9  excluding participants with top-coded age (80)", flush=True)
    d9 = d[d.age < 80].reset_index(drop=True)
    rb9, z9 = _build_adj(d9)
    rcmi_parts.append(_excess_table(d9, z9, "SA9 age <80", n_null=N_NULL, boot=B_SA))
    hr_rows("SA9 age <80", d9, _headline_hr(d9, rb9, fixed_level=fixed, parallel=True))

    # Sleep is a direct report of usual hours (SLD010H) to 2013-2014 and is
    # derived from reported sleep-onset and wake times (SLD012) from 2015-2016.
    print("SA10 cycles 2007-2014 only (one sleep instrument)", flush=True)
    d10 = d[d.cycle.isin(["2007-2008", "2009-2010", "2011-2012",
                          "2013-2014"])].reset_index(drop=True)
    rb10, z10 = _build_adj(d10)
    rcmi_parts.append(_excess_table(d10, z10, "SA10 cycles 2007-2014", n_null=N_NULL,
                                    boot=B_SA))
    hr_rows("SA10 cycles 2007-2014", d10,
            _headline_hr(d10, rb10, fixed_level=fixed, parallel=True))

    # The adjustment used in the first version of this analysis: a decile of
    # the prognostic score, without explicit age strata, split by a tertile of
    # the co-habit index.  It leaves residual age within deciles.
    print("SA11 prognostic-score decile x co-habit tertile (no age strata)", flush=True)
    rcmi_parts.append(_excess_table(
        d, {h: C.combine_codes(rb, C.lifestyle_index(d, h, rb)) for h in C.HABITS},
        "SA11 score decile, no age strata", n_null=N_NULL, boot=B_SA))

    # The primary adjustment set of the previous version of this analysis: age
    # band x tertile of a demographic score (survey period only inside the
    # score), each stratum split by a tertile of a co-habit risk index.
    print("SA12 previous primary set (co-habit tertile, period only in the score)",
          flush=True)
    demo12 = C.demographic_strata(d, C.crossfit_score(d))
    rcmi_parts.append(_excess_table(
        d, {h: C.adjustment_set(d, h, demo12) for h in C.HABITS},
        "SA12 previous primary set", n_null=N_NULL, boot=B_SA))

    print("SA13 score quintiles (90 strata)", flush=True)
    _, z13 = _build_adj(d, n_bins=5)
    rcmi_parts.append(_excess_table(d, z13, "SA13 score quintiles", n_null=N_NULL,
                                    boot=B_SA))

    # A fixed horizon: death within 5 years among the cycles in which every
    # survivor was followed for at least 5 years (2007-2014).
    print("SA14 death within 5 years, cycles 2007-2014", flush=True)
    d14 = d[d.cycle.isin(["2007-2008", "2009-2010", "2011-2012",
                          "2013-2014"])].reset_index(drop=True)
    d14["died"] = ((d14["died"] == 1) & (d14["permth"] <= 60)).astype(int)
    _, z14 = _build_adj(d14)
    rcmi_parts.append(_excess_table(d14, z14, "SA14 5-year horizon", n_null=N_NULL,
                                    boot=B_SA))

    # The information measures from MEC-weighted tables (pooled weight
    # WTMEC2YR / number of cycles), calibrated against a within-stratum
    # permutation null that keeps the weights on their rows; the null is
    # checked on data generated under conditional independence, and the
    # standard errors come from a smoothed bootstrap of the weighted analysis.
    print("SA15 MEC-weighted information measures", flush=True)
    dv15 = pd.read_csv(C.DESIGN_CSV, usecols=["SEQN", "WTMEC2YR"])
    w15 = d[["SEQN"]].merge(dv15, on="SEQN", how="left")["WTMEC2YR"].values
    w15 = w15 / d["cycle"].nunique()
    n_eff = float(w15.sum() ** 2 / (w15 ** 2).sum())
    xs15 = {h: d[h].values.astype(int) for h in C.HABITS}
    zs15 = {h: np.unique(zmap[h], return_inverse=True)[1] for h in C.HABITS}
    y15 = d["died"].values.astype(int)
    p1 = _fit_p(_outcome_design(d), y15)
    winit = (xs15, zs15, w15, p1, y15.astype(float), N_PERM // 5)
    wse = _pmap(_wboot_one, range(B_SA), _w_init, winit)
    wchk = _pmap(_wnull_one, [(h, r) for h in C.HABITS for r in range(R_WNULL)],
                 _w_init, winit)
    pv = np.array([p_ for _, p_ in wchk])
    extra_out["sa15_null_check"] = dict(
        datasets=int(pv.size), permutations=N_PERM // 5,
        reject_05=float((pv <= 0.05).mean()),
        reject_05_mcse=float(np.sqrt((pv <= 0.05).mean() * (1 - (pv <= 0.05).mean())
                                     / pv.size)),
        mean_p=float(pv.mean()), n_eff=n_eff)
    print("  null check under conditional independence: P <= 0.05 in %.3f of %d "
          "datasets (mean P %.2f)" % ((pv <= 0.05).mean(), pv.size, pv.mean()), flush=True)
    wdp15 = float(np.average(y15, weights=w15))
    extra_out["prevalence_reference_sa15"] = dict(
        death_proportion_weighted=wdp15,
        factor=_prevalence_factor(p1, wdp15, xs15, zs15))
    for h in C.HABITS:
        c = _wcal(xs15[h], y15, zs15[h], w15, N_PERM, [C.SEED + 47, C.HABITS.index(h)])
        yc = C.table_for(d, h, zmap[h]).counts()[:, 1, :].ravel()
        rcmi_parts.append(pd.DataFrame([dict(
            analysis="SA15 MEC-weighted", habit=h, label=C.LABEL[h], n=len(d),
            n_deaths=int(d.died.sum()), n_strata=len(np.unique(zmap[h])),
            deaths_per_cell=float(yc.mean()), cells_lt2=float((yc < 2).mean()),
            rcmi=c["rcmi"], null=c["null_mean"], excess=c["excess"],
            calibrated=c["calibrated"], p_null=c["p_null"], p_perm=c["p_null"],
            n_eff=n_eff,
            calibrated_se=float(np.std([o[h] for o in wse], ddof=1)))]))

    # Complete-case selection: Cox models weighted by the inverse probability
    # of being complete, from a logistic model among all eligible adults of
    # 2007-2018 (see _completeness_design); intervals from a bootstrap that
    # refits the completeness model in every resample.  The information
    # measures are recomputed from the completeness-weighted tables, with the
    # permutation null and smoothed bootstrap of SA15; because the weights
    # depend on death, both recompute them from the permuted or simulated
    # deaths (the completeness model's coefficients held fixed).
    print("SA16 inverse-probability-of-completeness weights", flush=True)
    el = C.load(common=False, complete_covariates=False)
    el = el[el.cycle.isin(C.PA_CYCLES)].reset_index(drop=True)
    comp = el[C.HABITS + list(C.COVARIATES)].notna().all(axis=1).values
    X16 = _completeness_design(el)
    pc = _fit_p(X16, comp.astype(float))
    # the weights as a function of death, for the null and the bootstrap of
    # the information measures: death enters the completeness model through
    # 'died' and 'py x died' (the last column; 'died' is third from last)
    b16, _ = C.irls(X16, comp.astype(float), link="logit")
    died_el = el["died"].values.astype(float)
    delta_el = b16[-3] + b16[-1] * X16[:, -2]
    eta_el = X16 @ b16 - died_el * delta_el
    assert np.allclose(1.0 / (1.0 + np.exp(-np.clip(eta_el + died_el * delta_el, -30, 30))), pc)
    d16 = el[comp].reset_index(drop=True)
    d16 = d16.astype({c: int for c in C.HABITS + list(C.COVARIATES)})
    w16 = 1.0 / pc[comp]
    w16 = w16 / w16.mean()
    extra_out["sa16_weights"] = dict(
        n_eligible=int(len(el)), n_complete=int(comp.sum()),
        n_missing_covariate=int(el[list(C.COVARIATES)].isna().any(axis=1).sum()),
        min=float(w16.min()), p01=float(np.percentile(w16, 1)),
        p99=float(np.percentile(w16, 99)), max=float(w16.max()))
    w16d = d[["SEQN"]].merge(pd.DataFrame({"SEQN": d16["SEQN"].values, "w": w16}),
                             on="SEQN", how="left")["w"].values
    assert np.isfinite(w16d).all() and len(d16) == len(d)
    n_eff16 = float(w16d.sum() ** 2 / (w16d ** 2).sum())
    wpd = d[["SEQN"]].merge(pd.DataFrame({"SEQN": el["SEQN"].values[comp],
                                          "eta": eta_el[comp], "delta": delta_el[comp]}),
                            on="SEQN", how="left")
    wpar16 = (wpd["eta"].values, wpd["delta"].values)
    w_chk = _wfun(wpar16, y15.astype(float))
    assert np.allclose(w_chk / w_chk.mean(), w16d)
    wse16 = _pmap(_wboot_one, range(B_SA), _w_init,
                  (xs15, zs15, w16d, p1, y15.astype(float), N_PERM // 5, wpar16))
    wchk16 = _pmap(_wnull_one, [(h, r) for h in C.HABITS for r in range(R_WNULL)],
                   _w_init, (xs15, zs15, w16d, p1, y15.astype(float), N_PERM // 5, wpar16))
    pv16 = np.array([p_ for _, p_ in wchk16])
    extra_out["sa16_null_check"] = dict(
        datasets=int(pv16.size), permutations=N_PERM // 5,
        reject_05=float((pv16 <= 0.05).mean()),
        reject_05_mcse=float(np.sqrt((pv16 <= 0.05).mean() * (1 - (pv16 <= 0.05).mean())
                                     / pv16.size)),
        mean_p=float(pv16.mean()),
        reject_05_by_habit={h: float(np.mean([p_ <= 0.05 for g, p_ in wchk16 if g == h]))
                            for h in C.HABITS})
    print("  SA16 null check (weights recomputed from the redrawn deaths): P <= 0.05 "
          "in %.3f of %d datasets (mean P %.2f)" % ((pv16 <= 0.05).mean(), pv16.size,
                                                   pv16.mean()), flush=True)
    for h in C.HABITS:
        c = _wcal(xs15[h], y15, zs15[h], w16d, N_PERM, [C.SEED + 53, C.HABITS.index(h)],
                  wpar=wpar16)
        yc = C.table_for(d, h, zmap[h]).counts()[:, 1, :].ravel()
        rcmi_parts.append(pd.DataFrame([dict(
            analysis="SA16 completeness-weighted", habit=h, label=C.LABEL[h], n=len(d),
            n_deaths=int(d.died.sum()), n_strata=len(np.unique(zmap[h])),
            deaths_per_cell=float(yc.mean()), cells_lt2=float((yc < 2).mean()),
            rcmi=c["rcmi"], null=c["null_mean"], excess=c["excess"],
            calibrated=c["calibrated"], p_null=c["p_null"], p_perm=c["p_null"],
            n_eff=n_eff16,
            calibrated_se=float(np.std([o[h] for o in wse16], ddof=1)))]))
        print("  %-6s completeness-weighted calibrated=%.4f" % (h, c["calibrated"]),
              flush=True)
    base16 = {}
    for h in C.HABITS:
        c0 = _wcal(xs15[h], y15, zs15[h], np.ones(len(d)), N_PERM,
                   [C.SEED + 53, C.HABITS.index(h)])
        base16[h] = dict(calibrated=c0["calibrated"], p_null=c0["p_null"])
    wdp = float(np.average(y15, weights=w16d))
    extra_out["sa16_reference"] = dict(
        unweighted_permutation=base16, death_proportion_weighted=wdp,
        death_proportion=float(y15.mean()),
        factor=_prevalence_factor(p1, wdp, xs15, zs15))
    print("  SA16 baseline (no weights, permutation null): " + ", ".join(
        "%s %.4f" % (h, v["calibrated"]) for h, v in base16.items())
        + "; weighted death proportion %.4f" % wdp, flush=True)
    rb16 = C.deciles(C.crossfit_score(d16), 10)
    anchor16 = _headline_hr(d16, rb16, weight=w16, fixed_level=fixed, parallel=True)
    out = _pmap(_ipw_one, range(B_IPW), _ipw_init,
                (el, comp, X16, d16, rb16,
                 {h: (anchor16[h]["beta"], anchor16[h]["hess"]) for h in C.HABITS}, fixed))
    ci16 = {}
    for h in C.HABITS:
        v = np.log(np.asarray([o[h] for o in out if h in o], float))
        se = float(v.std(ddof=1)) if v.size > 1 else np.nan
        lhr = np.log(anchor16[h]["hr"])
        ci16[h] = dict(hr_lo=float(np.exp(lhr - 1.96 * se)),
                       hr_hi=float(np.exp(lhr + 1.96 * se)),
                       se_log_hr=se, n_boot=int(v.size), ci="bootstrap")
    hr_rows("SA16 completeness-weighted", d16, anchor16, **ci16)

    # Prevalent disease (ever told of heart disease, heart failure, stroke,
    # emphysema/chronic bronchitis/COPD, cancer or diabetes) and fair or poor
    # self-rated health: a common cause of inactivity, low weight, quitting
    # alcohol or smoking, long sleep and death.  (a) participants free of both;
    # (b) everyone, with both added to the prognostic score and the Cox models;
    # (c) as (a) with score halves, a set coarser for its fewer deaths.
    # The healthy subgroup is sparse, so SA17a also gets the refinement
    # diagnostic, and a prevalence-matched reference: the primary smoothing
    # model with its intercept shifted to SA17a's death proportion (odds ratios
    # unchanged), whose rCMI shows how much a lower proportion of deaths alone
    # changes the measure.
    print("SA17 prevalent disease and self-rated health", flush=True)
    d17 = d[d["chronic"].notna() & d["poor_health"].notna()].reset_index(drop=True)
    d17 = d17.astype({"chronic": int, "poor_health": int})
    d17a = d17[(d17.chronic == 0) & (d17.poor_health == 0)].reset_index(drop=True)
    rb17a, z17a = _build_adj(d17a)
    rcmi_parts.append(_excess_table(d17a, z17a, "SA17a free of prevalent disease",
                                    n_null=N_NULL, boot=B_SA))
    hr_rows("SA17a free of prevalent disease", d17a,
            _headline_hr(d17a, rb17a, fixed_level=fixed, parallel=True))
    extra_out["sa17a_refinement"], extra_out["sa17a_refinement_mcse"] = _refine(
        d17a, z17a, N_SPLIT, N_NULL)
    # size of the two nulls at the sparser strata of SA17a (2.5-4.1 deaths
    # per cell), as in the check of the primary strata (measures)
    R_NC17 = _env("R_NULL_CHECK_SA17A", 80, 4)
    zc17 = {("PRIMARY", h): np.unique(z17a[h], return_inverse=True)[1] for h in C.HABITS}
    nc17 = _pmap(_nc_one, [("PRIMARY", h, 1000 + r_) for h in C.HABITS
                           for r_ in range(R_NC17)], _nc_init,
                 ({h: d17a[h].values.astype(int) for h in C.HABITS}, zc17,
                  d17a["died"].values.astype(float), 200, 100))
    pm_ = np.array([o[2] for o in nc17])
    pp_ = np.array([o[4] for o in nc17])
    extra_out["sa17a_null_check"] = dict(
        datasets=int(pm_.size), per_habit=R_NC17, null_draws=200, permutations=100,
        reject_05=float((pm_ <= 0.05).mean()), reject_10=float((pm_ <= 0.10).mean()),
        perm_reject_05=float((pp_ <= 0.05).mean()),
        perm_reject_10=float((pp_ <= 0.10).mean()),
        reject_05_mcse=float(np.sqrt((pm_ <= 0.05).mean() * (1 - (pm_ <= 0.05).mean())
                                     / pm_.size)))
    print("  SA17a null check (%d datasets): multinomial P<=0.05 in %.3f, <=0.10 in "
          "%.3f; permutation %.3f and %.3f" % (
              pm_.size, (pm_ <= 0.05).mean(), (pm_ <= 0.10).mean(),
              (pp_ <= 0.05).mean(), (pp_ <= 0.10).mean()), flush=True)
    _, z17c = _build_adj(d17a, n_bins=2)
    rcmi_parts.append(_excess_table(d17a, z17c, "SA17c free of prevalent disease, "
                                    "score halves", n_null=N_NULL, boot=B_SA))
    target = float(d17a.died.mean())
    extra_out["prevalence_reference"] = {
        "death_proportion": target, "primary_death_proportion": float(y15.mean()),
        "factor": _prevalence_factor(p1, target, xs15, zs15)}
    print("  prevalence-matched factor: " + ", ".join(
        "%s %.2f" % (h, v) for h, v in extra_out["prevalence_reference"]["factor"].items()))
    covs17 = list(C.DEMOG_COVARIATES) + ["chronic", "poor_health"]
    rb17b, z17b = _build_adj(d17, covariates=covs17)
    rcmi_parts.append(_excess_table(d17, z17b, "SA17b adjusted for prevalent disease",
                                    n_null=N_NULL, boot=B_SA))
    hr_rows("SA17b adjusted for prevalent disease", d17,
            _headline_hr(d17, rb17b, fixed_level=fixed, parallel=True,
                         extra=("chronic", "poor_health")))
    # (d) the complement of SA17a: participants with prevalent disease or fair
    # or poor health; and the habit x illness interaction in everyone with
    # both items known (the Cox model of Table 3 with 'ill' and its product
    # with the habit added), which tests whether the hazard ratios differ
    # between the healthy and the ill
    d17d = d17[(d17.chronic == 1) | (d17.poor_health == 1)].reset_index(drop=True)
    rb17d, z17d = _build_adj(d17d)
    rcmi_parts.append(_excess_table(d17d, z17d, "SA17d with prevalent disease",
                                    n_null=N_NULL, boot=B_SA))
    hr_rows("SA17d with prevalent disease", d17d,
            _headline_hr(d17d, rb17d, fixed_level=fixed, parallel=True))
    d17["ill"] = ((d17.chronic == 1) | (d17.poor_health == 1)).astype(int)
    rb17 = C.deciles(C.crossfit_score(d17), 10)
    extra_out["sa17_interaction"] = {h: _interaction_hr(d17, h, rb17, "ill", fixed[h])
                                     for h in C.HABITS}
    # the same with illness as a proportional covariate (one baseline hazard),
    # an assumption the Grambsch-Therneau test rejects
    extra_out["sa17_interaction_proportional"] = {
        h: _interaction_hr(d17, h, rb17, "ill", fixed[h], stratified=False)
        for h in C.HABITS}
    # the same with each item of the (not prespecified) composite alone
    for item in ("chronic", "poor_health"):
        extra_out["sa17_interaction_" + item] = {
            h: _interaction_hr(d17, h, rb17, item, fixed[h]) for h in C.HABITS}
    for h, v in extra_out["sa17_interaction"].items():
        print("  %-6s %s: HR healthy %.2f, ill %.2f; ratio %.2f, P %.3f (joint P %.3f, %d df)"
              % (h, v["level_name"], v["hr0"], v["hr1"], v["ratio"], v["p_level"],
                 v["p_joint"], v["df"]), flush=True)

    # Cox models stratified on the primary adjustment strata, the habit the
    # only covariate: how much of the difference between the information
    # measure and the directly adjusted hazard ratio is the coarseness of the
    # stratification.
    print("SA18 Cox models stratified on the primary adjustment strata", flush=True)
    sa18 = {h: _stratified_hr(d, h, zmap[h], fixed[h]) for h in C.HABITS}
    hr_rows("SA18 stratified on the primary strata", d, sa18)

    # Physical activity from the leisure and transport domains only, with the
    # same cut-points: total activity includes work, and the most active level
    # is mostly occupational, so the primary contrast partly compares manual
    # workers with people not working (employment is not measured here).
    print("SA19 physical activity from leisure and transport only", flush=True)
    assert d["pa_lt"].notna().all()
    d19 = d.copy()
    d19["pa"] = d19["pa_lt"].astype(int)
    rb19, z19 = _build_adj(d19)
    rcmi_parts.append(_excess_table(d19, z19, "SA19 leisure and transport activity",
                                    n_null=N_NULL, boot=B_SA))
    hr_rows("SA19 leisure and transport activity", d19,
            _headline_hr(d19, rb19, fixed_level=fixed, parallel=True))
    d17l = d17.copy()
    d17l["pa"] = d17l["pa_lt"].astype(int)
    rb17l = C.deciles(C.crossfit_score(d17l), 10)
    extra_out["sa19_interaction"] = {h: _interaction_hr(d17l, h, rb17l, "ill", fixed[h])
                                     for h in C.HABITS}
    v_ = extra_out["sa19_interaction"]["pa"]
    print("  pa (leisure and transport) %s: HR healthy %.2f, ill %.2f; P %.3f (joint %.3f)"
          % (v_["level_name"], v_["hr0"], v_["hr1"], v_["p_level"], v_["p_joint"]),
          flush=True)
    # Coarser codings of every habit at once (the information measures are
    # invariant to relabelling levels but not to merging them): current
    # smoking or not; no drinking (lifetime abstainers and former drinkers),
    # up to 14 and more than 14 drinks/week; BMI <25, 25-29.9, >=30 kg/m2;
    # any physical activity or none; sleep <7, 7-<9, >=9 h; sitting <420 or
    # >=420 min/day.  The strata of the primary analysis are kept, so only the
    # codings change.  Information measures only.
    print("SA20 coarser codings of every habit, primary strata", flush=True)
    d20 = d.copy()
    for h_, mp in COARSE.items():
        d20[h_] = d20[h_].map(mp).astype(int)
    rcmi_parts.append(_excess_table(d20, zmap, "SA20 coarser codings", n_null=N_NULL,
                                    boot=B_SA))

    # Follow-up from the examination rather than the household interview:
    # BMI requires attending the examination, 0-3 months after the interview,
    # so with entry at the interview everyone has a short immortal interval.
    # Delayed entry at the examination (hazard ratios only).
    print("SA21 follow-up from the examination", flush=True)
    d21 = d.copy()
    lag = (d21["permth"] - d21["permth_exm"]).clip(lower=0) / 12.0
    d21["age"] = np.minimum(d21["age"] + lag, d21["age_exit"] - 0.5 / 12)
    hr_rows("SA21 follow-up from the examination", d21,
            _headline_hr(d21, rb, fixed_level=fixed, parallel=True))
    extra_out["sa21_lag_months"] = {
        str(int(k)): int(v) for k, v in
        (d["permth"] - d["permth_exm"]).value_counts().sort_index().items()}
    extra_out["sa19_levels"] = {
        "pct_active_no_leisure": float(100 * ((d.pa == 3) & (d.pa_lt == 0)).sum()
                                       / (d.pa == 3).sum()),
        "pct_by_level": {int(k): float(v) for k, v in
                         (100 * d19.pa.value_counts(normalize=True)).sort_index().items()}}

    json.dump(extra_out, open(os.path.join(C.OUT_DIR, "sensitivity_extra.json"), "w",
                              encoding="utf-8", newline=chr(10)), indent=1)
    rc = pd.concat(rcmi_parts, ignore_index=True)
    rc.to_csv(os.path.join(C.OUT_DIR, "sensitivity_rcmi.csv"), index=False)
    pd.DataFrame(surv_rows).to_csv(os.path.join(C.OUT_DIR, "sensitivity_hr.csv"),
                                   index=False)
    for val in ("calibrated", "excess"):
        piv = rc.pivot_table(index="label", columns="analysis", values=val)
        order = primary.sort_values(val, ascending=False).label.tolist()
        print("\n%s UNDER EVERY SPECIFICATION" % val.upper())
        print(piv.reindex(order).round(4).to_string())
        print(piv.reindex(order).rank(ascending=False, method="min").astype("Int64").to_string())
    print("\n%.0fs" % (time.time() - t0))


# =====================================================================
# 4.  Simulation against a known truth
# =====================================================================
DX_SIM, DY_SIM = 4, 2
B_VEC = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
DX_VEC = [3, 5, 5, 4, 4, 4]        # the alphabets of the six real habits
CC_VEC = [0.6, 3.0, 0.5, 3.0, 1.2, 2.0]
#: marginal distributions of the six real habits in the primary sample
#: (smoking, alcohol, BMI, physical activity, sleep, sedentary time), so that
#: the simulated exposures reproduce their rare levels (e.g. BMI <18.5)
MARGIN_VEC = [[0.555, 0.242, 0.203],
              [0.142, 0.186, 0.457, 0.167, 0.049],
              [0.015, 0.268, 0.329, 0.212, 0.176],
              [0.265, 0.135, 0.192, 0.408],
              [0.135, 0.209, 0.530, 0.126],
              [0.280, 0.350, 0.203, 0.167]]
GRID_K = [10, 20, 50, 100, 200, 1000, 2000]
GRID_N = [10000, 29371]


def _population(K, b, rng, dx=DX_SIM, conc=3.0, margin=None):
    """True p(x, y, z): Z uniform on K strata, X|Z a stratum-specific Dirichlet
    draw (centred on ``margin`` when given, with total concentration conc*dx),
    Y|X,Z Bernoulli with a known dose coefficient, so the population rCMI is
    exact."""
    p_z = np.full(K, 1.0 / K)
    alpha = rng.normal(-2.2, 0.8, K)
    a = (np.full(dx, conc) if margin is None
         else conc * dx * np.asarray(margin, float) / np.sum(margin))
    p_xz = rng.dirichlet(a, size=K).T
    s = np.linspace(0.0, 1.0, dx)
    p3 = np.zeros((dx, DY_SIM, K))
    for k in range(K):
        py1 = 1.0 / (1.0 + np.exp(-(alpha[k] + b * s)))
        p3[:, 1, k] = p_xz[:, k] * py1 * p_z[k]
        p3[:, 0, k] = p_xz[:, k] * (1 - py1) * p_z[k]
    return p3


def _draw(p3, n, rng):
    flat = p3.ravel()
    return rng.multinomial(n, flat / flat.sum()).astype(float).reshape(p3.shape)


def _spearman(a, b):
    ra = pd.Series(a).rank().values - pd.Series(a).rank().mean()
    rb = pd.Series(b).rank().values - pd.Series(b).rank().mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def _estimates(cnt, n_null, seed):
    """Plug-in rCMI and its two calibrations for a count table."""
    r = C.rcmi_of(cnt)
    nv = C.null_rcmi(C._CountTable(cnt), n_null=n_null, seed=seed)
    return (r, r - float(nv.mean()),
            float(np.sqrt(max(r * r - float(np.mean(nv ** 2)), 0.0))), float(nv.mean()))


def _sim_abs_one(task):
    """One cell of the absolute-recovery grid: REPS datasets of size n from a
    population with K strata and dose b; each carries its own seed."""
    K, bi, b, n, reps, n_null = task
    p3 = _population(K, b, np.random.default_rng([C.SEED, K, bi]))
    truth = C.rcmi_of(p3 * 1e9)
    rng = np.random.default_rng([C.SEED + 1, K, bi, n])
    plug, exc, cal, nullm = [], [], [], []
    for _ in range(reps):
        r, e, c, nu = _estimates(_draw(p3, n, rng), n_null, int(rng.integers(1e9)))
        plug.append(r)
        exc.append(e)
        cal.append(c)
        nullm.append(nu)
    plug, exc, cal = np.asarray(plug), np.asarray(exc), np.asarray(cal)

    def rel(v):
        return (v.mean() - truth) / truth if truth > 1e-6 else np.nan

    def rel_mcse(v):
        return v.std(ddof=1) / np.sqrt(len(v)) / truth if truth > 1e-6 else np.nan
    p_death = float(p3[:, 1, :].sum())
    return dict(K=K, n=n, b=b, truth=truth,
                mean_per_cell=n / (K * DX_SIM * DY_SIM),
                deaths_per_cell=n * p_death / (K * DX_SIM),
                plugin=plug.mean(), excess=exc.mean(), calibrated=cal.mean(),
                null_mean=float(np.mean(nullm)),
                null_share=float(np.mean(nullm)) / plug.mean(),
                plugin_rel_err=rel(plug), excess_rel_err=rel(exc),
                calibrated_rel_err=rel(cal),
                excess_rel_err_mcse=rel_mcse(exc), calibrated_rel_err_mcse=rel_mcse(cal),
                excess_sd=exc.std(ddof=1), calibrated_sd=cal.std(ddof=1),
                calibrated_mcse=cal.std(ddof=1) / np.sqrt(len(cal)))


def _sim_pops(K):
    return [_population(K, b, np.random.default_rng(C.SEED + 100 + j),
                        dx=DX_VEC[j], conc=CC_VEC[j], margin=MARGIN_VEC[j])
            for j, b in enumerate(B_VEC)]


def _sim_rank_one(task):
    """Rank recovery for one (K, n): REPS datasets of each of the six
    exposures."""
    K, n, reps, n_null = task
    pops = _sim_pops(K)
    truths = np.array([C.rcmi_of(p * 1e9) for p in pops])
    rng = np.random.default_rng([C.SEED + 77, K, n])
    res = {e: {k: [] for k in ("rho", "top1", "top2")}
           for e in ("plugin", "excess", "calibrated")}
    for _ in range(reps):
        est = {e: [] for e in res}
        for p3 in pops:
            r, e, c, _ = _estimates(_draw(p3, n, rng), n_null, int(rng.integers(1e9)))
            est["plugin"].append(r)
            est["excess"].append(e)
            est["calibrated"].append(c)
        for e, v in est.items():
            v = np.array(v)
            res[e]["rho"].append(_spearman(truths, v))
            res[e]["top1"].append(float(np.argmax(v) == np.argmax(truths)))
            res[e]["top2"].append(float(set(np.argsort(truths)[-2:])
                                        == set(np.argsort(v)[-2:])))
    p_death = float(np.mean([p[:, 1, :].sum() for p in pops]))
    row = dict(K=K, n=n, mean_per_cell=n / (K * DX_SIM * DY_SIM),
               deaths_per_cell=n * p_death / (K * DX_SIM))
    for e in res:
        for k in ("rho", "top1", "top2"):
            v = np.asarray(res[e][k], float)
            row["%s_%s" % (k, e)] = float(v.mean())
            row["%s_%s_mcse" % (k, e)] = float(v.std(ddof=1) / np.sqrt(len(v)))
    return row


def simulation():
    """Does subtracting the null floor RECOVER the truth?

    An empirical null floor cannot answer that; only data whose population
    rCMI is known can.  Two questions are asked: how close each estimator gets
    to the true value (including a weak effect, b = 0.25), and whether it
    recovers the known ordering of six exposures.  Three estimators: the
    plug-in; the square-root-scale excess; the D_JS-scale calibrated rCMI.
    Every grid cell carries its own seed, so the cells run in parallel, and
    every performance measure is reported with its Monte-Carlo standard error.
    """
    REPS = _env("SIM_REPS", 100, 4)
    N_NULL = _env("SIM_NULL", 100, 25)
    B_ABS = [0.0, 0.25, 0.5, 1.0]
    _save_params("simulation", replicates=REPS, null_draws=N_NULL, strata=GRID_K,
                 sample_sizes=GRID_N, dose=B_VEC, dose_absolute=B_ABS, levels=DX_VEC)
    t0 = time.time()
    tasks = [(K, bi, b, n, REPS, N_NULL) for K in GRID_K for bi, b in enumerate(B_ABS)
             for n in GRID_N]
    rows = _pmap(_sim_abs_one, tasks)
    ab = pd.DataFrame(rows)
    ab.to_csv(os.path.join(C.OUT_DIR, "simulation_absolute.csv"), index=False)
    print(ab[["K", "n", "b", "truth", "deaths_per_cell", "excess_rel_err",
              "calibrated_rel_err", "calibrated_rel_err_mcse"]].round(3).to_string(index=False))

    print("\nRANK RECOVERY (six exposures with known, distinct population values)")
    truth_rows = []
    for K in GRID_K:
        truths = np.array([C.rcmi_of(p * 1e9) for p in _sim_pops(K)])
        o = np.argsort(truths)[::-1]
        truth_rows.append(dict(
            K=K, **{"exposure_%d" % j: round(float(truths[j]), 5)
                    for j in range(len(B_VEC))},
            gap_1_2=round(float((truths[o[0]] - truths[o[1]]) / truths[o[0]]), 3),
            gap_2_3=round(float((truths[o[1]] - truths[o[2]]) / truths[o[1]]), 3)))
    rank_rows = _pmap(_sim_rank_one, [(K, n, REPS, N_NULL) for K in GRID_K for n in GRID_N])
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(os.path.join(C.OUT_DIR, "simulation_rank.csv"), index=False)
    pd.DataFrame(truth_rows).to_csv(os.path.join(C.OUT_DIR, "simulation_truths.csv"),
                                    index=False)
    print(rk[["K", "n", "top2_plugin", "top2_excess", "top2_calibrated",
              "rho_calibrated"]].round(3).to_string(index=False))
    print("\n%.0fs" % (time.time() - t0))


# =====================================================================
# 5.  Semi-parametric check on the real data
# =====================================================================
_KW = {}
SCENARIOS = ["additive", "interaction"]


def _interaction_design(d):
    """Design of the richer truth model: the demographic covariates plus, for
    each habit, its levels crossed with age band and with survey period (so
    that each habit's effect differs by age and by period).  The crossed codes
    contain the habit main effects; the blocks are collinear, which the ridge
    term of the fit resolves (the fitted probabilities are unique)."""
    f = d[C.DEMOG_COVARIATES].copy()
    cols = list(C.DEMOG_COVARIATES)
    for h in C.HABITS:
        f[h + "_x_age"] = d[h].astype(int) * 10 + d["age_b"].astype(int)
        f[h + "_x_per"] = d[h].astype(int) * 10 + d["cyc5"].astype(int)
        cols += [h + "_x_age", h + "_x_per"]
    return C.design(f, cols)[0]


def _cal_init(d, X, p_true, zs_pop, n_null, n_inner, chunk):
    _KW.update(d=d, X=X, p_true=p_true, zs_pop=zs_pop, n_null=n_null,
               n_inner=n_inner, chunk=chunk,
               xs={h: d[h].values.astype(int) for h in C.HABITS})


def _cal_dataset(scen, r):
    """Simulated dataset r of a scenario: n individuals drawn from the
    analytic sample, death drawn from that scenario's population model."""
    p = _KW["p_true"][scen]
    rng = np.random.default_rng([C.SEED + 9, SCENARIOS.index(scen), r])
    idx = rng.integers(0, p.size, p.size)
    return idx, (rng.random(p.size) < p[idx]).astype(int)


def _cal_strata(idx, y):
    """The primary adjustment sets re-estimated on a simulated dataset, exactly
    as on the real data (cross-fitted prognostic scores, tertiles within age
    band x period)."""
    dr = _KW["d"].iloc[idx].reset_index(drop=True).copy()
    dr["died"] = y
    return {h: np.unique(C.direct_set(dr, h), return_inverse=True)[1] for h in C.HABITS}


def _table(x, y, z):
    dx, dz = int(x.max() + 1), int(z.max() + 1)
    c = np.bincount((x * 2 + y) * dz + z, minlength=dx * 2 * dz)
    return c.astype(float).reshape(dx, 2, dz)


def _cal_one(task):
    """Either the estimates on simulated dataset r (``chunk`` < 0), with the
    strata re-estimated on the dataset and, for comparison, held at their
    population values; or one chunk of its bootstrap: smoothed (death
    re-simulated from the additive model refitted to the dataset, strata held
    fixed at the dataset's values, as in the primary analysis) and plain (the
    dataset's own deaths resampled).  Chunk 0 also returns the association of
    the dataset's bootstrap population."""
    scen, r, chunk = task
    idx, y = _cal_dataset(scen, r)
    zs = _cal_strata(idx, y)
    xs = {h: _KW["xs"][h][idx] for h in C.HABITS}
    rng = np.random.default_rng([C.SEED + 10, SCENARIOS.index(scen), r, chunk + 1])
    if chunk < 0:
        out = dict(scenario=scen, rep=r)
        for h in C.HABITS:
            pl, ex, ca, _ = _estimates(_table(xs[h], y, zs[h]), _KW["n_null"],
                                       int(rng.integers(1e9)))
            zp = np.unique(_KW["zs_pop"][h][idx], return_inverse=True)[1]
            fixed = _estimates(_table(xs[h], y, zp), _KW["n_null"],
                               int(rng.integers(1e9)))[2]
            # the refinement diagnostic on this dataset: every stratum split at
            # random in two (does the change it shows predict the known error?)
            rs = np.random.default_rng([C.SEED + 11, SCENARIOS.index(scen), r,
                                        C.HABITS.index(h)]).integers(0, 2, idx.size)
            rs2 = np.random.default_rng([C.SEED + 12, SCENARIOS.index(scen), r,
                                         C.HABITS.index(h)])
            split = _estimates(_table(xs[h], y, zs[h] * 2 + rs), _KW["n_null"],
                               int(rs2.integers(1e9)))[2]
            out.update({h + ":plugin": pl, h + ":excess": ex, h + ":calibrated": ca,
                        h + ":calibrated_fixed": fixed, h + ":calibrated_split": split})
        return out
    p_hat = _fit_p(_KW["X"][idx], y)
    rows = []
    for b in range(_KW["chunk"]):
        j = rng.integers(0, idx.size, idx.size)
        ys = (rng.random(idx.size) < p_hat[j]).astype(int)
        row = dict(scenario=scen, rep=r, b=chunk * _KW["chunk"] + b)
        for kind, yy in (("smooth", ys), ("plain", y[j])):
            for h in C.HABITS:
                row[h + ":" + kind] = _estimates(_table(xs[h][j], yy, zs[h][j]),
                                                 _KW["n_inner"], int(rng.integers(1e9)))[2]
        rows.append(row)
    world = None
    if chunk == 0:
        world = {h: C.rcmi_of(_expected_table(xs[h], zs[h], p_hat)) for h in C.HABITS}
    return dict(scenario=scen, rep=r, rows=rows, world=world)


def _ratio_mcse(ratio, se_vals, sd, n_datasets):
    """Monte-Carlo SE of mean(bootstrap SE) / SD(estimate) by the delta method."""
    se_vals = np.asarray(se_vals, float)
    rel_num = se_vals.std(ddof=1) / np.sqrt(se_vals.size) / se_vals.mean()
    rel_den = 1.0 / np.sqrt(2.0 * (n_datasets - 1))
    return float(ratio * np.sqrt(rel_num ** 2 + rel_den ** 2))


def calibration_check():
    """Does the calibration recover the truth, and is the bootstrap standard
    error honest, in data like ours?

    The population is the analytic sample itself (individuals with their six
    habits and covariates), with death drawn from one of two models fitted to
    the data: 'additive', the smoothing model of the primary bootstrap
    (additive logistic on the demographic covariates and the six habits), and
    'interaction', which lies outside that class (every habit's effect differs
    by age band and by survey period).  The target is each habit's rCMI under
    the primary adjustment set of the real data, computed exactly from the
    table of expected deaths.  Datasets of the analytic size are drawn from the
    population and analysed as the real data are: the prognostic scores and
    strata are re-estimated on each dataset.  For the first datasets of each
    scenario the smoothed bootstrap is run as in the primary analysis (additive
    smoothing model refitted, strata held at the dataset's values), together
    with a plain bootstrap; their standard errors are compared with the true
    sampling SD over datasets, for each habit and each paired difference.
    Every performance measure carries its Monte-Carlo standard error."""
    R = _env("CAL_REPS", 400, 6)
    R_BOOT = _env("CAL_BOOT_REPS", 200, 2)
    B = _env("CAL_B", 100, 20)
    CHUNK = _env("CAL_CHUNK", 50, 10)
    N_NULL = _env("CAL_NULL", 100, 20)
    N_INNER = _env("CAL_INNER", 50, 10)
    B = max(CHUNK, B // CHUNK * CHUNK)
    _save_params("calcheck", datasets=R, bootstrapped_datasets=R_BOOT,
                 bootstrap_resamples=B, null_draws=N_NULL, inner_null_draws=N_INNER,
                 scenarios=SCENARIOS)
    t0 = time.time()
    d = C.load(common=True)
    sets, _ = adjustment_sets(d)
    X = _outcome_design(d)
    y0 = d["died"].values
    p_true = {"additive": _fit_p(X, y0)}
    Xi = _interaction_design(d)
    bi, _ = C.irls(Xi, y0.astype(float), link="logit", ridge=1e-4)
    p_true["interaction"] = 1.0 / (1.0 + np.exp(-np.clip(Xi @ bi, -30, 30)))
    zs_pop = {h: np.unique(sets["PRIMARY"][h], return_inverse=True)[1] for h in C.HABITS}
    xs = {h: d[h].values.astype(int) for h in C.HABITS}
    truth = {(s, h): C.rcmi_of(_expected_table(xs[h], zs_pop[h], p_true[s]))
             for s in SCENARIOS for h in C.HABITS}
    init = (d, X, p_true, zs_pop, N_NULL, N_INNER, CHUNK)
    tasks = [(s, r, c) for s in SCENARIOS for r in range(R_BOOT) for c in range(B // CHUNK)]
    tasks += [(s, r, -1) for s in SCENARIOS for r in range(R)]
    res = _pmap(_cal_one, tasks, _cal_init, init)
    est = pd.DataFrame([o for o in res if "rows" not in o])
    boot = pd.DataFrame([row for o in res if "rows" in o for row in o["rows"]])
    world = {(o["scenario"], o["rep"]): o["world"] for o in res
             if "rows" in o and o["world"] is not None}

    rows, pairs, ranks = [], [], []
    for s in SCENARIOS:
        e_s, b_s = est[est.scenario == s].set_index("rep"), boot[boot.scenario == s]

        def boot_sd(kind, cols):
            v = sum(sg * b_s[h + ":" + kind] for h, sg in cols)
            return v.groupby(b_s.rep).std(ddof=1)

        for h in C.HABITS:
            th = truth[(s, h)]
            cal = e_s[h + ":calibrated"]
            sd = float(cal.std(ddof=1))
            s_sm, s_pl = boot_sd("smooth", [(h, 1)]), boot_sd("plain", [(h, 1)])
            bm = b_s.groupby("rep")[h + ":smooth"].mean()
            wv = pd.Series({r: world[(s, r)][h] for r in s_sm.index})
            cover = ((cal.loc[s_sm.index] - th).abs() <= 1.96 * s_sm)
            rsm, rpl = float(s_sm.mean() / sd), float(s_pl.mean() / sd)
            rows.append(dict(
                scenario=s, habit=h, label=C.LABEL[h], truth=th,
                plugin=e_s[h + ":plugin"].mean(), excess=e_s[h + ":excess"].mean(),
                calibrated=cal.mean(), calibrated_fixed=e_s[h + ":calibrated_fixed"].mean(),
                excess_rel_err=(e_s[h + ":excess"].mean() - th) / th,
                calibrated_rel_err=(cal.mean() - th) / th,
                calibrated_rel_err_mcse=sd / np.sqrt(len(cal)) / th,
                calibrated_fixed_rel_err=(e_s[h + ":calibrated_fixed"].mean() - th) / th,
                split_change_rel=float((e_s[h + ":calibrated_split"] - cal).mean() / th),
                split_change_rel_mcse=float((e_s[h + ":calibrated_split"] - cal).std(ddof=1)
                                            / np.sqrt(len(cal)) / th),
                calibrated_sd=sd,
                smooth_se=float(s_sm.mean()), smooth_se_ratio=rsm,
                smooth_se_ratio_mcse=_ratio_mcse(rsm, s_sm, sd, len(cal)),
                plain_se=float(s_pl.mean()), plain_se_ratio=rpl,
                plain_se_ratio_mcse=_ratio_mcse(rpl, s_pl, sd, len(cal)),
                smooth_shift_in_sds=float(((bm - wv) / s_sm).mean()),
                wald_coverage=float(cover.mean()),
                wald_coverage_mcse=float(np.sqrt(cover.mean() * (1 - cover.mean())
                                                 / len(cover))),
                plugin_minus_calibrated=float((e_s[h + ":plugin"] - cal).mean()),
                reps=len(cal), boot_reps=int(s_sm.size)))
            print("  %-11s %-18s truth=%.4f cal=%+.0f%% (fixed strata %+.0f%%) excess=%+.0f%% "
                  "SE ratio smooth=%.2f plain=%.2f cover=%.2f"
                  % (s, C.LABEL[h], th, 100 * rows[-1]["calibrated_rel_err"],
                     100 * rows[-1]["calibrated_fixed_rel_err"],
                     100 * rows[-1]["excess_rel_err"], rsm, rpl, cover.mean()), flush=True)
        for i, a in enumerate(C.HABITS):
            for bb in C.HABITS[i + 1:]:
                diff = e_s[a + ":calibrated"] - e_s[bb + ":calibrated"]
                sd = float(diff.std(ddof=1))
                s_sm = boot_sd("smooth", [(a, 1), (bb, -1)])
                s_pl = boot_sd("plain", [(a, 1), (bb, -1)])
                tdiff = truth[(s, a)] - truth[(s, bb)]
                cover = ((diff.loc[s_sm.index] - tdiff).abs() <= 1.96 * s_sm)
                rsm, rpl = float(s_sm.mean() / sd), float(s_pl.mean() / sd)
                pairs.append(dict(
                    scenario=s, habit_a=a, habit_b=bb, truth=tdiff,
                    mean=float(diff.mean()), sd=sd,
                    smooth_se_ratio=rsm, smooth_se_ratio_mcse=_ratio_mcse(rsm, s_sm, sd, len(diff)),
                    plain_se_ratio=rpl, plain_se_ratio_mcse=_ratio_mcse(rpl, s_pl, sd, len(diff)),
                    wald_coverage=float(cover.mean()),
                    wald_coverage_mcse=float(np.sqrt(cover.mean() * (1 - cover.mean())
                                                     / len(cover)))))
        t = pd.Series({h: truth[(s, h)] for h in C.HABITS})
        lead, pair = t.idxmax(), set(t.sort_values(ascending=False).index[:2])
        for e in ("plugin", "excess", "calibrated"):
            v = e_s[[h + ":" + e for h in C.HABITS]].set_axis(C.HABITS, axis=1)
            top1 = (v.idxmax(axis=1) == lead).astype(float)
            top2 = v.apply(lambda r: set(r.sort_values(ascending=False).index[:2]) == pair,
                           axis=1).astype(float)
            rho = np.array([_spearman(r.values, t.values) for _, r in v.iterrows()])
            ranks.append(dict(
                scenario=s, estimator=e, top1=float(top1.mean()),
                top1_mcse=float(np.sqrt(top1.mean() * (1 - top1.mean()) / len(top1))),
                top2=float(top2.mean()),
                top2_mcse=float(np.sqrt(top2.mean() * (1 - top2.mean()) / len(top2))),
                rho=float(rho.mean()), rho_mcse=float(rho.std(ddof=1) / np.sqrt(rho.size)),
                true_order=" > ".join(t.sort_values(ascending=False).index),
                order_of_means=" > ".join(v.mean().sort_values(ascending=False).index)))
    pd.DataFrame(rows).to_csv(os.path.join(C.OUT_DIR, "calibration_check.csv"), index=False)
    pd.DataFrame(pairs).to_csv(os.path.join(C.OUT_DIR, "calibration_check_pairs.csv"),
                               index=False)
    rk = pd.DataFrame(ranks)
    rk.to_csv(os.path.join(C.OUT_DIR, "calibration_check_rank.csv"), index=False)
    pp = pd.DataFrame(pairs)
    for s in SCENARIOS:
        q = pp[pp.scenario == s]
        print("  %s paired differences: SE ratio smooth %.2f-%.2f, plain %.2f-%.2f, "
              "coverage %.2f-%.2f" % (s, q.smooth_se_ratio.min(), q.smooth_se_ratio.max(),
                                      q.plain_se_ratio.min(), q.plain_se_ratio.max(),
                                      q.wald_coverage.min(), q.wald_coverage.max()))
    print(rk[["scenario", "estimator", "top1", "top2", "rho", "true_order"]]
          .round(3).to_string(index=False))
    print("\n%.0fs" % (time.time() - t0))


# =====================================================================
STEPS = {"measures": measures, "survival": survival, "ph": ph_check,
         "sensitivity": sensitivity, "simulation": simulation,
         "calcheck": calibration_check}

if __name__ == "__main__":
    opts = [a for a in sys.argv[1:] if a.startswith("-")]
    if "-h" in opts or "--help" in opts:
        print(__doc__)
        sys.exit(0)
    bad_opts = sorted(set(opts) - {"--quick"})
    if bad_opts:
        raise SystemExit("unknown option(s): %s; the only option is --quick"
                         % ", ".join(bad_opts))
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__)
        raise SystemExit("no step given; use one of: %s" % ", ".join(["all"] + list(STEPS)))
    unknown = sorted(set(args) - set(STEPS) - {"all"})
    if unknown:
        raise SystemExit("unknown step(s): %s; use %s"
                         % (", ".join(unknown), ", ".join(["all"] + list(STEPS))))
    if "all" in args and len(args) > 1:
        raise SystemExit("'all' cannot be combined with other steps")
    todo = list(STEPS) if (not args or args[0] == "all") else args
    for name in todo:
        print("\n" + "=" * 72)
        print(name.upper())
        print("=" * 72, flush=True)
        STEPS[name]()
