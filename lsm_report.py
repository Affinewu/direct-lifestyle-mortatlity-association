# -*- coding: utf-8 -*-
"""
lsm_report.py -- tables, figures, typesetting and number checking.

    python lsm_report.py all        tables, figures, fill, render, arxiv, verify
    python lsm_report.py tables     main and supplementary tables + numbers.json
    python lsm_report.py figures    Figures 1-3 and S1-S3 at journal resolution
    python lsm_report.py fill       the documents from their templates + numbers.json
    python lsm_report.py render     Markdown -> LaTeX -> PDF for every document
    python lsm_report.py arxiv      the single-file preprint
    python lsm_report.py verify     the documents against the analysis
    python lsm_report.py compare    output/ against reference_output/

The Markdown files in submission/ are the single source of record.  The PDFs are
generated from them, so the two cannot drift -- the failure mode that put three
different values of one statistic into an earlier draft.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

import lsm_core as C

OUT, SUB, FIG = C.OUT_DIR, C.SUB_DIR, C.FIG_DIR
FMT = "%.4f"
MINUS = "−"


# =====================================================================
# 1.  Tables and numbers.json
# =====================================================================
def _num(v, fmt="%.1f"):
    """Format a number with a true minus sign and without a negative zero."""
    s = fmt % v
    if re.fullmatch(r"-0(\.0+)?", s):
        s = s[1:]
    return s.replace("-", MINUS)


def _pct(x, n):
    return "%s (%.1f)" % (_int(x), 100.0 * x / n)


def _int(v):
    """Integer with a thin-space thousands separator (rendered as \\, )."""
    return format(int(v), ",").replace(",", " ")


def _thin(t):
    """Thin-space thousands separator in the integer part of a formatted number."""
    head, dot, tail = t.partition(".")
    return _int(head) + dot + tail if head.lstrip("-").isdigit() else t


def _half_up(v):
    """Integer rounding half away from zero (Python's round() rounds half to
    even, which prints a median of 56.5 as 56)."""
    import decimal
    return str(decimal.Decimal(str(float(v))).quantize(decimal.Decimal("1"),
                                                       rounding=decimal.ROUND_HALF_UP))


def _pv(v):
    """A P value to three decimals, or <0.001."""
    return "" if pd.isna(v) else ("<0.001" if v < 0.001 else "%.3f" % v)


def _pv2(v):
    """A P value: three decimals, two significant figures below 0.01 (so that
    0.0017 can be compared with the Bonferroni threshold 0.0033) and one below
    0.001 (0.0007, 0.00003), in decimal notation."""
    if pd.isna(v):
        return ""
    v = float(v)
    if v < 0.001:
        return "%.*f" % (int(-np.floor(np.log10(v))), v) if v > 0 else "0"
    return "%#.2g" % v if v < 0.01 else "%.3f" % v


def _num_signed(v, fmt="%.4f"):
    """As _num, but a non-zero value that rounds to zero is printed to one
    significant figure with its sign (−0.00002), so that an interval shown as
    touching zero from below is not read as excluding it; an exact zero has no
    sign."""
    s = fmt % v
    if float(s) == 0.0:
        if v == 0:
            return fmt % 0.0
        d = int(-np.floor(np.log10(abs(v))))
        s = "%.*f" % (d, v)
    return s.replace("-", MINUS)


def _p(v, floor):
    """A Monte-Carlo or bootstrap P value: '≤floor' at the resolution of the
    add-one estimate (no draw beyond the observed value)."""
    return "≤%s" % floor if v <= float(floor) + 1e-12 else "%.3f" % v


def table1(d):
    rows, groups = [], [("Overall", d), ("Died", d[d.died == 1]),
                        ("Alive at censoring", d[d.died == 0])]

    def add(label, fn):
        rows.append(dict(Characteristic=label, **{g: fn(s) for g, s in groups}))

    add("*n*", lambda s: _int(len(s)))
    add("Age at screening, mean (SD), years†",
        lambda s: "%.1f (%.1f)" % (s.age.mean(), s.age.std()))
    add("Follow-up, median (IQR), months",
        lambda s: "%s (%s–%s)" % (_half_up(s.permth.median()),
                                  _half_up(s.permth.quantile(.25)),
                                  _half_up(s.permth.quantile(.75))))
    add("Person-years", lambda s: _int(round(s.py.sum())))
    add("Female, *n* (%)", lambda s: _pct(int((s.sex_b == 1).sum()), len(s)))
    for code, lab in enumerate(["non-Hispanic White", "non-Hispanic Black",
                                "Mexican American", "other Hispanic",
                                "other or multiracial"]):
        add("Race/ethnicity: %s, *n* (%%)" % lab,
            lambda s, c=code: _pct(int((s.race5 == c).sum()), len(s)))
    for code, lab in enumerate(["less than high school", "high school or some college",
                                "college graduate"]):
        add("Education: %s, *n* (%%)" % lab,
            lambda s, c=code: _pct(int((s.educ == c).sum()), len(s)))
    for code, lab in enumerate(["<1.3", "1.3–<3.5", "≥3.5", "not reported"]):
        add("Income-to-poverty ratio: %s, *n* (%%)" % lab,
            lambda s, c=code: _pct(int((s.pir == c).sum()), len(s)))
    for code, lab in enumerate(["married or living with a partner",
                                "widowed, divorced or separated", "never married"]):
        add("Marital status: %s, *n* (%%)" % lab,
            lambda s, c=code: _pct(int((s.marital == c).sum()), len(s)))
    for code, lab in zip((2, 3, 4), ("2007–2010", "2011–2014", "2015–2018")):
        add("Survey period: %s, *n* (%%)" % lab,
            lambda s, c=code: _pct(int((s.cyc5 == c).sum()), len(s)))
    for h in C.HABITS:
        for lv in sorted(pd.unique(d[h]).tolist()):
            add("%s: %s, *n* (%%)" % (_lab(h), C.LEVELS[h][int(lv)]),
                lambda s, hh=h, l=lv: _pct(int((s[hh] == l).sum()), len(s)))
    return pd.DataFrame(rows)


def table2(m):
    tot = m[m.spec == "total"].set_index("habit")
    dem = m[m.spec == "DEMO"].set_index("habit")
    pri = m[m.spec == "PRIMARY"].set_index("habit")
    order = pri.sort_values("calibrated", ascending=False).index.tolist()
    rt = pd.read_csv(os.path.join(OUT, "ratios.csv")).set_index(["habit", "scale"])
    rows = []
    for h in order:
        def ratio(sc):
            r = rt.loc[(h, sc)]
            return "%.2f (%s to %s)" % (r.ratio, _num(r.lo, "%.2f"), _num(r.hi, "%.2f"))
        rows.append({
            "Habit": C.LABEL[h], "Levels": int(pri.loc[h, "levels"]),
            "Total (SE)": "%s (%s)" % (FMT % tot.loc[h, "calibrated"],
                                       FMT % tot.loc[h, "calibrated_se"]),
            "Direct, demographics (SE)": "%s (%s)" % (FMT % dem.loc[h, "calibrated"],
                                                      FMT % dem.loc[h, "calibrated_se"]),
            "Direct, demographics and co-habits (SE)": "%s (%s)" % (
                FMT % pri.loc[h, "calibrated"], FMT % pri.loc[h, "calibrated_se"]),
            "Direct/total, $\\sqrt{D_{\\mathrm{JS}}}$ scale (95% CI)": ratio("sqrt"),
            "Direct/total, $D_{\\mathrm{JS}}$ scale (95% CI)": ratio("djs")})
    return pd.DataFrame(rows)


def _hr_ci(hr, lo, hi):
    """HR (95% CI) to two decimals; a limit that would round to 1.00 from the
    other side of 1 gets a third decimal, so the table never hides whether the
    interval includes 1."""
    def f(v):
        return "%.3f" % v if "%.2f" % v == "1.00" and v != 1 else "%.2f" % v
    return "%.2f (%s–%s)" % (hr, f(lo), f(hi))


def table3(hr, rk):
    m = hr.merge(rk[["habit", "level", "risk10", "rd10", "rd10_lo", "rd10_hi"]],
                 on=["habit", "level"])
    rows = []
    for _, r in m.iterrows():
        rows.append({
            "Habit": r.label + UNIT.get(r.habit, ""), "Level": r.level_name,
            "*n*": _int(r.n_level), "Deaths": _int(r.deaths_level),
            "Rate per 1000 person-years": "%.1f" % (1000 * r.deaths_level / r.py_level),
            "Hazard ratio (95% CI)": "1.00 (reference)" if r.reference else
            _hr_ci(r.hr, r.hr_lo, r.hr_hi),
            "10-year risk, %": "%.1f" % r.risk10,
            "Risk difference, percentage points (95% CI)": "0 (reference)"
            if r.reference else
            "%s (%s to %s)" % (_num(r.rd10), _num(r.rd10_lo), _num(r.rd10_hi))})
    return pd.DataFrame(rows)


def _nonsig(m):
    """', except <habit>, P = ...' for every primary estimate not beyond its
    null at the resolution of the Monte-Carlo P, or ''."""
    pr = m[m.spec == "PRIMARY"]
    out = ["%s, *P* = %.3f" % (C.LABEL[h].lower(), v)
           for h, v in zip(pr.habit, pr.p_null) if v > 0.0021]
    return (" except " + "; ".join(out)) if out else ""


#: the unit of the BMI levels, carried by the habit's label in tables and
#: figures (the other habits' levels carry their own units)
UNIT = {"bmi": " (kg/m²)"}


def _lab(h):
    """A habit's label with the unit of its levels where they have none."""
    return C.LABEL[h] + UNIT.get(h, "")


def _grouped(t, group="Habit", item="Level", label="Habit and level"):
    """A table with the group printed once, as a bold header row above its
    items (as in Table 1), instead of on every row."""
    rest = [c for c in t.columns if c not in (group, item)]
    rows = []
    for g, blk in t.groupby(group, sort=False):
        rows.append({label: "**%s**" % g, **{c: "" for c in rest}})
        for _, r_ in blk.iterrows():
            rows.append({label: r_[item], **{c: r_[c] for c in rest}})
    return pd.DataFrame(rows)


def _md(df, floatfmt="%.4f"):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else _num(v, floatfmt))
    cols = list(d.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c])
                                     for c in cols) + " |")
    return "\n".join(out)


def tables():
    d = C.load(common=True)
    m = pd.read_csv(os.path.join(OUT, "measures.csv"))
    hr = pd.read_csv(os.path.join(OUT, "cox.csv"))
    rk = pd.read_csv(os.path.join(OUT, "standardised_risk.csv"))
    diff = pd.read_csv(os.path.join(OUT, "pairwise.csv"))

    t1, t2, t3 = table1(d), table2(m), table3(hr, rk)
    for name, t in (("table1_cohort", t1), ("table2_measures", t2),
                    ("table3_doseresponse", t3)):
        t.to_csv(os.path.join(OUT, name + ".csv"), index=False)
    # as printed: the number of levels after the habit; Table 3 grouped
    t2d = t2.assign(Habit=["%s (%d)" % (a, b) for a, b in zip(t2.Habit, t2.Levels)]
                    ).drop(columns=["Levels"])
    t3d = _grouped(t3)
    n_top = int((d.age == 80).sum())

    # ---------------- main tables document
    parts = ["# Tables", "",
             "## Table 1. Characteristics of the analytic cohort", "",
             "NHANES 2007–2018 participants aged ≥20 years with complete "
             "data on all six habits and all covariates (a missing income-to-poverty "
             "ratio is kept as its own category), by vital status at 31 December "
             "2019. †Age is top-coded at 80 years in the public-use files; "
             "%s participants (%.1f%%) are recorded as 80."
             % (_int(n_top), 100.0 * n_top / len(d)), "", _md(t1), "",
             "<!-- PAGEBREAK -->", "",
             "## Table 2. Total and direct association between each habit and "
             "all-cause mortality", "",
             "Calibrated estimates $\\widetilde{\\mathrm{rCMI}} = "
             "[\\max(\\widehat{\\mathrm{rCMI}}{}^2 - "
             "\\mathbb{E}_0[\\widehat{\\mathrm{rCMI}}{}^2],\\,0)]^{1/2}$ (likewise for "
             "rMI), where $\\widehat{\\mathrm{rCMI}}$ is the plug-in estimate from "
             "the observed table and $\\mathbb{E}_0[\\cdot]$ the mean over 500 tables "
             "simulated under conditional independence of habit and death within "
             "strata, so that $\\mathbb{E}_0[\\widehat{\\mathrm{rCMI}}]$ is the null "
             "floor; on the $\\sqrt{D_{\\mathrm{JS}}}$ scale, "
             "bounded in [0, 1], whose ceiling for a binary outcome is 0.558 "
             "(Supplementary Methods S2–S3). The number of levels of each habit is "
             "in parentheses. *Total*: no adjustment (the calibrated rMI). *Direct, "
             "demographics*: conditional on 54 strata, the six age bands × three "
             "survey periods, each split into tertiles of a cross-fitted prognostic "
             "score for death built from the demographic covariates. *Direct, "
             "demographics and co-habits*: the same, with a prognostic score that also "
             "contains the other five habits (the primary analysis). Standard errors "
             "from 1000 smoothed bootstrap resamples (Supplementary Methods S3.2); "
             "comparisons between habits are in Supplementary Table S2. Intervals of "
             "±1.96 SE are not tests against zero, because the calibrated estimator "
             "is positive under conditional independence; the test is the "
             "Monte-Carlo *P* of Supplementary Table S1 (≤0.002 for every habit"
             + _nonsig(m) + "). *Direct/total*: "
             "the ratio of the direct to the total association on each scale; the "
             "interval is the bootstrap distribution of the ratio, divided by its "
             "value in the bootstrap population and multiplied by the estimate, so "
             "it cannot be negative and the $D_{\\mathrm{JS}}$-scale limits are the "
             "squares of the $\\sqrt{D_{\\mathrm{JS}}}$-scale limits (a "
             "percentile-type interval, oriented as the ranking probabilities of "
             "Supplementary Table S2d; the bootstrap *P* of Supplementary Table S2 "
             "is instead the dual of the basic interval). The "
             "$\\sqrt{D_{\\mathrm{JS}}}$-scale ratio is a ratio of distances and the "
             "$D_{\\mathrm{JS}}$-scale ratio its square; neither is a share of an "
             "additive decomposition, and either can exceed 1 when a within-stratum "
             "association is masked in the margin.",
             "", _md(t2d), "",
             "<!-- PAGEBREAK -->", "",
             "## Table 3. Fully adjusted dose–response", "",
             "Cox proportional-hazards models with attained age as the time scale "
             "(entry at age at the screening interview), adjusted for sex, "
             "race/ethnicity (five groups), education, income-to-poverty ratio, "
             "marital status, "
             "four-year survey period, prognostic-score decile and all five other "
             "habits. Hazard-ratio intervals are Wald intervals. Standardised "
             "10-year risks are the probability of death within 10 years of the "
             "baseline age, from the same model with the Breslow baseline hazard, "
             "averaged over the cohort's baseline ages and covariates (g-formula); "
             "risk-difference intervals are percentile intervals from %d bootstrap "
             "resamples with the Cox model refitted in each (prognostic-score "
             "deciles held fixed). Risk differences are computed from unrounded "
             "risks, so they can differ in the last digit from the difference of "
             "the rounded risks shown. Total person-time %s person-years."
             % (_design("survival", "survival_bootstrap_resamples", 400),
                _int(round(d.py.sum()))), "", _md(t3d), ""]
    os.makedirs(SUB, exist_ok=True)
    open(os.path.join(SUB, "tables.md"), "w", encoding="utf-8",
         newline="\n").write("\n".join(parts))

    _numbers(d, m, hr, rk, diff)
    _supplementary_tables(m, diff, d)
    _assemble_supplement()
    print("tables written (T1 %d rows, T2 %d, T3 %d)" % (len(t1), len(t2), len(t3)))


def _assemble_supplement():
    """Concatenate the supplementary methods, tables and figure legends into the
    single file that gets uploaded."""
    src = os.path.join(SUB, "supplementary_methods.md")
    if not os.path.exists(src):
        print("  supplementary_methods.md not found: supplement not assembled")
        return
    chunks = [open(src, encoding="utf-8").read()]
    p = os.path.join(SUB, "supplementary_tables.md")
    if os.path.exists(p):
        body = open(p, encoding="utf-8").read()
        body = body.split("\n", 1)[1] if body.startswith("# ") else body
        chunks.append("\n<!-- PAGEBREAK -->\n\n# Supplementary Tables\n" + body)
    chunks.append("""
<!-- PAGEBREAK -->

# Supplementary Figures

**Supplementary Figure S1.** Participant flow, from the ten pooled NHANES cycles
to the primary analytic sample. Apart from the restriction to adults, the largest
exclusion is of the four cycles before 2007–2008, in which the physical-activity
and sitting questions were not asked (the sleep questionnaire began in
2005–2006). Missing data among the eligible adults of 2007–2018 are given in
Supplementary Table S9.

<!-- FIG figS1_flow.pdf | **Supplementary Figure S1.** Participant flow. -->

**Supplementary Figure S2.** Directed acyclic graph for the analysis. *X*, the
habit of interest; *Y*, death; *D*, demographic covariates and survey period;
*C*, the other five habits; $X_0$, $C_0$, the same habits earlier in life;
*L* (dashed, unmeasured), a disposition to healthy
or unhealthy behaviour that causes both *X* and *C*, which makes *C* a
confounder of *X*; *M*, prevalent disease and self-rated health, measured but
used only in sensitivity analysis SA17: a consequence of earlier habits and a
cause of the current habits and of death (reverse causation), so that excluding
participants with it (SA17a) and adjusting for it (SA17b) can each bias the
habit–mortality association, in either direction; *U* (dashed), other
unmeasured common causes such as diet and
health-care access, which no observational adjustment can remove. $X_0$ and
$C_0$ are unmeasured except through the former-smoker and former-drinker
categories, so confounding by earlier habits ($X \\leftarrow X_0 \\rightarrow Y$,
$X \\leftarrow X_0 \\rightarrow M \\rightarrow Y$) is not removed. Dashed boxes,
unmeasured or largely unmeasured variables; dashed grey arrows, paths from the
wholly unmeasured *L* and *U*; arrows from the partly measured $X_0$, $C_0$ are
solid. Arrows from *D*, *L* and *U* into the earlier habits are omitted for
clarity. The primary analysis conditions on *D* and *C* through the stratum *Z*
(age band, survey period and a prognostic score): this removes confounding
through *D* and through
$L \\rightarrow C$, but also blocks $X \\rightarrow C \\rightarrow Y$, so the quantity estimated is a
direct association and not a total causal effect; and because *C* is also
affected by *U* and *M*, conditioning on it can open non-causal paths such as
$X \\rightarrow C \\leftarrow U \\rightarrow Y$ (collider stratification). The conditioning is a
coarsening into 54 strata (Supplementary Methods S1.6), so every adjustment is
partial.

<!-- FIG figS2_dag.pdf | **Supplementary Figure S2.** Directed acyclic graph. -->

**Supplementary Figure S3.** Direct association (calibrated rCMI) for each
habit in the primary analysis (highlighted, and marked by a dotted vertical
line) and in every sensitivity analysis
that re-estimates it (Supplementary Table S3), with ±1.96 smoothed-bootstrap
standard errors where available (truncated at zero); open markers, estimates
truncated at zero. The hazard-ratio analyses SA2, SA4a/b, SA7, SA18 and SA21 are in
Supplementary Table S4.

<!-- FIG figS3_sensitivity.pdf | **Supplementary Figure S3.** Direct association across the sensitivity analyses. -->
""")
    open(os.path.join(SUB, "supplementary.md"), "w",
         encoding="utf-8", newline="\n").write("\n".join(chunks) + "\n")


SPEC_NAME = {"total": "none", "AGE": "age band (6)",
             "DEMO": "demographics and period (54)",
             "PRIMARY": "primary: + co-habits (54)", "RAW": "exact strata"}


def _sect(n, title, body, note=""):
    s = ["", "## Supplementary Table S%s. %s" % (n, title), ""]
    if note:
        s += [note, ""]
    return "\n".join(s + [body, ""])


def _title():
    """The paper's title, from the H1 of the manuscript (so that no other file
    can carry a stale one)."""
    p = os.path.join(SUB, "manuscript.md")
    if os.path.exists(p):
        m = re.match(r"#\s+(.*)", open(p, encoding="utf-8").read())
        if m:
            return m.group(1).strip()
    return "How much of a lifestyle–mortality association is direct?"


def _r2(v, nd=2):
    """Round half up at ``nd`` decimals (float formatting rounds 0.975 down)."""
    import decimal
    q = decimal.Decimal(1).scaleb(-nd)
    return str(decimal.Decimal(repr(float(v))).quantize(q, rounding=decimal.ROUND_HALF_UP))


def _pct_hu(v):
    """A proportion as a whole percentage, rounded half up."""
    return int(_r2(100.0 * float(v), 0))


HATS = {"plug": "$\\widehat{\\mathrm{rCMI}}$",
        "floor": "$\\mathbb{E}_0[\\widehat{\\mathrm{rCMI}}]$",
        "cal": "$\\widetilde{\\mathrm{rCMI}} = [\\max(\\widehat{\\mathrm{rCMI}}{}^2 - "
               "\\mathbb{E}_0[\\widehat{\\mathrm{rCMI}}{}^2],\\,0)]^{1/2}$",
        "exc": "$\\widehat{\\mathrm{rCMI}} - \\mathbb{E}_0[\\widehat{\\mathrm{rCMI}}]$"}

SA_CODE = re.compile(r"^(SA\d+[a-d]?) .*")


def _sa_name(c):
    """An analysis name as displayed: year ranges with an en dash, and the
    weighting analyses named as in the text."""
    c = re.sub(r"(\d)-(\d)", r"\1–\2", c)
    for a_, b_ in (("free of prevalent disease", "healthy (no prevalent disease, "
                    "self-rated health good or better)"),
                   ("adjusted for prevalent disease", "adjusted for illness (prevalent "
                    "disease, self-rated health)"),
                   ("with prevalent disease", "ill (prevalent disease or fair or poor "
                    "self-rated health)")):
        c = c.replace(a_, b_)
    return c.replace("MEC-weighted", "survey-weighted (MEC weight)")


def _sa_code(c):
    return "Primary" if c == "primary" else SA_CODE.sub(r"\1", c)


def _sa_key(c):
    if c == "primary":
        return (0, "")
    return (int(re.match(r"SA(\d+)", c).group(1)), c)


def _supplementary_tables(m, diff, d):
    parts = ["# Supplementary Tables", "", "*%s*" % _title(), ""]
    order = (m[m.spec == "PRIMARY"].sort_values("calibrated", ascending=False)
             .habit.tolist())
    rank_of = {h: i for i, h in enumerate(order)}
    spec_order = {s: i for i, s in enumerate(["total", "AGE", "DEMO", "PRIMARY", "RAW"])}
    mm = m.copy()
    mm["_h"] = mm.habit.map(rank_of)
    mm["_s"] = mm.spec.map(spec_order)
    mm = mm.sort_values(["_h", "_s"])
    pri = mm[mm.spec == "PRIMARY"].set_index("habit")

    def se(col):
        return ["" if pd.isna(v) else FMT % v for v in mm[col]]

    raw_ = (mm.spec == "RAW").values
    bperm = _extra().get("sa16_reference", {}).get("unweighted_permutation", {})
    t = pd.DataFrame({
        "Habit": mm.label,
        "Adjustment": [SPEC_NAME[s_] + ("†" if s_ == "RAW" else "") for s_ in mm.spec],
        "Strata": [_int(v) for v in mm.n_strata],
        "Plug-in": [FMT % v for v in mm.rcmi],
        "Null floor": [FMT % v for v in mm.null_mean],
        "Calibrated": ["" if x_ else FMT % v for v, x_ in zip(mm.calibrated, raw_)],
        "SE (calibrated)": se("calibrated_se"),
        "Excess": ["" if x_ else FMT % v for v, x_ in zip(mm.excess, raw_)],
        "SE (excess)": se("excess_se"),
        "*P* (null)": ["" if x_ else _p(v, "0.002") for v, x_ in zip(mm.p_null, raw_)],
        "*P* (permutation null)": [
            _p(bperm[h]["p_null"], "0.002") if s_ == "PRIMARY" and h in bperm else ""
            for h, s_ in zip(mm.habit, mm.spec)],
        "Floor share": ["%.2f" % v for v in mm.null_share],
        "Deaths per cell: mean; median; % <2; % <5": [
            "" if s_ == "total" else "%.1f; %s; %d; %d" % (a, _half_up(b), _pct_hu(c),
                                                        _pct_hu(e))
            for s_, a, b, c, e in zip(mm.spec, mm.deaths_per_cell, mm.cell_deaths_median,
                                      mm.cells_lt2, mm.cells_lt5)]})
    parts.append("<!-- LANDSCAPE on -->")
    parts.append(_sect(
        1, "Every regularised measure, by exposure and adjustment set", _md(t),
        "All estimates on the primary analytic sample (*n* = %s). *Plug-in*: %s "
        "(for *none*, $\\widehat{\\mathrm{rMI}}$). *Null floor*: %s, the mean of 500 "
        "datasets generated under conditional independence. *Calibrated*: %s. "
        "*Excess*: %s. *SE*: bootstrap standard error (1000 smoothed paired "
        "resamples, null floor refitted in each with 100 draws). *P* (null): "
        "Monte-Carlo upper-tail *P* against the null, resolution 0.002; on datasets "
        "generated under conditional independence given the primary strata it was "
        "at most 0.05 in %s%% of them (Supplementary Methods S3.1). *P* "
        "(permutation null): against the within-stratum permutation null of "
        "SA15–SA16, without weights (primary set; Supplementary Table S3, 'SA16, no "
        "weights'). †Exact strata, with fewer than one death per cell: the null "
        "floor is underestimated there (in %d datasets generated under conditional "
        "independence given these strata the excess averaged %s, and *P* was at "
        "most 0.05 in %s%%), so the calibrated value, the excess and *P* are not "
        "shown and the floor share is a lower bound. Standard errors were not "
        "computed for the age-band set, and deaths per cell are not shown for the "
        "unadjusted (none) rows. *P* values are given to three decimals, to two "
        "significant figures below 0.01 and to one below 0.001, except Monte-Carlo "
        "and bootstrap *P* values, which are given to three decimals at their stated "
        "resolution. "
        "*Floor "
        "share*: null floor as a proportion of the plug-in value. *Deaths per "
        "cell*: deaths in each exposure-by-stratum cell, their mean and median and "
        "the percentages of cells with fewer than two and fewer than five. "
        "Adjustment sets are defined "
        "in Supplementary Methods S1.6."
        % (_int(len(d)), HATS["plug"], HATS["floor"], HATS["cal"], HATS["exc"],
           _nc_pct("PRIMARY", "reject_05"), _nc("RAW").get("datasets", 0),
           "%.3f" % np.mean(list(_nc("RAW").get("mean_excess", {"x": np.nan}).values())),
           _nc_pct("RAW", "reject_05"))))
    parts.append("<!-- LANDSCAPE off -->")

    bt = mm[mm.spec.isin(["total", "DEMO", "PRIMARY"])]
    rows = []
    for _, r in bt.iterrows():
        sc = "calibrated"
        rows.append({
            "Habit": r.label, "Adjustment": SPEC_NAME[r.spec],
            "Estimate": FMT % r[sc], "Model value": FMT % r.boot_world,
            "Model − estimate, SE": _num((r.boot_world - r[sc]) / r[sc + "_se"], "%.1f"),
            "Bootstrap mean": FMT % r[sc + "_boot_mean"], "SE": FMT % r[sc + "_se"],
            "SE, variance units": ("" if pd.isna(r.get("calibrated_se_psu", np.nan))
                                   else FMT % r.calibrated_se_psu),
            "Shift (SE)": _num(r[sc + "_shift_in_sds"], "%.2f"),
            "Estimate ± 1.96 SE": "%s to %s" % (_num(r[sc] - 1.96 * r[sc + "_se"], FMT),
                                                _num(r[sc] + 1.96 * r[sc + "_se"], FMT)),
            "Basic interval": "%s to %s" % (_num(r[sc + "_basic_lo"], FMT),
                                           _num(r[sc + "_basic_hi"], FMT))})
    parts.append("<!-- LANDSCAPE on -->")
    parts.append(_sect(
        "1b", "The smoothed bootstrap of the calibrated rCMI", _md(pd.DataFrame(rows)),
        "1000 smoothed paired resamples (Supplementary Methods S3.2). *Model value*: "
        "the rCMI of the bootstrap population, i.e. of the table of expected deaths "
        "under the additive logistic model from which deaths are re-simulated; "
        "*model − estimate, SE*: their difference in bootstrap standard errors. "
        "*Shift*: (bootstrap mean − model value) in bootstrap standard errors, the "
        "bias of the estimator in the bootstrap population. *Basic interval*: the "
        "2.5th and 97.5th bootstrap percentiles pivoted on the model value, "
        "estimate − (percentile − model value). *SE, variance units*: the same "
        "smoothed bootstrap with the masked variance units of the NHANES design, "
        "rather than individuals, resampled within each (cycle-specific) design "
        "stratum (500 resamples; primary set only)."))
    parts.append("<!-- LANDSCAPE off -->")

    p_ = os.path.join(OUT, "smoothing.csv")
    if os.path.exists(p_):
        sm = pd.read_csv(p_).set_index("habit").loc[order]
        gap_ = 1e4 * (sm.world_interaction ** 2 - sm.world_additive ** 2)
        s1e = pd.DataFrame({
                "Habit": sm.label,
                "Estimate": [FMT % pri.loc[h, "calibrated"] for h in order],
                "Model value, additive": [FMT % v for v in sm.world_additive],
                "Model value, interactions": [FMT % v for v in sm.world_interaction],
                "Model value, two-year cycles": [FMT % v for v in sm.world_cycle],
                "SE, additive smoother": [FMT % v for v in sm.se_additive],
                "SE, interaction smoother": [FMT % v for v in sm.se_interaction],
                "SE ratio (design factor), observed deaths": [
                    "%.2f" % v for v in sm.deff_plain]})
        if "overfit_djs" in sm:
            s1e.insert(5, "Interactions − additive, $D_{\\mathrm{JS}} \\times 10^{4}$",
                       [_num(v, "%.2f") for v in gap_])
            s1e.insert(6, "Expected from overfitting, $\\times 10^{4}$ (MC SE)", [
                "%s (%.2f)" % (_num(1e4 * v, "%.2f"), 1e4 * e)
                for v, e in zip(sm.overfit_djs, sm.overfit_djs_mcse)])
        ovf_note = ("" if "overfit_djs" not in sm else
                    " The likelihood-ratio test and AIC disagree: AIC favours the "
                    "additive model (by %.1f). *Interactions − additive*: the difference "
                    "between the two model values on the $D_{\\mathrm{JS}}$ scale. "
                    "*Expected from overfitting*: the mean, "
                    "over %d datasets whose deaths are simulated from the fitted "
                    "additive model (no interactions), of the interaction model's "
                    "value minus the additive model's, both refitted, on the "
                    "$D_{\\mathrm{JS}}$ scale; it is %d–%d%% of the observed "
                    "difference, so the interaction model's larger values, its "
                    "smaller standard errors and its flatter level shares "
                    "(Supplementary Table S1f) largely reflect overfitting."
                    % (float(sm.aic_diff.iloc[0]),
                       _design("measures", "overfit_datasets", 30),
                       int(round(100 * min(1e4 * sm.overfit_djs / gap_))),
                       int(round(100 * max(1e4 * sm.overfit_djs / gap_)))))
        parts.append("<!-- LANDSCAPE on -->")
        parts.append(_sect(
            "1e", "The smoothing model of the bootstrap", _md(s1e),
            "*Model value*: rCMI of the table of expected deaths under the additive "
            "smoothing model and under the model with habit × age band and habit × "
            "period terms, and under the additive model with the two-year survey "
            "cycle in place of the four-year period (the alcohol questionnaire "
            "changed between the two cycles of 2015–2018). The additive model is "
            "rejected against the interaction "
            "model (likelihood ratio %.1f on %d df, *P* %s). *SE*: the primary "
            "smoothed bootstrap (1000 resamples, additive smoother) and the same "
            "bootstrap with the interaction model as the smoother (%d resamples). "
            "*SE ratio (design factor), observed deaths*: standard error of the "
            "plug-in "
            "estimate over resamples of masked variance units divided by that over "
            "resamples of individuals, both keeping the observed deaths (%d "
            "resamples each), which includes clustering of the outcome that the "
            "smoothed bootstrap, re-simulating death, cannot see. Each ratio of "
            "bootstrap standard deviations has a Monte-Carlo SE of about 0.04, so "
            "ratios within about 0.08 of 1 do not differ from 1."
            % (sm.lr.iloc[0], int(sm.lr_df.iloc[0]),
               "= %s" % _pv2(sm.lr_p.iloc[0]),
               _design("measures", "interaction_smoother_resamples", 500),
               _design("measures", "plain_resamples", 500)) + ovf_note))
        parts.append("<!-- LANDSCAPE off -->")

    t2 = mm[["label", "spec", "domi", "race", "bound_alphabet", "bound_achievable"]]
    t2 = pd.DataFrame({"Habit": t2.label, "Adjustment": t2.spec.map(SPEC_NAME),
                       "do-MI": [FMT % v for v in t2.domi],
                       "RACE": [FMT % v for v in t2.race],
                       "Alphabet bound": [FMT % v for v in t2.bound_alphabet],
                       "Achievable bound": [FMT % v for v in t2.bound_achievable]})
    parts.append(_sect(
        "1c", "Interventional measures and the upper bounds", _md(t2),
        "do-MI and RACE are the intervention-family counterparts of rMI and rCMI "
        "(plug-in values, not null-calibrated). The alphabet bound "
        "$c_{\\max}(2) = 0.558$ applies to rMI, rCMI and do-MI but not to RACE, "
        "whose ceiling is 1. The achievable bound is the largest rCMI attainable "
        "with the observed $p(x, z)$ (Supplementary Methods S2.3); it never "
        "exceeds the alphabet bound."))

    for spec, sc, num, title in [
            ("PRIMARY", "calibrated", "2", "Pairwise differences in direct association "
             "(primary adjustment set, calibrated rCMI)"),
            ("PRIMARY", "excess", "2b", "Pairwise differences, square-root-scale excess"),
            ("DEMO", "calibrated", "2c", "Pairwise differences, demographic adjustment "
             "only")]:
        dd = diff[(diff.spec == spec) & (diff.scale == sc)].copy()
        flip = dd.habit_a.map(rank_of) > dd.habit_b.map(rank_of)
        for a, b in (("label_a", "label_b"), ("habit_a", "habit_b")):
            dd.loc[flip, [a, b]] = dd.loc[flip, [b, a]].values
        dd.loc[flip, "diff"] = -dd.loc[flip, "diff"]
        for lo, hi in (("wald_lo", "wald_hi"), ("pct_lo", "pct_hi"),
                       ("basic_lo", "basic_hi")):
            a_, b_ = dd.loc[flip, lo].copy(), dd.loc[flip, hi].copy()
            dd.loc[flip, lo], dd.loc[flip, hi] = -b_, -a_
        # order of Table 2: by the first habit, then by the second
        dd["_a"], dd["_b"] = dd.habit_a.map(rank_of), dd.habit_b.map(rank_of)
        dd = dd.sort_values(["_a", "_b"]).reset_index(drop=True)
        out = pd.DataFrame({
            "Habit A": dd.label_a, "Habit B": dd.label_b,
            "A − B": [_num(v, FMT) for v in dd["diff"]],
            "SE": [FMT % v for v in dd.se],
            "95% CI (bootstrap SE)": ["%s to %s" % (_num_signed(a, FMT),
                                                    _num_signed(b, FMT))
                                      for a, b in zip(dd.wald_lo, dd.wald_hi)],
            "*P*": [_pv2(v) for v in dd.p_wald],
            "*P*, bootstrap": [_p(v, "0.002") for v in dd.p_boot],
            "Basic interval": ["%s to %s" % (_num(a, FMT), _num(b, FMT))
                               for a, b in zip(dd.basic_lo, dd.basic_hi)]})
        if "p_wald_psu" in dd and dd.p_wald_psu.notna().any():
            out["*P*, variance units"] = [_pv2(v) for v in dd.p_wald_psu]
        if "p_wald_int" in dd and dd.p_wald_int.notna().any():
            out["*P*, interaction smoother"] = [_pv2(v) for v in dd.p_wald_int]
        if "p_boot_int" in dd and dd.p_boot_int.notna().any():
            out["*P*, bootstrap, interaction smoother"] = [
                _p(v, "%.3f" % (2.0 / (_design("measures", "interaction_smoother_resamples",
                                                500) + 1))) for v in dd.p_boot_int]
        note = ("All six habits are measured in the same individuals, so the "
                "resamples are drawn once over individuals and every habit is "
                "recomputed within the same resample (1000 smoothed resamples; "
                "Supplementary Methods S3). The 95%% CI and *P* use the bootstrap "
                "standard error of the paired difference and a normal reference; "
                "the basic interval, pivoted on the difference between the model "
                "values, is shown for comparison. *P*, bootstrap: the dual of the basic "
                "interval, twice the proportion of resamples whose deviation from the "
                "model value is at least as large as the estimate, on its side "
                "(add-one, resolution 0.002). *P*, variance units: as *P*, with the "
                "standard error from the "
                "smoothed bootstrap over masked variance units, which captures the "
                "clustering of exposures and strata but not of death (the SE ratio "
                "that includes death, from resampling the observed deaths, is up to "
                "%.2f for single habits, Supplementary Table S1e, and %s for the "
                "paired differences of the plug-in; with the SE of smoking against "
                "sleep inflated by its ratio, %s, that comparison's *P* is %s). *P*, "
                "interaction smoother: as *P*, with the standard error from the "
                "smoothed bootstrap whose smoothing model has habit × age band and "
                "habit × period terms (%d resamples), whose smaller standard errors "
                "partly reflect that model's overfitting (Supplementary Table S1e); "
                "*P*, bootstrap, interaction smoother: the bootstrap *P* from those "
                "resamples (resolution %.3f). With 15 comparisons the Bonferroni "
                "threshold is 0.0033, which the bootstrap *P* reaches only when no "
                "resample lies beyond the estimate, and never with %d resamples. "
                "Habit A precedes habit B in the order of Table 2, and the rows "
                "follow that order. Differences are computed from unrounded "
                "estimates, so they can differ in the last digit from differences of "
                "the rounded values of Table 2."
                % (_deff_max(), _pair_deff_txt()[0], _pair_deff_txt()[1],
                   _pair_deff_txt()[2],
                   _design("measures", "interaction_smoother_resamples", 500),
                   2.0 / (_design("measures", "interaction_smoother_resamples", 500) + 1),
                   _design("measures", "interaction_smoother_resamples", 500))
                if num == "2" else "As Supplementary Table S2.")
        if sc == "excess":
            # the excess has no population counterpart to pivot on
            out = out.drop(columns=[c for c in ("Basic interval", "*P*, bootstrap",
                                                "*P*, bootstrap, interaction smoother")
                                    if c in out])
            note = ("As Supplementary Table S2. The excess is a biased estimator of "
                    "the association and has no population counterpart on which a "
                    "bootstrap distribution could be pivoted, so only the "
                    "normal-reference interval and *P* are shown; they test a "
                    "difference between excesses, which is not a difference between "
                    "the associations, because the excess understates each habit by "
                    "a different amount (Supplementary Methods S3.1).")
        parts.append("<!-- LANDSCAPE on -->")
        parts.append(_sect(num, title, _md(out), note))
        parts.append("<!-- LANDSCAPE off -->")

    p_ = os.path.join(OUT, "ranks.csv")
    if os.path.exists(p_):
        rk_ = pd.read_csv(p_).set_index("habit").loc[order]
        parts.append(_sect(
            "2d", "How firmly the ranking is identified", _md(pd.DataFrame({
                "Habit": rk_.label,
                "Ranked first": ["%.2f" % v for v in rk_.p_first],
                "Ranked first or second": ["%.2f" % v for v in rk_.p_top2],
                "Ranked last": ["%.2f" % v for v in rk_.p_last],
                "95% range of rank": ["%d–%d" % (a, b) if a != b else "%d" % a
                                      for a, b in zip(rk_.rank_lo, rk_.rank_hi)],
                "Ranked first, interaction smoother": ["%.2f" % v for v in rk_.p_first_int],
                "Ranked last, interaction smoother": ["%.2f" % v for v in rk_.p_last_int],
                "95% range of rank, interaction smoother": [
                    "%d–%d" % (a, b) if a != b else "%d" % a
                    for a, b in zip(rk_.rank_lo_int, rk_.rank_hi_int)]})),
            "Proportions of the 1000 smoothed resamples (pivoted on the model values: "
            "resampled value − model value + estimate, the percentile orientation, "
            "which mirrors the basic orientation of the bootstrap *P* of "
            "Supplementary Table S2) in which each habit's "
            "calibrated rCMI ranks first, first or second, and last among the six, "
            "and the 2.5th–97.5th percentile range of its rank; primary set. "
            "*Interaction smoother*: the same from the %d resamples of the smoothed "
            "bootstrap whose smoothing model has habit × age band and habit × period "
            "terms (Supplementary Table S1e)."
            % _design("measures", "interaction_smoother_resamples", 500)))
    p_ = os.path.join(OUT, "refinement.csv")
    if os.path.exists(p_):
        rf = pd.read_csv(p_)
        g_ = rf.groupby("habit")
        mean_ = {h: g_.get_group(h).calibrated.mean() for h in order}
        base_ = {h: pri.loc[h, "calibrated"] for h in order}
        mc_ = {h: g_.get_group(h).calibrated.std(ddof=1)
               / np.sqrt(len(g_.get_group(h))) for h in order}
        parts.append(_sect(
            "1d", "Refinement diagnostic", _md(pd.DataFrame({
                "Habit": [C.LABEL[h] for h in order],
                "Primary (54 strata)": [FMT % base_[h] for h in order],
                "Each stratum split at random in two: mean (SD)": [
                    "%s (%s)" % (FMT % mean_[h], FMT % g_.get_group(h).calibrated.std(ddof=1))
                    for h in order],
                "MC SE of the mean": [FMT % mc_[h] for h in order],
                "Change": [_num(mean_[h] - base_[h], FMT) for h in order],
                "Change / MC SE": ["%.1f" % ((mean_[h] - base_[h]) / mc_[h]) for h in order],
                "Change on the $D_{\\mathrm{JS}}$ scale, %": [
                    _num(100 * ((g_.get_group(h).calibrated ** 2).mean() / base_[h] ** 2
                                - 1), "%.0f") for h in order]})),
            "Every stratum of the primary set is split in two by an independent "
            "random variable, which leaves the population rCMI unchanged; %d random "
            "splits. *MC SE*: Monte-Carlo standard error of the mean over splits "
            "(SD/√%d). A rise shows how the upward drift of the calibrated "
            "estimator grows when the strata are made sparser than the primary 54 "
            "(Supplementary Methods S5.2); it measures the bias at 108 strata minus "
            "that at 54, not the bias of the primary estimate, which the "
            "semi-parametric check puts within %d%% (Supplementary Table S8b). "
            "*Change on the $D_{\\mathrm{JS}}$ scale*: the mean over splits of the "
            "squared calibrated estimate relative to the square of the primary "
            "estimate, minus 1; as a mean of squares it includes the variance over "
            "splits, and so exceeds the change in the square of the mean. Changes are "
            "computed from unrounded estimates."
            % (rf.split.nunique(), rf.split.nunique(), _calcheck_absmax())))

    p_ = os.path.join(OUT, "levels.csv")
    if os.path.exists(p_):
        lv_ = pd.read_csv(p_)
        lv_["_h"] = lv_.habit.map(rank_of)
        lv_ = lv_.sort_values(["_h", "level"], kind="stable")
        sh_ = lambda v: _num(_pct_hu(v), "%d")                        # noqa: E731
        t_ = pd.DataFrame({
            "Habit": [_lab(h) for h in lv_.habit],
            "Level": list(lv_.level_name),
            "% of sample": ["%.1f" % v for v in lv_.pct],
            "Deaths": [_int(v) for v in lv_.deaths],
            "Mean age, years": ["%.1f" % v for v in lv_.mean_age],
            "Share, model (additive), %": [sh_(v) for v in lv_.share_additive],
            "Share, model (interactions), %": [sh_(v) for v in lv_.share_interaction],
            "Share, calibrated, %": [sh_(v) for v in lv_.share_calibrated],
            "Calibrated contribution $\\times 10^{4}$ (SE)": [
                "%s (%.2f)" % (_num(1e4 * v, "%.2f"), 1e4 * e)
                for v, e in zip(lv_.calibrated, lv_.calibrated_se)],
            "Share, plug-in, %": [sh_(v) for v in lv_.share_plugin]})
        parts.append("<!-- LANDSCAPE on -->")
        parts.append(_sect(
            "1f", "Where each habit's direct association comes from", _md(_grouped(t_)),
            "$\\mathrm{rCMI}^2$ ($D_{\\mathrm{JS}}$) is a sum over the cells of the "
            "habit × death × stratum table of non-negative terms, so it splits "
            "exactly over the habit's levels (Supplementary Methods S2.2). *Share, "
            "model*: each level's share of the $D_{\\mathrm{JS}}$ of the table of "
            "expected deaths under the additive smoothing model and under the model "
            "with habit × age band and habit × period terms (Supplementary Table "
            "S1e): the shares of the association itself, free of the null floor; "
            "the interaction model's are pulled towards equal shares by its "
            "overfitting (Supplementary Table S1e). "
            "*Share, calibrated*: the level's contribution to the plug-in value "
            "minus its contribution to the null floor (the same 500 null tables), as "
            "a percentage of the sum of these differences over levels, which is the "
            "calibrated $D_{\\mathrm{JS}}$; a level can have a negative share when "
            "its null contribution exceeds its observed one. *Calibrated "
            "contribution*: the level's calibrated $D_{\\mathrm{JS}}$ (plug-in minus "
            "null floor) $\\times 10^{4}$, with its standard error from the %d smoothed "
            "bootstrap resamples (Supplementary Methods S3.2); the contributions add "
            "up to the square of the calibrated rCMI of Table 2. The shares differ "
            "between the columns, most for levels with few deaths, and none of them "
            "is precise. *Share, plug-in*: of "
            "the plug-in value, most of which is null floor (Supplementary Table "
            "S1), so its shares say little about where the association lies. The "
            "measure has no direction: a level contributes when its death rate "
            "within strata differs from that of the other levels, in either "
            "direction (Table 3). Primary adjustment set."
            % _design("measures", "bootstrap_resamples", 1000)))
        parts.append("<!-- LANDSCAPE off -->")

    # ---- sensitivity analyses: one row per analysis
    sr = pd.read_csv(os.path.join(OUT, "sensitivity_rcmi.csv"))
    ana = sorted(sr.analysis.unique(), key=_sa_key)
    labs = {h: C.LABEL[h] for h in order}
    est_rows, p_rows = [], []
    for a in ana:
        g = sr[sr.analysis == a].set_index("habit")
        row = {"Analysis": _sa_code(a),
               "*n*": "/".join(_int(v) for v in sorted({int(x) for x in g.n})),
               "Deaths": "/".join(_int(v) for v in sorted({int(x) for x in g.n_deaths}))}
        if "deaths_per_cell" in g and g.deaths_per_cell.notna().any():
            row["Deaths per cell"] = "%.1f–%.1f" % (g.deaths_per_cell.min(),
                                                    g.deaths_per_cell.max())
        prow = {"Analysis": row["Analysis"]}
        for h in order:
            if h not in g.index:
                row[labs[h]] = prow[labs[h]] = ""
                continue
            se_ = (pri.loc[h, "calibrated_se"] if a == "primary"
                   else g.loc[h, "calibrated_se"] if "calibrated_se" in g else np.nan)
            row[labs[h]] = FMT % g.loc[h, "calibrated"] + (
                "" if pd.isna(se_) else " (%s)" % (FMT % se_))
            prow[labs[h]] = _p(g.loc[h, "p_perm"] if "p_perm" in g
                               and pd.notna(g.loc[h, "p_perm"]) else g.loc[h, "p_null"],
                               "0.002")
        est_rows.append(row)
        p_rows.append(prow)
        if a.startswith("SA15 ") and "prevalence_reference_sa15" in _extra():
            fac15 = _extra()["prevalence_reference_sa15"]["factor"]
            b15 = _extra()["sa16_reference"]["unweighted_permutation"]
            est_rows.append({"Analysis": "SA15, expected†", **{
                labs[h]: FMT % (b15[h]["calibrated"] * fac15[h]) for h in order}})
        if a.startswith("SA16 ") and "sa16_reference" in _extra():
            b16 = _extra()["sa16_reference"]["unweighted_permutation"]
            est_rows.append({"Analysis": "SA16, no weights§", **{
                labs[h]: FMT % b16[h]["calibrated"] for h in order}})
            p_rows.append({"Analysis": "SA16, no weights§", **{
                labs[h]: _p(b16[h]["p_null"], "0.002") for h in order}})
        if a.startswith("SA17a "):
            ex_ = _extra()
            if "prevalence_reference" in ex_:
                fac = ex_["prevalence_reference"]["factor"]
                est_rows.append({"Analysis": "SA17a, expected†", **{
                    labs[h]: FMT % (pri.loc[h, "calibrated"] * fac[h]) for h in order}})
            if "sa17a_refinement" in ex_:
                sp_, mc17 = ex_["sa17a_refinement"], ex_.get("sa17a_refinement_mcse", {})
                est_rows.append({"Analysis": "SA17a, strata split‡", **{
                    labs[h]: FMT % sp_[h] + ("" if h not in mc17 else " (%s)" % (FMT % mc17[h]))
                    for h in order}})
    legend = "; ".join("%s, %s" % (_sa_code(c), _sa_name(re.sub(r"^SA\d+[a-d]? ", "", c)))
                       for c in ana if c != "primary")
    sa1 = sr[sr.analysis.str.startswith("SA1 ")]
    big = sa1[sa1.n == sa1.n.max()]
    parts.append("<!-- LANDSCAPE on -->")
    parts.append(_sect(
        3, "Direct association (calibrated rCMI) under the sensitivity analyses",
        _md(pd.DataFrame(est_rows)) + "\n\nMonte-Carlo *P* against the "
        "within-stratum permutation null:\n\n" + _md(pd.DataFrame(p_rows)),
        "Calibrated rCMI (smoothed-bootstrap standard error). Analyses are defined "
        "in Supplementary Methods S1.8: %s. *Deaths per cell*: the range over the six "
        "habits of the mean number of deaths per exposure-by-stratum cell. Standard "
        "errors: 1000 "
        "resamples for the primary analysis, %d for the others (none for SA1); "
        "SA15 and SA16 are resampled with their weights (%d permutations per "
        "resample) and calibrated against a "
        "within-stratum permutation null (%d permutations; SA16's weights, which "
        "depend on death, recomputed from the permuted or simulated deaths; "
        "Supplementary Methods S1.8). SA19 classifies physical activity from the "
        "leisure and transport domains only, with the same cut-points; SA20 codes "
        "every habit more coarsely (current smoking or not; no drinking, up to 14 "
        "or more than 14 drinks/week; BMI <25, 25–29.9 or ≥30 kg/m²; any physical "
        "activity or none; sleep <7, 7–<9 or ≥9 h; sitting <420 or ≥420 min/day). "
        "SA1 uses, for smoking, alcohol use and BMI, "
        "the %s participants of "
        "1999–2018 complete on the three (%s deaths; %d co-habits), and for "
        "physical activity, sleep duration and sitting time the primary sample, so "
        "their SA1 values equal the primary ones (and SA1 is not counted among "
        "the sensitivity analyses of sleep in the main text, nor among the "
        "single-sample analyses in its statement on smoking's lead: its habits come "
        "from samples with different proportions of deaths, which alone changes "
        "the measures, Supplementary Methods S2.2, property 5). SA17d is the "
        "complement of SA17a. †Expected under the prevalence-matched reference: the primary "
        "estimate (for SA15, the unweighted estimate with the same permutation "
        "null, §) times the ratio of the smoothing model's rCMI at the analysis's "
        "proportion of deaths (weighted, %.1f%%, for SA15; %.1f%% for SA17a) to "
        "that at the primary proportion. ‡Every SA17a "
        "stratum split at random in two: mean over %d splits (Monte-Carlo SE). "
        "§The primary analysis calibrated against the within-stratum permutation "
        "null of SA15 and SA16 without weights: the like-for-like baseline for "
        "those two. "
        "SA2, SA4a/b, SA7, SA18 and SA21 are hazard-ratio analyses (Supplementary "
        "Table S4). Null floors from 500 draws. *P*: against the within-stratum "
        "permutation null (%d permutations, resolution 0.002); the multinomial "
        "null used for the null floors is liberal, more so in sparse strata (*P* at "
        "most 0.05 in %s%% of datasets generated under conditional independence "
        "given the primary strata and in %s%% given those of SA17a; Supplementary "
        "Methods S3.1)."
        % (legend, _design("sensitivity", "bootstrap_resamples", 200),
           _design("sensitivity", "sa15_null_check_permutations", 100),
           _design("sensitivity", "sa15_permutations", 500), _int(big.n.iloc[0]), _int(big.n_deaths.iloc[0]),
           int(big.n_cohabits.iloc[0]),
           100 * float(_extra().get("prevalence_reference_sa15", {}).get(
               "death_proportion_weighted", float("nan"))),
           100 * float(_extra().get("prevalence_reference", {}).get(
               "death_proportion", float("nan"))),
           _design("sensitivity", "refinement_splits", 100),
           _design("sensitivity", "permutations_all", 500),
           _nc_pct("PRIMARY", "reject_05"),
           (lambda v: "?" if v is None else "%d" % _pct_hu(v))(
               _extra().get("sa17a_null_check", {}).get("reject_05")))))
    parts.append("<!-- LANDSCAPE off -->")

    sh = pd.read_csv(os.path.join(OUT, "sensitivity_hr.csv"))
    sh = sh.iloc[sorted(range(len(sh)), key=lambda i: (_sa_key(sh.analysis.iloc[i]), i))]
    t4 = pd.DataFrame({
        "Analysis": [_sa_name(a) for a in sh.analysis],
        "Habit": sh.label, "*n*": [_int(v) for v in sh.n],
        "Deaths": [_int(v) for v in sh.n_deaths], "Contrast": sh.contrast,
        "HR (95% CI)": ["%.2f (%.2f–%.2f)" % (h, a, b)
                        for h, a, b in zip(sh.hr, sh.hr_lo, sh.hr_hi)]})
    parts.append("<!-- LANDSCAPE on -->")
    parts.append(_sect(
        4, "Hazard ratios under the sensitivity analyses", _md(t4),
        "For each habit the contrast is held fixed at the level with the highest "
        "hazard ratio in the primary fit, against the reference level (never "
        "smoker; light drinker; BMI 18.5–24.9 kg/m²; ≥1800 MET-min/week; 7–<9 h; "
        "<240 min/day); SA4 shows every BMI level. Intervals are Wald intervals, "
        "except SA2 and SA16: log hazard ratio ± 1.96 standard errors, the "
        "standard error from a Rao–Wu rescaling bootstrap (main-text reference "
        "40; reference S7) over masked variance "
        "units (SA2, %d replicates) or from a bootstrap of individuals that refits "
        "the completeness model in every resample (SA16, %d resamples). SA3 and "
        "SA4b count follow-up from a landmark two years after the interview, and "
        "SA3b from a landmark four years after it; SA7 uses time on study with a "
        "separate baseline hazard for each four-year period; SA8 censors "
        "follow-up at 10 years; SA16 weights by the inverse probability of being "
        "a complete case, from a model among all eligible adults of 2007–2018 "
        "(a missing education or marital status as its own category); SA17d is "
        "the complement of SA17a; SA18 has a separate baseline hazard for each "
        "stratum of the primary adjustment set and the habit as the only "
        "covariate; SA19 classifies physical activity from the leisure and "
        "transport domains only, with the same cut-points."
        % (_design("sensitivity", "cluster_bootstrap_resamples", 500),
           _design("sensitivity", "ipw_resamples", 200))))
    parts.append("<!-- LANDSCAPE off -->")
    ex_ = _extra()
    if "sa17_interaction" in ex_:
        it = ex_["sa17_interaction"]
        ci3 = lambda a, b, c: "%.2f (%.2f–%.2f)" % (a, b, c)          # noqa: E731
        v0 = next(iter(it.values()))
        parts.append(_sect(
            "4b", "Hazard ratios among the healthy and the ill, and their difference",
            _md(pd.DataFrame({
                "Habit": [_lab(h) for h in order],
                "Contrast": [it[h]["level_name"] for h in order],
                "HR, healthy (95% CI)": [ci3(it[h]["hr0"], it[h]["hr0_lo"], it[h]["hr0_hi"])
                                         for h in order],
                "HR, ill (95% CI)": [ci3(it[h]["hr1"], it[h]["hr1_lo"], it[h]["hr1_hi"])
                                     for h in order],
                "Ratio, healthy/ill (95% CI)": [
                    ci3(1 / it[h]["ratio"], 1 / it[h]["ratio_hi"], 1 / it[h]["ratio_lo"])
                    for h in order],
                "*P*, contrast": [_p4b(it[h]["p_level"]) for h in order],
                "*P*, all levels (df)": ["%s (%d)" % (_p4b(it[h]["p_joint"]), it[h]["df"])
                                         for h in order]}).pipe(_s4b_pa_rows, it, order, ci3))
            + ("" if "rd0" not in v0 else
               "\n\nThe same contrasts on the additive scale: g-formula 10-year "
               "standardised risk (%) of the reference level and of the contrast "
               "within each group, and their difference (percentage points, computed "
               "from unrounded risks, so it can differ in the last digit from the "
               "difference of the rounded risks shown):\n\n"
               + _md(pd.DataFrame({
                   "Habit": [_lab(h) for h in order],
                   "Contrast": [it[h]["level_name"] for h in order],
                   "Healthy: reference, contrast": [
                       "%.1f, %.1f" % (100 * it[h]["risk_ref0"], 100 * it[h]["risk_level0"])
                       for h in order],
                   "Healthy: difference": [_num(100 * it[h]["rd0"], "%.1f") for h in order],
                   "Ill: reference, contrast": [
                       "%.1f, %.1f" % (100 * it[h]["risk_ref1"], 100 * it[h]["risk_level1"])
                       for h in order],
                   "Ill: difference": [_num(100 * it[h]["rd1"], "%.1f") for h in order]}))),
            "One Cox model per habit among the %s participants with both prevalent "
            "disease and self-rated health known (%s deaths): the model of Table 3 "
            "with separate baseline hazards for the healthy and the ill (illness: "
            "prevalent heart disease, heart failure, stroke, chronic lung disease, "
            "cancer or diabetes, or fair or poor self-rated health) and the product "
            "of illness with each of the habit's contrasts. Separate baselines are "
            "needed because illness does not act proportionally over attained age "
            "(fourth panel). "
            "*Healthy*: %s participants (%s deaths); *ill*: %s (%s deaths). *P*, "
            "contrast: Wald test that the hazard ratio of the contrast shown is the "
            "same in the two groups; *P*, all levels: joint Wald test of all the "
            "habit's product terms. The fixed contrast is that of Supplementary "
            "Table S4; for physical activity every contrast is shown. The additive-"
            "scale panel uses the same model, with a Breslow baseline for each group "
            "(point estimates). The composite definition of illness was not "
            "prespecified; the next panel repeats the model with each of its two "
            "criteria alone as the modifier, in the same participants."
            % (_int(v0["n0"] + v0["n1"]), _int(v0["deaths0"] + v0["deaths1"]),
               _int(v0["n0"]), _int(v0["deaths0"]), _int(v0["n1"]), _int(v0["deaths1"]))))
        rows1, sizes1 = [], []
        for key_, name_ in (("sa17_interaction_chronic", "Diagnosed disease"),
                            ("sa17_interaction_poor_health", "Fair or poor self-rated health")):
            if key_ not in ex_:
                continue
            it1 = ex_[key_]
            w0 = next(iter(it1.values()))
            sizes1.append("%s: %s without (%s deaths), %s with (%s)" % (
                name_.lower(), _int(w0["n0"]), _int(w0["deaths0"]), _int(w0["n1"]),
                _int(w0["deaths1"])))
            for h in order:
                rows1 += _s4b_rows(it1[h], h, name_ if h == order[0] else "", ci3)
        if "sa19_interaction" in ex_:
            rows1 += _s4b_rows(ex_["sa19_interaction"]["pa"], "pa",
                               "Composite (SA19: physical activity from leisure and "
                               "transport only)", ci3)
        if rows1:
            # the second panel belongs to the S4b section (the sections are
            # later ordered by their headings)
            parts[-1] = (parts[-1].rstrip("\n") + "\n\nEach criterion alone as the "
                         "modifier (%s); and the composite with physical activity "
                         "from leisure and transport only (SA19). For physical "
                         "activity every contrast is shown, in this panel and the "
                         "first:\n\n" % "; ".join(sizes1)
                         + _md(pd.DataFrame(rows1)) + "\n")
        if "sa17_interaction_proportional" in ex_:
            ip = ex_["sa17_interaction_proportional"]
            parts[-1] = (parts[-1].rstrip("\n") + "\n\nIllness as a proportional "
                         "covariate (one baseline hazard for all), a specification the "
                         "Grambsch–Therneau test rejects for the illness term "
                         "(*P* ≤ %s in all six models):\n\n"
                         % ("%#.2g" % max(v["ph_mod_p"] for v in ip.values()))
                         + _md(pd.DataFrame({
                             "Habit": [_lab(h) for h in order],
                             "Contrast": [ip[h]["level_name"] for h in order],
                             "HR, healthy (95% CI)": [
                                 ci3(ip[h]["hr0"], ip[h]["hr0_lo"], ip[h]["hr0_hi"])
                                 for h in order],
                             "HR, ill (95% CI)": [
                                 ci3(ip[h]["hr1"], ip[h]["hr1_lo"], ip[h]["hr1_hi"])
                                 for h in order],
                             "*P*, contrast": [_p4b(ip[h]["p_level"]) for h in order],
                             "*P*, all levels (df)": [
                                 "%s (%d)" % (_p4b(ip[h]["p_joint"]), ip[h]["df"])
                                 for h in order]})) + "\n")

    bf = pd.read_csv(os.path.join(OUT, "biasfloor.csv"))
    t5 = pd.DataFrame({
        "Habit": bf.label, "Strata": [_int(v) for v in bf.k_strata],
        "Mean *n* per stratum": [_thin("%.1f" % v) for v in bf.mean_per_stratum],
        "Plug-in rCMI": [FMT % v for v in bf.rcmi],
        "Null floor": [FMT % v for v in bf.null_mean],
        "Excess": [_num(v, FMT) for v in bf.excess],
        "Calibrated": [FMT % v for v in bf.calibrated],
        "Floor share": ["%.2f" % v for v in bf.null_share]})
    parts.append(_sect(
        5, "The null floor as the prognostic score is cut into finer strata", _md(t5),
        "The cross-fitted demographic prognostic score alone, cut into 2 to 2 000 "
        "equal-count strata; null floors from 500 draws. Plug-in: %s; null floor: "
        "%s." % (HATS["plug"], HATS["floor"])))

    reps = _design("simulation", "replicates", 100)
    lv = ", ".join(str(v) for v in _design("simulation", "levels", [3, 5, 5, 4, 4, 4]))
    for num, fname, title, note in [
        (6, "simulation_absolute.csv", "Simulation: recovery of absolute values",
         "*K* strata, *n* observations, direct-effect coefficient β; the true rCMI is "
         "computed exactly from the generating distribution. *Obs. per cell*: "
         "*n*/(8*K*) for the four-level exposure; *deaths per cell*: expected deaths "
         "per exposure-by-stratum cell. Relative errors are (mean estimate − "
         "truth)/truth over %d replicates, with their Monte-Carlo standard errors "
         "(blank where the truth is 0)." % reps),
        (7, "simulation_rank.csv", "Simulation: recovery of a known ordering",
         "*K* strata, *n* observations and *Obs. per cell*, *n*/(8*K*), as in "
         "Supplementary Table S6. Six exposures whose population rCMIs are known "
         "and distinct (Supplementary "
         "Table S8), with the alphabets (%s levels) and margins of the six real "
         "habits. *Top 1*: proportion of %d replicates in which the largest "
         "estimate is the true largest. *Leading pair*: proportion in which the two "
         "largest estimates are the true two largest, in either order. ρ: mean "
         "Spearman correlation with the true ordering. Monte-Carlo standard errors "
         "in brackets. *Deaths per cell*: expected deaths per exposure-by-stratum "
         "cell, averaged over the six exposures. The plug-in's high recovery at "
         "fewer than one death per cell is an artefact of the design: there the two "
         "leading exposures also have the two largest null floors (Supplementary "
         "Methods S5.3)." % (lv, reps)),
        (8, "simulation_truths.csv", "Simulation: the population values estimated",
         "Exact population rCMI of each simulated exposure (direct-effect "
         "coefficients β = 0, 0.25, 0.5, 0.75, 1.0 and 1.5), with the relative gaps "
         "between the first and second and between the second and third "
         "largest.")]:
        p = os.path.join(OUT, fname)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        land = False
        if fname == "simulation_absolute.csv":
            def err(v, e):
                # an MC SE below one percentage point keeps one decimal, so
                # that none is printed as (0)
                return "" if pd.isna(v) else "%s%% (%s)" % (
                    _num(100 * v, "%.0f"), "%.1f" % (100 * e) if 100 * e < 0.95
                    else "%.0f" % (100 * e))
            df = pd.DataFrame({
                "*K*": [_int(v) for v in df.K], "*n*": [_int(v) for v in df.n],
                "β": ["%g" % v for v in df.b],
                "Obs. per cell": ["%.1f" % v for v in df.mean_per_cell],
                "Deaths per cell": ["%.2f" % v for v in df.deaths_per_cell],
                "True rCMI": [FMT % v for v in df.truth],
                "Plug-in": [FMT % v for v in df.plugin],
                "Excess": [_num(v, FMT) for v in df.excess],
                "Calibrated": [FMT % v for v in df.calibrated],
                "Plug-in error": ["" if pd.isna(v) else "%s%%" % _num(100 * v, "%.0f")
                                  for v in df.plugin_rel_err],
                "Excess error (MC SE)": [err(v, e) for v, e in zip(
                    df.excess_rel_err, df.excess_rel_err_mcse)],
                "Calibrated error (MC SE)": [err(v, e) for v, e in zip(
                    df.calibrated_rel_err, df.calibrated_rel_err_mcse)]})
            land = True
        elif fname == "simulation_rank.csv":
            out = {"*K*": [_int(v) for v in df.K], "*n*": [_int(v) for v in df.n],
                   "Obs. per cell": ["%.1f" % v for v in df.mean_per_cell],
                   "Deaths per cell": ["%.1f" % v for v in df.deaths_per_cell]}
            for k, lab in (("top1", "Top 1"), ("top2", "Leading pair"), ("rho", "ρ")):
                for e, el in (("plugin", "plug-in"), ("excess", "excess"),
                              ("calibrated", "calibrated")):
                    out["%s, %s" % (lab, el)] = [
                        "%.2f (%.2f)" % (v, s_) for v, s_ in zip(
                            df["%s_%s" % (k, e)], df["%s_%s_mcse" % (k, e)])]
            df = pd.DataFrame(out)
            land = True
        else:
            df = df.rename(columns={"exposure_%d" % j: "β = %s" % b
                                    for j, b in enumerate(["0", "0.25", "0.5", "0.75",
                                                           "1.0", "1.5"])})
            for g_c in ("gap_1_2", "gap_2_3"):
                df[g_c] = ["%d%%" % int(round(100 * v)) for v in df[g_c]]
            df = df.rename(columns={"gap_1_2": "Gap, 1st–2nd",
                                    "gap_2_3": "Gap, 2nd–3rd"})
            df["K"] = [_int(v) for v in df["K"]]
            df = df.rename(columns={"K": "*K*"})
        if land:
            parts.append("<!-- LANDSCAPE on -->")
        parts.append(_sect(num, title, _md(df, "%.4f"), note))
        if land:
            parts.append("<!-- LANDSCAPE off -->")

    p = os.path.join(OUT, "calibration_check.csv")
    if os.path.exists(p):
        ccall = pd.read_csv(p)
        prm = json.load(open(os.path.join(OUT, "params_calcheck.json"), encoding="utf-8"))
        nb, nd = int(prm["bootstrapped_datasets"]), int(prm["datasets"])
        lb = 0.025 ** (1.0 / nb)             # one-sided 97.5% lower bound for nb/nb
        pct = lambda v: "%s%%" % _num(100 * v, "%.0f")                  # noqa: E731

        def pm(v, e, n=nb):
            if e == 0 and v == 1.0:
                return "1.00 (≥%s)" % _r2(0.025 ** (1.0 / n))
            return "%s (%s)" % (_num(float(_r2(v)), "%.2f"), _r2(e))
        body = []
        for scen, lab in (("additive", "Population model inside the smoothing class "
                                       "(additive)"),
                          ("interaction", "Population model outside the smoothing class "
                                          "(habit × age band and habit × period "
                                          "interactions)")):
            cc = ccall[ccall.scenario == scen].set_index("habit").loc[order]
            body.append("*%s*\n\n" % lab + _md(pd.DataFrame({
                "Habit": cc.label, "True rCMI": [FMT % v for v in cc.truth],
                "Calibrated error (MC SE)": [
                    "%s (%.1f%%)" % (pct(v), 100 * e)
                    for v, e in zip(cc.calibrated_rel_err, cc.calibrated_rel_err_mcse)],
                "Error, strata held at population values": [
                    pct(v) for v in cc.calibrated_fixed_rel_err],
                "Change when strata are split (MC SE)": [
                    "%s (%.1f%%)" % (pct(v), 100 * e) for v, e in zip(
                        cc.split_change_rel, cc.split_change_rel_mcse)],
                "Excess error": [pct(v) for v in cc.excess_rel_err],
                "True SD": [FMT % v for v in cc.calibrated_sd],
                "SE ratio, smoothed (MC SE)": [pm(v, e) for v, e in zip(
                    cc.smooth_se_ratio, cc.smooth_se_ratio_mcse)],
                "SE ratio, plain (MC SE)": [pm(v, e) for v, e in zip(
                    cc.plain_se_ratio, cc.plain_se_ratio_mcse)],
                "Coverage (MC SE)": [pm(v, e) for v, e in zip(
                    cc.wald_coverage, cc.wald_coverage_mcse)]})))
        q = os.path.join(OUT, "calibration_check_rank.csv")
        if os.path.exists(q):
            rk_ = pd.read_csv(q)
            body.append("*Recovery of the ordering*\n\n" + _md(pd.DataFrame({
                "Population model": rk_.scenario,
                "Estimator": rk_.estimator.map({"plugin": "plug-in", "excess": "excess",
                                                "calibrated": "calibrated"}),
                "Leading habit identified (MC SE)": [pm(v, e, nd) for v, e in zip(
                    rk_.top1, rk_.top1_mcse)],
                "Leading pair identified (MC SE)": [pm(v, e, nd) for v, e in zip(
                    rk_.top2, rk_.top2_mcse)],
                "Mean Spearman ρ (MC SE)": [pm(v, e, nd) for v, e in zip(
                    rk_.rho, rk_.rho_mcse)]})))
        q = os.path.join(OUT, "calibration_check_pairs.csv")
        extra = ""
        if os.path.exists(q):
            cp = pd.read_csv(q)
            extra = " " + " ".join(
                "%s model: for the 15 paired differences the SE ratio was %s–%s "
                "(smoothed) and %s–%s (plain), and the coverage %s–%s."
                % (sc.capitalize(), _r2(g.smooth_se_ratio.min()), _r2(g.smooth_se_ratio.max()),
                   _r2(g.plain_se_ratio.min()), _r2(g.plain_se_ratio.max()),
                   _r2(g.wald_coverage.min()), _r2(g.wald_coverage.max()))
                for sc, g in cp.groupby("scenario", sort=False))
        parts.append("<!-- LANDSCAPE on -->")
        parts.append(_sect(
            "8b", "Semi-parametric check on the real data", "\n\n".join(body),
            "Population: the analytic sample, with death drawn from a model fitted to "
            "the data (Supplementary Methods S5.4); target: each habit's rCMI under "
            "the primary adjustment set of the real data. %s datasets per model, "
            "analysed as the real data are (prognostic scores and strata re-estimated "
            "on each dataset); errors are (mean − truth)/truth. *Error, strata "
            "held at population values*: the same datasets with the strata of the "
            "real data. *Change when strata are split*: mean change of the calibrated "
            "estimate when every stratum of the dataset is split at random in two "
            "(the refinement diagnostic of Supplementary Table S1d), relative to the "
            "truth; set against the calibrated error, it shows whether the "
            "diagnostic predicts the bias at 54 strata. *True SD*: SD of the "
            "calibrated estimate over datasets. For "
            "%d datasets per model the smoothed bootstrap of the primary analysis "
            "(additive smoothing model) and a plain bootstrap were run (%d resamples "
            "each): *SE ratio*, mean bootstrap SE divided by the true SD; *Coverage*, "
            "proportion of those datasets in which estimate ± 1.96 SE "
            "(smoothed) covered the target. MC SE: Monte-Carlo standard error.%s "
            "Proportions are rounded half up.%s"
            % (_int(prm["datasets"]), nb, prm["bootstrap_resamples"],
               (" (≥%s): where every one of the %d datasets was covered, the "
                "one-sided 97.5%% lower confidence bound." % (_r2(lb), nb))
               if (ccall.wald_coverage >= 1).any() else "", extra)))
        parts.append("<!-- LANDSCAPE off -->")

    p = os.path.join(OUT, "cox_by_age.csv")
    if os.path.exists(p):
        ca = pd.read_csv(p)
        ca["_h"] = ca.habit.map(rank_of)
        ca = ca.sort_values(["_h"], kind="stable")
        parts.append(_sect(
            "10b", "Hazard ratios below and above attained age 65", _md(pd.DataFrame({
                "Habit": ca.label, "Contrast": ca.level,
                "HR, age <65 (95% CI)": ["%.2f (%.2f–%.2f)" % t for t in zip(
                    ca.hr_young, ca.lo_young, ca.hi_young)],
                "HR, age ≥65 (95% CI)": ["%.2f (%.2f–%.2f)" % t for t in zip(
                    ca.hr_old, ca.lo_old, ca.hi_old)],
                "*P*, difference": [_pv2(v) for v in ca.p_interaction]})),
            "One Cox model per habit, with attained age as the time scale, follow-up "
            "split at attained age 65 and the habit's contrasts crossed with an "
            "indicator of age ≥65; adjusted as in Table 3. The hazard ratio of "
            "Table 3 is an average of the two, weighted by the ages at which deaths "
            "occur."))

    ph = pd.read_csv(os.path.join(OUT, "ph_check.csv"))
    g = ph[ph.level == "GLOBAL"].set_index("label")
    rows = []
    for h_ in order:
        lab = C.LABEL[h_]
        rows.append({"Habit": "**%s**" % _lab(h_), "Contrast": "all contrasts",
                     "Statistic": "χ² = %.2f on %d df" % (g.loc[lab, "chi2"],
                                                                   g.loc[lab, "df"]),
                     "*P*": _pv2(g.loc[lab, "p"])})
        for _, r in ph[(ph.label == lab) & (ph.level != "GLOBAL")].iterrows():
            rows.append({"Habit": "", "Contrast": r.level,
                         "Statistic": "*T* = %s" % _num(r.z, "%.2f"),
                         "*P*": _pv2(r.p)})
    parts.append(_sect(
        10, "Proportional-hazards score tests", _md(pd.DataFrame(rows)),
        "Score test for a coefficient that changes linearly with the rank of the "
        "event time (Grambsch and Therneau; main-text reference 44), jointly over "
        "each habit's contrasts "
        "against its reference level and for each contrast. *T*: the standardised "
        "score statistic of a contrast, approximately standard normal under "
        "proportional hazards; a negative *T* means a relative hazard that declines "
        "with attained age. With six joint tests the "
        "Bonferroni threshold is 0.0083."))

    _, miss, n_base = participant_flow()
    parts.append(_sect(
        9, "Missing data among eligible adults of 2007–2018", _md(miss),
        "Counts among the %s adults aged ≥20 years, eligible for mortality "
        "linkage, with known vital status and follow-up, in the six cycles "
        "2007–2018. Alcohol is missing when the questionnaire was not "
        "completed or the frequency or quantity was refused or unknown; "
        "respondents who reported fewer than 12 drinks in their life (2007–2016) "
        "or never having had a drink (2017–2018) are non-drinkers, not missing. "
        "A missing income-to-poverty ratio is kept as its own category, so the "
        "income variable itself has no missing values. Requiring completeness on "
        "all six habits and all covariates leaves the %s participants of the "
        "primary analysis (Supplementary Figure S1)."
        % (_int(n_base), _int(len(d)))))

    sel = selection_table()
    parts.append(_sect(
        "9b", "Participants included and excluded", _md(sel),
        "Among the eligible adults of 2007–2018 (Supplementary Table S9). "
        "Excluded participants are shown by the first reason for exclusion in the "
        "order of Supplementary Figure S1: body mass index not measured (the "
        "examination was not attended or not completed), alcohol use unknown, and "
        "any other habit or covariate missing. Deaths per 1000 person-years are "
        "crude."))

    # tables in numerical order (S1, S1b, ..., S10, S10b), each landscape
    # marker kept with its own table
    chunks, cur = [], []
    for p_ in parts:
        if p_ == "<!-- LANDSCAPE on -->":
            cur = [p_]
        elif p_ == "<!-- LANDSCAPE off -->":
            chunks[-1].append(p_)
        else:
            chunks.append(cur + [p_])
            cur = []

    def _key(chunk):
        m_ = re.search(r"Supplementary Table S(\d+)([a-z]?)\.", "\n".join(chunk))
        return (int(m_.group(1)), m_.group(2)) if m_ else (-1, "")
    parts = [p_ for ch in sorted(chunks, key=_key) for p_ in ch]
    open(os.path.join(SUB, "supplementary_tables.md"), "w", encoding="utf-8",
         newline="\n").write("\n".join(parts) + "\n")


def _p4b(v):
    """A Wald P of Table S4b (the convention of _pv2)."""
    return _pv2(v)


def _s4b_rows(v, h, modifier, ci3):
    """Rows of the second panel of Table S4b: the fixed contrast of habit h
    with the joint test, and for physical activity every other contrast."""
    rows = [{"Modifier": modifier, "Habit": _lab(h), "Contrast": v["level_name"],
             "HR, without (95% CI)": ci3(v["hr0"], v["hr0_lo"], v["hr0_hi"]),
             "HR, with (95% CI)": ci3(v["hr1"], v["hr1_lo"], v["hr1_hi"]),
             "*P*, contrast": _p4b(v["p_level"]),
             "*P*, all levels (df)": "%s (%d)" % (_p4b(v["p_joint"]), v["df"])}]
    if h == "pa":
        for c in v.get("by_level", []):
            if c["level_name"] == v["level_name"]:
                continue
            rows.append({"Modifier": "", "Habit": "", "Contrast": c["level_name"],
                         "HR, without (95% CI)": ci3(c["hr0"], c["hr0_lo"], c["hr0_hi"]),
                         "HR, with (95% CI)": ci3(c["hr1"], c["hr1_lo"], c["hr1_hi"]),
                         "*P*, contrast": _p4b(c["p"]), "*P*, all levels (df)": ""})
    return rows


def _s4b_pa_rows(t, it, order, ci3):
    """The first panel of Table S4b with every contrast of physical activity
    under its fixed one (the joint test covers them all)."""
    if "by_level" not in it.get("pa", {}):
        return t
    out = []
    for i, h in enumerate(order):
        out.append(t.iloc[i].to_dict())
        if h != "pa":
            continue
        for c in it["pa"]["by_level"]:
            if c["level_name"] == it["pa"]["level_name"]:
                continue
            out.append({"Habit": "", "Contrast": c["level_name"],
                        "HR, healthy (95% CI)": ci3(c["hr0"], c["hr0_lo"], c["hr0_hi"]),
                        "HR, ill (95% CI)": ci3(c["hr1"], c["hr1_lo"], c["hr1_hi"]),
                        "Ratio, healthy/ill (95% CI)": ci3(1 / c["ratio"], 1 / c["ratio_hi"],
                                                           1 / c["ratio_lo"]),
                        "*P*, contrast": _p4b(c["p"]), "*P*, all levels (df)": ""})
    return pd.DataFrame(out)


def _nc(design):
    """output/null_check.json for one design (PRIMARY or RAW), or {}."""
    p = os.path.join(OUT, "null_check.json")
    return (json.load(open(p, encoding="utf-8")).get(design, {})
            if os.path.exists(p) else {})


def _nc_pct(design, key):
    v = _nc(design).get(key)
    return "?" if v is None else "%d" % _pct_hu(v)


def _extra():
    """output/sensitivity_extra.json, or {}."""
    p = os.path.join(OUT, "sensitivity_extra.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def _pair_deff_txt():
    """The observed-deaths SE ratio of the paired differences (range), that of
    smoking against sleep, and the latter's P with the SE inflated by it."""
    p = os.path.join(OUT, "pairwise.csv")
    if not os.path.exists(p):
        return "?", "?", "?"
    d_ = pd.read_csv(p)
    d_ = d_[(d_.spec == "PRIMARY") & (d_.scale == "calibrated")]
    if "deff_plain_diff" not in d_:
        return "?", "?", "?"
    ss = d_[d_.habit_a.isin(["smoke", "sleep"]) & d_.habit_b.isin(["smoke", "sleep"])].iloc[0]
    return ("%.2f–%.2f" % (d_.deff_plain_diff.min(), d_.deff_plain_diff.max()),
            "%.2f" % ss.deff_plain_diff, _pv2(ss.p_wald_deff))


def _deff_max():
    """Largest SE ratio (design factor) of the plug-in with the observed deaths
    kept."""
    p = os.path.join(OUT, "smoothing.csv")
    return float(pd.read_csv(p).deff_plain.max()) if os.path.exists(p) else float("nan")


def _calcheck_absmax():
    """Largest absolute relative error (%) of the calibrated estimate in the
    semi-parametric check, over both population models."""
    p = os.path.join(OUT, "calibration_check.csv")
    if not os.path.exists(p):
        return 0
    return int(round(100 * float(pd.read_csv(p).calibrated_rel_err.abs().max())))


def _design(step, key, default):
    """A replicate count or other design constant recorded by an analysis step
    (output/params_<step>.json)."""
    p = os.path.join(OUT, "params_%s.json" % step)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8")).get(key, default)
    return default


def _rho(a, b):
    ra, rb = pd.Series(list(a)).rank(), pd.Series(list(b)).rank()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def _numbers(d, m, hr, rk, diff):
    """Every quantity the text quotes, in one file (output/numbers.json)."""
    L = C.LABEL
    by = {s: m[m.spec == s].set_index("habit")
          for s in ("total", "AGE", "DEMO", "PRIMARY", "RAW") if (m.spec == s).any()}
    pri, tot, dem, raw = by["PRIMARY"], by["total"], by["DEMO"], by["RAW"]
    order = pri.sort_values("calibrated", ascending=False).index.tolist()
    r4 = lambda v: round(float(v), 4)                               # noqa: E731
    N = dict(
        n_analytic=int(len(d)), n_deaths=int(d.died.sum()),
        person_years=float(d.py.sum()), median_fu_months=float(d.permth.median()),
        fu_iqr=[float(d.permth.quantile(.25)), float(d.permth.quantile(.75))],
        deaths_per_1000py=round(1000 * float(d.died.sum()) / float(d.py.sum()), 1),
        mean_age=round(float(d.age.mean()), 1), sd_age=round(float(d.age.std()), 1),
        pct_female=round(100 * float((d.sex_b == 1).mean()), 1),
        n_age_topcoded=int((d.age == 80).sum()),
        pct_age_topcoded=round(100 * float((d.age == 80).mean()), 1),
        deaths_age_topcoded=int(d[d.age == 80].died.sum()),
        pct_deaths_age_topcoded=round(100 * float(d[d.age == 80].died.sum())
                                      / float(d.died.sum()), 1),
        pct_pir_missing=round(100 * float((d.pir == 3).mean()), 1),
        ranking=[L[h] for h in order],
        calibrated={s: {L[h]: r4(by[s].loc[h, "calibrated"]) for h in order}
                    for s in by},
        calibrated_se={L[h]: r4(pri.loc[h, "calibrated_se"]) for h in order},
        excess={s: {L[h]: r4(by[s].loc[h, "excess"]) for h in order} for s in by},
        excess_se={L[h]: r4(pri.loc[h, "excess_se"]) for h in order},
        ranking_excess=[L[h] for h in pri.sort_values("excess", ascending=False).index],
        p_null={L[h]: round(float(pri.loc[h, "p_null"]), 3) for h in order},
        share_of_total={L[h]: int(round(100 * pri.loc[h, "calibrated"]
                                        / tot.loc[h, "calibrated"])) for h in order},
        share_of_total_excess={L[h]: int(round(100 * pri.loc[h, "excess"]
                                               / tot.loc[h, "excess"])) for h in order},
        primary_strata=int(pri.n_strata.max()), demo_strata=int(dem.n_strata.max()),
        primary_per_stratum=round(len(d) / float(pri.n_strata.max())),
        primary_per_cell=round(len(d) / float(pri.n_strata.max()) / 8.0),
        se_range=[r4(pri.calibrated_se.min()), r4(pri.calibrated_se.max())],
        shift_range=[round(float(pri.calibrated_shift_in_sds.min()), 2),
                     round(float(pri.calibrated_shift_in_sds.max()), 2)],
        boot_world={L[h]: r4(pri.loc[h, "boot_world"]) for h in order},
        boot_world_absdiff_max=r4((pri.boot_world - pri.calibrated).abs().max()),
        real_gaps=[int(round(100 * (pri.loc[order[i], "calibrated"]
                                    - pri.loc[order[i + 1], "calibrated"])
                             / pri.loc[order[i], "calibrated"])) for i in range(2)],
        raw_strata=int(raw.n_strata.iloc[0]),
        raw_mean_per_stratum=round(len(d) / float(raw.n_strata.iloc[0]), 1),
        raw_plugin=[round(float(raw.rcmi.min()), 3), round(float(raw.rcmi.max()), 3)],
        raw_null_share=[int(round(100 * float(raw.null_share.min()))),
                        int(round(100 * float(raw.null_share.max())))],
        raw_null_share_js=[int(round(100 * float(raw.null_share_js.min()))),
                           int(round(100 * float(raw.null_share_js.max())))],
        raw_excess=[r4(raw.excess.min()), r4(raw.excess.max())],
        raw_calibrated=[r4(raw.calibrated.min()), r4(raw.calibrated.max())],
        raw_over_primary=[round(float((raw.calibrated / pri.calibrated).min()), 1),
                          round(float((raw.calibrated / pri.calibrated).max()), 1)],
        raw_ranking_excess=[L[h] for h in raw.sort_values("excess", ascending=False).index],
        raw_ranking_calibrated=[L[h] for h in
                                raw.sort_values("calibrated", ascending=False).index],
        rho_raw_vs_primary=round(_rho(raw.loc[order, "calibrated"],
                                      pri.loc[order, "calibrated"]), 2),
        rho_raw_vs_primary_excess=round(_rho(raw.loc[order, "excess"],
                                             pri.loc[order, "excess"]), 2),
        alphabet_bound=round(C.cmax(2), 4),
        achievable_bound=[r4(m.bound_achievable.min()), r4(m.bound_achievable.max())],
        family_agreement=dict(
            rho_domi=round(_rho(pri.loc[order, "calibrated"], pri.loc[order, "domi"]), 2),
            rho_race=round(_rho(pri.loc[order, "calibrated"], pri.loc[order, "race"]), 2)),
        race_max=dict(habit=L[pri.race.idxmax()], value=round(float(pri.race.max()), 3)),
        race_max_all=dict(habit=L[m.loc[m.race.idxmax(), "habit"]],
                          spec=str(m.loc[m.race.idxmax(), "spec"]),
                          value=round(float(m.race.max()), 3)),
        n_bmi_underweight=int((d.bmi == 0).sum()))

    g = d.groupby("cycle").agg(n=("died", "size"), deaths=("died", "sum"),
                               py=("py", "sum"), medfu=("permth", "median"))
    g["risk"], g["rate"] = g.deaths / g.n, 1000 * g.deaths / g.py
    N.update(cycle_medfu=[int(round(g.medfu.max())), int(round(g.medfu.min()))],
             cycle_risk_fold=round(float(g.risk.max() / g.risk.min()), 1),
             cycle_rate_fold=round(float(g.rate.max() / g.rate.min()), 1),
             pct_current_smokers=int(round(100 * float((d.smoke == 2).mean()))),
             pct_below_activity=int(round(100 * float(d.pa.isin([0, 1]).mean()))))
    raw_all = pd.read_pickle(C.CLEAN_PKL)
    el = raw_all[raw_all.adult_ok]
    N.update(n_pooled_records=int(len(raw_all)), n_eligible_all_cycles=int(len(el)),
             n_eligible_2007_2018=int(el.cycle.isin(C.PA_CYCLES).sum()),
             n_deaths_eligible=int(el.died.sum()),
             death_pct_eligible_2007_2018=round(100 * float(
                 el[el.cycle.isin(C.PA_CYCLES)].died.mean()), 1),
             death_pct_analytic=round(100 * float(d.died.mean()), 1))

    # ---- paired comparisons
    N["pairs"] = {}
    for (s, sc), blk in diff.groupby(["spec", "scale"]):
        key = "%s/%s" % (s, sc)
        N["pairs"][key] = {}
        for r in blk.itertuples():
            a, b = (r.habit_a, r.habit_b)
            sign = 1.0
            if order.index(a) > order.index(b):          # report as leader minus other
                a, b, sign = b, a, -1.0
            lo, hi = sorted([sign * r.wald_lo, sign * r.wald_hi])
            N["pairs"][key]["%s vs %s" % (L[a], L[b])] = dict(
                diff=r4(sign * r.diff), se=r4(r.se), ci=[r4(lo), r4(hi)],
                p=float("%.3g" % r.p_wald),
                p_boot=(round(float(r.p_boot), 3) if "p_boot" in blk
                        and pd.notna(r.p_boot) else None),
                basic=(sorted([r4(sign * r.basic_lo), r4(sign * r.basic_hi)])
                       if "basic_lo" in blk and pd.notna(r.basic_lo) else None),
                deff=(round(float(r.deff_plain_diff), 2) if "deff_plain_diff" in blk
                      and pd.notna(r.deff_plain_diff) else None),
                p_deff=(float("%.3g" % r.p_wald_deff) if "p_wald_deff" in blk
                        and pd.notna(r.p_wald_deff) else None),
                sep_wald=bool(r.sep_wald),
                sep_pct=bool(r.sep_pct), sep_basic=bool(r.sep_basic))
        if "deff_plain_diff" in blk and blk.deff_plain_diff.notna().any():
            N["pair_deff"] = [round(float(blk.deff_plain_diff.min()), 2),
                              round(float(blk.deff_plain_diff.max()), 2)]
        N["pairs"][key + "/n_separable"] = dict(
            wald=int(blk.sep_wald.sum()), pct=int(blk.sep_pct.sum()),
            basic=int(blk.sep_basic.sum()), total=int(len(blk)))

    # ---- hazard ratios and standardised risks
    N["hazard_ratios"] = {}
    for h in C.HABITS:
        blk = hr[hr.habit == h]
        N["hazard_ratios"][L[h]] = {
            str(r.level_name): [round(float(r.hr), 2), round(float(r.hr_lo), 2),
                                round(float(r.hr_hi), 2)]
            for r in blk.itertuples() if not r.reference}
        N["hazard_ratios"][L[h]]["reference"] = str(blk[blk.reference].level_name.iloc[0])
    N["standardised_10y"] = {
        L[h]: {str(r.level_name): [round(float(r.risk10), 1), round(float(r.rd10), 1)]
               for r in rk[rk.habit == h].itertuples()} for h in C.HABITS}

    p = os.path.join(OUT, "ph_check.csv")
    if os.path.exists(p):
        ph = pd.read_csv(p)
        gl = ph[ph.level == "GLOBAL"]
        N["proportional_hazards"] = {str(r.label): dict(chi2=round(float(r.chi2), 2),
                                                        df=int(r.df),
                                                        p=round(float(r.p), 4))
                                     for r in gl.itertuples()}

    p = os.path.join(OUT, "sensitivity_rcmi.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        piv = s.pivot_table(index="label", columns="analysis", values="calibrated")
        N["sensitivity_calibrated"] = {c: {k: r4(v) for k, v in piv[c].items()}
                                       for c in piv.columns}
        N["sensitivity_ranking"] = {c: list(piv[c].sort_values(ascending=False).index)
                                    for c in piv.columns}
        pp_ = s.pivot_table(index="label", columns="analysis",
                            values="p_perm" if "p_perm" in s else "p_null")
        N["sensitivity_p"] = {c: {k: round(float(v), 3) for k, v in pp_[c].items()}
                              for c in pp_.columns}
        N["sensitivity_n"] = {c: sorted({int(v) for v in s[s.analysis == c].n})
                              for c in piv.columns}
    p = os.path.join(OUT, "sensitivity_hr.csv")
    if os.path.exists(p):
        sv = pd.read_csv(p)
        N["sensitivity_hr"] = {}
        for r in sv.itertuples():
            N["sensitivity_hr"].setdefault(r.analysis, {})["%s: %s" % (r.label, r.contrast)] = (
                [round(float(r.hr), 2)] + ([] if pd.isna(getattr(r, "hr_lo", np.nan))
                                           else [round(float(r.hr_lo), 2),
                                                 round(float(r.hr_hi), 2)]))
        # ranges of the (fixed) headline contrast of each exposure: over the
        # sensitivity analyses that keep the whole cohort, and over all of them
        # (SA4 is BMI in never-smokers and is reported separately)
        sa = sv[~sv.analysis.str.startswith("SA4") & (sv.analysis != "primary")]
        n_all = int(len(d))

        def rng_(g):
            return [round(float(g.hr.min()), 2), round(float(g.hr.max()), 2), int(len(g)),
                    str(g.loc[g.hr.idxmin(), "analysis"]).split(" ")[0],
                    str(g.loc[g.hr.idxmax(), "analysis"]).split(" ")[0]]
        N["sensitivity_hr_range"] = {"%s: %s" % k: rng_(g)
                                     for k, g in sa.groupby(["label", "contrast"])}
        N["sensitivity_hr_range_whole"] = {
            "%s: %s" % k: rng_(g) for k, g in sa[sa.n == n_all].groupby(["label", "contrast"])}

    p = os.path.join(OUT, "biasfloor.csv")
    if os.path.exists(p):
        bf = pd.read_csv(p)
        N["null_floor_curve"] = {
            L[h]: dict(first=int(round(100 * float(b.sort_values("k_strata").null_share.iloc[0]))),
                       peak=int(round(100 * float(b.null_share.max()))),
                       peak_k=int(b.loc[b.null_share.idxmax(), "k_strata"]),
                       tail=[int(round(100 * float(b[b.k_strata >= 500].null_share.min()))),
                             int(round(100 * float(b[b.k_strata >= 500].null_share.max())))],
                       tail_k=[500, int(b.k_strata.max())],
                       last=int(round(100 * float(b.sort_values("k_strata").null_share.iloc[-1]))),
                       calibrated_to_200=[round(float(b[b.k_strata <= 200].calibrated.min()), 3),
                                          round(float(b[b.k_strata <= 200].calibrated.max()), 3)],
                       excess=[r4(b.sort_values("k_strata").excess.iloc[0]),
                               r4(b.excess.min())])
            for h, b in bf.groupby("habit")}

    p = os.path.join(OUT, "simulation_absolute.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        a = s[s.b > 0]
        dense, sparse = a[a.mean_per_cell >= 18], a[a.mean_per_cell < 4]
        rng_ = lambda v: [int(round(100 * float(v.min()))),              # noqa: E731
                          int(round(100 * float(v.max())))]
        N["simulation_abs"] = dict(
            dense_excess_err=rng_(dense.excess_rel_err),
            dense_calibrated_err=rng_(dense.calibrated_rel_err),
            sparse_excess_err=rng_(sparse.excess_rel_err),
            sparse_calibrated_err=rng_(sparse.calibrated_rel_err),
            plugin_err_max=int(round(100 * float(a.plugin_rel_err.max()))),
            null_excess=[r4(s[s.b == 0].excess.min()), r4(s[s.b == 0].excess.max())],
            null_calibrated_dense=[r4(s[(s.b == 0) & (s.mean_per_cell >= 18)].calibrated.min()),
                                   r4(s[(s.b == 0) & (s.mean_per_cell >= 18)].calibrated.max())],
            dense_cut=18, sparse_cut=4)
    p = os.path.join(OUT, "simulation_rank.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        sp, de = s[s.mean_per_cell < 4], s[s.mean_per_cell >= 18]
        pr = lambda v: [int(round(100 * float(v.min()))),               # noqa: E731
                        int(round(100 * float(v.max())))]
        rr = lambda v: [round(float(v.min()), 2), round(float(v.max()), 2)]  # noqa: E731
        N["simulation_rank"] = {
            e: dict(sparse_pair=pr(sp["top2_%s" % e]), dense_pair=pr(de["top2_%s" % e]),
                    sparse_rho=rr(sp["rho_%s" % e]), dense_rho=rr(de["rho_%s" % e]),
                    top1=pr(s["top1_%s" % e]), all_pair=pr(s["top2_%s" % e]),
                    all_rho=rr(s["rho_%s" % e]))
            for e in ("plugin", "excess", "calibrated")}
    q = os.path.join(OUT, "simulation_truths.csv")
    if os.path.exists(q):
        t = pd.read_csv(q)
        N["simulation_truths"] = {int(r.K): dict(gap_1_2=float(r.gap_1_2),
                                                 gap_2_3=float(r.gap_2_3))
                                  for r in t.itertuples()}
        N["simulation_gaps"] = dict(
            gap_1_2=[int(round(100 * t.gap_1_2.min())), int(round(100 * t.gap_1_2.max()))],
            gap_2_3=[int(round(100 * t.gap_2_3.min())), int(round(100 * t.gap_2_3.max()))])
    pc = lambda v: [int(round(100 * float(v.min()))),                  # noqa: E731
                    int(round(100 * float(v.max())))]
    r2 = lambda v: [round(float(v.min()), 2), round(float(v.max()), 2)]  # noqa: E731
    p = os.path.join(OUT, "calibration_check.csv")
    if os.path.exists(p):
        N["calcheck"] = {}
        ccall = pd.read_csv(p)
        cpall = pd.read_csv(os.path.join(OUT, "calibration_check_pairs.csv"))
        rkall = pd.read_csv(os.path.join(OUT, "calibration_check_rank.csv"))
        for scen, cc in ccall.groupby("scenario"):
            cc = cc.set_index("habit")
            cp = cpall[cpall.scenario == scen]
            N["calcheck"][scen] = dict(
                calibrated_err=pc(cc.calibrated_rel_err),
                calibrated_abs_err_max=int(round(100 * float(cc.calibrated_rel_err.abs().max()))),
                calibrated_fixed_err=pc(cc.calibrated_fixed_rel_err),
                excess_err=pc(cc.excess_rel_err),
                smooth_se_ratio=r2(cc.smooth_se_ratio), plain_se_ratio=r2(cc.plain_se_ratio),
                smooth_se_ratio_mcse_max=round(float(cc.smooth_se_ratio_mcse.max()), 2),
                plain_se_shortfall=pc(1 - cc.plain_se_ratio),
                smooth_shift_in_sds=r2(cc.smooth_shift_in_sds),
                wald_coverage=r2(cc.wald_coverage),
                wald_coverage_pct=[_pct_hu(cc.wald_coverage.min()), _pct_hu(cc.wald_coverage.max())],
                wald_coverage_mcse_max=round(float(cc.wald_coverage_mcse.max()), 3),
                pair_smooth_se_ratio=r2(cp.smooth_se_ratio),
                pair_plain_se_ratio=r2(cp.plain_se_ratio),
                pair_plain_se_shortfall=pc(1 - cp.plain_se_ratio),
                pair_wald_coverage=r2(cp.wald_coverage),
                pair_wald_coverage_pct=[_pct_hu(cp.wald_coverage.min()),
                                        _pct_hu(cp.wald_coverage.max())],
                per_habit={L[h]: dict(truth=r4(r.truth), calibrated=r4(r.calibrated),
                                      calibrated_err=int(round(100 * r.calibrated_rel_err)),
                                      smooth_se_ratio=round(float(r.smooth_se_ratio), 2),
                                      plain_se_ratio=round(float(r.plain_se_ratio), 2),
                                      coverage=round(float(r.wald_coverage), 2))
                           for h, r in cc.iterrows()},
                rank={r.estimator: dict(
                    top1=int(round(100 * r.top1)), top2=int(round(100 * r.top2)),
                    rho=round(float(r.rho), 2),
                    true_order=[L[h] for h in r.true_order.split(" > ")],
                    order_of_means=[L[h] for h in r.order_of_means.split(" > ")])
                    for r in rkall[rkall.scenario == scen].itertuples()})

    # ---- ranking probabilities, direct/total ratios, refinement diagnostic
    p = os.path.join(OUT, "ranks.csv")
    if os.path.exists(p):
        N["rank_probability"] = {
            L[r.habit]: dict(first=round(float(r.p_first), 2), top2=round(float(r.p_top2), 2),
                             last=round(float(r.p_last), 2),
                             first_pct=int(round(100 * r.p_first)),
                             top2_pct=int(round(100 * r.p_top2)),
                             rank_range=[int(r.rank_lo), int(r.rank_hi)])
            for r in pd.read_csv(p).itertuples()}
    p = os.path.join(OUT, "ratios.csv")
    if os.path.exists(p):
        N["ratio"] = {}
        for r in pd.read_csv(p).itertuples():
            N["ratio"].setdefault(L[r.habit], {})[r.scale] = dict(
                ratio=round(float(r.ratio), 2), ci=[round(float(r.lo), 2),
                                                    round(float(r.hi), 2)],
                pct=int(round(100 * r.ratio)))
    p = os.path.join(OUT, "refinement.csv")
    if os.path.exists(p):
        rf = pd.read_csv(p)
        N["refinement"] = {}
        for h, g in rf.groupby("habit"):
            mc = float(g.calibrated.std(ddof=1) / np.sqrt(len(g)))
            chg = float(g.calibrated.mean() - pri.loc[h, "calibrated"])
            N["refinement"][L[h]] = dict(
                mean=r4(g.calibrated.mean()), sd=r4(g.calibrated.std(ddof=1)),
                mcse=r4(mc), change=r4(chg), change_in_mcse=round(chg / mc, 1),
                djs_pct=int(round(100 * ((g.calibrated ** 2).mean()
                                         / pri.loc[h, "calibrated"] ** 2 - 1))))
        ch = [v["change"] for v in N["refinement"].values()]
        N["refinement_change_range"] = [min(ch), max(ch)]
        rel = [100 * (g.calibrated.mean() / pri.loc[h, "calibrated"] - 1)
               for h, g in rf.groupby("habit")]
        N["refinement_rel_range"] = [int(round(min(rel))), int(round(max(rel)))]
        N["refinement_splits"] = int(rf.split.nunique())
        beyond = [h for h in order if N["refinement"][L[h]]["change_in_mcse"] > 2]
        within = [h for h in order if h not in beyond]
        N["refinement_beyond_mc"] = [L[h] for h in beyond]
        N["refinement_within_mc"] = [L[h] for h in within]
        if beyond:
            cb = [N["refinement"][L[h]]["change"] for h in beyond]
            N["refinement_beyond_range"] = [min(cb), max(cb)]
        if within:
            cw = [N["refinement"][L[h]]["change"] for h in within]
            N["refinement_within_range"] = [min(cw), max(cw)]
    if "calibrated_se_psu" in pri:
        N["calibrated_se_psu"] = {L[h]: r4(pri.loc[h, "calibrated_se_psu"]) for h in order}
        N["se_psu_over_individual"] = r2(pri.calibrated_se_psu / pri.calibrated_se)
    N["calibrated_se_total"] = {L[h]: r4(tot.loc[h, "calibrated_se"]) for h in order
                                if "calibrated_se" in tot and pd.notna(tot.loc[h, "calibrated_se"])}
    N["calibrated_se_demo"] = {L[h]: r4(dem.loc[h, "calibrated_se"]) for h in order
                               if "calibrated_se" in dem and pd.notna(dem.loc[h, "calibrated_se"])}
    N["deaths_per_cell"] = {L[h]: round(float(pri.loc[h, "deaths_per_cell"]), 1)
                            for h in order if "deaths_per_cell" in pri}
    p = os.path.join(OUT, "cox_by_age.csv")
    if os.path.exists(p):
        N["hr_by_age"] = {
            "%s: %s" % (L[r.habit], r.level): dict(
                young=[round(float(r.hr_young), 2), round(float(r.lo_young), 2),
                       round(float(r.hi_young), 2)],
                old=[round(float(r.hr_old), 2), round(float(r.lo_old), 2),
                     round(float(r.hi_old), 2)],
                p=float("%.3g" % r.p_interaction))
            for r in pd.read_csv(p).itertuples()}
        # the age-split contrasts are tested together: Bonferroni threshold
        N["hr_by_age_bonferroni"] = dict(
            contrasts=len(N["hr_by_age"]),
            threshold=float("%.2g" % (0.05 / len(N["hr_by_age"]))))
    p = os.path.join(OUT, "ph_check.csv")
    if os.path.exists(p):
        phc = pd.read_csv(p)
        N["ph_contrast_p"] = {"%s: %s" % (r.label, r.level): float("%.3g" % r.p)
                              for r in phc[phc.level != "GLOBAL"].itertuples()}
    # how far survey period was controlled inside a demographic score alone (the
    # previous primary set, SA12): share of participants in their stratum's
    # modal period
    demo12 = C.demographic_strata(d, C.crossfit_score(d))
    ct = pd.crosstab(demo12, d["cyc5"].values)
    N["sa12_modal_period_pct"] = int(round(100 * float(ct.max(axis=1).sum()) / float(len(d))))
    N["deaths_per_cell_nominal"] = [int(round(float(d.died.sum()) / (54 * 5))),
                                    int(round(float(d.died.sum()) / (54 * 3)))]

    # the included/excluded comparison (Supplementary Table S9b), as numbers
    sel = selection_table()

    def nums(t):
        return [float(v) for v in re.findall(r"\d+(?:\.\d+)?", str(t).replace(" ", ""))]
    N["selection"] = {col: {ch: nums(v) for ch, v in zip(sel.Characteristic, sel[col])}
                      for col in sel.columns if col != "Characteristic"}

    # ---- facts quoted in the Supplementary Methods
    raw_x = pd.read_pickle(C.RAW_PKL)
    N["zero_answers"] = {c: int((raw_x[c].abs() <= 1e-30).sum())
                         for c in ("ALQ120Q", "ALQ121") if c in raw_x}
    N["per_stratum_at_200"] = int(round(len(d) / 200.0))
    p = os.path.join(OUT, "simulation_absolute.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        N["simulation_cells"] = [round(float(s.mean_per_cell.min()), 1),
                                 int(round(float(s.mean_per_cell.max())))]
        N["simulation_deaths_per_cell"] = [round(float(s.deaths_per_cell.min()), 1),
                                           int(round(float(s.deaths_per_cell.max())))]
        # recovery by sparsity band (expected deaths per exposure-by-stratum
        # cell) and effect size; Monte-Carlo SEs of the relative errors
        a = s[s.b > 0]
        bands = {"dense": a.deaths_per_cell >= 10,
                 "middle": (a.deaths_per_cell >= 2) & (a.deaths_per_cell < 10),
                 "sparse": a.deaths_per_cell < 2}
        N["simulation_band"] = {}
        for band, mask in bands.items():
            for lab, bm in (("weak", a.b == 0.25), ("strong", a.b >= 0.5), ("all", a.b > 0)):
                g = a[mask & bm]
                if not len(g):
                    continue
                N["simulation_band"].setdefault(band, {})[lab] = dict(
                    calibrated_err=pc(g.calibrated_rel_err), excess_err=pc(g.excess_rel_err),
                    plugin_err_max=int(round(100 * float(g.plugin_rel_err.max()))),
                    calibrated_mcse_max=int(round(100 * float(g.calibrated_rel_err_mcse.max()))),
                    deaths_per_cell=[round(float(g.deaths_per_cell.min()), 1),
                                     round(float(g.deaths_per_cell.max()), 1)])
        z0 = s[s.b == 0]
        N["simulation_null_calibrated"] = [r4(z0.calibrated.min()), r4(z0.calibrated.max())]
    p = os.path.join(OUT, "simulation_rank.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        N["simulation_rank_all"] = {
            e: dict(top1=pc(s["top1_" + e]), top2=pc(s["top2_" + e]),
                    rho=r2(s["rho_" + e]),
                    top2_mcse_max=int(round(100 * float(s["top2_%s_mcse" % e].max()))))
            for e in ("plugin", "excess", "calibrated")}
    p = os.path.join(OUT, "ph_check.csv")
    if os.path.exists(p):
        ph = pd.read_csv(p)
        N["ph_z"] = {"%s: %s" % (r.label, r.level): round(float(r.z), 2)
                     for r in ph[ph.level != "GLOBAL"].itertuples()}
    man = C.manifest_path()
    if os.path.exists(man):
        mf = pd.read_csv(man)
        col = mf.columns[0]
        N["data_files"] = dict(total=int(len(mf)),
                               xpt=int(mf[col].astype(str).str.lower().str.endswith(".xpt").sum()),
                               lmf=int(mf[col].astype(str).str.lower().str.endswith(".dat").sum()))
    # every number printed by the self-tests (python lsm_core.py test), so that
    # Supplementary Methods S4 can be checked against the actual test output
    p = os.path.join(OUT, "log_selftest.txt")
    if os.path.exists(p):
        txt = open(p, encoding="utf-8").read()
        N["selftest"] = dict(
            checks=len(re.findall(r"^\s*\[(?:PASS|FAIL)\]", txt, re.M)),
            passed=len(re.findall(r"^\s*\[PASS\]", txt, re.M)),
            values=sorted({float(v) for v in re.findall(
                r"-?\d+(?:\.\d+)?", re.sub(r"(?<=\d) (?=\d{3}\b)", "", txt))}))
    # ---- round-2 additions: distribution of deaths per cell, the smoothing
    # model, the refinement bias, sensitivity-analysis precision, the SA15 null,
    # the SA16 weights, SA17a's sparsity and prevalence-matched reference
    if "cells_lt2" in pri:
        N["cell_deaths"] = {L[h]: dict(median=float(pri.loc[h, "cell_deaths_median"]),
                                       p10=float(pri.loc[h, "cell_deaths_p10"]),
                                       p90=round(float(pri.loc[h, "cell_deaths_p90"]), 1),
                                       lt2=_pct_hu(pri.loc[h, "cells_lt2"]),
                                       lt5=_pct_hu(pri.loc[h, "cells_lt5"]))
                            for h in order}
        N["cell_deaths_range"] = dict(
            median=[float(pri.cell_deaths_median.min()), float(pri.cell_deaths_median.max())],
            lt2=[_pct_hu(pri.cells_lt2.min()), _pct_hu(pri.cells_lt2.max())],
            lt5=[_pct_hu(pri.cells_lt5.min()), _pct_hu(pri.cells_lt5.max())])
    gap_se = (pri.boot_world - pri.calibrated) / pri.calibrated_se
    N["boot_world_gap"] = {L[h]: dict(diff=r4(pri.loc[h, "boot_world"] - pri.loc[h, "calibrated"]),
                                      in_se=round(float(gap_se.loc[h]), 1),
                                      pct=int(round(100 * float(pri.loc[h, "boot_world"]
                                                                / pri.loc[h, "calibrated"] - 1))))
                           for h in order}
    N["boot_world_gap_range"] = dict(
        in_se=[round(float(gap_se.min()), 1), round(float(gap_se.max()), 1)],
        absdiff_max=r4((pri.boot_world - pri.calibrated).abs().max()))
    p = os.path.join(OUT, "smoothing.csv")
    if os.path.exists(p):
        sm = pd.read_csv(p).set_index("habit")
        rat = sm.se_interaction / sm.se_additive
        N["smoothing"] = dict(
            lr=round(float(sm.lr.iloc[0]), 1), lr_df=int(sm.lr_df.iloc[0]),
            lr_p=float("%.2g" % sm.lr_p.iloc[0]),
            se_ratio=[round(float(rat.min()), 2), round(float(rat.max()), 2)],
            se_interaction={L[h]: r4(sm.loc[h, "se_interaction"]) for h in order},
            world_interaction={L[h]: r4(sm.loc[h, "world_interaction"]) for h in order},
            deff_plain={L[h]: round(float(sm.loc[h, "deff_plain"]), 2) for h in order},
            deff_plain_range=[round(float(sm.deff_plain.min()), 2),
                              round(float(sm.deff_plain.max()), 2)])
        if "overfit_djs" in sm:
            gap_ = sm.world_interaction ** 2 - sm.world_additive ** 2
            of_ = sm.overfit_djs
            N["smoothing"].update(
                aic_diff=round(float(sm.aic_diff.iloc[0]), 1),
                overfit_e4={L[h]: round(1e4 * float(of_[h]), 2) for h in order},
                overfit_e4_range=[round(1e4 * float(of_.min()), 1),
                                  round(1e4 * float(of_.max()), 1)],
                gap_e4={L[h]: round(1e4 * float(gap_[h]), 2) for h in order},
                overfit_share_pct=[int(round(100 * float((of_ / gap_).min()))),
                                   int(round(100 * float((of_ / gap_).max())))])
    p = os.path.join(OUT, "refinement.csv")
    if os.path.exists(p):
        rf = pd.read_csv(p)
        djs = {h: int(round(100 * ((rf[rf.habit == h].calibrated ** 2).mean()
                                   / pri.loc[h, "calibrated"] ** 2 - 1)))
               for h in order}
        N["refinement_djs_pct"] = {L[h]: djs[h] for h in order}
        N["refinement_djs_range"] = [min(djs.values()), max(djs.values())]
    p = os.path.join(OUT, "sensitivity_rcmi.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        if "calibrated_se" in s:
            N["sensitivity_se"] = {
                a: {r.label: r4(r.calibrated_se) for r in g.itertuples()
                    if pd.notna(r.calibrated_se)}
                for a, g in s.groupby("analysis")}
        if "deaths_per_cell" in s:
            N["sensitivity_deaths_per_cell"] = {
                a: [round(float(g.deaths_per_cell.min()), 1),
                    round(float(g.deaths_per_cell.max()), 1)]
                for a, g in s.groupby("analysis") if g.deaths_per_cell.notna().any()}
            N["sensitivity_cells_lt2"] = {
                a: [_pct_hu(g.cells_lt2.min()), _pct_hu(g.cells_lt2.max())]
                for a, g in s.groupby("analysis") if g.cells_lt2.notna().any()}
        N["sensitivity_deaths"] = {a: sorted({int(v) for v in g.n_deaths})
                                   for a, g in s.groupby("analysis")}
        pv_ = s.pivot_table(index="analysis", columns="habit", values="calibrated")
        if {"smoke", "pa"} <= set(pv_.columns):
            N["sensitivity_smoke_minus_pa"] = {a: r4(v) for a, v in
                                               (pv_["smoke"] - pv_["pa"]).items()}
    if {"chronic", "poor_health"} <= set(d.columns):
        h_ = d[(d.chronic == 0) & (d.poor_health == 0)]
        N["sa17a_counts"] = dict(n=int(len(h_)), deaths=int(h_.died.sum()),
                                 bmi_underweight=int((h_.bmi == 0).sum()),
                                 bmi_underweight_deaths=int(h_[h_.bmi == 0].died.sum()),
                                 death_share_pct=int(round(100 * h_.died.sum() / d.died.sum())))
    p = os.path.join(OUT, "sensitivity_extra.json")
    if os.path.exists(p):
        ex = json.load(open(p, encoding="utf-8"))
        if "sa15_null_check" in ex:
            c_ = ex["sa15_null_check"]
            N["sa15_null_check"] = dict(
                datasets=int(c_["datasets"]), permutations=int(c_["permutations"]),
                reject_05=round(float(c_["reject_05"]), 3),
                reject_05_pct=_pct_hu(c_["reject_05"]),
                reject_05_mcse=round(float(c_["reject_05_mcse"]), 3),
                mean_p=round(float(c_["mean_p"]), 2), n_eff=int(round(c_["n_eff"])))
        if "sa16_null_check" in ex:
            c_ = ex["sa16_null_check"]
            N["sa16_null_check"] = dict(
                datasets=int(c_["datasets"]), permutations=int(c_["permutations"]),
                reject_05_pct=_pct_hu(c_["reject_05"]),
                reject_05_mcse=round(float(c_["reject_05_mcse"]), 3),
                reject_05_by_habit_pct={L[h]: _pct_hu(v)
                                        for h, v in c_["reject_05_by_habit"].items()},
                mean_p=round(float(c_["mean_p"]), 2))
        if "sa19_interaction" in ex:
            N["sa19_interaction"] = {}
            for h, v in ex["sa19_interaction"].items():
                N["sa19_interaction"][L[h]] = dict(
                    contrast=v["level_name"],
                    healthy=[round(v["hr0"], 2), round(v["hr0_lo"], 2), round(v["hr0_hi"], 2)],
                    ill=[round(v["hr1"], 2), round(v["hr1_lo"], 2), round(v["hr1_hi"], 2)],
                    p=float("%.3g" % v["p_level"]), p_joint=float("%.3g" % v["p_joint"]),
                    df=int(v["df"]))
        if "sa19_levels" in ex:
            s_ = ex["sa19_levels"]
            N["sa19_levels"] = dict(
                pct_active_no_leisure=int(round(s_["pct_active_no_leisure"])),
                pct_none=round(float(s_["pct_by_level"]["0"]), 1))
        if "sa16_weights" in ex:
            w_ = ex["sa16_weights"]
            N["sa16_weights"] = {k: (int(v) if k.startswith("n_") else round(float(v), 2))
                                 for k, v in w_.items()}
        if "sa17a_refinement" in ex and "sensitivity_calibrated" in N:
            base17 = N["sensitivity_calibrated"].get("SA17a free of prevalent disease", {})
            mc17 = ex.get("sa17a_refinement_mcse", {})
            N["sa17a_refinement"] = {
                L[h]: dict(split=r4(v), change=r4(v - base17.get(L[h], np.nan)),
                           mcse=r4(mc17.get(h, np.nan)))
                for h, v in ex["sa17a_refinement"].items() if L[h] in base17}
            ch17 = [v["change"] for v in N["sa17a_refinement"].values()]
            N["sa17a_refinement_range"] = [min(ch17), max(ch17)]
        if "sa17_interaction" in ex:
            N["sa17_interaction"] = {}
            for h, v in ex["sa17_interaction"].items():
                N["sa17_interaction"][L[h]] = dict(
                    contrast=v["level_name"],
                    healthy=[round(v["hr0"], 2), round(v["hr0_lo"], 2), round(v["hr0_hi"], 2)],
                    ill=[round(v["hr1"], 2), round(v["hr1_lo"], 2), round(v["hr1_hi"], 2)],
                    ratio_healthy_ill=[round(1 / v["ratio"], 2), round(1 / v["ratio_hi"], 2),
                                       round(1 / v["ratio_lo"], 2)],
                    p=float("%.3g" % v["p_level"]), p_joint=float("%.3g" % v["p_joint"]),
                    df=int(v["df"]), n=[int(v["n0"]), int(v["n1"])],
                    deaths=[int(v["deaths0"]), int(v["deaths1"])],
                    rd_pct=([round(100 * v["rd0"], 1), round(100 * v["rd1"], 1)]
                            if "rd0" in v else None))
            if "sa17_interaction_proportional" in ex:
                N["sa17_interaction_proportional"] = {}
                for h, v in ex["sa17_interaction_proportional"].items():
                    N["sa17_interaction_proportional"][L[h]] = dict(
                        healthy=[round(v["hr0"], 2), round(v["hr0_lo"], 2),
                                 round(v["hr0_hi"], 2)],
                        ill=[round(v["hr1"], 2), round(v["hr1_lo"], 2), round(v["hr1_hi"], 2)],
                        p=float("%.3g" % v["p_level"]),
                        p_joint=float("%.3g" % v["p_joint"]),
                        ph_ill_p=float("%.2g" % v["ph_mod_p"]))
                N["sa17_ill_ph_p_range"] = [
                    float("%.2g" % min(v["ph_mod_p"] for v in
                                       ex["sa17_interaction_proportional"].values())),
                    float("%.2g" % max(v["ph_mod_p"] for v in
                                       ex["sa17_interaction_proportional"].values()))]
            pj = [(v["p_joint"], L[h]) for h, v in ex["sa17_interaction"].items()]
            N["sa17_interaction_min_other_p_joint"] = float("%.2g" % min(
                p_ for p_, lab in pj if lab != L["pa"]))
        if "prevalence_reference_sa15" in ex:
            p15 = ex["prevalence_reference_sa15"]
            # SA15 is calibrated against the within-stratum permutation null,
            # so its reference is the unweighted analysis with that null
            b15 = ex["sa16_reference"]["unweighted_permutation"]
            N["prevalence_reference_sa15"] = dict(
                death_pct=round(100 * float(p15["death_proportion_weighted"]), 1),
                factor={L[h]: round(float(v), 2) for h, v in p15["factor"].items()},
                factor_range=[round(float(min(p15["factor"].values())), 2),
                              round(float(max(p15["factor"].values())), 2)],
                expected={L[h]: r4(b15[h]["calibrated"] * float(v))
                          for h, v in p15["factor"].items()})
            # how far each weighted estimate lies below its expected value,
            # in its own standard errors
            s15 = pd.read_csv(os.path.join(OUT, "sensitivity_rcmi.csv"))
            s15 = s15[s15.analysis.str.startswith("SA15 ")].set_index("habit")
            N["prevalence_reference_sa15"]["shortfall_se"] = {
                L[h]: round(float((b15[h]["calibrated"] * float(v)
                                   - s15.loc[h, "calibrated"]) / s15.loc[h, "calibrated_se"]), 1)
                for h, v in p15["factor"].items() if h in s15.index}
        for item in ("chronic", "poor_health"):
            key_ = "sa17_interaction_" + item
            if key_ in ex:
                N[key_] = {}
                for h, v in ex[key_].items():
                    N[key_][L[h]] = dict(
                        contrast=v["level_name"],
                        without=[round(v["hr0"], 2), round(v["hr0_lo"], 2),
                                 round(v["hr0_hi"], 2)],
                        with_=[round(v["hr1"], 2), round(v["hr1_lo"], 2),
                               round(v["hr1_hi"], 2)],
                        p=float("%.3g" % v["p_level"]), p_joint=float("%.3g" % v["p_joint"]),
                        df=int(v["df"]), n=[int(v["n0"]), int(v["n1"])],
                        deaths=[int(v["deaths0"]), int(v["deaths1"])],
                        by_level={c["level_name"]: dict(
                            without=round(c["hr0"], 2), with_=round(c["hr1"], 2),
                            p=float("%.3g" % c["p"])) for c in v.get("by_level", [])})
        if "prevalence_reference" in ex:
            pr_ = ex["prevalence_reference"]
            N["prevalence_reference"] = dict(
                death_pct=round(100 * float(pr_["death_proportion"]), 1),
                primary_death_pct=round(100 * float(pr_["primary_death_proportion"]), 1),
                factor={L[h]: round(float(v), 2) for h, v in pr_["factor"].items()},
                factor_range=[round(float(min(pr_["factor"].values())), 2),
                              round(float(max(pr_["factor"].values())), 2)],
                expected={L[h]: r4(pri.loc[h, "calibrated"] * float(v))
                          for h, v in pr_["factor"].items()})
    p = os.path.join(OUT, "simulation_absolute.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        z0 = s[s.b == 0]
        bands0 = {"dense": z0.deaths_per_cell >= 10,
                  "middle": (z0.deaths_per_cell >= 2) & (z0.deaths_per_cell < 10),
                  "sparse": z0.deaths_per_cell < 2}
        N["simulation_null_by_band"] = {
            b: dict(calibrated=[r4(z0[m_].calibrated.min()), r4(z0[m_].calibrated.max())],
                    deaths_per_cell=[round(float(z0[m_].deaths_per_cell.min()), 1),
                                     round(float(z0[m_].deaths_per_cell.max()), 1)])
            for b, m_ in bands0.items() if m_.any()}
    if "calcheck" in N:
        cc_ = N["calcheck"]
        both = [cc_[s_] for s_ in ("additive", "interaction") if s_ in cc_]
        N["calcheck_pooled"] = dict(
            excess_err=[min(b["excess_err"][0] for b in both),
                        max(b["excess_err"][1] for b in both)],
            calibrated_abs_err_max=max(b["calibrated_abs_err_max"] for b in both),
            smooth_se_ratio=[min(b["smooth_se_ratio"][0] for b in both),
                             max(b["smooth_se_ratio"][1] for b in both)],
            plain_se_shortfall_max=max(b["plain_se_shortfall"][1] for b in both))
    # ---- round-3 additions: each level's share of D_JS, the interaction
    # smoother's comparisons, the alcohol questionnaire change, the excess of
    # the stratified over the adjusted hazard ratio
    p = os.path.join(OUT, "levels.csv")
    if os.path.exists(p):
        lv = pd.read_csv(p)
        N["level_share"] = {
            L[h]: {str(r_.level_name): dict(
                additive=_pct_hu(r_.share_additive), interaction=_pct_hu(r_.share_interaction),
                calibrated=_pct_hu(r_.share_calibrated), plugin=_pct_hu(r_.share_plugin),
                pct=round(float(r_.pct), 1), deaths=int(r_.deaths),
                mean_age=round(float(r_.mean_age), 1),
                contribution=round(1e4 * float(r_.calibrated), 2),
                contribution_se=round(1e4 * float(r_.calibrated_se), 2)
                if "calibrated_se" in lv else None)
                for r_ in g.itertuples()}
            for h, g in lv.groupby("habit")}
        # across the additive-model and calibrated splits, which differ most
        # for levels with few deaths (the interaction model's shares are
        # pulled towards equal shares by its overfitting: Table S1e)
        cols3 = ("share_additive", "share_calibrated")
        N["level_share_range"] = {
            L[h]: {str(r_.level_name): [min(_pct_hu(getattr(r_, c)) for c in cols3),
                                        max(_pct_hu(getattr(r_, c)) for c in cols3)]
                   for r_ in g.itertuples()}
            for h, g in lv.groupby("habit")}
        # as text, a single value when the splits agree ("44", not "44–44")
        N["level_share_range_txt"] = {
            lab: {lvl: (_num(v[0], "%d") if v[0] == v[1]
                        else "%s–%s" % (_num(v[0], "%d"), _num(v[1], "%d")))
                  for lvl, v in g.items()}
            for lab, g in N["level_share_range"].items()}
        b_ = lv[lv.habit == "bmi"].set_index("level")
        lean_ow = [_pct_hu(b_.loc[[0, 2, 3], c].sum()) for c in cols3]
        N["bmi_share_lean_25_35"] = [min(lean_ow), max(lean_ow)]
        N["bmi_share_ge35"] = [min(_pct_hu(b_.loc[4, c]) for c in cols3),
                               max(_pct_hu(b_.loc[4, c]) for c in cols3)]
    if "p_wald_int" in diff:
        pp_ = diff[(diff.spec == "PRIMARY") & (diff.scale == "calibrated")]
        N["pairs_int"] = {}
        for r_ in pp_.itertuples():
            a, b = r_.habit_a, r_.habit_b
            if order.index(a) > order.index(b):
                a, b = b, a
            N["pairs_int"]["%s vs %s" % (L[a], L[b])] = dict(
                se=r4(r_.se_int), p=float("%.3g" % r_.p_wald_int),
                p_boot=round(float(r_.p_boot_int), 3))
        N["pairs_bonferroni"] = dict(additive=int((pp_.p_wald < 0.05 / 15).sum()),
                                     interaction=int((pp_.p_wald_int < 0.05 / 15).sum()),
                                     total=int(len(pp_)))
    p = os.path.join(OUT, "ranks.csv")
    if os.path.exists(p):
        rr_ = pd.read_csv(p)
        if "p_first_int" in rr_:
            N["rank_probability_int"] = {
                L[r_.habit]: dict(first_pct=_pct_hu(r_.p_first_int),
                                  last_pct=_pct_hu(r_.p_last_int),
                                  rank_range=[int(r_.rank_lo_int), int(r_.rank_hi_int)])
                for r_ in rr_.itertuples()}
    p = os.path.join(OUT, "smoothing.csv")
    if os.path.exists(p) and "smoothing" in N:
        sm = pd.read_csv(p).set_index("habit")
        N["smoothing"]["world_additive"] = {L[h]: r4(sm.loc[h, "world_additive"])
                                            for h in order}
        if "world_cycle" in sm:
            N["smoothing"]["world_cycle"] = {L[h]: r4(sm.loc[h, "world_cycle"])
                                             for h in order}
    # SA16 against a like-for-like baseline: the same permutation null
    # without weights, and the prevalence-matched reference at the weighted
    # proportion of deaths
    ex16 = _extra().get("sa16_reference")
    p = os.path.join(OUT, "sensitivity_rcmi.csv")
    if ex16 and os.path.exists(p):
        s16 = pd.read_csv(p)
        w16 = s16[s16.analysis.str.startswith("SA16")].set_index("habit")
        base = ex16["unweighted_permutation"]
        rise = [100 * (w16.loc[h, "calibrated"] / base[h]["calibrated"] - 1)
                for h in w16.index]
        fac = [100 * (v - 1) for v in ex16["factor"].values()]
        N["sa16_reference"] = dict(
            unweighted={L[h]: r4(base[h]["calibrated"]) for h in C.HABITS},
            unweighted_p={L[h]: round(float(base[h]["p_null"]), 3) for h in C.HABITS},
            rise_pct=[int(round(min(rise))), int(round(max(rise)))],
            change=[r4(min(w16.loc[h, "calibrated"] - base[h]["calibrated"]
                           for h in w16.index)),
                    r4(max(w16.loc[h, "calibrated"] - base[h]["calibrated"]
                           for h in w16.index))],
            death_pct_weighted=round(100 * ex16["death_proportion_weighted"], 1),
            factor_pct=[round(min(fac), 1), round(max(fac), 1)],
            weighted_p={L[h]: round(float(w16.loc[h, "p_null"]), 3) for h in w16.index})
        rise_h = {h: 100 * (w16.loc[h, "calibrated"] / base[h]["calibrated"] - 1)
                  for h in w16.index}
        N["sa16_reference"].update(
            rise_pct_by_habit={L[h]: round(v, 1) for h, v in rise_h.items()},
            factor_pct_by_habit={L[h]: round(100 * (float(v) - 1), 1)
                                 for h, v in ex16["factor"].items()},
            rise_pct_others=[int(round(min(v for h, v in rise_h.items() if h != "smoke"))),
                             int(round(max(v for h, v in rise_h.items() if h != "smoke")))],
            factor_pct_others=[round(100 * (min(float(v) for h, v in ex16["factor"].items()
                                                if h != "smoke") - 1), 1),
                               round(100 * (max(float(v) for h, v in ex16["factor"].items()
                                                if h != "smoke") - 1), 1)])
    p = os.path.join(OUT, "sensitivity_rcmi.csv")
    if os.path.exists(p):
        sd_ = pd.read_csv(p)
        sd_ = sd_[(sd_.habit == "sleep") & (sd_.analysis != "primary")
                  & ~sd_.analysis.str.startswith("SA1 ")]
        pcol_ = sd_.p_perm if "p_perm" in sd_ else sd_.p_null
        N["sleep_detected"] = dict(n=int((pcol_ < 0.05).sum()), analyses=int(len(sd_)))
        c17_ = _extra().get("sa17a_null_check")
        if c17_:
            N["sa17a_null_check"] = dict(
                datasets=int(c17_["datasets"]), per_habit=int(c17_["per_habit"]),
                reject_05_pct=_pct_hu(c17_["reject_05"]),
                reject_10_pct=_pct_hu(c17_["reject_10"]),
                perm_reject_05_pct=_pct_hu(c17_["perm_reject_05"]),
                perm_reject_10_pct=_pct_hu(c17_["perm_reject_10"]))
        hs_ = pd.read_csv(os.path.join(OUT, "sensitivity_hr.csv"))
        h21 = hs_[hs_.analysis.str.startswith("SA21 ")].set_index("habit")
        h0 = hs_[hs_.analysis == "primary"].set_index("habit")
        if len(h21):
            N["sa21_max_hr_change"] = round(float((h21.hr - h0.hr.reindex(h21.index))
                                                  .abs().max()), 3)
        lag_ = _extra().get("sa21_lag_months", {})
        if lag_:
            tot_ = sum(lag_.values())
            N["sa21_lag"] = dict(max=max(int(k) for k in lag_),
                                 mean=round(sum(int(k) * v for k, v in lag_.items()) / tot_, 1))
    if os.path.exists(p):
        s20 = pd.read_csv(p)
        s20 = s20[s20.analysis.str.startswith("SA20 ")].set_index("habit")
        if len(s20):
            rk20 = s20.calibrated.rank(ascending=False, method="min")
            N["sa20_rank"] = {L[h]: int(rk20[h]) for h in s20.index}
            N["primary_rank"] = {L[h]: i + 1 for i, h in enumerate(order)}
    if os.path.exists(p):
        s11 = pd.read_csv(p)
        s11 = s11[s11.analysis.str.startswith("SA11 ")].set_index("habit")
        if len(s11):
            N["sa11_change_pct"] = {
                L[h]: int(round(100 * (s11.loc[h, "calibrated"] / pri.loc[h, "calibrated"]
                                       - 1))) for h in s11.index}
    # published NHANES unweighted response rates, all ages, interviewed and
    # examined (NCHS response-rate tables; 2017-2018 adjusted for the screener
    # response rate of 90.9%); quoted in Supplementary Methods S1.1
    N["response_rates"] = {"2007-2008": [78.4, 75.4], "2017-2018": [51.9, 48.8]}
    cy = {}
    for c_ in ("2015-2016", "2017-2018"):
        g_ = d[d.cycle == c_]
        if len(g_):
            cy[c_] = dict(abstainer=round(100 * float((g_.alc == 0).mean()), 1),
                          former=round(100 * float((g_.alc == 1).mean()), 1),
                          died=round(100 * float(g_.died.mean()), 1))
    N["alcohol_by_cycle"] = cy
    p = os.path.join(OUT, "sensitivity_rcmi.csv")
    if os.path.exists(p):
        s16 = pd.read_csv(p)
        w16 = s16[s16.analysis.str.startswith("SA16")].set_index("habit")
        if len(w16):
            ch16 = [float(w16.loc[h, "calibrated"] - pri.loc[h, "calibrated"])
                    for h in w16.index]
            N["sa16_change_max"] = r4(max(abs(v) for v in ch16))
            N["sa16_change_range"] = [r4(min(ch16)), r4(max(ch16))]
    p = os.path.join(OUT, "sensitivity_hr.csv")
    if os.path.exists(p):
        sv = pd.read_csv(p)
        pr0 = sv[sv.analysis == "primary"].set_index("label")
        s18 = sv[sv.analysis.str.startswith("SA18")].set_index("label")
        if len(s18):
            N["sa18_log_hr_increase_pct"] = {
                lab: int(round(100 * (np.log(s18.loc[lab, "hr"]) / np.log(pr0.loc[lab, "hr"])
                                      - 1)))
                for lab in [L[h] for h in order] if lab in s18.index}
    if "calcheck" in N:
        ccall = pd.read_csv(os.path.join(OUT, "calibration_check.csv"))
        cpall = pd.read_csv(os.path.join(OUT, "calibration_check_pairs.csv"))
        for scen, cc in ccall.groupby("scenario"):
            if "split_change_rel" in cc:
                N["calcheck"][scen]["split_change"] = pc(cc.split_change_rel)
                N["calcheck"][scen]["split_change_mcse_max"] = round(
                    100 * float(cc.split_change_rel_mcse.max()), 1)
            N["calcheck"][scen]["pair_wald_coverage_mcse_max"] = round(float(
                cpall[cpall.scenario == scen].wald_coverage_mcse.max()), 3)
            N["calcheck"][scen]["bootstrapped_datasets"] = int(
                cc.boot_reps.iloc[0]) if "boot_reps" in cc else None
        both = [N["calcheck"][s_] for s_ in ("additive", "interaction") if s_ in N["calcheck"]]
        N["calcheck_pooled"].update(
            wald_coverage_pct=[min(b["wald_coverage_pct"][0] for b in both),
                               max(b["wald_coverage_pct"][1] for b in both)],
            pair_wald_coverage_pct=[min(b["pair_wald_coverage_pct"][0] for b in both),
                                    max(b["pair_wald_coverage_pct"][1] for b in both)],
            coverage_mcse_max=max(max(b["wald_coverage_mcse_max"],
                                      b["pair_wald_coverage_mcse_max"]) for b in both))
        if all("split_change" in b for b in both):
            N["calcheck_pooled"]["split_change"] = [min(b["split_change"][0] for b in both),
                                                    max(b["split_change"][1] for b in both)]
            N["calcheck_pooled"]["calibrated_err"] = [
                min(b["calibrated_err"][0] for b in both),
                max(b["calibrated_err"][1] for b in both)]

    for s_ in ("PRIMARY", "RAW"):
        o_ = _nc(s_)
        if o_:
            N.setdefault("null_check", {})[s_] = dict(
                datasets=o_["datasets"], per_habit=o_["per_habit"],
                null_draws=o_.get("null_draws"), permutations=o_.get("permutations"),
                reject_05_pct=_pct_hu(o_["reject_05"]),
                reject_10_pct=_pct_hu(o_["reject_10"]),
                reject_mcse_05_pp=round(100 * o_["reject_mcse_05"], 1),
                reject_mcse_10_pp=round(100 * o_["reject_mcse_10"], 1),
                mean_excess=[r4(min(o_["mean_excess"].values())),
                             r4(max(o_["mean_excess"].values()))],
                mean_excess_avg=round(float(np.mean(list(o_["mean_excess"].values()))), 3),
                excess_in_sd=[round(float(min(o_["mean_excess"][h] / o_["sd_excess"][h]
                                              for h in o_["mean_excess"])), 1),
                              round(float(max(o_["mean_excess"][h] / o_["sd_excess"][h]
                                              for h in o_["mean_excess"])), 1)],
                perm_reject_05_pct=(_pct_hu(o_["perm_reject_05"])
                                    if "perm_reject_05" in o_ else None),
                perm_reject_10_pct=(_pct_hu(o_["perm_reject_10"])
                                    if "perm_reject_10" in o_ else None))
    N["design"] = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith("params_") and f.endswith(".json"):
            N["design"][f[7:-5]] = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
    # design strata of the analytic sample by their number of masked variance
    # units (Supplementary Methods S3.2)
    if os.path.exists(C.DESIGN_CSV):
        dv_ = pd.read_csv(C.DESIGN_CSV, usecols=["SEQN", "stratum", "psu"])
        dv_ = C.load(common=True)[["SEQN"]].merge(dv_, on="SEQN", how="left")
        nu_ = dv_.groupby("stratum").psu.nunique()
        N["design"].update(strata_in_sample=int(len(nu_)),
                           strata_three_units=int((nu_ == 3).sum()),
                           strata_units_max=int(nu_.max()))
    json.dump(N, open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8",
                      newline="\n"), indent=2, ensure_ascii=False)
    print("numbers.json: %d keys; ranking = %s" % (len(N), N["ranking"]))


# =====================================================================
# 2.  Figures
# =====================================================================
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DPI, WIDTH_IN = 400, 300.0 / 25.4          # >=300 mm at >=300 dpi
    # Figures 1 and 3 are printed at the text width (about 160 mm); drawn on a
    # 300-mm canvas their smallest text printed at about 5 pt, so they are
    # drawn at 230 mm, where 10.5-pt text prints at about 7.3 pt
    W_MAIN = 230.0 / 25.4
    C_TOTAL, C_DIRECT, C_NULL, C_EXC = "#b9c6d6", "#b2182b", "#e8e2d8", "#2166ac"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 16, "axes.titlesize": 17, "xtick.labelsize": 14,
        "ytick.labelsize": 14, "legend.fontsize": 14, "legend.frameon": False,
        # DejaVu Sans maths shifts \widehat to the right of its base; STIX sans
        # centres it and matches the sans-serif text
        "mathtext.fontset": "stixsans",
        # Embed TrueType rather than matplotlib's default Type 3.  Type 3 text
        # blurs when zoomed in some readers, and journal production and arXiv
        # both flag it; type 42 keeps the text selectable and searchable.
        "pdf.fonttype": 42, "ps.fonttype": 42})

    os.makedirs(FIG, exist_ok=True)

    def save(fig, name):
        for ext in ("png", "pdf"):
            # no creation timestamp, so that a rerun gives identical bytes
            meta = {"CreationDate": None} if ext == "pdf" else {"Software": None}
            fig.savefig(os.path.join(FIG, "%s.%s" % (name, ext)), dpi=DPI,
                        bbox_inches="tight", facecolor="white", metadata=meta)
        h_px, w_px = plt.imread(os.path.join(FIG, name + ".png")).shape[:2]
        print("  %-26s %d dpi, %d x %d px (%.0f mm wide)"
              % (name, DPI, w_px, h_px, 25.4 * w_px / DPI))
        plt.close(fig)

    m = pd.read_csv(os.path.join(OUT, "measures.csv"))
    tot = m[m.spec == "total"].set_index("habit")
    pri = m[m.spec == "PRIMARY"].set_index("habit")
    dem = m[m.spec == "DEMO"].set_index("habit")
    V = "calibrated"
    order = pri.sort_values(V, ascending=False).index.tolist()

    # ---------------- Figure 1
    fig, axes = plt.subplots(1, 2, figsize=(W_MAIN, 6.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    ax.barh(y + 0.20, [tot.loc[h, V] for h in order], height=0.38,
            color=C_TOTAL, label="Total (calibrated rMI)")
    ax.barh(y - 0.20, [pri.loc[h, V] for h in order], height=0.38,
            color=C_DIRECT,
            label="Direct (calibrated rCMI given\ndemographics, period and five\n"
                  "co-habits; $\\pm$1.96 SE)")
    err = [1.96 * pri.loc[h, V + "_se"] for h in order]
    err_lo = [min(e, pri.loc[h, V]) for e, h in zip(err, order)]     # truncated at 0
    ax.errorbar([pri.loc[h, V] for h in order], y - 0.20, xerr=[err_lo, err],
                fmt="none", ecolor="#3b0a10", elinewidth=1.8, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels([C.LABEL[h] for h in order])
    ax.set_xlabel("Calibrated rMI (total) or rCMI (direct)\n"
                  "$\\sqrt{D_{\\mathrm{JS}}}$ scale, ceiling 0.558")
    ax.set_title("a  Total versus direct association", loc="left", fontweight="bold",
                 fontsize=15.5)
    xmax = max(tot.loc[h, V] for h in order)
    rt = pd.read_csv(os.path.join(OUT, "ratios.csv")).set_index(["habit", "scale"])
    for i, h in enumerate(order):
        ax.text(tot.loc[h, V] + xmax * 0.02, y[i] + 0.20,
                "$\\sqrt{D_{\\mathrm{JS}}}$ %.2f; $D_{\\mathrm{JS}}$ %.2f"
                % (rt.loc[(h, "sqrt"), "ratio"], rt.loc[(h, "djs"), "ratio"]),
                va="center", fontsize=12, color="#5a6b7c")
    ax.set_xlim(0, xmax * 1.62)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30))

    ax = axes[1]
    xs = np.array([dem.loc[h, V] for h in order])
    ys = np.array([pri.loc[h, V] for h in order])
    hi = max(xs.max(), ys.max()) * 1.18
    hi = max(hi, xs.max() + hi * 0.55)
    ax.plot([0, hi], [0, hi], color="#bbbbbb", lw=1.2, ls="--")
    ax.scatter(xs, ys, s=55, color=C_EXC, edgecolor="white", lw=0.8, zorder=3)
    # labels in a column to the right of the points, each as close to the
    # height of its own point as a minimum spacing allows (the least-squares
    # spread, by pooling adjacent violators), so that the leader lines stay
    # nearly horizontal and neither cross nor pass through another marker
    g_ = hi * 0.095
    srt = np.argsort(ys)
    vals, wts = [], []
    for v in ys[srt] - g_ * np.arange(len(srt)):
        vals.append(float(v))
        wts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w2 = wts[-2] + wts[-1]
            vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w2]
            wts[-2:] = [w2]
    pos = np.repeat(vals, wts) + g_ * np.arange(len(srt))
    placed = {order[j]: pos[k] for k, j in enumerate(srt)}
    xlab = xs.max() + hi * 0.10
    for h, a, b in zip(order, xs, ys):
        ax.annotate(C.LABEL[h], xy=(a, b), xytext=(xlab, placed[h]),
                    fontsize=12, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.8, color="#999999",
                                    shrinkA=2, shrinkB=2, relpos=(0, 0.5)))
    ax.set_xlabel("Calibrated rCMI given\ndemographics and period")
    ax.set_ylabel("Calibrated rCMI given\ndemographics, period\nand five co-habits")
    ax.set_title("b  What co-habits account for", loc="left", fontweight="bold",
                 fontsize=15.5)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    tk = np.arange(0, hi + 1e-9, 0.02)
    ax.set_xticks(tk)
    ax.set_yticks(tk)
    fig.tight_layout(w_pad=4.0)
    save(fig, "fig1_total_vs_direct")

    # ---------------- Figure 2
    hr = pd.read_csv(os.path.join(OUT, "cox.csv"))
    fig, ax = plt.subplots(figsize=(WIDTH_IN * 0.82, 8.6))
    xhi = float(hr.hr_hi.max()) * 1.04
    xtext = xhi * 1.08
    ypos, labels, yy = [], [], 0.0
    # the order of Tables 1 and 3, the other hazard-ratio display
    for h in C.HABITS[::-1]:                 # y grows upward: the first on top
        for _, r in hr[hr.habit == h].iloc[::-1].iterrows():
            ypos.append(yy)
            labels.append("    " + str(r.level_name))
            col = "#8a8a8a" if r.reference else C_DIRECT
            if r.reference:
                ax.plot([1.0], [yy], marker="|", ms=15, color=col, mew=2.5)
                ax.text(xtext, yy, "1.00 (reference)", va="center", fontsize=11.5,
                        color="#6a6a6a")
            else:
                ax.plot([r.hr_lo, r.hr_hi], [yy, yy], color=col, lw=2.4,
                        solid_capstyle="round")
                ax.plot([r.hr], [yy], marker="o", ms=8.5, color=col, zorder=3)
                ax.text(xtext, yy, _hr_ci(r.hr, r.hr_lo, r.hr_hi),
                        va="center", fontsize=11.5)
            yy += 1
        ypos.append(yy)
        labels.append(_lab(h))
        yy += 1.2
    ax.axvline(1.0, color="#999999", lw=1.2, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if not lab.startswith("    "):
            t.set_fontweight("bold")
    ax.set_xlabel("Hazard ratio for all-cause mortality (95% CI)")
    # ratio measures on a log axis, so that 0.75 and 1.33 are equidistant from 1
    ax.set_xscale("log")
    xlo = float(hr.hr_lo.min()) * 0.96
    ax.set_xlim(xlo, xtext * 1.75)
    ax.set_ylim(-1, yy)
    ticks = [t for t in (0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
             if xlo <= t <= xhi]
    ax.spines["bottom"].set_bounds(min(min(ticks), float(hr.hr_lo.min())),
                                   max(max(ticks), float(hr.hr_hi.max())))
    ax.set_xticks(ticks)
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.minorticks_off()
    fig.tight_layout()
    save(fig, "fig2_dose_response")

    # ---------------- Figure 3
    cur = pd.read_csv(os.path.join(OUT, "biasfloor.csv"))
    # two rows: a and b above, c (with its longer legend) below
    fig = plt.figure(figsize=(W_MAIN, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0])]
    ax = axes[0]
    c = cur[(cur.habit == "smoke") & cur.get("note", pd.Series(np.nan,
                                                              index=cur.index)).isna()]
    c = c.sort_values("k_strata")
    ax.fill_between(c.k_strata, 0, c.null_mean, color=C_NULL,
                    label="Null floor  $\\mathbb{E}_0[\\widehat{\\mathrm{rCMI}}]$")
    ax.fill_between(c.k_strata, c.null_mean, c.rcmi, color=C_EXC, alpha=.85,
                    label="Excess (square-root scale)")
    ax.plot(c.k_strata, c.rcmi, color="#123", lw=2,
            label="Plug-in $\\widehat{\\mathrm{rCMI}}$")
    ax.set_xscale("log")
    ax.set_xlabel("Strata of the demographic score")
    ax.set_ylabel("Plug-in $\\widehat{\\mathrm{rCMI}}$, smoking")
    ax.set_title("a  Null floor versus signal", loc="left", fontweight="bold")
    ax.set_ylim(0, float(c.rcmi.max()) * 1.55)
    ax.legend(loc="upper left", fontsize=11.5)

    ax = axes[1]
    for h, col in zip(["smoke", "pa", "bmi"], [C_DIRECT, "#1b7837", "#6a3d9a"]):
        cc = cur[(cur.habit == h) & cur.get(
            "note", pd.Series(np.nan, index=cur.index)).isna()].sort_values(
            "mean_per_stratum", ascending=False)
        ax.plot(cc.mean_per_stratum, 100 * cc.null_share, marker="o", ms=6,
                color=col, lw=2, label=C.LABEL[h])
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.axhline(50, color="#999999", ls="--", lw=1.2)
    ax.set_xlabel("Participants per stratum\n(axis reversed)")
    ax.set_ylabel("Null floor as % of\nplug-in $\\widehat{\\mathrm{rCMI}}$")
    ax.set_title("b  Finer strata, larger floor", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=11.5)

    ax = axes[2]
    p = os.path.join(OUT, "simulation_rank.csv")
    if os.path.exists(p):
        from matplotlib.lines import Line2D
        rk = pd.read_csv(p).sort_values("deaths_per_cell")
        # one line per sample size: joined in a single line, the settings of
        # the two sizes interleave and the curve zig-zags
        sizes = sorted(rk.n.unique(), reverse=True)
        styles = dict(zip(sizes, [("-", True), ("--", False)]))
        for n_ in sizes:
            sub = rk[rk.n == n_]
            ls_, filled = styles[n_]
            for col, lab, mk, cl in (
                    ("top2_plugin", "Plug-in $\\widehat{\\mathrm{rCMI}}$", "s", "#777777"),
                    ("top2_excess", "Excess (square-root scale)", "^", "#e08214"),
                    ("top2_calibrated", "Calibrated $\\widetilde{\\mathrm{rCMI}}$", "o",
                     C_EXC)):
                ax.plot(sub.deaths_per_cell, 100 * sub[col], marker=mk, ms=7, color=cl,
                        lw=2, ls=ls_, mfc=cl if filled else "white", mew=1.6,
                        label=lab if n_ == sizes[0] else None)
            if "top1_calibrated" in sub:
                top1 = sub[["top1_plugin", "top1_excess", "top1_calibrated"]].min(axis=1)
                ax.plot(sub.deaths_per_cell, 100 * top1, lw=1.4, ls=":", color="#1b7837",
                        label="Worst of the three, single largest"
                        if n_ == sizes[0] else None)
        h_, l_ = ax.get_legend_handles_labels()
        for n_ in sizes:
            ls_, filled = styles[n_]
            h_.append(Line2D([], [], color="#555555", lw=2, ls=ls_, marker="o", ms=6,
                             mfc="#555555" if filled else "white"))
            l_.append("$n$ = %s" % format(int(n_), ",").replace(",", "\u2009"))
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.invert_xaxis()
        ax.axvline(10, color="#bbbbbb", lw=1.0, ls="--", zorder=0)
        ax.set_xlabel("Expected deaths per cell\n(axis reversed)")
        ax.set_ylabel("% of replicates recovering\nthe true leading pair")
        ax.set_title("c  Known-truth simulation", loc="left", fontweight="bold")
        # the legend in the empty lower-right cell, as an axis of its own, so
        # that it takes no width from the panels
        axl = fig.add_subplot(gs[1, 1])
        axl.axis("off")
        axl.legend(h_, l_, loc="center left", fontsize=12.5, frameon=False)
    fig.tight_layout(w_pad=3.0, h_pad=2.5)
    save(fig, "fig3_bias_floor")

    _figure_flow(plt, WIDTH_IN, save)
    _figure_dag(plt, WIDTH_IN, save)
    _figure_sensitivity(plt, W_MAIN, save, C_DIRECT)


def participant_flow():
    """Counts for the participant-flow diagram (STROBE item 13, RECORD 6.3)
    and the missing-data table, computed from the harmonised table so that
    neither can drift from the analysis.  Returns (steps, missing-data table,
    number of eligible adults in 2007-2018)."""
    raw = pd.read_pickle(C.CLEAN_PKL)
    steps = []
    n0 = len(raw)
    age_ok = raw.age >= 20
    el = age_ok & raw.eligstat.eq(1) & raw.mortstat.isin([0, 1]) & raw.permth.notna()
    pa = el & raw.cycle.isin(C.PA_CYCLES)
    steps.append(("Pooled NHANES records, ten cycles 1999–2018", n0, None))
    steps.append(("Aged ≥20 years at screening", int(age_ok.sum()),
                  int((~age_ok).sum())))
    steps.append(("Eligible for mortality linkage,\nvital status and follow-up known",
                  int(el.sum()), int((age_ok & ~el).sum())))
    steps.append(("Cycles 2007–2018 (activity and\nsitting questions fielded)",
                  int(pa.sum()), int((el & ~pa).sum())))
    d = raw[pa].copy()
    d["age_b"] = pd.cut(d["age"], C.AGE_BINS, right=False, labels=False)
    d["sex_b"] = (d["sex"] == 2).astype(float)
    order = [c for c, _, _ in C.CYCLES]
    d["cyc5"] = d["cycle"].map({c: i for i, c in enumerate(order)}) // 2
    need = C.HABITS + list(C.COVARIATES)
    bmi_ok = d["bmi"].notna()
    steps.append(("Body mass index measured\n(examination attended)",
                  int(bmi_ok.sum()), int((~bmi_ok).sum())))
    alc_ok = bmi_ok & d["alc"].notna()
    steps.append(("Alcohol use known", int(alc_ok.sum()), int((bmi_ok & ~alc_ok).sum())))
    final = d.dropna(subset=need)
    steps.append(("Complete on the other four habits\nand all covariates",
                  len(final), int(alc_ok.sum()) - len(final)))
    pretty = dict(C.LABEL)
    pretty.update({"age_b": "Age band", "sex_b": "Sex",
                   "race": "Race/ethnicity", "race5": "Race/ethnicity",
                   "educ": "Education",
                   "pir": "Income-to-poverty ratio",
                   "marital": "Marital status", "cyc5": "Survey period"})
    miss = pd.DataFrame(
        [{"Variable": pretty.get(c, c),
          "Missing, *n*": _int(int(d[c].isna().sum())),
          "Missing, %": "%.1f" % (100 * d[c].isna().mean())}
         for c in need])
    return steps, miss, len(d)


def selection_table():
    """Included versus excluded among the eligible adults of 2007-2018, by
    reason for exclusion (Supplementary Table S9b)."""
    raw = pd.read_pickle(C.CLEAN_PKL)
    el = (raw.age >= 20) & raw.eligstat.eq(1) & raw.mortstat.isin([0, 1]) & \
        raw.permth.notna() & raw.cycle.isin(C.PA_CYCLES)
    d = raw[el].copy()
    d["cyc5"] = d["cycle"].map({c: i for i, c in enumerate(
        [c for c, _, _ in C.CYCLES])}) // 2
    need = C.HABITS + [c for c in C.COVARIATES if c not in ("age_b", "sex_b")]
    inc = d[need].notna().all(axis=1)
    groups = [("Included", d[inc]), ("Excluded, all", d[~inc]),
              ("no BMI", d[~inc & d.bmi.isna()]),
              ("alcohol unknown", d[~inc & d.bmi.notna() & d.alc.isna()]),
              ("other", d[~inc & d.bmi.notna() & d.alc.notna()])]
    rows = []

    def add(label, fn):
        rows.append(dict(Characteristic=label, **{g: fn(s) for g, s in groups}))

    add("*n*", lambda s: _int(len(s)))
    add("Age, mean (SD), years", lambda s: "%.1f (%.1f)" % (s.age.mean(), s.age.std()))
    add("Recorded at the top-coded age of 80, %",
        lambda s: "%.1f" % (100 * (s.age == 80).mean()))
    add("Female, %", lambda s: "%.1f" % (100 * (s.sex == 2).mean()))
    add("Non-Hispanic Black, %", lambda s: "%.1f" % (100 * (s.race == 1).mean()))
    add("College graduate, % of known", lambda s: "%.1f" % (100 * (s.educ == 2).sum()
                                                          / max(s.educ.notna().sum(), 1)))
    add("Deaths", lambda s: _int(int((s.mortstat == 1).sum())))
    add("Deaths per 1000 person-years",
        lambda s: "%.1f" % (1000 * (s.mortstat == 1).sum() / (s.permth.sum() / 12.0)))
    return pd.DataFrame(rows)


def _figure_flow(plt, WIDTH_IN, save):
    """Participant flow.  Geometry is set explicitly — boxes of a fixed width on
    a single axis, straight vertical connectors between them and exclusion
    labels on a parallel column — so nothing overlaps whatever the counts are."""
    from matplotlib.patches import FancyBboxPatch
    steps, _, _ = participant_flow()
    k = len(steps)
    fig, ax = plt.subplots(figsize=(WIDTH_IN * 0.70, 1.25 * k + 0.7))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def num(v):
        return format(v, ",").replace(",", " ")       # thin space

    x0, w, h = 0.03, 0.62, 0.100                           # box geometry
    gap = (1.0 - 0.06 - k * h) / max(k - 1, 1)
    tops = [1.0 - 0.03 - i * (h + gap) for i in range(k)]
    for (label, n, excl), top in zip(steps, tops):
        ax.add_patch(FancyBboxPatch((x0, top - h), w, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.02",
                                    facecolor="white", edgecolor="#2b2b2b",
                                    linewidth=1.5, zorder=2))
        ax.text(x0 + w / 2, top - h / 2, "%s\n$n$ = %s" % (label, num(n)),
                ha="center", va="center", fontsize=10.5, linespacing=1.35, zorder=3)
        if excl:
            ymid = top + gap / 2
            ax.annotate("", xy=(x0 + w + 0.10, ymid), xytext=(x0 + w / 2, ymid),
                        arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2),
                        zorder=1)
            ax.text(x0 + w + 0.115, ymid, "excluded\n%s" % num(excl),
                    ha="left", va="center", fontsize=10.5, color="#555555",
                    linespacing=1.4, zorder=3)
    for a, b in zip(tops[:-1], tops[1:]):
        ax.annotate("", xy=(x0 + w / 2, b), xytext=(x0 + w / 2, a - h),
                    arrowprops=dict(arrowstyle="-|>", color="#2b2b2b", lw=1.6),
                    zorder=1)
    save(fig, "figS1_flow")


def _figure_dag(plt, WIDTH_IN, save):
    """Directed acyclic graph.  Boxes are drawn first; every arrow then runs
    between the points where the centre-to-centre line leaves the source box
    and enters the target box, and is drawn ABOVE the boxes, so each arrowhead
    is visible (anchoring on the text artists hid several heads under boxes)."""
    fig, ax = plt.subplots(figsize=(WIDTH_IN * 0.86, 6.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    nodes = {
        "D": (0.27, 0.90, "Demographics and period $D$\n"
                          "age \u00b7 sex \u00b7 race \u00b7 education\n"
                          "income \u00b7 marital status \u00b7 period", False),
        "M": (0.60, 0.90, "Prevalent disease,\nself-rated health $M$\n"
                          "(measured; SA17 only)", False),
        "U": (0.90, 0.88, "Unmeasured $U$\ndiet \u00b7 health care", True),
        "P": (0.40, 0.06, "Earlier habits $X_0$, $C_0$\n(largely unmeasured)", True),
        "L": (0.08, 0.58, "Unmeasured\nhealth-behaviour\ndisposition $L$", True),
        "X": (0.30, 0.34, "Habit of interest\n$X$", False),
        "C": (0.62, 0.34, "Other five habits\n$C$", False),
        "Y": (0.91, 0.55, "All-cause\nmortality $Y$", False)}
    texts = {}
    for k, (x, y, lab, dashed) in nodes.items():
        texts[k] = ax.text(x, y, lab, ha="center", va="center", fontsize=11,
                           zorder=3, linespacing=1.45,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                     edgecolor="#8c8c8c" if dashed else "#2b2b2b",
                                     linewidth=1.6,
                                     linestyle="--" if dashed else "-"))
    fig.canvas.draw()
    inv = ax.transData.inverted()
    box = {}
    for k, t in texts.items():
        bb = t.get_bbox_patch().get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        box[k] = (x0, y0, x1, y1)

    def edge(k, towards):
        """Point where the ray from box k's centre towards ``towards`` exits
        the box (plus a small gap)."""
        x0, y0, x1, y1 = box[k]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = towards[0] - cx, towards[1] - cy
        tx = (x1 - cx) / abs(dx) if dx else np.inf
        ty = (y1 - cy) / abs(dy) if dy else np.inf
        t = min(tx, ty) * 1.0 + 0.012 / max(np.hypot(dx, dy), 1e-9)
        return cx + t * dx, cy + t * dy

    def centre(k):
        x0, y0, x1, y1 = box[k]
        return (x0 + x1) / 2, (y0 + y1) / 2

    def arrow(a, b, dashed=False, rad=0.0):
        pa, pb = edge(a, centre(b)), edge(b, centre(a))
        ax.annotate("", xy=pb, xytext=pa, zorder=4,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.3",
                                    color="#9a9a9a" if dashed else "#2b2b2b",
                                    lw=1.6, linestyle="--" if dashed else "-",
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3,rad=%.2f" % rad))
    for a, b, rad in [("D", "X", 0.0), ("D", "C", 0.0), ("D", "Y", 0.0),
                      ("D", "M", 0.0), ("M", "X", 0.0), ("M", "C", 0.0),
                      ("M", "Y", 0.0), ("X", "C", 0.0), ("X", "Y", -0.22),
                      ("C", "Y", 0.0), ("P", "M", 0.0), ("P", "X", 0.0),
                      ("P", "C", 0.0), ("P", "Y", 0.30)]:
        arrow(a, b, rad=rad)
    for a, b, rad in [("U", "X", 0.18), ("U", "C", 0.0), ("U", "Y", -0.15),
                      ("L", "X", 0.0), ("L", "C", 0.0)]:
        arrow(a, b, dashed=True, rad=rad)
    save(fig, "figS2_dag")


def _figure_sensitivity(plt, WIDTH_IN, save, C_DIRECT):
    """One panel per habit and one labelled row per analysis, with +/-1.96
    smoothed-bootstrap standard errors where the analysis has them, so every
    point can be identified without a legend."""
    p = os.path.join(OUT, "sensitivity_rcmi.csv")
    if not os.path.exists(p):
        return
    sr = pd.read_csv(p)
    m = pd.read_csv(os.path.join(OUT, "measures.csv"))
    pri = m[m.spec == "PRIMARY"].set_index("habit")
    order = pri.sort_values("calibrated", ascending=False).index.tolist()
    ana = sorted(sr.analysis.unique(), key=_sa_key)
    names = [_sa_code(a) for a in ana]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_IN, 11.0), sharey=True)
    y = np.arange(len(ana))[::-1]
    xmax = float(sr.calibrated.max()) * 1.45
    for ax, h in zip(axes.ravel(), order):
        g = sr[sr.habit == h].set_index("analysis")
        for yi, a in zip(y, ana):
            # SA1 uses the primary sample for physical activity, sleep and
            # sitting, so it repeats their primary estimates
            if a not in g.index or (a.startswith("SA1 ") and h not in ("smoke", "alc", "bmi")):
                continue
            v = float(g.loc[a, "calibrated"])
            se_ = (float(pri.loc[h, "calibrated_se"]) if a == "primary"
                   else float(g.loc[a, "calibrated_se"]) if "calibrated_se" in g
                   and pd.notna(g.loc[a, "calibrated_se"]) else np.nan)
            col = C_DIRECT if a == "primary" else "#4d5d6c"
            if np.isfinite(se_):
                ax.plot([max(v - 1.96 * se_, 0), v + 1.96 * se_], [yi, yi], color=col,
                        lw=1.6, alpha=0.8, clip_on=False)
            # an estimate truncated at zero is drawn open, on the axis
            ax.plot([v], [yi], marker="o", ms=6.5, color=col, zorder=3, clip_on=False,
                    mfc="white" if v <= 0 else col, mew=1.5)
        ax.axvline(float(pri.loc[h, "calibrated"]), color=C_DIRECT, lw=1.0, ls=":",
                   zorder=0)
        ax.set_title(C.LABEL[h], loc="left", fontweight="bold")
        ax.set_xlim(-0.002, xmax)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=11.5)
        ax.tick_params(axis="x", labelsize=11.5)
    for ax in axes[1]:
        ax.set_xlabel("Calibrated rCMI\n($\\sqrt{D_{\\mathrm{JS}}}$ scale)")
    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    save(fig, "figS3_sensitivity")


# =====================================================================
# 3.  Markdown -> LaTeX -> PDF
# =====================================================================
#: The TeX engine.  pdfLaTeX, not XeLaTeX: arXiv's default toolchain is
#: pdfLaTeX, and every attempt to make fontspec find a font there failed.
ENGINE = "pdflatex"

# Built with pdfLaTeX, not XeLaTeX.  fontspec's \setmainfont looks a font up by
# name through fontconfig, which arXiv's build container does not provide, so
# the preprint failed there however the font was supplied.  mathpazo gives the
# same Palatino letterforms from Type 1 fonts in psnfss, which is part of every
# LaTeX installation, needs no font files shipped with the source, and compiles
# on arXiv's default engine.  Every non-ASCII character is turned into a LaTeX
# command by inline(), so the source that reaches LaTeX is pure ASCII.
PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{mathpazo}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{array}
\usepackage{ragged2e}
\usepackage[font=small,labelfont=bf,justification=raggedright,
            singlelinecheck=false,labelformat=empty]{caption}
\usepackage{microtype}
\usepackage{setspace}
\usepackage{lineno}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\pdfsuppressptexinfo=-1
\usepackage{pdflscape}
\usepackage{needspace}
\newsavebox{\legendbox}
\doublespacing
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.15}
\begin{document}
\linenumbers
"""

SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s):
    """Escape LaTeX specials outside math and verbatim spans.

    Display math ($$...$$) must be matched BEFORE inline math, or the opening
    ``$$`` is read as an empty inline span and the formula body is escaped into
    nonsense -- which is exactly what garbled the equations in an earlier draft.
    """
    MDESC = {"*": r"\textsuperscript{*}", "_": r"\_", "|": r"\textbar{}",
             "\\": r"\textbackslash{}", "`": r"\textasciigrave{}",
             "[": "{[}", "]": "{]}", "#": r"\#"}
    out = []
    for part in re.split(r"(\$\$[^$]*\$\$|\$[^$]*\$|``[^`]+``|`[^`]*`)", s):
        if part.startswith("$") and part.endswith("$") and len(part) > 2:
            out.append(part)
        elif part.startswith("``") and part.endswith("``") and len(part) > 4:
            out.append(r"\texttt{%s}" % "".join(SPECIAL.get(c, c) for c in part[2:-2]))
        elif part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(r"\texttt{%s}" % "".join(SPECIAL.get(c, c) for c in part[1:-1]))
        else:
            # Markdown backslash escapes (\* \| \_ ...) become the literal glyph;
            # a lone backslash becomes \textbackslash{} -- whose braces must NOT
            # then be escaped again (that printed "\{}" on the title page).
            buf, i = [], 0
            while i < len(part):
                c = part[i]
                if c == "\\" and i + 1 < len(part) and part[i + 1] in MDESC:
                    buf.append(MDESC[part[i + 1]])
                    i += 2
                    continue
                buf.append(r"\textbackslash{}" if c == "\\" else SPECIAL.get(c, c))
                i += 1
            out.append("".join(buf))
    return "".join(out)


def inline(s):
    # URLs are held back before escaping (an escaped '_' or '%' would break
    # them) and typeset with \url, which makes them clickable and lets them
    # break without inserting a hyphen that is not part of the address.
    links, urls = [], []

    def _hold_link(m):
        links.append((m.group(1), m.group(2)))
        return "\x02%d\x02" % (len(links) - 1)

    def _hold_url(m):
        urls.append(m.group(0))
        return "\x01%d\x01" % (len(urls) - 1)

    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", _hold_link, s)
    s = re.sub(r"https?://[^\s<>()]*[^\s<>().,;:]", _hold_url, s)
    s = esc(s)
    # Protect maths spans: every substitution below is for text only, and a
    # Unicode or '*' replacement inside $...$ would corrupt the formula.
    maths = []

    def _hold(m):
        maths.append(m.group(0))
        return "\x00%d\x00" % (len(maths) - 1)

    s = re.sub(r"\$\$[^$]*\$\$|\$[^$]+\$", _hold, s)
    # bold may contain single-star emphasis or maths:
    # "**RACE is not bounded by $c_{\max}$.**"
    s = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"\\textbf{\1}", s)
    # straight double quotes -> TeX quotes
    s = re.sub(r'(^|[\s(\[{/\u2014\u2013-])"(?=\S)', r"\1``", s)
    # an opening single quote (an apostrophe inside a word follows a letter)
    s = re.sub(r"(^|[\s(\[{/\u2014\u2013-])'(?=\S)", r"\1`", s)
    s = s.replace('"', "''")
    # a digit group split by a plain space must not break across lines: 31 003
    s = re.sub(r"(?<![\d.,])(\d{1,3}) (?=\d{3}(?!\d|[.,]\d))", r"\1\\,", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", s)
    # a single-letter statistic stays on the line of its relation and value:
    # P = 0.0017, n = 10 000
    s = re.sub(r"(\\emph\{[PnT]\}) (=|<|>|≤|≥) ", r"\1~\2~", s)
    # citation markers -> superscripts; numbers start at 1 so that an interval
    # such as [0,1] is left alone
    s = re.sub(r"(?<![\w\\])\[([1-9]\d{0,2}(?:\s*[,\u2013-]\s*[1-9]\d{0,2})*)\]",
               lambda m: r"\textsuperscript{%s}" % m.group(1).replace("-", "--"), s)
    # A DOI or arXiv identifier is one long token that LaTeX will not break, so
    # it runs into the margin -- reference 26 overflowed by 64pt.  Offer a break
    # opportunity after each separator; \allowbreak adds no width and no hyphen.
    def _breakable(m):
        return re.sub(r"([./:])", r"\1\\allowbreak{}", m.group(0))

    s = re.sub(r"(?:doi:|arXiv:)[\w./:-]{6,}", _breakable, s)

    # multi-character sequences first
    s = s.replace("θ̂", r"$\hat{\theta}$")        # theta + combining hat
    # runs of Unicode super/subscripts -> \textsuperscript / \textsubscript
    SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
           "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+"}
    SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
           "₇": "7", "₈": "8", "₉": "9", "₋": "-"}
    # typeset as maths so that a superscript minus is a minus, not a hyphen
    for table, op in ((SUP, "^"), (SUB, "_")):
        s = re.sub("[" + "".join(table) + "]+",
                   lambda m, t=table, o=op: (r"\smash{\ensuremath{%s{%s}}}"
                                             if o == "^" else r"\ensuremath{%s{%s}}")
                   % (o, "".join(t[ch] for ch in m.group(0))), s)
    for a, b in [("–", "--"), ("—", "---"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
                 ("×", r"$\times$"), ("⫫", r"$\perp\!\!\!\perp$"),
                 ("√", r"$\surd$"), ("‖", r"$\|$"), ("−", "$-$"),
                 ("→", r"$\rightarrow$"), ("←", r"$\leftarrow$"),
                 ("η", r"$\eta$"), ("φ", r"$\varphi$"),
                 ("ρ", r"$\rho$"), ("θ", r"$\theta$"), ("χ", r"$\chi$"),
                 ("Φ", r"$\Phi$"), ("σ", r"$\sigma$"), ("μ", r"$\mu$"),
                 ("α", r"$\alpha$"), ("β", r"$\beta$"), ("γ", r"$\gamma$"),
                 ("δ", r"$\delta$"), ("λ", r"$\lambda$"), ("Δ", r"$\Delta$"),
                 ("∈", r"$\in$"), ("∉", r"$\notin$"), ("⊂", r"$\subset$"),
                 ("∞", r"$\infty$"), ("≈", r"$\approx$"), ("≠", r"$\neq$"),
                 ("∑", r"$\sum$"), ("∏", r"$\prod$"), ("∫", r"$\int$"),
                 ("π", r"$\pi$"), ("τ", r"$\tau$"), ("ε", r"$\varepsilon$"),
                 ("ω", r"$\omega$"), ("Σ", r"$\Sigma$"), ("±", r"$\pm$"),
                 # Everything below used to be left to the font, which
                 # XeLaTeX could do and pdfLaTeX cannot: an unmapped byte
                 # there is a "Unicode character not set up" error.
                 ("§", r"\S{}"), ("·", r"$\cdot$"), ("…", r"\dots{}"),
                 ("½", "$1/2$"), ("¼", "$1/4$"), ("¾", "$3/4$"),
                 ("°", r"$^\circ$"), ("‰", r"\textperthousand{}"),
                 ("†", r"$\dagger$"), ("‡", r"$\ddagger$"),
                 ("‘", "`"), ("’", "'"), ("“", "``"), ("”", "''"),
                 ("é", r"\'e"), ("á", r"\'a"), ("í", r"\'i"),
                 ("ó", r"\'o"), ("ú", r"\'u"), ("à", r"\`a"),
                 ("è", r"\`e"), ("ì", r"\`i"), ("ò", r"\`o"),
                 ("ù", r"\`u"), ("â", r"\^a"), ("ê", r"\^e"),
                 ("î", r"\^i"), ("ô", r"\^o"), ("û", r"\^u"),
                 ("ä", r'\"a'), ("ë", r'\"e'), ("ï", r'\"i'),
                 ("ö", r'\"o'), ("ü", r'\"u'), ("ç", r"\c{c}"),
                 ("ñ", r"\~n"), ("å", r"\aa{}"), ("ø", r"\o{}"),
                 ("æ", r"\ae{}"), ("œ", r"\oe{}"), ("ß", r"\ss{}"),
                 ("Á", r"\'A"), ("É", r"\'E"), ("Í", r"\'I"),
                 ("Ó", r"\'O"), ("Ú", r"\'U"), ("Ä", r'\"A'),
                 ("Ö", r'\"O'), ("Ü", r'\"U'), ("Ç", r"\c{C}"),
                 ("Ñ", r"\~N"),
                 (" ", r"\,"), (" ", "~")]:      # thin / non-breaking space
        s = s.replace(a, b)
    s = re.sub("\x00(\\d+)\x00", lambda m: maths[int(m.group(1))], s)
    s = re.sub("\x01(\\d+)\x01", lambda m: r"\url{%s}" % urls[int(m.group(1))], s)
    return re.sub("\x02(\\d+)\x02", lambda m: r"\href{%s}{%s}" % (
        links[int(m.group(1))][1], inline(links[int(m.group(1))][0])), s)


#: Characters allowed to reach the .tex unmapped.  Empty, deliberately:
#: the documents are built with pdfLaTeX, which has no Unicode input, so
#: every non-ASCII character must have become a command in inline() above.
_TYPESET_OK = set()


def unmapped_characters(paths):
    """Report non-ASCII characters that neither ``inline`` maps nor the font
    renders.  A silent gap here shows up as a box in the PDF — which is how
    10⁻⁷⁹ was lost from an earlier supplement."""
    import collections
    found = collections.Counter()
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            for ch in inline(line):
                if ord(ch) > 127 and ch not in _TYPESET_OK:
                    found[ch] += 1
    return found


# Tails that a collapsed backslash escape leaves behind inside maths.  Only
# sequences of three or more letters are listed, so that ordinary two-letter
# variable names cannot trip the check.
_ORPHAN_TAILS = (
    "lpha", "pprox", "inom", "ullet", "rac", "orall", "abla", "ewline",
    "olimits", "onumber", "ormalsize", "ight", "ightarrow", "imes", "heta",
    "ilde", "arepsilon", "arphi", "space", "dots", "ert", "angle", "ext",
    "extbf", "extit", "aggedright", "eft", "ambda", "igma", "elta",
)

_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.S)


def latex_sanity(paths):
    """Faults that compile *without error* and so cannot be found in the log.

    The motivating case: this workstation's shell collapses a doubled backslash
    inside a heredoc, so a script writing ``\\alpha`` emits ``\\a`` -- the BEL
    control character -- followed by ``lpha``.  XeLaTeX then typesets four
    italic letters without a murmur.  Five documents carried exactly that in
    Supplementary S5.1, and no log check or number check could have found it.

    Checked here: control characters anywhere; a tab inside maths; the visible
    residue of a collapsed escape; brace and dollar balance.
    """
    bad = []
    for p in paths:
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8").read()
        base = os.path.basename(p)

        def at(i):
            return "%s:%d" % (base, t.count("\n", 0, i) + 1)

        for i, ch in enumerate(t):
            o = ord(ch)
            if (o < 32 and ch not in "\n\t") or o == 127:
                bad.append("%s control character U+%04X" % (at(i), o))
        for m in _MATH_RE.finditer(t):
            seg = m.group(1) if m.group(1) is not None else m.group(2)
            where = at(m.start())
            if "\t" in seg:
                bad.append("%s tab inside maths" % where)
            if seg.count("{") != seg.count("}"):
                bad.append("%s unbalanced braces in maths: %s" % (where, seg[:40]))
            for tail in _ORPHAN_TAILS:
                if re.search(r"(?<![A-Za-z\\])" + tail + r"(?![A-Za-z])", seg):
                    bad.append("%s bare '%s' in maths -- a collapsed escape?"
                               % (where, tail))
        if p.endswith(".md"):
            # dollar delimiters are the Markdown convention; the generated LaTeX
            # uses \[ \] for display maths, so parity there means nothing
            stripped = re.sub(r"^```.*?^```", "", t, flags=re.S | re.M)
            if stripped.count("$$") % 2:
                bad.append("%s odd number of $$" % base)
            inline_only = re.sub(r"\$\$.+?\$\$", "", stripped, flags=re.S)
            if len(re.findall(r"(?<!\\)\$", inline_only)) % 2:
                bad.append("%s odd number of $" % base)
        else:
            no, nc = len(re.findall(r"\\\[", t)), len(re.findall(r"\\\]", t))
            if no != nc:
                bad.append("%s display maths mismatch: %d \\[ vs %d \\]"
                           % (base, no, nc))
    return bad


_LOG_PATTERNS = (
    ("error", re.compile(r"^! (.+)$", re.M)),
    ("undefined control sequence", re.compile(r"^! Undefined control sequence", re.M)),
    ("missing $", re.compile(r"Missing \$ inserted", re.M)),
    ("runaway argument", re.compile(r"^Runaway argument", re.M)),
    ("missing character", re.compile(r"Missing character: There is no (.+?) in font", re.M)),
    # not a TeX error: the PDF is open in a viewer, so the driver cannot write it.
    # Without this the build reports a bland "0 pages" and leaves the stale PDF.
    ("output not written", re.compile(r"(xdvipdfmx:fatal: Unable to open .*)", re.M)),
    ("output not written", re.compile(r"(No output PDF file written)", re.M)),
)


def driver_problems(stdout):
    """Failures the driver reports on stdout and never writes to the log.

    The one that matters in practice: if the PDF is open in a viewer, xdvipdfmx
    cannot replace it, the log stops mid-page, and the build would otherwise
    print a bland "0 pages" while leaving yesterday's PDF in place.
    """
    out = []
    for rx, hint in ((r"xdvipdfmx:fatal: Unable to open \"(.+?)\"",
                      "PDF is open in another program -- close it and re-run"),
                     (r"(No output PDF file written)", "")):
        m = re.search(rx, stdout or "")
        if m:
            out.append("output NOT written: %s%s"
                       % (m.group(1), "  (" + hint + ")" if hint else ""))
    return out


def log_problems(log):
    """Everything in a XeLaTeX log that a reader of the PDF would regret.

    ``render`` deletes the log once it has counted the pages, so anything worth
    knowing has to be extracted before that happens -- which is why this is a
    function and not a manual step.
    """
    if not os.path.exists(log):
        return ["%s: no log" % os.path.basename(log)]
    t = open(log, encoding="utf-8", errors="replace").read()
    out, seen = [], set()
    for label, rx in _LOG_PATTERNS:
        hits = rx.findall(t)
        if not hits:
            continue
        for h in hits[:4]:
            msg = "%s: %s" % (label, (h if isinstance(h, str) else h[0]).strip()[:80])
            if msg not in seen:
                seen.add(msg)
                out.append("%s  %s" % (os.path.basename(log), msg))
    # text overprinting a neighbouring column, and content pushed off a page
    # (a figure and its caption taller than the text block)
    wide = [float(w) for w in re.findall(
        r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)", t) if float(w) > 15]
    if wide:
        out.append("%s  overfull hbox: %d lines more than 15pt too wide (largest %.0fpt)"
                   % (os.path.basename(log), len(wide), max(wide)))
    tall = re.findall(r"Overfull \\vbox \((\d+(?:\.\d+)?)pt too high\) has occurred "
                      r"while \\output is active", t)
    if tall:
        out.append("%s  overfull vbox in the output routine: content cut off at a "
                   "page edge (%spt)" % (os.path.basename(log), tall[0]))
    return out


def _row(cells):
    r"""Join one table row and guard its opening bracket.

    A row beginning with '[' follows the previous row's ``\\``, and LaTeX reads
    the bracket as that command's optional vertical-space argument -- "Missing
    number, treated as zero" followed by "Illegal unit of measure".  The cover
    letter's reviewer-suggestion table, whose cells are ``[ ]`` for the authors
    to fill, hit this on four rows.
    """
    s = " & ".join(cells)
    if s.lstrip().startswith("["):
        s = "{}" + s
    return s + r" \\"


def _cells(ln):
    """Split a Markdown table row on '|' that is neither escaped (\\|) nor
    inside $...$ maths, so that a cell may hold p(y \\| do(x)) or $p(y|z)$."""
    cells, cur, in_math, i = [], "", False, 0
    body = ln.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body) and body[i + 1] == "|":
            cur += "\\|"
            i += 2
            continue
        if ch == "$":
            in_math = not in_math
        if ch == "|" and not in_math:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += ch
        i += 1
    return cells + [cur.strip()]


#: approximate advance widths, in em, of the characters of a numeric cell
_EMW = {".": 0.28, ",": 0.28, " ": 0.33, "(": 0.39, ")": 0.39, "–": 0.5, "−": 0.78,
        "-": 0.33, "%": 0.83, "<": 0.78, ">": 0.78, "≤": 0.78, "≥": 0.78, ";": 0.28}


def _em(s):
    """Printed width of a table cell in em of its font (digits 0.5 em)."""
    return sum(_EMW.get(ch, 0.5) for ch in s)


def _table(lines, landscape=False):
    rows = [_cells(ln) for ln in lines]
    header, body = rows[0], rows[2:]
    n = len(header)
    body = [(r + [""] * n)[:n] for r in body]
    widest = [max([len(header[j])] + [len(r[j]) for r in body]) for j in range(n)]
    numeric = [all(re.match(r"^[\d.,%()–−\-\s<>≤≥;]*(?:to[\d.,()–−\-\s]*)?$",
                            (r[j] or "").replace("(reference)", ""))
                   for r in body)
               for j in range(n)]
    if max(widest) <= 22 and n <= 7:
        spec = "".join("r" if numeric[j] else "l" for j in range(n))
        out = [r"\begin{center}", r"\footnotesize\singlespacing",
               r"\begin{tabular}{%s}" % spec, r"\toprule"]
        out.append(_row([r"\textbf{%s}" % inline(h) for h in header]))
        out.append(r"\midrule")
        for r in body:
            out.append(_row([inline(c) for c in r]))
        return out + [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    # Numeric columns get a fixed width, in em of the table font, that fits
    # their widest entry and the longest word of their header, so that numbers
    # never run into the next column and a header word never overprints its
    # neighbour.  The text columns share what is left of the line (a \dimexpr,
    # so the arithmetic is exact) and wrap without hyphenation.  A floor plus
    # longtable keeps LaTeX from looping on a degenerate width.
    em = {}
    for j in range(n):
        if numeric[j]:
            content = max([_em(r[j]) for r in body] or [0])
            # a header word is measured as printed: maths as its glyphs
            hdr = re.sub(r"\$[^$]*\$", "xxxx", header[j])
            # (a header may break after a hyphen: person-/years)
            hword = max([len(w) for w in re.split(r"[\s]+|(?<=-)", hdr) if w] or [0])
            em[j] = max(content, 0.66 * hword) + 0.4
    line_em = 100.0 if landscape else 64.0
    tot = sum(em.values())
    text = [j for j in range(n) if not numeric[j]]
    if tot > 0.8 * line_em and text:
        em = {j: e * 0.8 * line_em / tot for j, e in em.items()}
        tot = sum(em.values())
    weights = {j: max(6.0, min(float(widest[j]), 34.0)) for j in text}
    wsum = sum(weights.values())
    for j in text:
        room = (line_em - tot - 2 * n * 0.375) * weights[j] / wsum
        if room < 4.0:
            raise SystemExit("table '%s': text column '%s' would be %.1f em wide; "
                             "set the table in landscape or narrow its numeric "
                             "columns" % (header[0], header[j], room))
    spec = ""
    for j in range(n):
        if numeric[j]:
            spec += (">{\\RaggedLeft\\arraybackslash\\hyphenpenalty=10000}"
                     "p{%.2fem}" % em[j]
                     if text else ">{\\RaggedLeft\\arraybackslash\\hyphenpenalty=10000"
                     "\\exhyphenpenalty=10000}p{\\dimexpr "
                     "(\\linewidth-%d\\tabcolsep)*%d/1000\\relax}"
                     % (2 * n, round(1000 * em[j] / tot)))
        else:
            spec += (">{\\RaggedRight\\arraybackslash\\hyphenpenalty=10000"
                     "\\exhyphenpenalty=10000}"
                     "p{\\dimexpr (\\linewidth-%.2fem-%d\\tabcolsep)*%d/1000\\relax}"
                     % (tot, 2 * n, round(1000 * weights[j] / wsum)))
    hdr = _row([r"\textbf{%s}" % inline(h) for h in header])
    nb = len(body)
    if nb <= 20:
        # a short table is set as one unbreakable tabular: a longtable that
        # starts near a page foot can leave its header behind on its own
        rows_ = [_row([inline(c) for c in r]) for r in body]
        return [r"{\scriptsize\singlespacing\renewcommand{\arraystretch}{1.05}",
                r"\setlength{\tabcolsep}{3pt}", r"\begin{center}",
                r"\begin{tabular}{%s}" % spec, r"\toprule", hdr, r"\midrule"] + rows_ + [
                r"\bottomrule", r"\end{tabular}", r"\end{center}", r"}"]
    # a long table (Table 1 has 48 rows) is set a little tighter so that it
    # does not leave a few rows alone on the next page
    out = [r"{\scriptsize\singlespacing\renewcommand{\arraystretch}{%s}"
           % ("0.92" if nb > 40 else "1.05"),
           r"\setlength{\tabcolsep}{3pt}",
           r"\begin{longtable}{%s}" % spec, r"\toprule", hdr,
           r"\midrule", r"\endfirsthead", r"\toprule", hdr, r"\midrule", r"\endhead"]
    # no page break within the first three rows or the last three, and none
    # at all in a table of six rows or fewer, so that no row is stranded on a
    # page
    nb = len(body)
    # group heading rows (a bold first cell and nothing else), as in Tables 3
    # and S1f: page breaks are then allowed only before a group heading
    grp = [bool(re.match(r"^\*\*.+\*\*$", r[0] or "")) and not any(c for c in r[1:])
           for r in body]
    grouped = sum(grp) >= 2
    for k, r in enumerate(body):
        s = _row([inline(c) for c in r])
        if grouped:
            keep = k < nb - 1 and not grp[k + 1]
        else:
            keep = k < nb - 1 and (nb <= 6 or k <= 1 or k >= nb - 3)
        if keep:
            s = s[:-2] + r"\\*"
        out.append(s)
    return out + [r"\bottomrule", r"\end{longtable}", r"}"]


def convert(md, figdir=None, preamble=None, skip_h1=False):
    return ((preamble if preamble is not None else PREAMBLE)
            + "\n".join(convert_body(md, figdir=figdir, skip_h1=skip_h1))
            + "\n\\end{document}\n")


def convert_body(md, figdir=None, skip_h1=False, figplace="htbp"):
    """Markdown body -> LaTeX lines.

    ``figdir`` overrides where figures are looked for; pass a relative path when
    building a package whose .tex and images must sit in one directory (arXiv).
    ``skip_h1`` drops the level-1 title, for documents that set their own.
    ``figplace`` is the float specifier.  The preprint passes ``H``, which pins
    each figure exactly where it is written: with ``htbp`` a figure sitting
    among the long supplementary tables cannot be placed and LaTeX defers it to
    the end of the document, which is the opposite of what is wanted there.
    """
    # relative to submission/, where the .tex is compiled: no absolute path of
    # the machine that built it ends up in the .tex or the PDF
    figroot = (os.path.relpath(FIG, SUB).replace("\\", "/") if figdir is None
               else figdir)
    lines = md.split("\n")
    out, i, in_list = [], 0, None
    # the number the next item of an ordered list must carry to open a new item
    # inside a paragraph of continuation lines (see the paragraph collector)
    enum_next = None
    landscape, legend_label, legend_at, table_legend = False, None, 0, False
    lab_rx = re.compile(r"\*\*((?:Supplementary )?Figure S?\d+)\.\*\*")
    while i < len(lines):
        s = lines[i].rstrip()
        m = re.match(r"<!--\s*FIG\s+([\w.]+)\s*\|\s*(.*?)\s*-->", s)
        if m:
            # a figure placed directly after its own full legend gets no second,
            # short caption
            lm = lab_rx.match(m.group(2))
            own = bool(lm and lm.group(1) == legend_label)
            if own and figplace != "H" and not any(x_.strip()
                                                   for x_ in out[legend_at + 1:]):
                # the legend, already emitted, moves into the figure's float
                # below the image, which is shrunk to leave it room, so that
                # the two are never on different pages
                leg = " ".join(out[legend_at:]).strip()
                del out[legend_at:]
                out += [r"\begin{figure}[p]", r"\centering",
                        r"\sbox{\legendbox}{\parbox{\textwidth}{\small %s}}" % leg,
                        r"\includegraphics[width=\textwidth,height=\dimexpr "
                        r"0.9\textheight-\ht\legendbox-\dp\legendbox\relax,"
                        r"keepaspectratio]{%s}"
                        % os.path.join(figroot, m.group(1)).replace("\\", "/"),
                        r"\par\medskip", r"\usebox{\legendbox}", r"\end{figure}"]
                legend_label = None
                i += 1
                continue
            cap = ([] if own else [r"\caption{%s}" % inline(m.group(2))])
            out += [r"\begin{figure}[%s]" % figplace, r"\centering",
                    r"\includegraphics[width=\textwidth,height=0.8\textheight,"
                    r"keepaspectratio]{%s}"
                    % os.path.join(figroot, m.group(1)).replace("\\", "/")] + cap + [
                    r"\end{figure}"]
            legend_label = None
            i += 1
            continue
        if "<!-- PAGEBREAK -->" in s:
            out.append(r"\clearpage")
            i += 1
            continue
        m = re.match(r"<!--\s*LANDSCAPE\s+(on|off)\s*-->", s)
        if m:
            landscape = m.group(1) == "on"
            out.append(r"\begin{landscape}" if landscape else r"\end{landscape}")
            i += 1
            continue
        if "<!-- NOINDENT -->" in s:
            out.append(r"\setlength{\parindent}{0pt}")
            i += 1
            continue
        if s.startswith("<!--"):
            i += 1
            continue
        # fenced code block -> verbatim, passed through unescaped and with its
        # line breaks preserved (otherwise the commands run together)
        if s.lstrip().startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                block.append(lines[i].rstrip())
                i += 1
            i += 1
            if in_list:
                out.append(r"\end{%s}" % in_list)
                in_list = None
            out += [r"\begin{quote}", r"\small", r"\begin{verbatim}"]
            out += block
            out += [r"\end{verbatim}", r"\end{quote}"]
            continue
        # display math on its own line
        if s.strip().startswith("$$") and s.strip().endswith("$$") and len(s.strip()) > 4:
            out += [r"\[", s.strip()[2:-2], r"\]"]
            i += 1
            continue
        # display math over several lines: collect up to the closing $$ (the
        # paragraph branch below stops at a line starting with $$, and would
        # otherwise make no progress)
        if s.strip().startswith("$$"):
            block = [s.strip()[2:]]
            i += 1
            while i < len(lines) and not lines[i].rstrip().endswith("$$"):
                block.append(lines[i].strip())
                i += 1
            if i < len(lines):
                block.append(lines[i].strip()[:-2])
                i += 1
            out += [r"\["] + [b for b in block if b] + [r"\]"]
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(
                r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            blk = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                blk.append(lines[i])
                i += 1
            if in_list:
                out.append(r"\end{%s}" % in_list)
                in_list = None
            out += _table(blk, landscape=landscape)
            legend_label = None
            table_legend = False
            continue
        if re.match(r"^---+$", s):
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            if in_list:
                out.append(r"\end{%s}" % in_list)
                in_list = None
            lvl, txt = len(m.group(1)), inline(m.group(2))
            if lvl == 1:
                if not skip_h1:
                    out += [r"\begin{center}", r"{\LARGE\bfseries %s}" % txt,
                            r"\end{center}", r"\vspace{1em}"]
            else:
                # a table's heading, legend and first rows start on one page:
                # the room needed is estimated from the legend's length and, for
                # a table set unbreakable (20 rows or fewer), all of its rows
                if lvl == 2 and re.match(r"(?:Supplementary )?Table S?\d", m.group(2)):
                    j_, chars_ = i + 1, 0
                    while j_ < len(lines) and not lines[j_].strip().startswith("|"):
                        if not lines[j_].strip().startswith("<!--"):
                            chars_ += len(lines[j_])
                        j_ += 1
                    k_ = j_
                    while k_ < len(lines) and lines[k_].strip().startswith("|"):
                        k_ += 1
                    rows_ = max(k_ - j_ - 2, 0)
                    need_ = 40 + 12 * (chars_ // 100 + 1) + 11 * (
                        (rows_ + 3) if rows_ <= 20 else 8)
                    out.append(r"\needspace{%dpt}" % min(need_, 560))
                    table_legend = True
                out.append("\\%s{\\raggedright\\hyphenpenalty=10000 %s}"
                           % ({2: "section*", 3: "subsection*",
                               4: "subsubsection*"}[lvl], txt))
            i += 1
            continue
        for pat, env in ((r"^\s*[-*]\s+(.*)$", "itemize"),
                         (r"^\s*\d+\.\s+(.*)$", "enumerate")):
            m = re.match(pat, s)
            if m:
                if in_list != env:
                    if in_list:
                        out.append(r"\end{%s}" % in_list)
                    out.append(r"\begin{%s}" % env)
                    in_list = env
                out.append(r"\item %s" % inline(m.group(1)))
                if env == "enumerate":
                    enum_next = int(re.match(r"^\s*(\d+)", s).group(1)) + 1
                break
        else:
            if in_list and not s.strip():
                out.append(r"\end{%s}" % in_list)
                in_list = None
            if s.startswith(">"):
                out += [r"\begin{quote}\itshape", inline(s.lstrip("> ")), r"\end{quote}"]
                i += 1
                continue
            if not s.strip():
                out.append("")
                i += 1
                continue
            para = []
            while i < len(lines):
                t = lines[i].rstrip()
                # as in CommonMark, only a bullet or an ordered item numbered 1
                # interrupts a paragraph (a wrapped line that starts with "80."
                # is text, not a list); inside an ordered list the item with the
                # next number does too, so that items whose text runs over
                # several lines are not merged into the first
                if (not t.strip() or t.startswith((">", "|", "<!--", "$$"))
                        or re.match(r"^#{1,4}\s", t)
                        or re.match(r"^\s*([-*]|1\.)\s+", t) or re.match(r"^---+$", t)
                        or (in_list == "enumerate" and enum_next is not None
                            and re.match(r"^\s*%d\.\s+" % enum_next, t))):
                    break
                para.append(t.strip())
                i += 1
            if not para:
                # a line that ends a paragraph but that no branch above takes
                # is text; consuming it guarantees progress
                para.append(lines[i].strip())
                i += 1
            joined = " ".join(para)
            lm = lab_rx.match(joined)
            legend_label = lm.group(1) if lm else None
            k_ = i
            while k_ < len(lines) and not lines[k_].strip():
                k_ += 1
            if (len(joined) < 300 and k_ < len(lines)
                    and lines[k_].strip().startswith("|")):
                j_ = k_
                while j_ < len(lines) and lines[j_].strip().startswith("|"):
                    j_ += 1
                rows_ = [_cells(ln) for ln in lines[k_:j_]]
                nbody = len(rows_) - 2
                whole = nbody <= 6 or (len(rows_[0]) <= 7 and max(
                    len(c) for r_ in rows_ for c in r_) <= 22)
                # table rows are single-spaced (about 13 pt); the paragraph
                # itself takes about two body lines
                out.append(r"\needspace{%dpt}" % ((nbody + 4) * 13 + 50) if whole
                           else r"\needspace{8\baselineskip}")
            legend_at = len(out)
            if table_legend:
                # a table legend is set small and single-spaced, like its table
                out.append(r"{\small\singlespacing " + inline(joined) + r"\par}")
                table_legend = False
                continue
            out.append(inline(joined))
            continue
        i += 1
    if in_list:
        out.append(r"\end{%s}" % in_list)
    return out


def render(paths=None):
    # supplementary_methods is the hand-maintained part of the supplement and is
    # also circulated on its own, so it is built here too -- left out, its PDF
    # silently kept an earlier day's text.
    paths = paths or [os.path.join(SUB, n + ".md") for n in
                      ("manuscript", "tables", "supplementary",
                       "supplementary_methods", "cover_letter", "title_page")]
    for src in paths:
        if not os.path.exists(src):
            print("  skipped (not found): %s" % os.path.basename(src))
            continue
        tex = os.path.splitext(src)[0] + ".tex"
        open(tex, "w", encoding="utf-8", newline="\n").write(
            convert(open(src, encoding="utf-8").read()))
        d, base = os.path.dirname(tex) or ".", os.path.basename(tex)
        for _ in range(2):
            res = subprocess.run([ENGINE, "-interaction=nonstopmode", base],
                                 cwd=d, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
        pdf, log = (os.path.splitext(tex)[0] + e for e in (".pdf", ".log"))
        pages = 0
        if os.path.exists(log):
            mm = re.search(r"Output written on .*? \((\d+) pages?",
                           open(log, encoding="utf-8", errors="replace").read())
            pages = int(mm.group(1)) if mm else 0
        if pages > 200:
            print("REFUSED %s: %d pages -- a LaTeX runaway, check table widths"
                  % (os.path.basename(pdf), pages))
            if os.path.exists(pdf):
                os.remove(pdf)
        elif os.path.exists(pdf):
            print("  %-26s %d pages, %d bytes" % (os.path.basename(pdf), pages,
                                                  os.path.getsize(pdf)))
        for msg in driver_problems(res.stdout) + log_problems(log):
            print("  LATEX  %-26s %s" % (os.path.basename(pdf), msg))
        for ext in (".aux", ".log", ".out"):
            f = os.path.splitext(tex)[0] + ext
            if os.path.exists(f):
                os.remove(f)


# =====================================================================
# 3b.  arXiv preprint package
# =====================================================================
ARXIV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arxiv")

ARXIV_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{mathpazo}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{array}
\usepackage{ragged2e}
\usepackage[font=small,labelfont=bf,justification=raggedright,
            singlelinecheck=false,labelformat=empty]{caption}
\usepackage{microtype}
\usepackage{setspace}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\pdfsuppressptexinfo=-1
\usepackage{pdflscape}
\usepackage{float}
\usepackage{needspace}
\usepackage{authblk}
\onehalfspacing
\setlength{\parskip}{2pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.15}
\renewcommand\Affilfont{\small\normalfont}
\title{\bfseries @@TITLE@@}
\author[1]{Shengjun Wu\thanks{Corresponding author: \texttt{sjwu@nju.edu.cn}}}
\author[1,2]{Jeffery Wu}
\affil[1]{National Laboratory of Solid State Microstructures and School of
Physics, and Collaborative Innovation Center of Advanced Microstructures,
Nanjing University, Nanjing 210093, China}
\affil[2]{School of Physics and Astronomy, Shanghai Jiao Tong University,
Shanghai 200240, China}
\date{\today}
\begin{document}
\maketitle
"""


# Supplementary figures are cited from the main text, so there is no citing
# paragraph inside the appendix to attach them to.  Each is placed instead at the
# end of the supplementary section whose subject it illustrates, named here by
# the heading it should sit in front of.
ARXIV_SUPP_FIG_ANCHOR = {
    "figS1_flow.pdf": "## S2. ",             # end of S1, data sources and cohort
    "figS2_dag.pdf": "## S3. ",              # end of S2, what the measures identify
    "figS3_sensitivity.pdf": "## Supplementary Table S4. ",   # after Table S3
}


def _arxiv_place_figures(md, legend_heading, anchors=None):
    """Move each figure from the collected legend section to where it is cited.

    A journal wants the legends listed together and the figures supplied as
    separate files; a preprint is read as one document, where that arrangement
    puts every figure pages away from the sentence that needs it.  So the same
    single source is *rearranged* for the preprint rather than duplicated: the
    full legend becomes the figure's caption, the figure goes after the
    paragraph that first cites it, and the separate legend section is dropped so
    that nothing is stated twice.
    """
    figrx = re.compile(r"<!--\s*FIG\s+([\w.]+)\s*\|\s*(.*?)\s*-->")
    figs = [(m.group(1), m.group(2)) for m in figrx.finditer(md)]
    if not figs:
        return md

    # the legend section, and everything in it, leaves the document
    cut = md.find("\n" + legend_heading)
    if cut < 0:
        return md
    head, section = md[:cut], md[cut:]
    nxt = re.search(r"\n#{1,2} (?!.*(?:Figure legends|Supplementary Figures))",
                    section[1:])
    tail = section[1:][nxt.start():] if nxt else ""
    md = head.rstrip() + "\n" + tail
    md = re.sub(r"\n<!-- PAGEBREAK -->\s*$", "", md)

    # full legends, keyed by the label they open with
    legends = {}
    for para in re.split(r"\n\s*\n", section):
        m = re.match(r"\*\*((?:Supplementary )?Figure S?\d+)\.\*\*", para.strip())
        if m:
            legends[m.group(1)] = " ".join(para.strip().split())

    blocks = re.split(r"\n\s*\n", md)
    for path, short in figs:
        m = re.match(r"\*\*((?:Supplementary )?Figure S?\d+)\.\*\*", short)
        label = m.group(1) if m else None
        caption = legends.get(label, short)
        marker = "<!-- FIG %s | %s -->" % (path, caption)

        at = None
        anchor = (anchors or {}).get(path)
        if anchor:
            at = next((i for i, b in enumerate(blocks)
                       if b.lstrip().startswith(anchor)), None)
        elif label:
            # first paragraph that cites it; "Figure 3" also matches "Figure 3a"
            rx = re.compile(r"\b" + re.escape(label) + r"[a-c]?\b")
            at = next((i + 1 for i, b in enumerate(blocks)
                       if rx.search(b) and not b.lstrip().startswith("**")), None)
        if at is None:
            blocks.append(marker)            # never cited: keep it at the end
        else:
            blocks.insert(at, marker)
    return "\n\n".join(b for b in blocks if b.strip())


def arxiv():
    """Assemble the single self-contained preprint: author block, abstract, main
    text, references, figures near where they are cited, tables, and the
    supplementary material as an appendix.

    Tables are deliberately *not* moved next to their citations, unlike the
    figures: Tables 1 and 3 run to dozens of rows, and dropping one into the
    middle of Results breaks the argument for a page and a half.  They stay in
    their own section after the text, as in the submission.  The wording is the submitted
    manuscript verbatim -- the preprint and the submission must not differ in
    what they say, only in where the figures sit."""
    os.makedirs(ARXIV_DIR, exist_ok=True)
    doc, abst = _arxiv_tex()
    tex = os.path.join(ARXIV_DIR, "lifestyle_direct_association.tex")
    open(tex, "w", encoding="utf-8", newline="\n").write(doc)
    return _arxiv_compile(tex, abst)


def _arxiv_tex():
    """The preprint's LaTeX source, as a string, and its abstract."""
    man = open(os.path.join(SUB, "manuscript.md"), encoding="utf-8").read()

    # pull the abstract out; it becomes a LaTeX abstract environment
    abst = re.search(r"## Abstract\n(.*?)\n## Key Messages", man, re.S).group(1)
    body = man[man.index("## Key Messages"):]

    tables = open(os.path.join(SUB, "tables.md"), encoding="utf-8").read()
    supp = open(os.path.join(SUB, "supplementary.md"), encoding="utf-8").read()

    # the preprint is published material in its own right, so it carries the
    # declarations of the title page (ethics, data and code availability, the
    # disclaimer NCHS asks for, conflicts, funding) before the references
    tp = os.path.join(SUB, "title_page.md")
    if os.path.exists(tp):
        st = open(tp, encoding="utf-8").read()
        paras = [p_.strip() for p_ in re.split(r"\n\s*\n", st)]
        keep = [p_ for p_ in paras if re.match(
            r"\*\*(Ethics approval|Data availability|Code availability|Disclaimer|"
            r"Conflict of interest|Funding)\.\*\*", p_)]
        if keep:
            decl = "## Declarations\n\n" + "\n\n".join(keep) + "\n\n"
            body = body.replace("## References", decl + "## References", 1)

    # the title comes from the manuscript itself, broken after its question mark
    title = inline(re.match(r"# (.*)", man).group(1).strip())
    doc = ARXIV_PREAMBLE.replace("@@TITLE@@", title.replace("? ", "?\\\\\n", 1))
    doc += "\\begin{abstract}\n\\noindent\n"
    doc += "\n".join(convert_body(abst, figdir=".")) + "\n\\end{abstract}\n\n"
    body = _arxiv_place_figures(body, "## Figure legends")
    supp = _arxiv_place_figures(supp, "# Supplementary Figures",
                                ARXIV_SUPP_FIG_ANCHOR)
    doc += "\n".join(convert_body(body, figdir=".", skip_h1=True, figplace="H"))
    doc += "\n\\clearpage\n"
    doc += "\n".join(convert_body(tables, figdir="."))
    doc += "\n\\clearpage\n\\appendix\n"
    doc += "\n".join(convert_body(supp, figdir=".", figplace="H"))
    doc += "\n\\end{document}\n"
    return doc, abst


def _arxiv_compile(tex, abst):
    # figures must sit beside the .tex for an arXiv source upload
    import shutil
    for f in os.listdir(FIG):
        if f.endswith(".pdf"):
            shutil.copy(os.path.join(FIG, f), os.path.join(ARXIV_DIR, f))

    for _ in range(2):
        res = subprocess.run([ENGINE, "-interaction=nonstopmode",
                              os.path.basename(tex)], cwd=ARXIV_DIR,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
    pdf, log = (os.path.splitext(tex)[0] + e for e in (".pdf", ".log"))
    pages = 0
    if os.path.exists(log):
        mm = re.search(r"Output written on .*? \((\d+) pages?",
                       open(log, encoding="utf-8", errors="replace").read())
        pages = int(mm.group(1)) if mm else 0
    print("  %-34s %d pages, %d bytes" % (os.path.basename(pdf), pages,
                                          os.path.getsize(pdf)
                                          if os.path.exists(pdf) else 0))
    for msg in driver_problems(res.stdout) + log_problems(log):
        print("  LATEX  " + msg)
    for ext in (".aux", ".log", ".out"):
        f = os.path.splitext(tex)[0] + ext
        if os.path.exists(f):
            os.remove(f)
    # plain-text abstract for the arXiv metadata field
    plain = re.sub(r"\*\*(.*?)\*\*", r"\1", abst)
    plain = re.sub(r"\*(.*?)\*", r"\1", plain)
    plain = re.sub(r"\$[^$]*\$", "", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    open(os.path.join(ARXIV_DIR, "abstract.txt"), "w",
         encoding="utf-8", newline="\n").write(plain + "\n")
    ascii_ = plain
    for a, b in [("–", "-"), ("—", " -- "), ("≥", ">="), ("≤", "<="),
                 ("−", "-"), (" ", " "), (" ", " "),
                 ("×", "x"), ("’", "'"), ("“", '"'), ("”", '"'), ("·", "."),
                 ("²", "^2")]:
        ascii_ = ascii_.replace(a, b)
    if any(ord(ch) > 127 for ch in ascii_):
        print("  WARNING abstract_ascii.txt: unmapped characters %s"
              % sorted({ch for ch in ascii_ if ord(ch) > 127}))
    ascii_ = ascii_.encode("ascii", "replace").decode("ascii")
    open(os.path.join(ARXIV_DIR, "abstract_ascii.txt"), "w",
         encoding="utf-8", newline="\n").write(ascii_ + "\n")
    print("  abstract.txt / abstract_ascii.txt  %d characters" % len(plain))
    return pages


# =====================================================================
# 4.  Documents from templates, and verification
# =====================================================================
#: The manuscript, the Supplementary Methods and the cover letter are rendered
#: from templates (submission/templates/) in which every result is written as
#: a reference into output/numbers.json, {{path|format}}: path keys separated
#: by '::' (list items by index); format a printf format such as '.4f', or
#: 'int' (thin-space thousands), 'pct' (100 x, integer), 'p' (<0.001 or three
#: decimals), 'p2' (two significant figures), 'pp1' (a proportion in percentage
#: points, one decimal), 'd' (integer, keeping the sign),
#: optionally prefixed 'abs'.  verify re-renders them and compares.
TEMPLATE_DIR = os.path.join(SUB, "templates")
TEMPLATED = ("manuscript.md", "supplementary_methods.md", "cover_letter.md")
_TPL_RX = re.compile(r"\{\{([^}|]+)(?:\|([^}]+))?\}\}")


def _tget(N, path):
    v = N
    for k in path.split("::"):
        k = k.strip()
        v = v[int(k)] if isinstance(v, list) else v[k]
    return v


def _tfmt(v, f):
    if not f:
        return str(v)
    if f.startswith("abs"):
        v, f = abs(float(v)), f[3:]
    if f == "int":
        return _int(round(float(v)))
    if f == "pct":
        return "%d" % int(round(100 * float(v)))
    if f == "p":
        return "<0.001" if float(v) < 0.001 else "%.3f" % float(v)
    if f == "pp1":
        return "%.1f" % (100 * float(v))
    if f == "p2":
        return _pv2(float(v))
    out = ("%" + f) % float(v)
    return out.replace("-", MINUS) if out.startswith("-") else out


def fill_text(N, text):
    """Render one template; returns (text, unresolved references, number of
    references)."""
    missing, count = [], [0]

    def sub(m):
        count[0] += 1
        try:
            return _tfmt(_tget(N, m.group(1)), m.group(2))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            missing.append("%s (%s)" % (m.group(0), exc))
            return m.group(0)
    return _TPL_RX.sub(sub, text), missing, count[0]


def fill():
    """Render the documents from their templates with output/numbers.json."""
    N = json.load(open(os.path.join(OUT, "numbers.json"), encoding="utf-8"))
    done = 0
    for name in TEMPLATED:
        src = os.path.join(TEMPLATE_DIR, name)
        if not os.path.exists(src):
            print("  %s: no template" % name)
            continue
        out, missing, k = fill_text(N, open(src, encoding="utf-8").read())
        if missing:
            raise SystemExit("unresolved in %s:\n  %s" % (name, "\n  ".join(missing)))
        open(os.path.join(SUB, name), "w", encoding="utf-8", newline="\n").write(out)
        print("  %-28s %d numbers from numbers.json" % (name, k))
        done += 1
    if done:
        _assemble_supplement()


def _plain(v):
    """A numpy scalar as the Python number it holds (numpy 2 prints its repr
    as np.float64(...))."""
    return v.item() if hasattr(v, "item") else v


def compare(ref=None, rtol=1e-6):
    """Compare output/ with reference_output/: every cell of every CSV, and
    every value of every JSON file except the recorded run parameters
    (params_*.json), by its key path -- numbers within ``rtol``, strings,
    flags and nulls exactly.  Deterministic outputs agree bit for bit on the
    platform the reference was made on; across platforms, within ``rtol``."""
    here = os.path.dirname(os.path.abspath(__file__))
    ref = ref or next((p for p in (os.path.join(here, "reference_output"),
                                   os.path.join(here, "shared", "reference_output"))
                       if os.path.isdir(p)), os.path.join(here, "reference_output"))
    if not os.path.isdir(ref):
        print("compare: %s not found" % ref)
        return ["no reference_output"]
    bad, n = [], 0
    for f in sorted(os.listdir(ref)):
        a, b = os.path.join(OUT, f), os.path.join(ref, f)
        if f.endswith(".csv"):
            if not os.path.exists(a):
                bad.append("%s missing" % f)
                continue
            x, y = pd.read_csv(a), pd.read_csv(b)
            if list(x.columns) != list(y.columns) or len(x) != len(y):
                bad.append("%s: different shape or columns" % f)
                continue
            for c in x.columns:
                n += len(x)
                if x[c].dtype.kind in "fi" and y[c].dtype.kind in "fi":
                    xv, yv = x[c].to_numpy(float), y[c].to_numpy(float)
                    both = np.isnan(xv) & np.isnan(yv)
                    diff = ~both & ~np.isclose(xv, yv, rtol=rtol, atol=1e-12)
                else:
                    diff = (x[c].astype(str) != y[c].astype(str)).to_numpy()
                if diff.any():
                    ids = [k for k in ("analysis", "spec", "scale", "scenario", "habit",
                                       "habit_a", "habit_b", "level", "label", "K", "n", "b")
                           if k in y.columns and k != c][:4]
                    bad.append("\n      ".join(
                        ["%s: column %s differs in %d rows" % (f, c, int(diff.sum()))]
                        + ["row %d (%s): %r, reference %r"
                           % (i_, ", ".join("%s=%s" % (k, y[k].iloc[i_]) for k in ids),
                              _plain(x[c].iloc[i_]), _plain(y[c].iloc[i_]))
                           for i_ in np.flatnonzero(diff)[:5]]))
        elif f.endswith(".json") and not f.startswith("params_"):
            if not os.path.exists(a):
                bad.append("%s missing" % f)
                continue
            vx = dict(_flat_items(json.load(open(a, encoding="utf-8"))))
            vy = dict(_flat_items(json.load(open(b, encoding="utf-8"))))
            n += len(vy)
            for k_ in sorted(set(vx) | set(vy)):
                p_, q_ = vx.get(k_, "<absent>"), vy.get(k_, "<absent>")
                num_ = (isinstance(p_, (int, float)) and isinstance(q_, (int, float))
                        and not isinstance(p_, bool) and not isinstance(q_, bool))
                same_ = (np.isclose(float(p_), float(q_), rtol=rtol, atol=1e-12)
                         or (np.isnan(float(p_)) and np.isnan(float(q_)))) if num_ \
                    else p_ == q_
                if not same_:
                    bad.append("%s: %s is %r, reference %r" % (f, k_, p_, q_))
    print("compare    : %d values against %s" % (n, ref))
    print("RESULT     : %s" % ("identical within rtol %g" % rtol if not bad
                               else "%d differences" % len(bad)))
    for x_ in bad:
        print("   " + x_)
    return bad


# =====================================================================
# 5.  Verification
# =====================================================================
def _norm(t):
    """Normalise a text for number matching: thin and non-breaking spaces and
    thousands separators removed, dashes and minus signs made hyphens."""
    t = t.replace("\u2009", " ").replace("\u00a0", " ")
    for a in "\u2013\u2014\u2212":
        t = t.replace(a, "-")
    return re.sub(r"(?<=\d)[ ,](?=\d{3}\b)", "", t)


def _flat_values(obj, out):
    """Every number in a nested numbers.json structure."""
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flat_values(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flat_values(v, out)
    return out


def _flat_items(obj, prefix=""):
    """(key path, value) for every leaf of a nested JSON structure: numbers,
    strings, flags and nulls alike."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat_items(v, "%s/%s" % (prefix, k) if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flat_items(v, "%s[%d]" % (prefix, i))
    else:
        yield prefix, obj


def _renderings(values):
    """The strings under which a value may legitimately be quoted."""
    out = set()
    for v in values:
        for x in (v, abs(v)):
            if float(x).is_integer():
                out.add("%d" % x)
            for k in range(0, 5):
                out.add(("%." + str(k) + "f") % x)
            # two or three significant figures, the form of a small P
            for f in ("%.2g", "%#.2g", "%.3g"):
                s = f % x
                if "e" not in s:
                    out.add(s)
    return out


#: numbers that may appear in the prose without being analysis results: the
#: design constants, cut-points, calendar dates and cycle years
def _constants():
    c = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "14", "15",
         "18", "20", "24", "25", "30", "34", "35", "44", "45", "54", "55", "64", "108",
         "65", "74", "75", "79", "80", "100", "150", "200", "240", "419", "420",
         "500", "599", "600", "1000", "1799", "1800", "2000",
         "0.05", "0.5", "0.25", "0.75", "1.0", "1.5", "0.8", "2.2", "3.5",
         "18.5", "24.9", "29.9", "34.9", "0.558", "0.5579", "1.96", "0.0033",
         "0.975", "0.025", "0.008", "0.0083", "95", "97.5", "2.5", "31", "2019",
         "2022", "2026",
         "2.2", "10000", "29371",
         # upper 5% points of chi-square on 1, 2, 5, 9 df (Supplementary S4)
         "0.050", "3.841", "5.991", "11.070", "16.919"}
    for y in range(1999, 2019):
        c.add(str(y))
    return c


def _body(md):
    """Prose only: without the reference list, maths, code, comments, figure
    markers and citation brackets."""
    t = re.split(r"\n## (?:Supplementary r|R)eferences", md)[0]
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$|\$[^$]*\$", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"\[([1-9]\d{0,2}(?:\s*[,\u2013-]\s*[1-9]\d{0,2})*)\]", " ", t)
    t = re.sub(r"\[@[^\]]*\]", " ", t)
    t = re.sub(r"(?:Supplementary )?(?:Table|Figure|Methods?)\s+S?\d+[a-f]?"
               r"(?:\.\d+)?(?:\s*[\u2013-]\s*S?\d+[a-f]?)?", " ", t)
    t = re.sub(r"\bS\d+[a-f]?(?:\.\d+)?\b|\bSA\d+[a-d]?\b|\u00a7\s*S?\d+(?:\.\d+)?",
               " ", t)
    # identifiers rather than quantities: variable lists (PAQ605/610/PAD615),
    # missing-value codes (7/9, 77/99), SHA-256, arXiv/DOI identifiers,
    # cross-references to numbered references, postal codes
    t = re.sub(r"\b[A-Z]{2,}\w*(?:/\w+)+", " ", t)
    t = re.sub(r"\b7+/9+\b", " ", t)
    t = re.sub(r"SHA-256|(?:arXiv|doi):\S+", " ", t)
    t = re.sub(r"references?\s+\d+(?:\s*(?:,|and|–|-)\s*\d+)*", " ", t)
    # sections and chapters of a cited book
    t = re.sub(r"\b(?:sections?|chapters?)\s+\d+(?:\.\d+)*", " ", t)
    t = re.sub(r"\b\d{6}(?=, China)", " ", t)
    # ordinals label percentiles and positions (the 1st-99th percentile)
    t = re.sub(r"\b\d+(?:\.\d+)?(?:st|nd|rd|th)\b", " ", t)
    return t


def _word_count(md, start, stop):
    """Whitespace tokens between two headings, excluding the heading lines and
    layout comments (<!-- PAGEBREAK -->)."""
    i = md.index(start)
    j = md.index(stop, i + len(start))
    lines = [ln for ln in md[i:j].split("\n")[1:]
             if not ln.startswith("#") and not ln.strip().startswith("<!--")]
    return len(" ".join(lines).split())


def verify(path=None):
    """Assert that the manuscript agrees with the analysis and is well formed.

    0. The manuscript, the Supplementary Methods and the cover letter are
       exactly their templates rendered with output/numbers.json, so every
       templated number is tied to the quantity it names.
    1. Every other number in the prose of those documents, and in the table
       legends, occurs among the values in output/numbers.json (an analysis
       result or a recorded design constant) or is a cut-point or date.
    2. Every hazard ratio with its interval matches cox.csv, and every paired
       difference with its interval matches pairwise.csv.
    3. References are cited in order of first appearance, and every listed
       reference is cited.
    4. Each generated .tex is exactly what its Markdown source converts to.
    5. shared/ holds the same code as the project root.
    6. No LaTeX that compiles but typesets wrongly; no untypesettable character.
    7. The word counts on the title page are the counts of the text.
    """
    path = path or os.path.join(SUB, "manuscript.md")
    if not os.path.exists(path):
        print("verify: %s not found -- nothing to check" % path)
        return []
    fails, checks = [], [0]
    N = json.load(open(os.path.join(OUT, "numbers.json"), encoding="utf-8"))
    ok = _renderings(_flat_values(N, [])) | _constants()
    tok = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.]*\d)")

    def check_numbers(p):
        if not os.path.exists(p):
            return
        body = _norm(_body(open(p, encoding="utf-8").read()))
        for m in tok.finditer(body):
            checks[0] += 1
            t = m.group(0)
            if t in ok or t.lstrip("-") in ok:
                continue
            ctx = body[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")
            fails.append("%-24s %s: '%s' in '...%s...'" % (
                "number not in numbers.json", os.path.basename(p), t, ctx))

    skipped = []
    for name in ("manuscript.md", "supplementary_methods.md", "cover_letter.md"):
        p_ = os.path.join(SUB, name) if name != "manuscript.md" else path
        tp_ = os.path.join(TEMPLATE_DIR, name)
        if os.path.exists(tp_):
            # templated numbers are checked against their keys below; here the
            # numbers written literally in the template
            lit = os.path.join(OUT, "_literal_%s" % name)
            open(lit, "w", encoding="utf-8", newline="\n").write(
                _TPL_RX.sub(" ", open(tp_, encoding="utf-8").read()))
            n_before = len(fails)
            check_numbers(lit)
            os.remove(lit)
            for j in range(n_before, len(fails)):
                fails[j] = fails[j].replace("_literal_", "template of ")
        elif os.path.exists(p_):
            check_numbers(p_)
        else:
            skipped.append("numbers in %s (file not present)" % name)

    # ---- the documents are their templates rendered with numbers.json
    bound = 0
    for name in TEMPLATED:
        src, doc = os.path.join(TEMPLATE_DIR, name), os.path.join(SUB, name)
        if not (os.path.exists(src) and os.path.exists(doc)):
            skipped.append("template binding of %s (template or document absent)" % name)
            continue
        out, missing, k = fill_text(N, open(src, encoding="utf-8").read())
        bound += k
        checks[0] += k
        if missing:
            fails.append("%-24s %s: %s" % ("template key missing", name, missing[:3]))
            continue
        have = open(doc, encoding="utf-8").read()
        if out != have:
            la, lb = out.split("\n"), have.split("\n")
            j = next((j for j in range(min(len(la), len(lb))) if la[j] != lb[j]),
                     min(len(la), len(lb)))
            fails.append("%-24s %s line %d: document '%s', rendered '%s'"
                         % ("number not from its key", name, j + 1,
                            (lb[j] if j < len(lb) else "")[:70],
                            (la[j] if j < len(la) else "")[:70]))

    # ---- numbers in the legends of the generated tables (prose lines only)
    for name in ("tables.md", "supplementary_tables.md"):
        p_ = os.path.join(SUB, name)
        if not os.path.exists(p_):
            skipped.append("legend numbers of %s (file not present)" % name)
            continue
        prose = "\n".join(ln for ln in open(p_, encoding="utf-8").read().split("\n")
                          if not ln.lstrip().startswith(("|", "#", "*How much")))
        tmp = os.path.join(OUT, "_legends_%s" % name)
        open(tmp, "w", encoding="utf-8", newline="\n").write(prose)
        n_before = len(fails)
        check_numbers(tmp)
        os.remove(tmp)
        for j in range(n_before, len(fails)):
            fails[j] = fails[j].replace("_legends_", "legend of ")

    # ---- hazard ratios and paired differences, checked in context
    cox = pd.read_csv(os.path.join(OUT, "cox.csv"))
    hr_ok = {("%.2f" % r.hr, "%.2f" % r.hr_lo, "%.2f" % r.hr_hi)
             for r in cox.itertuples() if not r.reference}
    # hazard ratios quoted from the sensitivity analyses and the age-split models
    p_ = os.path.join(OUT, "sensitivity_hr.csv")
    if os.path.exists(p_):
        hr_ok |= {("%.2f" % r.hr, "%.2f" % r.hr_lo, "%.2f" % r.hr_hi)
                  for r in pd.read_csv(p_).itertuples() if pd.notna(r.hr_lo)}
    # the habit x illness model of Supplementary Table S4b
    for v in [v_ for k_ in ("sa17_interaction", "sa17_interaction_chronic",
                            "sa17_interaction_poor_health", "sa19_interaction")
              for v_ in _extra().get(k_, {}).values()]:
        hr_ok.add(("%.2f" % v["hr0"], "%.2f" % v["hr0_lo"], "%.2f" % v["hr0_hi"]))
        hr_ok.add(("%.2f" % v["hr1"], "%.2f" % v["hr1_lo"], "%.2f" % v["hr1_hi"]))
    p_ = os.path.join(OUT, "cox_by_age.csv")
    if os.path.exists(p_):
        for r in pd.read_csv(p_).itertuples():
            hr_ok.add(("%.2f" % r.hr_young, "%.2f" % r.lo_young, "%.2f" % r.hi_young))
            hr_ok.add(("%.2f" % r.hr_old, "%.2f" % r.lo_old, "%.2f" % r.hi_old))
    pw = pd.read_csv(os.path.join(OUT, "pairwise.csv"))
    pw_ok = set()
    for r in pw.itertuples():
        for s in (1, -1):
            lo, hi = sorted([s * r.wald_lo, s * r.wald_hi])
            pw_ok.add(("%.4f" % (s * r.diff), "%.4f" % lo, "%.4f" % hi))
    hr_pat = re.compile(r"\b(\d\.\d\d)(?:\s*\(|,\s+)(?:95% CI\s*)?(\d\.\d\d)-(\d\.\d\d)\b")
    pw_pat = re.compile(r"\b(-?0\.\d{4})\s*\((?:95% CI\s*)?(-?0\.\d{4}) to (-?0\.\d{4})\)")
    for p in (path, os.path.join(SUB, "tables.md")):
        if not os.path.exists(p):
            continue
        t = _norm(open(p, encoding="utf-8").read())
        for mm in hr_pat.finditer(t):
            checks[0] += 1
            if mm.groups() not in hr_ok:
                fails.append("%-24s %s: %s (%s-%s)" % (("hazard ratio not in cox.csv",
                                                        os.path.basename(p)) + mm.groups()))
        for mm in pw_pat.finditer(t):
            checks[0] += 1
            if mm.groups() not in pw_ok:
                fails.append("%-24s %s: %s (%s to %s)" % (("difference not in pairwise",
                                                          os.path.basename(p)) + mm.groups()))

    # ---- references: all cited, cited in order of first appearance
    raw = open(path, encoding="utf-8").read()
    body = raw.split("## References")[0]
    first = []
    for mm in re.finditer(r"\[([1-9]\d{0,2}(?:\s*[,\u2013-]\s*[1-9]\d{0,2})*)\]", body):
        for part in mm.group(1).split(","):
            part = part.strip()
            r2 = re.match(r"^(\d+)\s*[\u2013-]\s*(\d+)$", part)
            nums = range(int(r2.group(1)), int(r2.group(2)) + 1) if r2 else [int(part)]
            for k in nums:
                if k not in first:
                    first.append(k)
    listed = [int(mm.group(1)) for mm in re.finditer(r"^(\d+)\. ", raw.split(
        "## References")[1] if "## References" in raw else "", re.M)]
    checks[0] += 3
    if sorted(listed) != list(range(1, len(listed) + 1)):
        fails.append("%-24s %s" % ("reference list not 1..n", listed))
    if set(listed) - set(first):
        fails.append("%-24s %s" % ("listed but never cited", sorted(set(listed) - set(first))))
    if set(first) - set(listed):
        fails.append("%-24s %s" % ("cited but not listed", sorted(set(first) - set(listed))))
    if first != sorted(first):
        fails.append("%-24s first citations in the order %s" % ("citation order", first))
    if re.search(r"[A-Za-z0-9)]\[[1-9]", body):
        fails.append("%-24s a citation directly after a word is not superscripted"
                     % "citation format")

    # ---- generated LaTeX is current
    for name in ("manuscript", "tables", "supplementary", "supplementary_methods",
                 "cover_letter", "title_page"):
        md, tex = (os.path.join(SUB, name + e) for e in (".md", ".tex"))
        if os.path.exists(md) and os.path.exists(tex):
            checks[0] += 1
            if open(tex, encoding="utf-8").read() != convert(open(md, encoding="utf-8").read()):
                fails.append("%-24s %s.tex is not the conversion of %s.md -- run render"
                             % ("stale LaTeX", name, name))

    # ---- shared/ holds the same code and document sources
    here = os.path.dirname(os.path.abspath(__file__))
    same = ["lsm_core.py", "lsm_analysis.py", "lsm_report.py",
            "crosscheck_statsmodels.py"] + [
        os.path.join("submission", f) for f in
        ("manuscript.md", "supplementary_methods.md", "title_page.md")] + [
        os.path.join("submission", "templates", f) for f in TEMPLATED
        if f != "cover_letter.md"]
    if os.path.isdir(os.path.join(here, "shared")):
        for f in same:
            other, mine = os.path.join(here, "shared", f), os.path.join(here, f)
            if not os.path.exists(mine):
                continue
            checks[0] += 1
            if not os.path.exists(other):
                fails.append("%-24s shared/%s is missing" % ("shared package", f))
            elif open(other, "rb").read() != open(mine, "rb").read():
                fails.append("%-24s shared/%s differs from %s" % ("shared package", f, f))

    # ---- the preprint is the conversion of the current sources
    atex = os.path.join(ARXIV_DIR, "lifestyle_direct_association.tex")
    if os.path.exists(atex) and os.path.exists(os.path.join(SUB, "supplementary.md")):
        checks[0] += 1
        if open(atex, encoding="utf-8").read() != _arxiv_tex()[0]:
            fails.append("%-24s arxiv .tex is not built from the current sources -- "
                         "run arxiv" % "stale preprint")
    else:
        skipped.append("preprint currency (arxiv .tex or supplement absent)")
    tp_md = os.path.join(SUB, "title_page.md")
    if not os.path.exists(tp_md):
        skipped.append("preprint Declarations and word counts (title_page.md absent)")

    # ---- no Markdown list artefact in the manuscript (a wrapped line that
    # begins "80." once became an enumerate); the reference list is the only
    # numbered list
    mtex = os.path.join(SUB, "manuscript.tex")
    if os.path.exists(mtex):
        checks[0] += 1
        k_ = open(mtex, encoding="utf-8").read().count("\\begin{enumerate}")
        if k_ > 1:
            fails.append("%-24s manuscript.tex has %d numbered lists; only the "
                         "reference list should be one" % ("list artefact", k_))

    # ---- every numbered list is rendered item by item: a list whose items
    # run over several lines was once merged into its first item
    for name in ("manuscript", "supplementary", "supplementary_methods",
                 "cover_letter", "title_page"):
        md, tex = (os.path.join(SUB, name + e) for e in (".md", ".tex"))
        if not (os.path.exists(md) and os.path.exists(tex)):
            continue
        checks[0] += 1
        want = len(re.findall(r"^\d+\.[ \t]", open(md, encoding="utf-8").read(), re.M))
        got = sum(b_.count("\\item") for b_ in re.findall(
            r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}",
            open(tex, encoding="utf-8").read(), re.S))
        if want != got:
            fails.append("%-24s %s: %d numbered items in the Markdown, %d in the "
                         "LaTeX" % ("list items merged", name, want, got))
    # ---- Z is the stratum; a test statistic must not be written Z
    for p_ in [os.path.join(SUB, n_ + ".tex") for n_ in
               ("manuscript", "tables", "supplementary", "supplementary_methods")] + [atex]:
        if os.path.exists(p_):
            checks[0] += 1
            if re.search(r"\\emph\{Z\} =|\$Z\$ =", open(p_, encoding="utf-8").read()):
                fails.append("%-24s %s: a statistic written Z = ..."
                             % ("symbol reuse", os.path.basename(p_)))

    # ---- placeholders the authors must fill before posting or submitting
    holders = {}
    for p_ in [os.path.join(SUB, f) for f in ("manuscript.md", "title_page.md",
                                             "cover_letter.md", "supplementary_methods.md",
                                             "STROBE_checklist.md", "RECORD_checklist.md")] + [
            atex, os.path.join(ARXIV_DIR, "ARXIV_SUBMISSION.md"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "HOW_TO_REPRODUCE.md"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared",
                         "HOW_TO_REPRODUCE.md")]:
        if os.path.exists(p_):
            for mm in re.finditer(r"\[(?:[A-Z][A-Z ]+[A-Z])\]", open(p_, encoding="utf-8").read()):
                holders.setdefault(mm.group(0), set()).add(os.path.basename(p_))

    # ---- typesetting
    docs = [os.path.join(SUB, n + ".md") for n in
            ("manuscript", "tables", "supplementary", "supplementary_methods",
             "cover_letter", "title_page")]
    gaps = unmapped_characters(docs)
    checks[0] += 1
    if gaps:
        fails.append("%-24s %s" % ("untypesettable", ["U+%04X %s" % (ord(c), c) for c in gaps]))
    for p in docs:
        checks[0] += 1
        if os.path.exists(p):
            t = open(p, encoding="utf-8").read()
            # a range whose two ends are the same number ('44–44%')
            for mm in re.finditer(r"(?<![\d.,\w])(\d+(?:\.\d+)?)[–-]\1(?![\d.])", t):
                fails.append("%-24s %s:%d %s" % ("degenerate range", os.path.basename(p),
                                                 t.count("\n", 0, mm.start()) + 1,
                                                 mm.group(0)))
            # a standard error rounded to zero ('(0)', '(0.0)') in a table; the
            # binomial SE of an observed proportion of 0 or 1, '(0.00)', is
            # exactly zero and allowed
            for mm in re.finditer(r"\| [^|\n]*\((?:0|0\.0)\)[^|\n]*(?=\|)", t):
                fails.append("%-24s %s:%d %s" % ("zero SE", os.path.basename(p),
                                                 t.count("\n", 0, mm.start()) + 1,
                                                 mm.group(0).strip("| ")))
            for mm in re.finditer(r"[−-]0\.0+(?![0-9])", t):
                fails.append("%-24s %s:%d %s" % ("signed zero", os.path.basename(p),
                                                 t.count("\n", 0, mm.start()) + 1,
                                                 mm.group(0)))
            # an exponent on a hatted or tilded rMI/rCMI sits above the accent
            # unless it is written {}^2
            for mm in re.finditer(r"\\(?:widehat|widetilde)\{\\mathrm\{rC?MI\}\}\^", t):
                fails.append("%-24s %s:%d %s" % ("raised exponent", os.path.basename(p),
                                                 t.count("\n", 0, mm.start()) + 1,
                                                 mm.group(0)))
    gen = [os.path.splitext(p)[0] + ".tex" for p in docs]
    gen.append(os.path.join(ARXIV_DIR, "lifestyle_direct_association.tex"))
    checks[0] += 1
    for msg in latex_sanity(docs + gen):
        fails.append("%-24s %s" % ("LaTeX sanity", msg))
    for p in gen:
        if not os.path.exists(p):
            continue
        t = open(p, encoding="utf-8").read()
        # the preamble (\documentclass[11pt]...) is not prose; keep the line
        # numbers of the body by blanking rather than cutting it
        k = t.find("\\begin{document}")
        if k > 0:
            t = re.sub(r"[^\n]", " ", t[:k]) + t[k:]
        t = re.sub(r"\\texttt\{[^}]*\}|\\href\{[^}]*\}|\\includegraphics[^\n]*|"
                   r"\\\[.*?\\\]|\$[^$]*\$|\\begin\{verbatim\}.*?\\end\{verbatim\}", " ",
                   t, flags=re.S)
        for pat, what in ((r"\\_", "literal underscore in prose"),
                          (r"\*\*", "unconverted bold marker"),
                          (r"textbackslash\\\{\\\}", "escaped backslash residue"),
                          (r"[A-Za-z)]\[\d", "unsuperscripted citation")):
            checks[0] += 1
            for mm in re.finditer(pat, t):
                fails.append("%-24s %s:%d %s" % ("LaTeX prose", os.path.basename(p),
                                                 t.count("\n", 0, mm.start()) + 1, what))

    # ---- word counts on the title page
    tp = os.path.join(SUB, "title_page.md")
    if os.path.exists(tp):
        abstract = _word_count(raw, "## Abstract", "## Key Messages")
        main = _word_count(raw, "## Introduction", "## References")
        t = _norm(open(tp, encoding="utf-8").read())
        for lab, v, lim, rx in (("abstract", abstract, 250, r"Abstract: \*\*%d\*\*"),
                                ("main text", main, 3000, r"Main text[^*]*\*\*%d\*\*")):
            checks[0] += 1
            if not re.search(rx % v, t):
                fails.append("%-24s %s is %d words; the title page does not say so"
                             % ("word count", lab, v))
            if v > lim:
                fails.append("%-24s %s is %d words, over the IJE limit of %d"
                             % ("word count", lab, v, lim))

    print("=" * 70)
    print("manuscript : %s" % path)
    print("checks     : %d  (%d references; %d numbers bound to their keys)"
          % (checks[0], len(listed), bound))
    for s_ in skipped:
        print("skipped    : %s" % s_)
    for k_, v_ in sorted(holders.items()):
        print("placeholder: %s in %s (fill before posting)" % (k_, ", ".join(sorted(v_))))
    if fails:
        print("FAILURES   : %d" % len(fails))
        for f in fails:
            print("   " + f)
    else:
        print("RESULT     : every templated number is rendered from its key in "
              "numbers.json, and every other number in the prose occurs among its "
              "values")
    print("=" * 70)
    return fails


# =====================================================================
STEPS = {"tables": tables, "figures": figures, "fill": fill, "render": render,
         "arxiv": arxiv, "verify": verify, "compare": compare}
ALL = ["tables", "figures", "fill", "render", "arxiv", "verify"]

if __name__ == "__main__":
    opts = [a for a in sys.argv[1:] if a.startswith("-")]
    if "-h" in opts or "--help" in opts:
        print(__doc__)
        sys.exit(0)
    if opts:
        raise SystemExit("unknown option(s): %s; lsm_report.py takes no options"
                         % ", ".join(sorted(set(opts))))
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
    todo = ALL if args[0] == "all" else args
    bad = []
    for name in todo:
        print("\n== %s ==" % name)
        r = STEPS[name]()
        if name in ("verify", "compare"):
            bad = bad + list(r or [])
    sys.exit(1 if bad else 0)
