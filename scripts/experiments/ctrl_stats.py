"""Paired differences with 95% confidence intervals for the closed-loop Q3 and Q4 runs. Prints markdown tables.

Usage: PYTHONPATH=. python scripts/experiments/ctrl_stats.py

Every comparison is paired by site seed (same site, same drone detections), so the interval is on the per-site
difference. A difference is called detectable when its interval excludes zero; intervals are not corrected for
multiple comparisons.
"""
import os

import numpy as np

from rescue.ctrl_experiments import load_rows
from rescue.experiments import mean_ci


def paired(rows, a, b, key, **where):
    """(mean, half-width, n) of key[a] - key[b] over seeds present in both, restricted to rows matching ``where``."""
    sel = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    A = {r["seed"]: r[key] for r in sel if r["variant"] == a}
    B = {r["seed"]: r[key] for r in sel if r["variant"] == b}
    d = np.array([A[s] - B[s] for s in A if s in B and not (np.isnan(A[s]) or np.isnan(B[s]))])
    m, h = mean_ci(d)
    return m, h, len(d)


def fmt(m, h, n=None, digits=3):
    s = f"{m:+.{digits}f} ± {h:.{digits}f}"
    return s + ("*" if abs(m) > h else "")


def mean(rows, variant, key, **where):
    v = [r[key] for r in rows if r["variant"] == variant and all(r.get(k) == x for k, x in where.items())]
    return float(np.nanmean(v)), len(v)


def q3():
    if not os.path.exists("runs/q3_sigma.csv"):
        return
    rows = load_rows("runs/q3_sigma.csv")
    n = len({r["seed"] for r in rows})
    print(f"### Q3: safety filter vs. no filter, paired over {n} sites (* = interval excludes 0)\n")
    print("| sigma_meas (m) | violations, full | violations, no filter | score, full | score diff (full - no filter) "
          "| rescue-rate diff | time-to-critical diff (s) |")
    print("|---|---|---|---|---|---|---|")
    for sm in sorted({r["sigma_meas"] for r in rows}):
        w = {"sigma_meas": sm}
        print(f"| {sm:g} | {mean(rows, 'full', 'any_clear_viol', **w)[0]:.3f} | {mean(rows, 'no_cbf', 'any_clear_viol', **w)[0]:.3f} "
              f"| {mean(rows, 'full', 'norm_class', **w)[0]:.3f} "
              f"| {fmt(*paired(rows, 'full', 'no_cbf', 'norm_class', **w)[:2])} "
              f"| {fmt(*paired(rows, 'full', 'no_cbf', 'rescue_rate', **w)[:2])} "
              f"| {fmt(*paired(rows, 'full', 'no_cbf', 'time_to_critical', **w)[:2], digits=1)} |")
    print()


def q4_table(name, title, extra=None):
    path = f"runs/{name}.csv"
    if not os.path.exists(path):
        return
    rows = load_rows(path)
    keys = sorted({r.get("p_state_cue", 0.0) for r in rows}) if "p_state_cue" in rows[0] else [None]
    variants = [v for v in ("full", "oracle", "oracle_global") if any(r["variant"] == v for r in rows)]
    print(f"### {title}\n")
    print("| drone state cue | n sites | no_state score | " + " | ".join(f"{v} - no_state" for v in variants) + " |")
    print("|---|---|---|" + "---|" * len(variants))
    for c in keys:
        w = {} if c is None else {"p_state_cue": c}
        m0, n0 = mean(rows, "no_state", "norm_state", **w)
        cells = [fmt(*paired(rows, v, "no_state", "norm_state", **w)[:2]) for v in variants]
        print(f"| {'none' if not c else f'{c:g}'} | {n0} | {m0:.3f} | " + " | ".join(cells) + " |")
    print()


def q4():
    q4_table("q4_default", "Q4, default scene (state-weighted score, paired differences vs. no inference)")
    q4_table("q4_triage", "Q4, triage regime (8 survivors, T = 300 s, state-dependent handling time)")
    path = "runs/q4_triage_sweep.csv"
    if os.path.exists(path):
        rows = load_rows(path)
        print("### Q4, triage regime vs. number of survivors (no drone cue)\n")
        print("| survivors | no_state score | full - no_state | oracle - no_state | oracle_global - no_state |")
        print("|---|---|---|---|---|")
        for n in sorted({r["n_survivors"] for r in rows}):
            w = {"n_survivors": n}
            print(f"| {n:g} | {mean(rows, 'no_state', 'norm_state', **w)[0]:.3f} | "
                  + " | ".join(fmt(*paired(rows, v, "no_state", "norm_state", **w)[:2]) for v in ("full", "oracle", "oracle_global"))
                  + " |")
        print()
    triage_replication()


def triage_replication():
    """Original 200 triage sites vs. the fresh 500 (runs/q4_triage_large.csv) vs. both pooled, no drone cue."""
    if not (os.path.exists("runs/q4_triage.csv") and os.path.exists("runs/q4_triage_large.csv")):
        return
    old = [r for r in load_rows("runs/q4_triage.csv") if r.get("p_state_cue") == 0.0]
    new = load_rows("runs/q4_triage_large.csv")
    print("### Q4, triage regime replication (8 survivors, T = 300 s, no drone cue)\n")
    print("| sites | n | no_state score | full - no_state | oracle - no_state | oracle_global - no_state |")
    print("|---|---|---|---|---|---|")
    for name, rows in (("original", old), ("fresh", new), ("pooled", old + new)):
        m0, n0 = mean(rows, "no_state", "norm_state")
        print(f"| {name} | {n0} | {m0:.3f} | "
              + " | ".join(fmt(*paired(rows, v, "no_state", "norm_state")[:2]) for v in ("full", "oracle", "oracle_global"))
              + " |")
    print()


if __name__ == "__main__":
    q3()
    q4()
