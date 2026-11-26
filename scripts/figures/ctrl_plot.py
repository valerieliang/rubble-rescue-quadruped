"""Figures and summary tables for Q3-Q5 from runs/*.csv. Writes runs/figures/ and prints markdown tables.

Usage: PYTHONPATH=. python scripts/figures/ctrl_plot.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rescue.ctrl_experiments import load_rows
from rescue.experiments import mean_ci

OUT = "runs/figures"
COL = {"full": "#1f77b4", "no_cbf": "#d62728", "no_state": "#7f7f7f", "oracle": "#2ca02c"}
NAME = {"full": "full (MPC+CBF, inference)", "no_cbf": "baseline 3: no safety filter",
        "no_state": "baseline 4: no state inference", "oracle": "baseline 5: comm. oracle"}


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def series(rows, variant, xkey, ykey):
    xs = sorted({r[xkey] for r in rows if xkey in r})
    out = [mean_ci([r[ykey] for r in sel(rows, variant=variant, **{xkey: x})]) for x in xs]
    return xs, np.array([o[0] for o in out]), np.array([o[1] for o in out])


def line(ax, rows, variant, xkey, ykey):
    xs, m, h = series(rows, variant, xkey, ykey)
    ax.plot(xs, m, "o-", color=COL[variant], label=NAME[variant])
    ax.fill_between(xs, m - h, m + h, color=COL[variant], alpha=0.2)


def table(rows, variants, xkey, ykeys):
    xs = sorted({r[xkey] for r in rows if xkey in r})
    print(f"| {xkey} | " + " | ".join(f"{v}: {k}" for v in variants for k in ykeys) + " |")
    print("|" + "---|" * (1 + len(variants) * len(ykeys)))
    for x in xs:
        cells = [f"{mean_ci([r[k] for r in sel(rows, variant=v, **{xkey: x})])[0]:.2f}" for v in variants for k in ykeys]
        print(f"| {x} | " + " | ".join(cells) + " |")
    print()


def q3():
    if not os.path.exists("runs/q3_sigma.csv"):
        return
    rows = load_rows("runs/q3_sigma.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for v in ("full", "no_cbf"):
        line(axs[0], rows, v, "sigma_meas", "any_clear_viol")
        line(axs[1], rows, v, "sigma_meas", "norm_class")
        line(axs[2], rows, v, "sigma_meas", "rescue_rate")
    for ax, t in zip(axs, ("episodes with a clearance violation", "oracle-normalized score", "rescue rate within T")):
        ax.set_xlabel("onboard localization noise sigma_meas (m)")
        ax.set_title(t)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/q3_uncertainty.png", dpi=150)
    print("### Q3: vs localization noise\n")
    table(rows, ["full", "no_cbf"], "sigma_meas", ["any_clear_viol", "norm_class", "rescue_rate"])
    for name in ("q3_confp", "q3_naive"):
        if os.path.exists(f"runs/{name}.csv"):
            r = load_rows(f"runs/{name}.csv")
            print(f"### {name}\n")
            key = "conf_p" if name == "q3_confp" else "hazard_aware_planner"
            table(r, sorted({x["variant"] for x in r}), key, ["any_clear_viol", "any_haz_viol", "norm_class", "rescue_rate"])


def q3_conf():
    if not os.path.exists("runs/q3_conf_sweep.csv"):
        return
    rows = load_rows("runs/q3_conf_sweep.csv")
    sms = sorted({r["sigma_meas"] for r in rows})
    confs = sorted({r["conf_p"] for r in rows if r["variant"] == "full"})
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    print("### Q3: confidence level of the safety region (paired vs. no filter)\n")
    print("| sigma_meas | conf_p | episodes with violation | score | score diff vs. no filter |")
    print("|---|---|---|---|---|")
    for sm in sms:
        nb = {r["seed"]: r["norm_class"] for r in rows if r["variant"] == "no_cbf" and r["sigma_meas"] == sm}
        v, d, e = [], [], []
        for c in confs:
            f = [r for r in rows if r["variant"] == "full" and r["sigma_meas"] == sm and r["conf_p"] == c]
            m, h = mean_ci([r["norm_class"] - nb[r["seed"]] for r in f])
            v.append(np.mean([r["any_clear_viol"] for r in f]))
            d.append(m)
            e.append(h)
            print(f"| {sm:g} | {c:g} | {v[-1]:.3f} | {np.mean([r['norm_class'] for r in f]):.3f} | "
                  f"{m:+.3f} ± {h:.3f}{'*' if abs(m) > h else ''} |")
        axs[0].plot(confs, v, "o-", label=f"sigma_meas {sm:g}")
        axs[1].errorbar(confs, d, yerr=e, fmt="o-", capsize=3, label=f"sigma_meas {sm:g}")
    axs[0].set_title("episodes with a clearance violation")
    axs[1].set_title("score difference vs. no filter")
    axs[1].axhline(0, color="k", lw=0.5)
    for ax in axs:
        ax.set_xlabel("confidence level of the safety region")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/q3_confidence.png", dpi=150)
    print()


def q4_triage():
    out = {n: load_rows(f"runs/{n}.csv") for n in ("q4_triage", "q4_triage_sweep") if os.path.exists(f"runs/{n}.csv")}
    if not out:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    vs = ("full", "oracle", "oracle_global")
    cols = ("#1f77b4", "#2ca02c", "#7f7f7f")
    lab = {"full": "full (inference)", "oracle": "in-range oracle", "oracle_global": "global oracle (bound)"}

    def diff(rows, v, **w):
        A = {r["seed"]: r["norm_state"] for r in rows if r["variant"] == v and all(r[k] == x for k, x in w.items())}
        B = {r["seed"]: r["norm_state"] for r in rows if r["variant"] == "no_state" and all(r[k] == x for k, x in w.items())}
        return mean_ci([A[k] - B[k] for k in A if k in B])

    if "q4_triage" in out:
        rows = out["q4_triage"]
        cues = sorted({r["p_state_cue"] for r in rows})
        w = 0.25
        for j, (v, col) in enumerate(zip(vs, cols)):
            for i, c in enumerate(cues):
                m, h = diff(rows, v, p_state_cue=c)
                axs[0].bar(i + (j - 1) * w, m, w, yerr=h, capsize=3, color=col, label=lab[v] if i == 0 else None)
        axs[0].set_xticks(range(len(cues)))
        axs[0].set_xticklabels(["no drone cue" if not c else f"drone cue {c:g}" for c in cues])
        axs[0].axhline(0, color="k", lw=0.5)
        axs[0].set_ylabel("state-weighted score minus no inference")
        axs[0].set_title("Triage regime (8 survivors, T = 300 s)")
        axs[0].legend(fontsize=7)
    if "q4_triage_sweep" in out:
        rows = out["q4_triage_sweep"]
        ns = sorted({r["n_survivors"] for r in rows})
        for v, col in zip(vs, cols):
            res = [diff(rows, v, n_survivors=n) for n in ns]
            axs[1].errorbar(ns, [m for m, _ in res], yerr=[h for _, h in res], fmt="o-", capsize=3, color=col, label=lab[v])
        axs[1].axhline(0, color="k", lw=0.5)
        axs[1].set_xlabel("survivors per site")
        axs[1].set_title("Gain vs. number of survivors (no drone cue)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/q4_triage.png", dpi=150)


def q4():
    studies = [("q4_sigma", "sigma_pos", "prior noise (m)"), ("q4_fp", "n_false_pos", "false positives"),
               ("q4_nsurv", "n_survivors", "survivors"), ("q4_mobile", "frac_mobile", "fraction mobile"),
               ("q4_reliability", "reliability", "prompt reliability q"), ("q4_oom", "p_oom", "erratic fraction")]
    studies = [s for s in studies if os.path.exists(f"runs/{s[0]}.csv")]
    if not studies:
        return
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (name, key, lab) in zip(axs.ravel(), studies):
        rows = load_rows(f"runs/{name}.csv")
        for v in ("full", "no_state", "oracle"):
            line(ax, rows, v, key, "norm_state")
        ax.set_xlabel(lab)
        ax.set_ylabel("state-weighted score / oracle")
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/q4_ablations.png", dpi=150)
    print("### Q4: default config\n")
    if os.path.exists("runs/q4_default.csv"):
        d = load_rows("runs/q4_default.csv")
        for v in ("full", "no_state", "oracle"):
            rs = sel(d, variant=v)
            n = sum(r["n_tracked"] for r in rs)
            acc = sum(r["n_correct"] for r in rs) / n if n else float("nan")
            tn = sum(r["t_correct_n"] for r in rs)
            t = sum(r["t_correct_sum"] for r in rs) / tn if tn else float("nan")
            print(f"- {v}: score {mean_ci([r['norm_state'] for r in rs])[0]:.3f}, "
                  f"time to critical {mean_ci([r['time_to_critical'] for r in rs])[0]:.0f} s, "
                  f"state accuracy {acc:.2f} (n={n}), mean time to settle {t:.1f} s")
        print()
    if os.path.exists("runs/q4_oom.csv"):
        d = sel(load_rows("runs/q4_oom.csv"), variant="full")
        print("### Q4: out-of-model (erratic) robustness, full variant\n")
        for p in sorted({r["p_oom"] for r in d}):
            rs = sel(d, p_oom=p)
            en = sum(r["erratic_n"] for r in rs)
            n = sum(r["n_tracked"] for r in rs)
            print(f"- p_oom={p}: state accuracy {sum(r['n_correct'] for r in rs) / max(n, 1):.2f}, "
                  f"on erratic survivors {sum(r['erratic_correct'] for r in rs) / max(en, 1):.2f} (n={en})")
        print()


def q5():
    if not os.path.exists("runs/q5.csv"):
        return
    allrows = load_rows("runs/q5.csv")
    rows = [r for r in allrows if r.get("tag") not in ("two_distractors", "heading_term")]
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    names = {0: "unresponsive", 1: "trapped", 2: "mobile"}
    print("### Q5: legibility (mean over scenarios)\n")
    print("| lambda | state | blind | approach time (s) | cost-observer id time (s) | mean P cost-obs | mean P heading-obs | mean P learned-obs |")
    print("|---|---|---|---|---|---|---|---|")
    for state, c in zip((2, 1, 0), ("#1f77b4", "#ff7f0e", "#2ca02c")):
        for blind, ls in ((0, "-"), (1, "--")):
            rs = [r for r in rows if r["state"] == state and (r["blind"] == blind or r["lam"] == 0)]
            lams = sorted({r["lam"] for r in rs})
            f = lambda k: [np.mean([r[k] for r in rs if r["lam"] == l]) for l in lams]
            # State-weighted (thick, solid) and state-blind (thin, dashed, x) curves coincide for some states
            # (e.g. mobile), so draw the blind ones thinner on top to keep both visible.
            kw = dict(color=c, ls=ls, marker="x" if blind else "o", lw=1.2 if blind else 3.0,
                      ms=7 if blind else 5, zorder=3 if blind else 2)
            axs[0].plot(lams, f("t_total"), label=f"{names[state]}{' (state-blind)' if blind else ''}", **kw)
            axs[1].plot(lams, f("mean_p_cost"), **kw)
            axs[2].plot(lams, f("mean_p_heading"), **kw)
            axs[3].plot(lams, f("mean_p_learned"), **kw)
            for l in lams:
                q = [r for r in rs if r["lam"] == l]
                print(f"| {l:g} | {names[state]} | {blind} | {np.mean([r['t_total'] for r in q]):.1f} | "
                      f"{np.mean([min(r['ident_cost'], r['t_total']) for r in q]):.1f} | "
                      f"{np.mean([r['mean_p_cost'] for r in q]):.2f} | {np.mean([r['mean_p_heading'] for r in q]):.2f} | "
                      f"{np.mean([r['mean_p_learned'] for r in q]):.2f} |")
    for ax, t in zip(axs, ("approach time (s)", "mean P(true goal), cost observer", "mean P(true goal), heading observer", "mean P(true goal), learned observer")):
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("legibility weight lambda")
        ax.set_title(t)
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f"{OUT}/q5_legibility.png", dpi=150)
    print()
    ht = [r for r in allrows if r.get("tag") == "heading_term"]
    base = [r for r in rows if r["state"] == 2 and r["lam"] == 0]
    if ht and base:
        print("### Q5: heading-legibility term (mobile survivor; differences vs. no legibility, paired by scenario)\n")
        print("| lambda (cost) | lambda_head | approach time diff (s) | P cost-obs diff | P heading-obs diff | P learned-obs diff |")
        print("|---|---|---|---|---|---|")
        b = {int(r["seed"]): r for r in base}
        pool = {(300, 0): [r for r in rows if r["state"] == 2 and r["lam"] == 300 and r["blind"] == 0]}
        combos = [(300, 0)] + sorted({(r["lam"], r["lam_head"]) for r in ht})
        for (l, h) in combos:
            rs = pool.get((l, h)) or [r for r in ht if r["lam"] == l and r["lam_head"] == h]
            cells = []
            for k in ("t_total", "mean_p_cost", "mean_p_heading", "mean_p_learned"):
                m, hw = mean_ci([r[k] - b[int(r["seed"])][k] for r in rs if int(r["seed"]) in b])
                cells.append(f"{m:+.2f} ± {hw:.2f}{'*' if abs(m) > hw else ''}")
            print(f"| {l:g} | {h:g} | " + " | ".join(cells) + " |")
        print()


def main():
    os.makedirs(OUT, exist_ok=True)
    q3()
    q3_conf()
    q4()
    q4_triage()
    q5()


if __name__ == "__main__":
    main()
