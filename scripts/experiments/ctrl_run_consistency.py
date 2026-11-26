"""Consistency check between the two simulators: the closed-loop sim (ctrl_sim, Q3-Q5) under the Q1 prior sweeps,
against the frozen Q1 results from the waypoint-level Go2 sim (sim.py, README Table 1-2). Writes runs/consistency.csv
and prints a markdown table.

Usage: PYTHONPATH=. python scripts/ctrl_run_consistency.py [n_sites] [workers] [sensor_radius]

With sensor_radius = 5 the closed-loop sim uses the same sensing range as sim.py (default 10 m).

The two sims differ on purpose (sensor radius 10 m vs 5 m, rescue radius 3.5 m vs 1.5 m, 10 Hz MPC vs waypoint
tracker, survivors can move), so absolute scores are not comparable. What should agree is the trend: the score falls
monotonically as the prior gets worse, and the relative loss (score / score at the best setting) is similar.
Class-weighted score, safety filter off, so only the search and sensing differ from sim.py's setting.
"""
import sys

import numpy as np
from scipy.stats import spearmanr

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds
from rescue.experiments import mean_ci

# Frozen Q1 numbers (README Tables 1 and 2: prior + priority, Go2 sim, 40 sites per column).
Q1_SIGMA = {1: 0.81, 3: 0.73, 6: 0.56, 10: 0.39, 15: 0.26, 25: 0.10}
Q1_FP = {0: 0.79, 1: 0.73, 3: 0.68, 6: 0.62, 10: 0.53, 20: 0.46}


def table(rows, key, frozen, label):
    xs = sorted(frozen)
    cl = [mean_ci([r["norm_class"] for r in rows if r[key] == x]) for x in xs]
    m = np.array([c[0] for c in cl])
    q = np.array([frozen[x] for x in xs])
    rho = spearmanr(xs, m).statistic
    print(f"### {label}\n")
    print(f"| {key} | closed-loop score | relative to best | frozen Q1 score | relative to best |")
    print("|---|---|---|---|---|")
    for x, (mm, hh), qq in zip(xs, cl, q):
        print(f"| {x} | {mm:.3f} ± {hh:.3f} | {mm / m.max():.2f} | {qq:.2f} | {qq / q.max():.2f} |")
    rel_gap = np.abs(m / m.max() - q / q.max()).max()
    print(f"\nSpearman(closed-loop score, {key}) = {rho:+.2f}; Spearman(frozen Q1, {key}) = "
          f"{spearmanr(xs, q).statistic:+.2f}; max |relative-score difference| = {rel_gap:.2f}\n")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    radius = [float(sys.argv[3])] if len(sys.argv) > 3 else []
    extra = {"sensor_radius": radius} if radius else {}
    seeds = usable_seeds(n)
    rows = run_jobs(make_jobs(seeds, ["no_cbf"], "cons_sigma", sigma_pos=list(map(float, Q1_SIGMA)), **extra)
                    + make_jobs(seeds, ["no_cbf"], "cons_fp", n_false_pos=list(map(float, Q1_FP)), **extra), workers)
    save_rows(rows, "runs/consistency.csv" if not radius else f"runs/consistency_r{radius[0]:g}.csv")
    table([r for r in rows if r["label"] == "cons_sigma"], "sigma_pos", {float(k): v for k, v in Q1_SIGMA.items()},
          "Prior position noise")
    table([r for r in rows if r["label"] == "cons_fp"], "n_false_pos", {float(k): v for k, v in Q1_FP.items()},
          "False positives per site")


if __name__ == "__main__":
    main()
