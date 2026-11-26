"""Does fast_detect hurt when the onboard detector produces false detections (clutter)?

Usage (repo root):  PYTHONPATH=. python scripts/experiments/ctrl_run_fast_detect_fp.py [n_sites] [workers]
Unicycle, sigma_meas 0.5 m, full stack, fast_detect off / on (third argument "gate": off / on / on with fast_gate 2 / 3). False detections arrive at a Poisson rate (per second) at a
uniform random point within fp_radius of the robot (10 m = the whole sensor disc; 4 m = harsh, close to its path).
The belief turns a false detection into a confirmed track as it always did; fast_detect additionally puts a provisional
barrier in the filter for up to 5 s. Violations are against TRUE survivors only. Writes runs/ctrl_fast_detect_fp.csv.
"""
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds

SETTINGS = [(0.0, 10.0)] + [(r, d) for r in (0.1, 0.5, 2.0) for d in (10.0, 4.0)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    gate = len(sys.argv) > 3 and sys.argv[3] == "gate"
    seeds = usable_seeds(n)
    jobs = []
    for rate, rad in SETTINGS:
        for fd in (0, 1, 2, 3) if gate else (0, 1):      # 0 = off, 1 = on, 2 / 3 = on with that fast_gate
            jobs += make_jobs(seeds, ["full"], label=f"fp{rate}_r{rad}_fast{fd}", fast_detect=[fd > 0],
                              fast_gate=[max(fd, 1)], sigma_meas=[0.5], fp_rate=[rate], fp_radius=[rad])
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/ctrl_fast_detect_fp_gate.csv" if gate else "runs/ctrl_fast_detect_fp.csv")
    print(f"{'rate/s':>6s} {'radius':>6s} {'fast':>4s} {'score':>6s} {'rescue':>6s} {'viol':>5s} {'haz':>5s} {'min clear':>9s} {'paired diff':>16s}")
    for rate, rad in SETTINGS:
        base = {x["seed"]: x["norm_class"] for x in rows if x["label"] == f"fp{rate}_r{rad}_fast0"}
        for fd in (0, 1, 2, 3) if gate else (0, 1):
            r = [x for x in rows if x["label"] == f"fp{rate}_r{rad}_fast{fd}"]
            m = lambda k: float(np.mean([x[k] for x in r]))
            d = np.array([x["norm_class"] - base[x["seed"]] for x in r])
            print(f"{rate:6.1f} {rad:6.0f} {fd:4d} {m('norm_class'):6.3f} {m('rescue_rate'):6.2f} {m('any_clear_viol'):5.2f} "
                  f"{m('any_haz_viol'):5.2f} {min(x['min_clearance'] for x in r):9.2f} {d.mean():+8.3f} +/- {1.96 * d.std(ddof=1) / np.sqrt(len(d)) if fd else 0:.3f}")


if __name__ == "__main__":
    main()
