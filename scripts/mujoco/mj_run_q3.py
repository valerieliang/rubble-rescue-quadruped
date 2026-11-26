"""Q3 on the MuJoCo Go2: does the safety filter still work when a learned legged plant replaces the unicycle?

Usage (repo root):  PYTHONPATH=. python scripts/mj_run_q3.py [n_sites] [workers]

Runs the same sites and drone detections with the full system (MPC + CBF-QP, pad eps_track = 0.25 m) and
baseline 3 (same stack, no safety filter), once on the reduced-order unicycle and once on the MuJoCo Go2
driven by the trained policy. Writes runs/mj_q3.csv (one row per episode) and prints a summary.

Reading it: with the filter on, violations should stay at zero on both plants; without it they should not.
A violation on the MuJoCo plant that the unicycle did not have means the pad is too small for the legged robot.
"""
import os
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs("runs", exist_ok=True)
    jobs = make_jobs(usable_seeds(n), ["full", "no_cbf"], label="mj_q3", plant=["unicycle", "mujoco"])
    print(f"{len(jobs)} episodes ...")
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/mj_q3.csv")

    print(f"\n{'plant':9s} {'variant':7s} {'n':>3s} {'score':>6s} {'rescued':>8s} {'sites w/ clear viol':>20s} "
          f"{'sites w/ hazard viol':>21s} {'min clear (m)':>14s} {'falls':>5s}")
    for plant in ("unicycle", "mujoco"):
        for variant in ("full", "no_cbf"):
            r = [x for x in rows if x["plant"] == plant and x["variant"] == variant]
            if not r:
                continue
            m = lambda k: float(np.mean([x[k] for x in r]))
            print(f"{plant:9s} {variant:7s} {len(r):3d} {m('norm_class'):6.3f} {m('rescue_rate'):8.2f} "
                  f"{m('any_clear_viol'):20.2f} {m('any_haz_viol'):21.2f} {min(x['min_clearance'] for x in r):14.2f} "
                  f"{sum(x['fell'] for x in r):5d}")


if __name__ == "__main__":
    main()
