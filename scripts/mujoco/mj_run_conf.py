"""Thin-margin test on the MuJoCo Go2: shrink the safety region until the filter starts to fail.

Usage (repo root):  PYTHONPATH=. python scripts/mj_run_conf.py [n_sites] [workers]

With the default region (conf_p 0.95) the clearance margin is metres, so the 0.25 m tracking-error pad is
never the binding term and the unicycle-vs-legged comparison in mj_run_q3.py cannot show whether the pad is
big enough. Here the full system runs with smaller regions (conf_p 0.5 ... 0.9) at sigma_meas 1.0 m, on both
plants and with the pad on (0.25 m) and off (0 m). Writes runs/mj_conf.csv.

Reading it: find the smallest conf_p with zero violating sites on each plant. If the MuJoCo plant needs a
larger region than the unicycle, or violates with the pad off but not on, the legged robot's tracking error
matters and the pad earns its keep.
"""
import os
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs("runs", exist_ok=True)
    jobs = make_jobs(usable_seeds(n), ["full"], label="mj_conf", plant=["unicycle", "mujoco"],
                     conf_p=[0.5, 0.7, 0.9], eps_track=[0.0, 0.25], sigma_meas=[1.0])
    print(f"{len(jobs)} episodes ...")
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/mj_conf.csv")

    print(f"\n{'plant':9s} {'conf_p':>6s} {'pad':>5s} {'n':>3s} {'score':>6s} {'rescued':>8s} "
          f"{'sites w/ viol':>14s} {'viol steps/site':>16s} {'min clear (m)':>14s} {'falls':>5s}")
    for plant in ("unicycle", "mujoco"):
        for cp in (0.5, 0.7, 0.9):
            for pad in (0.0, 0.25):
                r = [x for x in rows if x["plant"] == plant and x["conf_p"] == cp and x["eps_track"] == pad]
                if not r:
                    continue
                m = lambda k: float(np.mean([x[k] for x in r]))
                print(f"{plant:9s} {cp:6.2f} {pad:5.2f} {len(r):3d} {m('norm_class'):6.3f} {m('rescue_rate'):8.2f} "
                      f"{m('any_clear_viol'):14.2f} {m('clear_viol_steps'):16.1f} "
                      f"{min(x['min_clearance'] for x in r):14.2f} {sum(x['fell'] for x in r):5d}")


if __name__ == "__main__":
    main()
