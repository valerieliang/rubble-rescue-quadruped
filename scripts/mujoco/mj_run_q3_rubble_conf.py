"""Cost of a high safety-region confidence on rubble (experiment 3, cost side).

Usage (repo root):  PYTHONPATH=. python scripts/mujoco/mj_run_q3_rubble_conf.py [n_sites] [workers]

Rough policy on the MuJoCo rubble heightfield, sigma_meas 0.5 m, pad 0.25 m, full stack, confidence 0.9 / 0.95 / 0.99
applied to every track (an upper bound on the cost of raising it for fresh tracks only). The 0.7 row is in
runs/mj_q3_rubble.csv. Writes runs/mj_q3_rubble_conf.csv.
"""
import os
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds

ROUGH = os.path.abspath("sim/go2/policy_rough.onnx")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seeds = usable_seeds(n)
    jobs = make_jobs(seeds, ["full"], label="conf", conf_p=[0.9, 0.95, 0.99], plant=["mujoco_rubble"], policy=[ROUGH])
    print(len(jobs), "episodes ...", flush=True)
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/mj_q3_rubble_conf.csv")
    print(f"\n{'conf':>5s} {'n':>3s} {'score':>6s} {'rescued':>8s} {'clear viol':>11s} {'haz viol':>9s} {'min clear (m)':>14s} {'falls':>5s}")
    for c in (0.9, 0.95, 0.99):
        r = [x for x in rows if x["conf_p"] == c]
        m = lambda k: float(np.mean([x[k] for x in r]))
        print(f"{c:5.2f} {len(r):3d} {m('norm_class'):6.3f} {m('rescue_rate'):8.2f} {m('any_clear_viol'):11.2f} "
              f"{m('any_haz_viol'):9.2f} {min(x['min_clearance'] for x in r):14.2f} {sum(x['fell'] for x in r):5d}")


if __name__ == "__main__":
    main()
