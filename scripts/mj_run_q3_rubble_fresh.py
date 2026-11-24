"""Fresh-track confidence in the stack, on rubble (experiment 3, cost side).

Usage (repo root):  PYTHONPATH=. python scripts/mj_run_q3_rubble_fresh.py [n_sites] [workers]

StackConfig.conf_fresh / n_fresh: a robot-detected track with fewer than n_fresh detections fused uses confidence
conf_fresh instead of 0.7 (the belief updates at 1 Hz, so n_fresh = 5 is about 5 s). Rough policy on the MuJoCo rubble
heightfield, sigma_meas 0.5 m, pad 0.25 m, full stack. The off row (0.7 everywhere) is runs/mj_q3_rubble.csv.
Writes runs/mj_q3_rubble_fresh.csv.
"""
import os
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds

ROUGH = os.path.abspath("sim/go2/policy_rough.onnx")
SETTINGS = [(0.95, 5), (0.99, 5), (0.99, 10)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seeds = usable_seeds(n)
    jobs = []
    for c, k in SETTINGS:
        jobs += make_jobs(seeds, ["full"], label=f"fresh{c}_{k}", conf_fresh=[c], n_fresh=[k],
                          plant=["mujoco_rubble"], policy=[ROUGH])
    print(len(jobs), "episodes ...", flush=True)
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/mj_q3_rubble_fresh.csv")
    print(f"\n{'conf_fresh':>10s} {'n_fresh':>7s} {'n':>3s} {'score':>6s} {'rescued':>8s} {'clear viol':>11s} {'haz viol':>9s} "
          f"{'min clear (m)':>14s} {'falls':>5s}")
    for c, k in SETTINGS:
        r = [x for x in rows if x["label"] == f"fresh{c}_{k}"]
        m = lambda key: float(np.mean([x[key] for x in r]))
        print(f"{c:10.2f} {k:7d} {len(r):3d} {m('norm_class'):6.3f} {m('rescue_rate'):8.2f} {m('any_clear_viol'):11.2f} "
              f"{m('any_haz_viol'):9.2f} {min(x['min_clearance'] for x in r):14.2f} {sum(x['fell'] for x in r):5d}")


if __name__ == "__main__":
    main()
