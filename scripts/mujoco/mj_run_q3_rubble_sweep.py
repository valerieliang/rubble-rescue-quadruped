"""Q3 on rubble at other noise levels and a bigger pad (rough policy, MuJoCo rubble heightfield only).

Usage (repo root):  PYTHONPATH=. python scripts/mj_run_q3_rubble_sweep.py [n_sites] [workers]

Settings (all confidence 0.7):
  sigma_meas 1.0 m, pad 0.25 m   full + no filter
  sigma_meas 2.0 m, pad 0.25 m   full + no filter
  sigma_meas 0.5 m, pad 0.40 m   full only (the no-filter run does not depend on the pad; see runs/mj_q3_rubble.csv)
The 0.25 m pad is below the measured 0.36 m rubble tracking error; 0.40 m is above it.
Writes runs/mj_q3_rubble_sweep.csv and prints a summary.
"""
import os
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds

ROUGH = os.path.abspath("sim/go2/policy_rough.onnx")
BASE = dict(plant=["mujoco_rubble"], policy=[ROUGH])


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs("runs", exist_ok=True)
    seeds = usable_seeds(n)
    jobs = []
    for s in (1.0, 2.0):
        jobs += make_jobs(seeds, ["full", "no_cbf"], label=f"sigma{s}_pad0.25", sigma_meas=[s], eps_track=[0.25], **BASE)
    jobs += make_jobs(seeds, ["full"], label="sigma0.5_pad0.40", sigma_meas=[0.5], eps_track=[0.40], **BASE)
    print(f"{len(jobs)} episodes ...", flush=True)
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/mj_q3_rubble_sweep.csv")

    print(f"\n{'setting':18s} {'variant':7s} {'n':>3s} {'score':>6s} {'rescued':>8s} {'clear viol':>11s} "
          f"{'haz viol':>9s} {'min clear (m)':>14s} {'falls':>5s}")
    for label in dict.fromkeys(r["label"] for r in rows):
        for variant in ("full", "no_cbf"):
            r = [x for x in rows if x["label"] == label and x["variant"] == variant]
            if not r:
                continue
            m = lambda k: float(np.mean([x[k] for x in r]))
            print(f"{label:18s} {variant:7s} {len(r):3d} {m('norm_class'):6.3f} {m('rescue_rate'):8.2f} "
                  f"{m('any_clear_viol'):11.2f} {m('any_haz_viol'):9.2f} {min(x['min_clearance'] for x in r):14.2f} "
                  f"{sum(x['fell'] for x in r):5d}")


if __name__ == "__main__":
    main()
