"""Q3 on rubble with smaller safety margins: when do clearance violations appear?

Usage (repo root):  PYTHONPATH=. python scripts/mj_run_q3_rubble_margins.py [n_sites] [workers]

Rough policy on the MuJoCo rubble heightfield, sigma_meas 0.5 m, full stack only (the no-filter run is in
runs/mj_q3_rubble.csv). The survivor barrier sits at  c_p * sigma + d_clear + eps_track + cp_offset  from the belief
mean (c_p = chi radius of the confidence); a violation is the robot centre within 1 m of a survivor.
Settings shrink, in turn: the confidence (the ellipse term), the tracking pad, and finally d_clear itself.
Writes runs/mj_q3_rubble_margins.csv and prints a summary.
"""
import os
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds

ROUGH = os.path.abspath("sim/go2/policy_rough.onnx")
BASE = dict(plant=["mujoco_rubble"], policy=[ROUGH])
# (conf_p, eps_track, d_clear)
SETTINGS = [(0.7, 0.25, 1.0)] + [(c, e, 1.0) for c in (0.5, 0.3, 0.1, 0.01) for e in (0.25, 0.0)] + \
           [(0.01, 0.0, 0.75), (0.01, 0.0, 0.5)]
# "low" (third argument): continue below d_clear 0.5, where the closest approach (about d_clear + 0.5 m) reaches the 1 m line
LOW = [(0.01, 0.0, 0.4), (0.01, 0.0, 0.25), (0.01, 0.0, 0.0)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    low = len(sys.argv) > 3 and sys.argv[3] == "low"
    settings = LOW if low else SETTINGS
    out = "runs/mj_q3_rubble_margins_low.csv" if low else "runs/mj_q3_rubble_margins.csv"
    os.makedirs("runs", exist_ok=True)
    seeds = usable_seeds(n)
    jobs = []
    for c, e, d in settings:
        jobs += make_jobs(seeds, ["full"], label=f"conf{c}_pad{e}_dclear{d}", conf_p=[c], eps_track=[e], d_clear=[d],
                          sigma_meas=[0.5], **BASE)
    print(f"{len(jobs)} episodes ...", flush=True)
    rows = run_jobs(jobs, workers)
    save_rows(rows, out)

    print(f"\n{'conf':>5s} {'pad':>5s} {'d_clear':>7s} {'fixed margin':>12s} {'n':>3s} {'score':>6s} {'rescued':>8s} "
          f"{'clear viol':>11s} {'haz viol':>9s} {'min clear (m)':>14s} {'falls':>5s}")
    for c, e, d in settings:
        r = [x for x in rows if x["label"] == f"conf{c}_pad{e}_dclear{d}"]
        m = lambda k: float(np.mean([x[k] for x in r]))
        print(f"{c:5.2f} {e:5.2f} {d:7.2f} {d + e + 0.25:12.2f} {len(r):3d} {m('norm_class'):6.3f} {m('rescue_rate'):8.2f} "
              f"{m('any_clear_viol'):11.2f} {m('any_haz_viol'):9.2f} {min(x['min_clearance'] for x in r):14.2f} "
              f"{sum(x['fell'] for x in r):5d}")


if __name__ == "__main__":
    main()
