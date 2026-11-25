"""fast_detect on physical rubble: cost and safety on the MuJoCo Go2 (rough policy), off vs on.

Usage (repo root):  PYTHONPATH=. python scripts/mj_run_fast_detect_rubble.py [n_sites] [workers]
Full stack, sigma_meas 0.5 m, pad 0.25 m, confidence 0.7. Writes runs/mj_fast_detect_rubble.csv.
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
    jobs = []
    for fd in (False, True):
        jobs += make_jobs(seeds, ["full"], label=f"fast_{int(fd)}", fast_detect=[fd], sigma_meas=[0.5],
                          plant=["mujoco_rubble"], policy=[ROUGH])
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/mj_fast_detect_rubble.csv")
    a = {x["seed"]: x["norm_class"] for x in rows if x["label"] == "fast_0"}
    for fd in (0, 1):
        r = [x for x in rows if x["label"] == f"fast_{fd}"]
        m = lambda k: float(np.mean([x[k] for x in r]))
        d = np.array([x["norm_class"] - a[x["seed"]] for x in r])
        print(f"fast_detect={fd} n={len(r)} score {m('norm_class'):.3f} rescue {m('rescue_rate'):.2f} viol {m('any_clear_viol'):.2f} "
              f"haz {m('any_haz_viol'):.2f} min clear {min(x['min_clearance'] for x in r):.2f} falls {sum(x['fell'] for x in r)} "
              f"paired diff {d.mean():+.3f} +/- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}")


if __name__ == "__main__":
    main()
