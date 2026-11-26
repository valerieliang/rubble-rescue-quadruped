"""Cost of a lower speed cap on the closed-loop benchmark (the price of the only fix left for a 1.5 m first sighting).

Usage (repo root):  PYTHONPATH=. python scripts/ctrl_run_speed_cost.py [n_sites] [workers]
Unicycle, sigma_meas 0.5 m, full stack with fast_detect on, speed cap 1.0 / 0.75 / 0.5 of the terrain speed.
Writes runs/ctrl_speed_cost.csv.
"""
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds

SCALES = (1.0, 0.75, 0.5)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seeds = usable_seeds(n)
    jobs = []
    for sc in SCALES:
        jobs += make_jobs(seeds, ["full"], label=f"speed_{sc}", fast_detect=[True], fast_gate=[1], sigma_meas=[0.5], speed_scale=[sc])
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/ctrl_speed_cost.csv")
    base = {x["seed"]: x["norm_class"] for x in rows if x["label"] == "speed_1.0"}
    for sc in SCALES:
        r = [x for x in rows if x["label"] == f"speed_{sc}"]
        m = lambda k: float(np.mean([x[k] for x in r]))
        d = np.array([x["norm_class"] - base[x["seed"]] for x in r])
        print(f"speed {sc}: score {m('norm_class'):.3f} rescue {m('rescue_rate'):.2f} viol {m('any_clear_viol'):.2f} "
              f"min clear {min(x['min_clearance'] for x in r):.2f} paired diff {d.mean():+.3f} +/- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}")


if __name__ == "__main__":
    main()
