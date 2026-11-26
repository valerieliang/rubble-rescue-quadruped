"""Cost of the 10 Hz first-detection path (StackConfig.fast_detect) on the closed-loop benchmark.

Usage (repo root):  PYTHONPATH=. python scripts/ctrl_run_fast_detect.py [n_sites] [workers]
Unicycle plant, sigma_meas 0.5 m, 100 sites by default; full stack with fast_detect off and on. Writes runs/ctrl_fast_detect.csv.
"""
import sys
import numpy as np
from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seeds = usable_seeds(n)
    jobs = []
    for fd in (False, True):
        jobs += make_jobs(seeds, ["full"], label=f"fast_{int(fd)}", fast_detect=[fd], fast_gate=[1], sigma_meas=[0.5])
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/ctrl_fast_detect.csv")
    for fd in (0, 1):
        r = [x for x in rows if x["label"] == f"fast_{fd}"]
        m = lambda k: float(np.mean([x[k] for x in r]))
        print(f"fast_detect={fd} n={len(r)} score {m('norm_class'):.3f} rescue {m('rescue_rate'):.2f} viol {m('any_clear_viol'):.2f} "
              f"haz {m('any_haz_viol'):.2f} min clear {min(x['min_clearance'] for x in r):.2f}")
    a = {x["seed"]: x["norm_class"] for x in rows if x["label"] == "fast_0"}
    d = np.array([x["norm_class"] - a[x["seed"]] for x in rows if x["label"] == "fast_1"])
    print(f"paired score difference {d.mean():+.3f} +/- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}")


if __name__ == "__main__":
    main()
