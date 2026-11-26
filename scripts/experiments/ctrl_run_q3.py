"""Q3: safety under uncertainty. Writes runs/q3_*.csv.

Usage: PYTHONPATH=. python scripts/ctrl_run_q3.py [n_sites] [workers]

Compares the full system (MPC + CBF-QP) with baseline 3 (same stack, no safety filter) as the robot's
onboard localization of survivors gets worse (sigma_meas), as the confidence level of the clearance
region changes, and with a hazard-blind planner (stress test: the filter alone must keep the robot out
of hazards).
"""
import os
import sys

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seeds = usable_seeds(n)
    os.makedirs("runs", exist_ok=True)
    both = ["full", "no_cbf"]

    studies = {
        "q3_sigma": dict(variants=both, grid={"sigma_meas": [0.25, 0.5, 1.0, 1.5, 2.0]}),
        "q3_confp": dict(variants=["full"], grid={"conf_p": [0.5, 0.8, 0.9, 0.95, 0.99]}),
        # confidence level x localization noise: the smallest region that still gives zero violations
        "q3_conf_sweep": dict(variants=both, grid={"sigma_meas": [0.5, 1.0, 2.0], "conf_p": [0.5, 0.7, 0.8, 0.9]}),
        "q3_naive": dict(variants=both, grid={"hazard_aware_planner": [False]}),
    }
    for name, s in studies.items():
        rows = run_jobs(make_jobs(seeds, s["variants"], name, **s["grid"]), workers)
        save_rows(rows, f"runs/{name}.csv")
        print(name, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
