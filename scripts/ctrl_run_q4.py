"""Q4: survivor-state inference and its ablations. Writes runs/q4_*.csv.

Usage: PYTHONPATH=. python scripts/ctrl_run_q4.py [n_sites] [workers]

Variants: full (inference), no_state (baseline 4: priority fixed by the drone), oracle (baseline 5:
communication oracle reveals z once the survivor is sensed). Ablations vary one setting at a time:
prior noise, false positives, number of survivors, fraction mobile, prompt-response reliability,
and the fraction of mobile survivors that move erratically (out-of-model).
"""
import os
import sys

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    seeds = usable_seeds(n)
    os.makedirs("runs", exist_ok=True)
    V = ["full", "no_state", "oracle"]
    studies = {
        "q4_default": {},
        "q4_sigma": {"sigma_pos": [1.0, 3.0, 6.0, 10.0]},
        "q4_fp": {"n_false_pos": [0, 1, 3, 6]},
        "q4_nsurv": {"n_survivors": [3, 5, 8]},
        "q4_mobile": {"frac_mobile": [0.0, 0.33, 0.66, 1.0]},
        "q4_reliability": {"reliability": [0.4, 0.6, 0.8, 1.0]},
        "q4_oom": {"p_oom": [0.0, 0.5, 1.0], "frac_mobile": [0.66]},
    }
    for name, grid in studies.items():
        rows = run_jobs(make_jobs(seeds, V, name, **grid), workers)
        save_rows(rows, f"runs/{name}.csv")
        print(name, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
