"""Q5: legible approach. Writes runs/q5.csv.

Usage: PYTHONPATH=. python scripts/ctrl_run_q5.py [n_scenarios] [workers]

Each scenario is one approach to a target among distractors on open ground, scored by two simulated
observers (cost-model and heading-extrapolation). Sweeps the legibility weight lambda for each true
survivor state, with the state-weighted planner (weight by state) and a state-blind one (full weight
whatever the state), plus a two-distractor variant.
"""
import os
import sys

from rescue.ctrl_experiments import run_approach_jobs, save_rows
from rescue.interfaces import ResponseState


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs("runs", exist_ok=True)
    lams = [0, 100, 300, 1000, 3000]
    jobs = []
    for seed in range(n):
        for lam in lams:
            for state in ResponseState:
                for blind in (False, True):
                    if lam == 0 and blind:
                        continue
                    jobs.append({"seed": seed, "lam": lam, "state": int(state), "blind": int(blind)})
        for lam in (0, 300):
            jobs.append({"seed": seed, "lam": lam, "state": int(ResponseState.MOBILE), "blind": 0,
                         "n_distractors": 2, "tag": "two_distractors"})
    rows = run_approach_jobs(jobs, workers)
    save_rows(rows, "runs/q5.csv")
    print("q5", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
