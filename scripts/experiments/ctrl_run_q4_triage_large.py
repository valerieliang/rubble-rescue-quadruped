"""Q4 triage replication on fresh sites: is the full system's gain over no inference real or null?

Usage: PYTHONPATH=. python scripts/experiments/ctrl_run_q4_triage_large.py [n_sites] [workers] [start_seed]
Same regime as runs/q4_triage.csv (reward triage, 8 survivors, T = 300 s, no drone state cue) on
n_sites usable seeds starting at start_seed (default 1000, disjoint from the original 200 sites).
Writes runs/q4_triage_large.csv.
"""
import sys

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seeds = usable_seeds(n, start)
    jobs = make_jobs(seeds, ["full", "no_state", "oracle", "oracle_global"], "q4_triage_large",
                     reward=["triage"], T=[300.0], n_survivors=[8], p_state_cue=[0.0])
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/q4_triage_large.csv")
    print("q4_triage_large", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
