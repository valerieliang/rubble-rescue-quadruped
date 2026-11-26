"""Q4: survivor-state inference and its ablations. Writes runs/q4_*.csv.

Usage: PYTHONPATH=. python scripts/experiments/ctrl_run_q4.py [n_sites] [workers]

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
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    # Triage regime: more survivors than the robot can reach in T, and state-dependent handling time
    # (unresponsive 30 s, trapped 15 s, mobile 0 s), faster decay for unresponsive survivors and
    # self-evacuation of mobile ones (interfaces.TRIAGE_REWARD). oracle_global = upper bound (all z known at t=0).
    VT = ["full", "no_state", "oracle", "oracle_global"]
    triage = {"q4_triage": dict(variants=VT, grid={"reward": ["triage"], "T": [300.0], "n_survivors": [8],
                                                   "p_state_cue": [0.0, 0.7]}),
              "q4_triage_sweep": dict(variants=VT, grid={"reward": ["triage"], "T": [300.0], "n_survivors": [4, 6, 8],
                                                         "p_state_cue": [0.0]})}
    if which in ("all", "triage", "sweep"):
        for name, s_ in triage.items():
            if which == "sweep" and name != "q4_triage_sweep":
                continue
            rows = run_jobs(make_jobs(seeds, s_["variants"], name, **s_["grid"]), workers)
            save_rows(rows, f"runs/{name}.csv")
            print(name, len(rows), "rows", flush=True)
    if which in ("triage", "sweep"):
        return
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
