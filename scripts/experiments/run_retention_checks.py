"""Checks on the 'keep' remedy: fresh sites, region size, and false-positive cost.

Usage (repo root):  RESCUE_SIM=go2 PYTHONPATH=. python scripts/experiments/run_retention_checks.py [n_sites] [workers]

1. Fresh sites (seeds 1000+, disjoint from the 0-159 of run_retention.py): priority vs keep across sigma.
2. Region size: keep with k = 1, 2 (default), 3 sigma.
3. False positives: keep retains a track until its region is searched, so spurious tracks cost time.
Writes runs/retention_fresh.csv and runs/retention_fp.csv.
"""
import sys

from rescue.drone_prior import PriorConfig
from rescue.experiments import paired_diff, run_grid, save_csv, sweep_cfgs

METRICS = ["norm_score", "rescue_rate", "time_to_critical"]


def table(rows, key, values, variants, base="priority"):
    for metric in METRICS:
        print(f"\n{metric}: variant minus {base}")
        print(f"{key:>10s} " + " ".join(f"{v:>18s}" for v in variants))
        for x in values:
            sub = [r for r in rows if r[key] == x]
            cells = [paired_diff(sub, v, base, metric) for v in variants]
            print(f"{x:10g} " + " ".join(f"{c['mean']:+8.3f}+/-{c['ci']:.3f}".rjust(18) for c in cells))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = range(1000, 1000 + n)
    variants = ["prio_keep", "prio_keep_k1", "prio_keep_k3"]
    rows = run_grid(sweep_cfgs(PriorConfig(), sigma_pos=[1, 3, 6, 10, 15, 25]), ["priority", *variants], seeds,
                    workers=workers)
    save_csv(rows, "runs/retention_fresh.csv")
    print("fresh sites,", n, "seeds from 1000")
    table(rows, "sigma_pos", [1, 3, 6, 10, 15, 25], variants)

    rows = run_grid(sweep_cfgs(PriorConfig(), n_false_pos=[0, 1, 6, 20]), ["priority", "prio_keep"], seeds,
                    workers=workers)
    save_csv(rows, "runs/retention_fp.csv")
    print("\nfalse positives (sigma 3 m)")
    table(rows, "n_false_pos", [0, 1, 6, 20], ["prio_keep"])


if __name__ == "__main__":
    main()
