"""Track-retention ablation: do the three remedies for lost survivors help, and where?

Usage (repo root):  RESCUE_SIM=go2 PYTHONPATH=. python scripts/experiments/run_retention.py [n_sites] [workers]

Runs the priority policy and each retention variant (search.PriorSearch ``retention``) on the same sites and
drone detections across position noise sigma, writes runs/retention.csv, and prints paired differences
(variant minus priority, mean +/- 95% half-width over sites). Use few workers (<= 8): every worker imports scipy.
"""
import sys

from rescue.drone_prior import PriorConfig
from rescue.experiments import RETENTION, paired_diff, run_grid, save_csv, sweep_cfgs

SIGMAS = [1, 3, 6, 10, 15, 25]
METRICS = ["norm_score", "rescue_rate", "time_to_critical"]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    methods = ["coverage", "priority", *RETENTION]
    rows = run_grid(sweep_cfgs(PriorConfig(), sigma_pos=SIGMAS), methods, range(n), workers=workers)
    save_csv(rows, "runs/retention.csv")
    print(len(rows), "rows")
    for metric in METRICS:
        print(f"\n{metric}: variant minus priority (n = {n} sites)")
        print(f"{'sigma':>6s} " + " ".join(f"{m:>18s}" for m in RETENTION))
        for s in SIGMAS:
            sub = [r for r in rows if r["sigma_pos"] == s]
            cells = []
            for m in RETENTION:
                d = paired_diff(sub, m, "priority", metric)
                cells.append(f"{d['mean']:+8.3f}+/-{d['ci']:.3f}")
            print(f"{s:6d} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
