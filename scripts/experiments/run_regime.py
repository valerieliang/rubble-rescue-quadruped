"""Q2 regime map: when does priority-weighted ordering beat nearest-first?

Usage (repo root):  RESCUE_SIM=go2 PYTHONPATH=. python scripts/experiments/run_regime.py [n_sites] [workers]

Priority vs nearest-first (same prior, same sites) over horizon T and survivors per site, at the default 80%
class accuracy and at 100%. Ordering can only matter when the robot cannot reach everyone, so the gain should
grow as T shrinks and survivors multiply. Writes runs/regime_q2.csv; prints paired score differences.
"""
import sys

import numpy as np

from rescue.drone_prior import PriorConfig
from rescue.experiments import paired_diff, run_grid, save_csv

TS = [100.0, 150.0, 300.0, 600.0]
NS = [3, 5, 8]
ACC = [0.8, 1.0]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rows = []
    for acc in ACC:
        for ns in NS:
            for T in TS:
                cfg = PriorConfig(p_class_correct=acc)
                r = run_grid([cfg], ["nearest", "priority"], range(n), T=T, workers=workers, n_survivors=ns)
                rows += r
                print(f"acc {acc} n {ns} T {T:.0f} done", flush=True)
    save_csv(rows, "runs/regime_q2.csv")
    for metric in ("norm_score", "time_to_critical"):
        for acc in ACC:
            print(f"\n{metric}: priority minus nearest, class accuracy {acc}")
            print(f"{'survivors':>10s} " + " ".join(f"T={T:<14.0f}" for T in TS))
            for ns in NS:
                cells = []
                for T in TS:
                    sub = [r for r in rows if r["p_class_correct"] == acc and r["n_survivors"] == ns and r["T"] == T]
                    d = paired_diff(sub, "priority", "nearest", metric)
                    cells.append(f"{d['mean']:+.3f}+/-{d['ci']:.3f}".ljust(16))
                print(f"{ns:10d} " + " ".join(cells))
    base = [r for r in rows if r["method"] == "nearest" and r["p_class_correct"] == 0.8]
    print("\nnearest-first score by (survivors, T):")
    for ns in NS:
        print(ns, [round(float(np.mean([r['norm_score'] for r in base if r['n_survivors'] == ns and r['T'] == T])), 3) for T in TS])


if __name__ == "__main__":
    main()
