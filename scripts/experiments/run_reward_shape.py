"""Why is priority ordering null? Sweep the shape of the class reward and see when ordering starts to matter.

Usage (repo root):  RESCUE_SIM=go2 PYTHONPATH=. python scripts/run_reward_shape.py [n_sites] [workers] [first_seed]

The reward for reaching survivor i at time t is w_class * exp(-decay_class * t). With the defaults (weights
1/2/4, decays 0.002/0.005/0.01 per s) a critical survivor is worth less than a stable one after 173 s, so the
decay cancels the weight for most rescues and class tells the planner little. This varies weights and decays
(these are modelling choices, not calibrated to real triage outcomes), with 8 survivors and T = 300 s, and
compares priority-weighted with nearest-first, paired by site, at 80% and 100% class accuracy. Each setting
is scored against its own oracle. Writes runs/reward_shape.csv.
"""
import sys

from rescue.drone_prior import PriorConfig
from rescue.experiments import paired_diff, run_grid, save_csv

D0 = (0.002, 0.005, 0.01)
SETTINGS = {   # name: (weights stable/urgent/critical, decays)
    "default (w 1/2/4, decay x1)": ((1, 2, 4), D0),
    "no decay": ((1, 2, 4), (0, 0, 0)),
    "decay x0.3": ((1, 2, 4), tuple(0.3 * d for d in D0)),
    "decay x3": ((1, 2, 4), tuple(3 * d for d in D0)),
    "decay x10": ((1, 2, 4), tuple(10 * d for d in D0)),
    "weights 1/4/16, decay x1": ((1, 4, 16), D0),
    "weights 1/4/16, decay x3": ((1, 4, 16), tuple(3 * d for d in D0)),
    "class irrelevant (w 1/1/1, no decay)": ((1, 1, 1), (0, 0, 0)),
}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rows = []
    for name, (w, d) in SETTINGS.items():
        for acc in (0.8, 1.0):
            r = run_grid([PriorConfig(p_class_correct=acc)], ["nearest", "priority"], range(first, first + n), T=300.0,
                         workers=workers, n_survivors=8, reward=(w, d))
            for x in r:
                x["setting"] = name
            rows += r
            print(name, acc, "done", flush=True)
    import csv
    with open("runs/reward_shape.csv" if first == 0 else f"runs/reward_shape_{first}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\npriority minus nearest-first, 8 survivors, T = 300 s, {n} sites")
    print(f"{'setting':40s} {'acc':>4s} {'nearest score':>14s} {'score diff':>18s} {'t_crit diff (s)':>18s}")
    for name in SETTINGS:
        for acc in (0.8, 1.0):
            sub = [x for x in rows if x["setting"] == name and x["p_class_correct"] == acc]
            near = sum(x["norm_score"] for x in sub if x["method"] == "nearest") / n
            a, b = paired_diff(sub, "priority", "nearest", "norm_score"), paired_diff(sub, "priority", "nearest", "time_to_critical")
            star = "*" if abs(a["mean"]) > a["ci"] else ""
            print(f"{name:40s} {acc:4.1f} {near:14.3f} {a['mean']:+9.3f}+/-{a['ci']:.3f}{star:1s} {b['mean']:+9.1f}+/-{b['ci']:.1f}")


if __name__ == "__main__":
    main()
