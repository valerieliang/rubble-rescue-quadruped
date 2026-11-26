"""What if the class weights are made bigger? Priority ordering and retention as the critical weight grows.

Usage (repo root):  RESCUE_SIM=go2 PYTHONPATH=. python scripts/experiments/run_reward_weights.py [n_sites] [workers]

Class weights (stable/urgent/critical) 1/2/4 (default), 1/4/16, 1/8/64, 1/16/256, with the default decay rates,
in the default scene (3-8 survivors, T = 600 s) and in a capacity-limited one (8 survivors, T = 300 s), at 80%,
90% and 100% class accuracy. Prints priority minus nearest-first and (priority + keep) minus priority, paired by
site, each setting scored against its own oracle. Writes runs/reward_weights.csv. Weights are modelling choices.
"""
import csv
import sys

from rescue.drone_prior import PriorConfig
from rescue.experiments import paired_diff, run_grid

D0 = (0.002, 0.005, 0.01)
WEIGHTS = {"1/2/4 (default)": (1, 2, 4), "1/4/16": (1, 4, 16), "1/8/64": (1, 8, 64), "1/16/256": (1, 16, 256)}
SCENES = {"default scene (3-8 survivors, T=600)": dict(T=600.0, n_survivors=None),
          "capacity-limited (8 survivors, T=300)": dict(T=300.0, n_survivors=8)}
ACCS = (0.8, 0.9, 1.0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rows = []
    for scene, kw in SCENES.items():
        for wname, w in WEIGHTS.items():
            for acc in ACCS:
                r = run_grid([PriorConfig(p_class_correct=acc)], ["nearest", "priority", "prio_keep"], range(n),
                             workers=workers, reward=(w, D0), **kw)
                for x in r:
                    x.update(scene=scene, weights=wname)
                rows += r
                print(scene, wname, acc, "done", flush=True)
    with open("runs/reward_weights.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for scene in SCENES:
        print(f"\n{scene}, {n} sites: priority minus nearest-first | (priority+keep) minus priority")
        print(f"{'weights':18s} {'acc':>4s} {'nearest':>8s} {'prio-near score':>17s} {'t_crit (s)':>15s} {'keep-prio score':>17s}")
        for wname in WEIGHTS:
            for acc in ACCS:
                sub = [x for x in rows if x["scene"] == scene and x["weights"] == wname and x["p_class_correct"] == acc]
                near = sum(x["norm_score"] for x in sub if x["method"] == "nearest") / n
                a, b = paired_diff(sub, "priority", "nearest", "norm_score"), paired_diff(sub, "priority", "nearest", "time_to_critical")
                c = paired_diff(sub, "prio_keep", "priority", "norm_score")
                st = lambda d: "*" if abs(d["mean"]) > d["ci"] else " "
                print(f"{wname:18s} {acc:4.1f} {near:8.3f} {a['mean']:+8.3f}+/-{a['ci']:.3f}{st(a)} "
                      f"{b['mean']:+7.1f}+/-{b['ci']:5.1f}{st(b)} {c['mean']:+8.3f}+/-{c['ci']:.3f}{st(c)}")


if __name__ == "__main__":
    main()
