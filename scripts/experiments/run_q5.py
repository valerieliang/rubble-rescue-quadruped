"""Q5: does a legibility term let an observer identify the goal earlier, and at what cost?

Standalone closed-loop approach scenarios (MPC + unicycle on open ground, no simulator): the robot
starts at a random point, 2-4 survivors are scattered at 12-35 m, and it approaches one of them.
A simulated observer (``observer.goal_posterior``, rationality beta) watches the trajectory and
tracks a posterior over the survivors. Legibility weight 0 is the no-legibility baseline.

Metrics per approach: time for the observer's posterior on the true goal to reach 0.8 and stay
there, as a fraction of the approach time; extra time the approach takes vs. weight 0; and path
length over straight-line distance.

The MPC and the observer share beta, so this measures what the term can do when the observer model
is right; ``--beta-obs`` evaluates against a different observer to see how that degrades.

Usage: PYTHONPATH=. python scripts/experiments/run_q5.py [n_trials] [--beta-obs B]
"""
import csv
import os
import sys

import numpy as np

from rescue.mpc import ApproachController, MPCPlanner
from rescue.observer import legibility_cost, time_to_identify

DT = 0.1
WEIGHTS = [0.0, 6.0, 20.0, 60.0, 150.0]
THRESH = 0.8
BETA = 0.5


def euclid(a, b):
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))


def make_trial(seed: int):
    rng = np.random.default_rng(seed)
    start = rng.uniform(10, 90, 2)
    n = int(rng.integers(2, 5))
    goals = []
    while len(goals) < n:
        ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(12, 35)
        g = start + r * np.array([np.cos(ang), np.sin(ang)])
        if (0 < g).all() and (g < 100).all() and all(euclid(g, h) > 6 for h in goals):
            goals.append(g)
    return start, np.array(goals), int(rng.integers(n))


def rollout(start, goals, true_idx, weight, seed, t_max=120.0):
    ctrl = ApproachController(MPCPlanner(seed=seed), legibility_base=weight, beta=BETA)
    others = [g for i, g in enumerate(goals) if i != true_idx]
    target = goals[true_idx]
    pos, heading, v = start.copy(), float(np.arctan2(*(target - start)[::-1])), 0.0
    traj = [(0.0, *pos)]
    for k in range(int(t_max / DT)):
        # Mobile survivors are watching: full legibility weight (see mpc.legibility_weight).
        v_cmd, om = ctrl.command(pos, heading, v, target, None, (), (), others,
                                 state_posterior=np.array([0.0, 0.0, 1.0]))
        v += float(np.clip(v_cmd - v, -DT, DT))
        heading = (heading + om * DT + np.pi) % (2 * np.pi) - np.pi
        pos = pos + v * DT * np.array([np.cos(heading), np.sin(heading)])
        traj.append(((k + 1) * DT, *pos))
        if euclid(pos, target) < 1.5:
            break
    return np.array(traj)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 60
    beta_obs = float(sys.argv[sys.argv.index("--beta-obs") + 1]) if "--beta-obs" in sys.argv else BETA
    os.makedirs("runs/figures", exist_ok=True)
    tag = "" if beta_obs == BETA else f"_obs{beta_obs:g}"
    rows = []
    for seed in range(n):
        start, goals, true_idx = make_trial(seed)
        base_T = None
        for w in WEIGHTS:
            traj = rollout(start, goals, true_idx, w, seed)
            T = traj[-1, 0]
            if w == 0.0:
                base_T = T
            sub = traj[::5]                                     # observer samples at 2 Hz
            tid = time_to_identify(sub, list(goals), true_idx, euclid, beta_obs, THRESH)
            rows.append({"seed": seed, "weight": w, "n_goals": len(goals), "approach_time": T,
                         "extra_time": T - base_T, "t_identify": tid,
                         "frac_identify": tid / T if np.isfinite(tid) else float("nan"),
                         "identified": float(np.isfinite(tid)), "path_ratio": legibility_cost(sub, euclid)})
    with open(f"runs/q5_legibility{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{n} approaches, observer beta = {beta_obs}, identified at posterior >= {THRESH}")
    print("weight  identified  frac of approach  extra time (s)  path ratio")
    stats = []
    for wt in WEIGHTS:
        r = [x for x in rows if x["weight"] == wt]
        frac = np.array([x["frac_identify"] for x in r], float)
        stats.append((wt, np.mean([x["identified"] for x in r]), np.nanmean(frac),
                      np.mean([x["extra_time"] for x in r]), np.mean([x["path_ratio"] for x in r])))
        print(f"{wt:6.0f}  {stats[-1][1]:9.2f}  {stats[-1][2]:15.2f}  {stats[-1][3]:13.1f}  {stats[-1][4]:9.3f}")
    plot(stats, tag)


def plot(stats, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": grid,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})
    st = np.array(stats)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    a.plot(st[:, 3], st[:, 2], color="#2a78d6", marker="o", ms=4, lw=1.5)
    for wt, x, y in zip(st[:, 0], st[:, 3], st[:, 2]):
        a.annotate(f"{wt:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7.5, color=muted)
    a.set_xlabel("Extra approach time vs. no legibility (s)")
    a.set_ylabel("Fraction of approach before goal identified")
    b.plot(range(len(st)), st[:, 4], color="#eb6834", marker="s", ms=4, lw=1.5)
    b.set_xticks(range(len(st)), [f"{w:g}" for w in st[:, 0]])
    b.set_xlabel("Legibility weight")
    b.set_ylabel("Path length / straight-line distance")
    fig.savefig(f"runs/figures/q5_legibility{tag}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
