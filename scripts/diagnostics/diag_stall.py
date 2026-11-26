"""Diagnose episodes where the robot stops making progress: old router vs the region-avoiding router.

Usage (repo root):  PYTHONPATH=. python scripts/diagnostics/diag_stall.py

Runs a few unicycle episodes (seconds each) that lost rescues when the router started avoiding safety
regions, with the old stack, the new router only, and the new router plus forget_stale_tracks, recording the stack's state
at every control step. For each, prints the rescues and the longest stretch without a rescue: how far the
robot actually moved, what it was commanded, whether the planner/filter stopped it, and where the route
and waypoint pointed. Writes runs/diag_<case>_<setup>.npz.
"""
import os
import sys
from dataclasses import replace

import numpy as np

from rescue import ctrl_sim
from rescue.ctrl_behavior import BehaviorConfig, make_site
from rescue.drone_prior import PriorConfig
from rescue.interfaces import REWARDS
from rescue.stack import VARIANTS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "figures"))
from render_mj import diag_stack_class  # noqa: E402

# (name, seed, StackConfig overrides, sigma_meas): cases whose rescue rate fell with the new router
CASES = [
    ("s16_c07_p25", 16, dict(conf_p=0.7, eps_track=0.25), 1.0),     # 5/6 -> 1/6
    ("s18_c09_p25", 18, dict(conf_p=0.9, eps_track=0.25), 1.0),     # 7/7 -> 2/7
    ("s32_q3", 32, {}, 0.5),                                         # 6/6 -> 4/6 (Q3 default settings)
]


SETUPS = {"old": dict(route_around_regions=False, forget_stale_tracks=False),
          "router": dict(route_around_regions=True, forget_stale_tracks=False),
          "router+forget": dict(route_around_regions=True, forget_stale_tracks=True)}


def run(seed, over, sigma_meas, setup):
    site = make_site(seed)
    diag = []
    real = ctrl_sim.RescueStack
    ctrl_sim.RescueStack = diag_stack_class(real, diag)
    try:
        cfg = replace(VARIANTS["full"], **SETUPS[setup], **over)
        log = ctrl_sim.run_episode_ctrl(site, PriorConfig(), cfg, seed, "full", sigma_meas=sigma_meas,
                                        behavior=BehaviorConfig(), dwell=REWARDS["base"].dwell)
    finally:
        ctrl_sim.RescueStack = real
    return site, log, {k: np.array([d[k] for d in diag]) for k in diag[0]}


def report(name, setup, site, log, d):
    rt = sorted(log.rescue_times.values())
    print(f"\n=== {name}  {setup}: rescued {len(rt)}/{len(site.survivors)} "
          f"at {[round(x) for x in rt]}, min clearance {log.min_clearance:.2f} m")
    t = d["t"]
    edges = [0.0] + rt + [t[-1]]
    gaps = [(b - a, a, b) for a, b in zip(edges[:-1], edges[1:])]
    g, a, b = max(gaps)
    sl = (t >= a) & (t <= b)
    xy = np.c_[d["x"], d["y"]][sl]
    moved = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    print(f"   longest gap without a rescue: t = {a:.0f}..{b:.0f} s ({g:.0f} s); walked {moved:.0f} m, "
          f"net {np.linalg.norm(xy[-1] - xy[0]):.0f} m")
    still = (np.abs(d["u_v"][sl]) < 0.02)
    print(f"   commanded speed ~0 for {0.1 * still.sum():.0f} s; nominal speed ~0 for "
          f"{0.1 * (np.abs(d['nom_v'][sl]) < 0.02).sum():.0f} s; filter modified {d['modified'][sl].mean():.0%}, "
          f"infeasible {1 - d['feasible'][sl].mean():.0%}")
    tid = d["target_id"][sl]
    changes = int(np.sum(np.diff(np.nan_to_num(tid, nan=-1)) != 0))
    print(f"   target changes {changes}; no target {np.mean(np.isnan(tid)):.0%}; target confirmed "
          f"{np.nanmean(d['target_confirmed'][sl]):.0%}; median target sigma {np.nanmedian(d['target_sigma'][sl]):.2f}")
    print(f"   median goal distance {np.nanmedian(d['goal_dist'][sl]):.1f} m; median route vertices "
          f"{np.median(d['path_len'][sl]):.0f}; tracks {np.median(d['n_tracks'][sl]):.0f}, "
          f"keep-outs {np.median(d['n_keepout'][sl]):.0f}")
    nmin = lambda x: np.nanmin(x) if np.any(~np.isnan(x)) else float("nan")
    print(f"   min margin to other regions {nmin(d['other_margin'][sl]):.2f} m, to keep-outs "
          f"{nmin(d['keepout_margin'][sl]):.2f} m")
    idx = np.flatnonzero(sl)
    print("      t      x      y   nom_v  nom_w    u_v    u_w  target(x,y)      goal(x,y)       wp(x,y)    gd  other  keep")
    for j in idx[:: max(1, len(idx) // 14)]:
        print(f"   {t[j]:5.0f} {d['x'][j]:6.1f} {d['y'][j]:6.1f} {d['nom_v'][j]:6.2f} {d['nom_w'][j]:6.2f} "
              f"{d['u_v'][j]:6.2f} {d['u_w'][j]:6.2f}  ({d['target_x'][j]:5.1f},{d['target_y'][j]:5.1f})  "
              f"({d['goal_x'][j]:5.1f},{d['goal_y'][j]:5.1f})  ({d['wp_x'][j]:5.1f},{d['wp_y'][j]:5.1f}) "
              f"{d['goal_dist'][j]:5.1f} {d['other_margin'][j]:6.2f} {d['keepout_margin'][j]:5.2f}")


def main():
    os.makedirs("runs", exist_ok=True)
    for name, seed, over, sm in CASES:
        for setup in SETUPS:
            site, log, d = run(seed, over, sm, setup)
            report(name, setup, site, log, d)
            np.savez(f"runs/diag_{name}_{setup.replace('+', '_')}.npz", **d)


if __name__ == "__main__":
    main()
