"""Per-layer compute cost of the stack, measured (not estimated).

Usage (repo root, on an otherwise idle machine):  PYTHONPATH=. python scripts/diagnostics/profile_layers.py [n_sites] [T]

Runs control-rate episodes (unicycle plant) and times each layer's calls with perf_counter, single process:
  plan   1 Hz   whole search/belief/route update          (RescueStack.plan)
  select 1 Hz   belief update + target selection         (StateAwareSearch.select, inside plan)
  route  1 Hz   terrain path planner                     (TerrainGrid.path, inside plan)
  infer  1 Hz   survivor-state motion inference          (RescueStack._motion_updates, inside plan)
  mpc    10 Hz  approach MPC                             (ApproachMPC.solve)
  cbf    10 Hz  safety filter                            (CBFFilter.filter)
for the full stack and for each layer removed or added (no state inference, no filter, cost-model legibility,
heading legibility). A 10 Hz layer has a 100 ms budget. Writes runs/profile.csv.
"""
import csv
import sys
import time
from dataclasses import replace

import numpy as np

import rescue.ctrl_mpc as ctrl_mpc
import rescue.safety as safety
import rescue.stack as stack
from rescue.ctrl_behavior import make_site
from rescue.ctrl_experiments import usable_seeds
from rescue.ctrl_sim import run_episode_ctrl
from rescue.drone_prior import PriorConfig
from rescue.stack import VARIANTS

LAYERS = {"plan": (stack.RescueStack, "plan"), "select": (stack.StateAwareSearch, "select"),
          "route": (ctrl_mpc.TerrainGrid, "path"), "infer": (stack.RescueStack, "_motion_updates"),
          "mpc": (ctrl_mpc.ApproachMPC, "solve"), "cbf": (safety.CBFFilter, "filter")}
TIMES: dict[str, list[float]] = {k: [] for k in LAYERS}


def instrument():
    for name, (cls, meth) in LAYERS.items():
        orig = getattr(cls, meth)

        def wrapped(*a, __o=orig, __n=name, **kw):
            t = time.perf_counter()
            try:
                return __o(*a, **kw)
            finally:
                TIMES[__n].append(time.perf_counter() - t)
        setattr(cls, meth, wrapped)


CONFIGS = {"full": VARIANTS["full"],
           "no_state": VARIANTS["no_state"],
           "no_cbf": VARIANTS["no_cbf"],
           "full + cost legibility (300)": replace(VARIANTS["full"], legibility=300.0),
           "full + heading legibility (3)": replace(VARIANTS["full"], legibility_heading=3.0)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    instrument()
    rows = []
    print(f"{'config':32s} {'layer':7s} {'calls':>7s} {'mean ms':>8s} {'p95 ms':>8s} {'max ms':>8s} {'share of wall':>14s}")
    for name, cfg in CONFIGS.items():
        for k in TIMES:
            TIMES[k].clear()
        wall = 0.0
        for seed in usable_seeds(n):
            t0 = time.perf_counter()
            run_episode_ctrl(make_site(seed), PriorConfig(), cfg, seed, name, T=T)
            wall += time.perf_counter() - t0
        for layer, ts in TIMES.items():
            if not ts:
                continue
            a = np.array(ts) * 1e3
            nested = layer in ("select", "route", "infer")
            row = dict(config=name, layer=layer, calls=len(a), mean_ms=a.mean(), p95_ms=np.percentile(a, 95),
                       max_ms=a.max(), share_of_wall=a.sum() / 1e3 / wall)
            rows.append(row)
            print(f"{name:32s} {layer:7s} {len(a):7d} {a.mean():8.2f} {row['p95_ms']:8.2f} {a.max():8.2f} "
                  f"{row['share_of_wall']:13.1%}{'*' if nested else ''}")
        print(f"{name:32s} wall {wall:.1f} s for {n} episodes of up to {T:.0f} s simulated")
    print("* nested inside plan")
    with open("runs/profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
