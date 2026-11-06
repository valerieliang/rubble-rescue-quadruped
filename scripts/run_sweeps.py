"""Q1 / Q2 sweeps on the stand-in sim. Writes CSVs to runs/.

Usage: RESCUE_SIM=go2 PYTHONPATH=. python scripts/run_sweeps.py [n_sites]
"""
import os
import sys

from rescue.drone_prior import PriorConfig
from rescue.experiments import run_grid, save_csv, sweep_cfgs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEEDS = range(N)
BASE = PriorConfig()
os.makedirs("runs", exist_ok=True)
ALL = ["coverage", "nearest", "priority"]

# Q1: value of the prior vs. position noise and false-positive rate (coverage is prior-free, run once per cfg for pairing).
for name, grid in {"q1_sigma": {"sigma_pos": [1, 3, 6, 10, 15, 25]},
                   "q1_fp": {"n_false_pos": [0, 1, 3, 6, 10, 20]}}.items():
    rows = run_grid(sweep_cfgs(BASE, **grid), ALL, SEEDS)
    save_csv(rows, f"runs/{name}.csv")
    print(name, len(rows), "rows")

# Q1 2-D: noise x false positives (prior methods only; coverage level taken from the 1-D sweeps).
rows = run_grid(sweep_cfgs(BASE, sigma_pos=[1, 3, 6, 10, 15, 25], n_false_pos=[0, 1, 3, 6, 10, 20]),
                ["priority"], SEEDS)
save_csv(rows, "runs/q1_grid.csv")
print("q1_grid", len(rows), "rows")

# Q2: priority vs nearest-first at default config, more sites for tighter paired CIs.
rows = run_grid([BASE], ["nearest", "priority"], range(4 * N))
save_csv(rows, "runs/q2_default.csv")
print("q2_default", len(rows), "rows")

# Q2 vs. class accuracy: priority weighting should only help when the drone's classes are reliable.
rows = run_grid(sweep_cfgs(BASE, p_class_correct=[0.4, 0.6, 0.8, 0.9, 1.0]), ["nearest", "priority"], range(4 * N))
save_csv(rows, "runs/q2_class.csv")
print("q2_class", len(rows), "rows")
