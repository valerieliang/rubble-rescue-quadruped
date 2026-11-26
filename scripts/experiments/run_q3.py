"""Q3 baseline (baseline 3: no safety filter). Writes runs/q3_baseline.csv and runs/figures/q3_baseline.png.

Sweeps drone position noise and records safety violations of the unfiltered policies (time in a
hazard region, or within the survivor keep-out radius), next to the radius of the 95% clearance
region a filter would have to respect at that noise. The filtered runs need the CBF-QP filter
and are not produced here.

Usage: RESCUE_SIM=go2 PYTHONPATH=. python scripts/experiments/run_q3.py [n_sites]
"""
import os
import sys

from rescue.drone_prior import PriorConfig
from rescue.experiments import run_grid, save_csv, sweep_cfgs

SIGMAS = [1, 3, 6, 10, 15, 25]


def run(n_sites: int) -> None:
    os.makedirs("runs", exist_ok=True)
    rows = run_grid(sweep_cfgs(PriorConfig(), sigma_pos=SIGMAS), ["coverage", "nearest", "priority"],
                    range(n_sites))
    save_csv(rows, "runs/q3_baseline.csv")
    print("q3_baseline", len(rows), "rows", flush=True)


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from rescue.clearance import chi2_level
    from rescue.experiments import load_csv, mean_ci
    from rescue.interfaces import STANDOFF_RADIUS, SURVIVOR_KEEPOUT

    rows = load_csv("runs/q3_baseline.csv")
    xs = sorted({r["sigma_pos"] for r in rows})
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": grid,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})
    style = {"priority": ("#2a78d6", "o", "priority-weighted"), "nearest": ("#eb6834", "s", "nearest-first"),
             "coverage": ("#8a8984", "^", "no prior (coverage)")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    for m, (color, marker, label) in style.items():
        y, h = np.array([mean_ci([r["violation_rate"] for r in rows if r["method"] == m and r["sigma_pos"] == x])
                         for x in xs]).T
        a.plot(xs, y, color=color, marker=marker, ms=4, lw=1.5, label=label, ls="--" if m == "coverage" else "-")
        a.fill_between(xs, y - h, y + h, color=color, alpha=0.15, lw=0)
    a.set_xscale("log")
    a.set_xlabel("Drone position noise sigma (m)")
    a.set_ylabel("Fraction of steps in violation")
    a.set_title("No safety filter (baseline 3)", fontsize=9, color=ink)
    a.legend(frameon=False, fontsize=8)
    sg = np.array(xs, float)
    for conf, ls in ((0.95, "-"), (0.99, ":")):
        b.plot(sg, sg * np.sqrt(chi2_level(conf)) + SURVIVOR_KEEPOUT, color="#2a78d6", ls=ls, lw=1.5,
               label=f"{int(conf * 100)}% clearance radius")
    b.axhline(STANDOFF_RADIUS, color=muted, lw=1, ls="--", label="rescue standoff radius")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("Drone position noise sigma (m)")
    b.set_ylabel("Keep-out radius around a track (m)")
    b.set_title("Region a filter must respect", fontsize=9, color=ink)
    b.legend(frameon=False, fontsize=8)
    os.makedirs("runs/figures", exist_ok=True)
    fig.savefig("runs/figures/q3_baseline.png")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        run(int(next((a for a in sys.argv[1:] if a.isdigit()), 40)))
    plot()
