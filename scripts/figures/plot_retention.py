"""Figure for the track-retention ablation: score, time to critical and rescue rate vs position noise.

Usage: PYTHONPATH=. python scripts/figures/plot_retention.py [out.png]    (needs runs/retention.csv from run_retention.py)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rescue.experiments import load_csv, mean_ci

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {"priority": ("#2a78d6", "o", "prior + priority (as in Q1)", "-"),
         "prio_keep": ("#eb6834", "s", "+ keep tracks until searched", "-"),
         "prio_region_keep": ("#3a9d5d", "^", "+ keep + search unobserved area", ":"),
         "coverage": ("#8a8984", "v", "no prior (coverage)", "--")}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 150})
METRICS = [("norm_score", "Oracle-normalized score"), ("time_to_critical", "Time to critical survivors (s)"),
           ("rescue_rate", "Rescue rate within T")]


def main(out="figures/metrics/retention.png"):
    rows = load_csv("runs/retention.csv")
    xs = sorted({r["sigma_pos"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for m, (color, marker, label, ls) in STYLE.items():
            stats = [mean_ci([r[metric] for r in rows if r["method"] == m and r["sigma_pos"] == x]) for x in xs]
            y, h = np.array(stats).T
            ax.plot(xs, y, color=color, marker=marker, ms=4, lw=1.5, ls=ls, label=label)
            ax.fill_between(xs, y - h, y + h, color=color, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("position noise sigma (m)")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=4, loc="outside lower center")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
