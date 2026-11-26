"""Figures and tables from runs/*.csv -> runs/figures/*.png.

Usage: PYTHONPATH=. python scripts/figures/plot_results.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rescue.experiments import load_csv, mean_ci, paired_diff

os.makedirs("runs/figures", exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# Validated categorical slots 1-2 (blue, orange); gray = prior-free reference.
STYLE = {"priority": ("#2a78d6", "o", "priority-weighted"),
         "nearest": ("#eb6834", "s", "nearest-first"),
         "coverage": ("#8a8984", "^", "no prior (coverage)")}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})

METRICS = [("norm_score", "Oracle-normalized score", False),
           ("time_to_critical", "Time to critical survivors (s)", True),
           ("rescue_rate", "Rescue rate within T", False)]


def sweep_figure(path, xkey, xlabel, out):
    rows = load_csv(path)
    xs = sorted({r[xkey] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    for ax, (metric, ylabel, _) in zip(axes, METRICS):
        for m, (color, marker, label) in STYLE.items():
            stats = [mean_ci([r[metric] for r in rows if r["method"] == m and r[xkey] == x]) for x in xs]
            y, h = np.array(stats).T
            ax.plot(xs, y, color=color, marker=marker, ms=4, lw=1.5, label=label,
                    ls="--" if m == "coverage" else "-")
            ax.fill_between(xs, y - h, y + h, color=color, alpha=0.15, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xkey == "sigma_pos":
            ax.set_xscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(handles),
               loc="outside lower center")
    fig.savefig(out)
    plt.close(fig)


def heatmap(path, out):
    grid = load_csv(path)
    cov = [r["norm_score"] for r in load_csv("runs/q1_sigma.csv") if r["method"] == "coverage"]
    cov_mean = mean_ci(cov)[0]
    sig = sorted({r["sigma_pos"] for r in grid})
    fps = sorted({r["n_false_pos"] for r in grid})
    Z = np.array([[mean_ci([r["norm_score"] for r in grid
                            if r["sigma_pos"] == s and r["n_false_pos"] == f])[0] - cov_mean
                   for s in sig] for f in fps])
    lim = np.nanmax(np.abs(Z))
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    im = ax.imshow(Z, origin="lower", cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(sig)), [f"{s:g}" for s in sig])
    ax.set_yticks(range(len(fps)), [f"{f:g}" for f in fps])
    ax.set_xlabel("position noise sigma (m)")
    ax.set_ylabel("expected false positives")
    ax.grid(False)
    for i in range(len(fps)):
        for j in range(len(sig)):
            ax.text(j, i, f"{Z[i, j]:+.2f}", ha="center", va="center", fontsize=6.5, color=INK)
    fig.colorbar(im, ax=ax, label="score vs. no-prior coverage")
    fig.savefig(out)
    plt.close(fig)


def q2_figure(path, out):
    rows = load_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9), constrained_layout=True)
    print("Q2: priority - nearest-first, paired over sites (mean +/- 95% CI, win rate)")
    for ax, (metric, label, lower_better) in zip(axes, METRICS):
        d = paired_diff(rows, "priority", "nearest", metric)
        print(f"  {label:34s} {d['mean']:+.3f} +/- {d['ci']:.3f}   priority higher on {d['win_rate']:.0%} of sites (rest tied or lower)   n={d['n']}")
        ax.errorbar([d["mean"]], [0], xerr=[d["ci"]], fmt="o", color=STYLE["priority"][0], capsize=3, ms=5)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks([])
        ax.set_xlabel(f"{label}\n(priority - nearest{', lower is better' if lower_better else ''})")
        ax.grid(axis="y", visible=False)
    fig.savefig(out)
    plt.close(fig)


def q2_class_figure(path, out):
    rows = load_csv(path)
    ps = sorted({r["p_class_correct"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), constrained_layout=True)
    print("Q2 vs class accuracy: priority - nearest (mean +/- 95% CI)")
    for ax, (metric, label, lower_better) in zip(axes, METRICS):
        ds = [paired_diff([r for r in rows if r["p_class_correct"] == p], "priority", "nearest", metric) for p in ps]
        for p, d in zip(ps, ds):
            print(f"  {metric:18s} p_class={p:.1f}: {d['mean']:+.3f} +/- {d['ci']:.3f}  higher on {d['win_rate']:.0%}")
        ax.errorbar(ps, [d["mean"] for d in ds], yerr=[d["ci"] for d in ds], fmt="o-", color=STYLE["priority"][0], capsize=3, ms=4)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("drone class accuracy")
        ax.set_ylabel(f"{label}\n(priority - nearest{', lower better' if lower_better else ''})", fontsize=7)
    fig.savefig(out)
    plt.close(fig)


sweep_figure("runs/q1_sigma.csv", "sigma_pos", "position noise sigma (m)", "runs/figures/q1_sigma.png")
sweep_figure("runs/q1_fp.csv", "n_false_pos", "expected false positives", "runs/figures/q1_fp.png")
heatmap("runs/q1_grid.csv", "runs/figures/q1_heatmap.png")
q2_figure("runs/q2_default.csv", "runs/figures/q2_paired.png")
q2_class_figure("runs/q2_class.csv", "runs/figures/q2_class.png")
print("figures in runs/figures/")
