"""Draw a seeded site and, optionally, one run's trajectory. Saves PNGs to figures/path_planning/.

Usage: python scripts/render_site.py SEED [SEED ...] [--method {coverage,nearest,priority}]
                                     [--sim {toy,go2}] [--out figures/path_planning]

Without --method it draws the site and drone detections only. Each seed is used for both the
site and the episode, as in rescue/experiments.py, so the detections shown are the ones the run saw.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from rescue import baselines, sim, toy_sim
from rescue.drone_prior import PriorConfig, generate_detections
from rescue.interfaces import Priority

T = 600.0
METHODS = {"coverage": baselines.coverage_policy,
           "nearest": baselines.nearest_first_policy,
           "priority": baselines.priority_policy}
METHOD_LABELS = {"coverage": "coverage (no prior)", "nearest": "prior + nearest-first",
                 "priority": "prior + priority"}
SIMS = {"toy": toy_sim, "go2": sim}
SIM_LABELS = {"toy": "toy sim", "go2": "Go2 sim"}

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
RUBBLE, HAZARD = "#e6e5e1", "#c3c2b7"
# Colour plus marker shape, so class never depends on colour alone.
CLASS_STYLE = {Priority.STABLE: ("#2a78d6", "o", "stable"),
               Priority.URGENT: ("#1baf7a", "s", "urgent"),
               Priority.CRITICAL: ("#eb6834", "D", "critical")}
UNCERTAINTY_SIGMAS = 2.0   # circle radius in units of the drone's position noise
HALO = [pe.withStroke(linewidth=2.5, foreground=SURFACE)]


def draw_regions(ax, site):
    for r in site.regions:   # rubble first so hazards stay visible where they overlap
        if r.kind == "rubble":
            ax.add_patch(Rectangle((r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin,
                                   facecolor=RUBBLE, edgecolor="none", zorder=1))
    for r in site.regions:
        if r.kind == "hazard":
            ax.add_patch(Rectangle((r.xmin, r.ymin), r.xmax - r.xmin, r.ymax - r.ymin,
                                   facecolor=HAZARD, edgecolor=MUTED, hatch="///", lw=0.6, zorder=2))


def draw_detections(ax, detections, sigma):
    for d in detections:
        color = CLASS_STYLE[d.priority][0]
        ax.add_patch(Circle(d.pos, UNCERTAINTY_SIGMAS * sigma, fill=False, edgecolor=color,
                            ls="--", lw=1.0, zorder=3, clip_on=True))
        ax.plot(*d.pos, marker="x", ms=6, mew=1.5, color=color, zorder=4)
        if d.source_id is None:
            ax.annotate("false positive", d.pos, xytext=(6, -10), textcoords="offset points",
                        fontsize=6.5, color=MUTED, path_effects=HALO, zorder=7)


def draw_survivors(ax, site):
    for s in site.survivors:
        color, marker, _ = CLASS_STYLE[s.priority]
        ax.plot(*s.pos, marker=marker, ms=8, color=color, mec=SURFACE, mew=1.2, zorder=6)


def draw_run(ax, site, log):
    start = site.start
    xy = np.array([start] + [(x, y) for _, x, y in log.trajectory])
    ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=1.2, zorder=5, solid_joinstyle="round")
    for s in site.survivors:
        if s.id in log.rescue_times:
            ax.plot(*s.pos, marker="o", ms=13, mfc="none", mec=INK, mew=1.2, zorder=6)
            ax.annotate(f"{log.rescue_times[s.id]:.0f} s", s.pos, xytext=(8, 7),
                        textcoords="offset points", fontsize=7.5, color=INK,
                        path_effects=HALO, zorder=7)


def legend_handles(sigma, with_run):
    handles = [Line2D([], [], ls="", marker=m, ms=7, color=c, mec=SURFACE, label=f"survivor: {name}")
               for c, m, name in CLASS_STYLE.values()]
    handles += [Line2D([], [], ls="--", marker="x", ms=6, color=MUTED,
                       label=f"drone detection, {UNCERTAINTY_SIGMAS:.0f}σ = {UNCERTAINTY_SIGMAS * sigma:g} m\n"
                             "(colour = reported class)"),
                Line2D([], [], ls="", marker="^", ms=7, color=INK, label="robot start"),
                Patch(facecolor=RUBBLE, label="rubble (0.5 m/s)"),
                Patch(facecolor=HAZARD, edgecolor=MUTED, hatch="///", label="hazard (0.25 m/s)")]
    if with_run:
        handles += [Line2D([], [], color=INK, lw=1.2, label="robot path"),
                    Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=INK,
                           label="rescued, with time")]
    return handles


def render(seed, cfg, method=None, sim_name="go2", out="figures/path_planning"):
    simulator = SIMS[sim_name]
    site = simulator.generate_site(seed)
    # run_episode draws the detections first from default_rng(seed), so this matches the run.
    detections = generate_detections(site, cfg, np.random.default_rng(seed))
    log = None
    if method:
        factory = METHODS[method]
        log = simulator.run_episode(site, lambda s, d: factory(s, d, cfg, horizon=T), cfg, seed,
                                    method, T=T)

    fig, ax = plt.subplots(figsize=(7.6, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    draw_regions(ax, site)
    draw_detections(ax, detections, cfg.sigma_pos)
    if log:
        draw_run(ax, site, log)
    draw_survivors(ax, site)
    ax.plot(*site.start, marker="^", ms=8, color=INK, mec=SURFACE, mew=1.0, zorder=8)

    xmin, ymin, xmax, ymax = site.bounds
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)

    title = f"Site {seed}: {len(site.survivors)} survivors, {len(detections)} drone detections"
    if log:
        title = (f"Site {seed}, {SIM_LABELS[sim_name]}, {METHOD_LABELS[method]}: "
                 f"{len(log.rescue_times)} of {len(site.survivors)} rescued by {T:.0f} s")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.legend(handles=legend_handles(cfg.sigma_pos, bool(log)), loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8, labelcolor=INK,
              columnspacing=1.6, handletextpad=0.5)

    os.makedirs(out, exist_ok=True)
    name = f"site_{seed:03d}_{sim_name}_{method}.png" if method else f"site_{seed:03d}.png"
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seeds", nargs="+", type=int)
    parser.add_argument("--method", choices=tuple(METHODS))
    parser.add_argument("--sim", choices=tuple(SIMS), default="go2")
    parser.add_argument("--out", default="figures/path_planning")
    args = parser.parse_args()
    for seed in args.seeds:
        print(render(seed, PriorConfig(), args.method, args.sim, args.out))


if __name__ == "__main__":
    main()
