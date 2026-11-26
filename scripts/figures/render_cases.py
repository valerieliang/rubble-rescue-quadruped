"""Six-panel case-study figure: one rollout per error source of the drone prior.

Usage: PYTHONPATH=. python scripts/figures/render_cases.py [--out figures/path_planning/cases_grid.png]
                                                    [--stats]

Each panel is a seeded Go2-sim site run with the default prior (sigma 3 m, 10% miss, 1 false
positive, 80% class accuracy) and the priority policy. Panels E and F also draw the nearest-first
path on the same site. --stats prints the numbers quoted in the report (scores, rescue times,
counterfactuals with one error source removed) so the paper table can be checked against the code.
"""
import argparse
import dataclasses
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from render_site import (CLASS_STYLE, HALO, INK, MUTED, SURFACE, UNCERTAINTY_SIGMAS,
                         draw_regions)
from rescue import baselines, sim
from rescue.drone_prior import PriorConfig, generate_detections
from rescue.metrics import summarize

T = 600.0
CFG = PriorConfig()
FACTORY = {"priority": baselines.priority_policy, "nearest": baselines.nearest_first_policy}

# seed, panel title, whether to compare against the nearest-first run (times shown as p/n), the
# survivor id whose dropped track is annotated with the sensor footprint (or None), the data-frame
# position of the "missed" tag (None = just below the marker), and whether the nearest-first path
# is drawn when it is compared (draw_other_path, default True).
CASES = [
    dict(tag="A", seed=0, title="Drone miss, found en route", overlay=False, footprint_for=None,
         missed_xy=(10, 96)),
    dict(tag="B", seed=1, title="Drone miss, never found", overlay=False, footprint_for=None,
         missed_xy=None),
    dict(tag="C", seed=139, title="Position error, track dropped", overlay=False, footprint_for=0,
         missed_xy=None),
    dict(tag="D", seed=48, title="False positives, time wasted", overlay=False, footprint_for=None,
         missed_xy=None),
    dict(tag="E", seed=152, title="Priority ordering helps", overlay=True, footprint_for=None,
         missed_xy=None),
    dict(tag="F", seed=155, title="Wrong class label hurts", overlay=True, footprint_for=None,
         missed_xy=None, draw_other_path=False),
]


def run_case(seed, method, tweak=None):
    """Run one episode; returns site, detections, log and the first time each reported track is
    dropped (existence probability below the policy cutoff), keyed by survivor id."""
    site = sim.generate_site(seed)
    dets = generate_detections(site, CFG, np.random.default_rng(seed))
    drops = {}

    def make(s, d):
        d = tweak(s, d) if tweak else d
        policy = FACTORY[method](s, d, CFG, horizon=T)
        by_pos = {tuple(np.round(x.pos, 9)): x.source_id for x in d if x.source_id is not None}
        inner = policy.act

        def act(obs):
            wp = inner(obs)
            for tr in policy.belief.tracks:
                sid = by_pos.get(tuple(np.round(tr.mean, 9)))
                if sid is not None and not tr.confirmed and tr.r < policy.r_min and sid not in drops:
                    drops[sid] = (obs.t, np.array(obs.robot_pos))
            return wp

        policy.act = act
        return policy

    log = sim.run_episode(site, make, CFG, seed, method, T=T)
    return site, dets, log, drops


def path_xy(site, log):
    return np.array([site.start] + [(x, y) for _, x, y in log.trajectory])


def rescue_labels(log, other=None):
    """Label text per survivor id: time in s, or 'p/n' times when another run is overlaid."""
    out = {}
    for sid, t in log.rescue_times.items():
        out[sid] = f"{t:.0f}"
    if other is not None:
        ids = set(log.rescue_times) | set(other.rescue_times)
        out = {i: "{}/{}".format(f"{log.rescue_times[i]:.0f}" if i in log.rescue_times else "-",
                                 f"{other.rescue_times[i]:.0f}" if i in other.rescue_times else "-")
               for i in ids}
    return out


def draw_panel(ax, case):
    seed = case["seed"]
    site, dets, log, drops = run_case(seed, "priority")
    log_n = run_case(seed, "nearest")[2] if case["overlay"] else None
    det_ids = {d.source_id for d in dets if d.source_id is not None}

    draw_regions(ax, site)
    if log_n is not None and case.get("draw_other_path", True):
        xy = path_xy(site, log_n)
        ax.plot(xy[:, 0], xy[:, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=4)
    xy = path_xy(site, log)
    ax.plot(xy[:, 0], xy[:, 1], color=INK, lw=0.9, zorder=5, solid_joinstyle="round")

    for d in dets:
        color = CLASS_STYLE[d.priority][0]
        ax.add_patch(Circle(d.pos, UNCERTAINTY_SIGMAS * CFG.sigma_pos, fill=False, edgecolor=color,
                            ls="--", lw=0.6, zorder=3))
        ax.plot(*d.pos, marker="x", ms=4, mew=1.0, color=color, zorder=4)
        if d.source_id is None:
            ax.annotate("FP", d.pos, xytext=(3, -8), textcoords="offset points", fontsize=5.5,
                        color=MUTED, path_effects=HALO, zorder=7)

    labels = rescue_labels(log, log_n)
    for s in site.survivors:
        color, marker, _ = CLASS_STYLE[s.priority]
        ax.plot(*s.pos, marker=marker, ms=5, color=color, mec=SURFACE, mew=0.8, zorder=6)
        ring = INK if s.id in log.rescue_times else "#b3261e"   # red ring = not rescued
        ax.plot(*s.pos, marker="o", ms=9, mfc="none", mec=ring, mew=0.9, zorder=6)
        if s.id in labels:
            ax.annotate(labels[s.id], s.pos, xytext=(5, 4), textcoords="offset points",
                        fontsize=5.5, color=INK, path_effects=HALO, zorder=8)
        if s.id not in det_ids:
            tag_xy = case["missed_xy"]
            if tag_xy is None:
                ax.annotate("missed by drone", s.pos, xytext=(-4, -11), textcoords="offset points",
                            fontsize=5.5, color="#b3261e", path_effects=HALO, zorder=8)
            else:
                ax.annotate("missed by drone", s.pos, xytext=tag_xy, textcoords="data",
                            fontsize=5.5, color="#b3261e", path_effects=HALO, zorder=8,
                            arrowprops=dict(arrowstyle="-", color="#b3261e", lw=0.5))

    fid = case["footprint_for"]
    if fid is not None and fid in drops:
        t_drop, p_drop = drops[fid]
        ax.add_patch(Circle(p_drop, 5.0, fill=False, edgecolor="#b3261e", lw=0.8, zorder=7))
        ax.annotate(f"track dropped at {t_drop:.0f} s\n(footprint 5 m)", p_drop, xytext=(-14, 16),
                    textcoords="offset points", fontsize=5.5, color="#b3261e",
                    path_effects=HALO, zorder=8, ha="center")

    ax.plot(*site.start, marker="^", ms=5, color=INK, mec=SURFACE, mew=0.6, zorder=8)
    sm = summarize(site, log, T)
    n = len(site.survivors)
    ax.set_title(f"{case['tag']}  {case['title']}", fontsize=7, color=INK, loc="left", pad=12)
    ax.text(0, 1.012, f"site {seed}: {len(log.rescue_times)}/{n} rescued, score {sm['norm_score']:.2f}",
            transform=ax.transAxes, fontsize=5.8, color=MUTED, va="bottom")
    xmin, ymin, xmax, ymax = site.bounds
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.tick_params(colors=MUTED, labelsize=5.5, length=2, pad=1.5)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
        sp.set_linewidth(0.5)
    ax.set_facecolor(SURFACE)


def legend_handles():
    h = [Line2D([], [], ls="", marker=m, ms=5, color=c, mec=SURFACE, label=name)
         for c, m, name in CLASS_STYLE.values()]
    h += [Line2D([], [], ls="--", marker="x", ms=4, color=MUTED, lw=0.6,
                 label="drone detection, 2$\\sigma$ circle"),
          Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=INK, label="rescued (time, s)"),
          Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec="#b3261e", label="not rescued"),
          Line2D([], [], color=INK, lw=0.9, label="priority path"),
          Line2D([], [], color=MUTED, lw=0.7, ls=(0, (3, 2)), label="nearest-first path"),
          Patch(facecolor="#e6e5e1", label="rubble"),
          Patch(facecolor="#c3c2b7", edgecolor=MUTED, hatch="///", label="hazard")]
    return h


def render(out):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), facecolor=SURFACE)
    for ax, case in zip(axes.ravel(), CASES):
        draw_panel(ax, case)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.915, bottom=0.115, wspace=0.12, hspace=0.36)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, frameon=False, fontsize=6,
               labelcolor=INK, columnspacing=1.2, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return out


def _no_fp(site, dets):
    return [d for d in dets if d.source_id is not None]


def _true_labels(site, dets):
    return [dataclasses.replace(d, priority=site.survivors[d.source_id].priority)
            if d.source_id is not None else d for d in dets]


def print_stats():
    for case in CASES:
        seed = case["seed"]
        for method in ("priority", "nearest"):
            site, dets, log, drops = run_case(seed, method)
            sm = summarize(site, log, T)
            order = sorted((round(t), i) for i, t in log.rescue_times.items())
            print(f"site {seed:3d} {method:8s} rescued {len(log.rescue_times)}/{len(site.survivors)} "
                  f"score {sm['norm_score']:.3f} ttc {sm['time_to_critical']:.0f} order(t,id) {order}")
    for name, seed, tweak in [("no false positives", 48, _no_fp), ("true class labels", 155, _true_labels)]:
        site, dets, log, _ = run_case(seed, "priority", tweak)
        sm = summarize(site, log, T)
        print(f"counterfactual site {seed} {name}: rescued {len(log.rescue_times)}/{len(site.survivors)} "
              f"score {sm['norm_score']:.3f} ttc {sm['time_to_critical']:.0f}")
    for sigma in (1, 2, 3, 6, 10, 25):
        cfg = dataclasses.replace(CFG, sigma_pos=float(sigma))
        site = sim.generate_site(139)
        log = sim.run_episode(site, lambda s, d: baselines.priority_policy(s, d, cfg, horizon=T),
                              cfg, 139, "priority", T=T)
        sm = summarize(site, log, T)
        print(f"site 139 sigma {sigma:2d}: rescued {len(log.rescue_times)}/{len(site.survivors)} "
              f"score {sm['norm_score']:.3f}")


def print_budget(n_sites=160, sigmas=(1, 3, 6, 10, 15)):
    """Why survivors are not rescued (priority policy, pooled over sites): not reported by the
    drone, reported but track dropped (r < cutoff), or other (still active at T / deleted)."""
    global CFG
    base = CFG
    try:
        for sigma in sigmas:
            CFG = dataclasses.replace(base, sigma_pos=float(sigma))
            n_surv = unresc = missed = dropped = 0
            for seed in range(n_sites):
                site, dets, log, drops = run_case(seed, "priority")
                reported = {d.source_id for d in dets if d.source_id is not None}
                n_surv += len(site.survivors)
                for s in site.survivors:
                    if s.id in log.rescue_times:
                        continue
                    unresc += 1
                    missed += s.id not in reported
                    dropped += s.id in reported and s.id in drops
            other = unresc - missed - dropped
            print(f"sigma {sigma:2d}: unrescued {unresc}/{n_surv}  not reported {missed / unresc:.0%}  "
                  f"dropped {dropped / unresc:.0%}  other {other / unresc:.0%}")
    finally:
        CFG = base


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="figures/path_planning/cases_grid.png")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--budget", action="store_true",
                        help="print the unrescued-survivor breakdown over 160 sites per sigma")
    args = parser.parse_args()
    print(render(args.out))
    if args.stats:
        print_stats()
    if args.budget:
        print_budget()


if __name__ == "__main__":
    main()
