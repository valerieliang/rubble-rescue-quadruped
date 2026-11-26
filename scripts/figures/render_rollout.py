"""Closed-loop rollout on one site, with and without the CBF safety filter (report/talk figure).

Usage: python scripts/render_rollout.py [SEED] [--sigma-meas 0.5] [--out figures/closed_loop]

Both variants see the same site and the same drone detections (paired, as in rescue/ctrl_experiments.py).
Without SEED, the seed with the smallest unfiltered clearance among those where the unfiltered run
violates the 1 m clearance and the filtered one does not is picked from runs/q3_sigma.csv, so the figure
shows the filter working rather than a typical episode. Per-step robot and survivor positions are recorded
by wrapping the simulator's step functions from outside; nothing in rescue/ is modified.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render_site import CLASS_STYLE, HALO, INK, MUTED, SURFACE, draw_regions   # noqa: E402

from rescue import ctrl_sim                                                    # noqa: E402
from rescue.ctrl_behavior import make_site                                     # noqa: E402
from rescue.ctrl_experiments import load_rows                                  # noqa: E402
from rescue.drone_prior import PriorConfig                                     # noqa: E402
from rescue.stack import VARIANTS                                              # noqa: E402

D_CLEAR = ctrl_sim.D_CLEAR_EVAL
VIOL = "#d62728"
OK = "#1f6fb4"
LABELS = {"no_cbf": "No safety filter", "full": "With CBF safety filter"}


def pick_seed(sigma_meas, csv="runs/q3_sigma.csv"):
    by = {}
    for r in load_rows(csv):
        if r.get("sigma_meas") == sigma_meas:
            by.setdefault(r["seed"], {})[r["variant"]] = r
    cand = [(v["no_cbf"]["min_clearance"], int(s)) for s, v in by.items()
            if "no_cbf" in v and "full" in v and v["no_cbf"]["any_clear_viol"] and not v["full"]["any_clear_viol"]
            and v["no_cbf"]["rescue_rate"] == 1.0 and v["full"]["rescue_rate"] == 1.0]
    return min(cand)[1]


def rollout(site, variant, seed, sigma_meas):
    """Run one episode and return (log, robot [N,2], survivors [N,S,2], dist [N])."""
    robot, surv = [], []
    real_surv, real_step = ctrl_sim.step_survivors, ctrl_sim.step_unicycle

    def step_survivors(site_, spos, *a, **k):
        out = real_surv(site_, spos, *a, **k)
        surv.append(np.array([spos[s.id] for s in site_.survivors]))
        return out

    def step_unicycle(*a, **k):
        out = real_step(*a, **k)
        robot.append(np.array(out[0], float))
        return out

    ctrl_sim.step_survivors, ctrl_sim.step_unicycle = step_survivors, step_unicycle
    try:
        log = ctrl_sim.run_episode_ctrl(site, PriorConfig(), VARIANTS[variant], seed, variant,
                                        sigma_meas=sigma_meas)
    finally:
        ctrl_sim.step_survivors, ctrl_sim.step_unicycle = real_surv, real_step
    n = min(len(robot), len(surv))
    robot, surv = np.array(robot[:n]), np.array(surv[:n])
    dist = np.linalg.norm(surv - robot[:, None, :], axis=2).min(axis=1)
    assert abs(dist.min() - log.min_clearance) < 1e-6, (dist.min(), log.min_clearance)
    return log, robot, surv, dist


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
        sp.set_linewidth(0.6)


def draw_map(ax, site, variant, res):
    log, robot, surv, dist = res
    style(ax)
    draw_regions(ax, site)
    for k in range(surv.shape[1]):        # survivor tracks (mobile ones move)
        ax.plot(surv[:, k, 0], surv[:, k, 1], color=MUTED, lw=0.7, ls=":", zorder=3)
    viol = dist < D_CLEAR
    ax.plot(robot[:, 0], robot[:, 1], color=INK, lw=1.1, zorder=5, solid_joinstyle="round")
    if viol.any():                         # thick red where the robot was inside 1 m
        seg = robot.copy()
        seg[~viol] = np.nan
        ax.plot(seg[:, 0], seg[:, 1], color=VIOL, lw=3.2, zorder=6, solid_capstyle="round")
        i = int(dist.argmin())
        ax.plot(*robot[i], marker="o", ms=15, mfc="none", mec=VIOL, mew=1.6, zorder=9)
        ax.annotate("worst violation\n(zoom below)", robot[i], xytext=(-48, 30),
                    textcoords="offset points", fontsize=8, color=VIOL, ha="right", path_effects=HALO,
                    zorder=9, arrowprops=dict(arrowstyle="-", color=VIOL, lw=0.8))
    for k, s in enumerate(site.survivors):
        color, marker, _ = CLASS_STYLE[s.priority]
        ax.plot(*surv[-1, k], marker=marker, ms=8, color=color, mec=SURFACE, mew=1.2, zorder=8)
    ax.plot(*site.start, marker="^", ms=8, color=INK, mec=SURFACE, mew=1.0, zorder=8)
    xmin, ymin, xmax, ymax = site.bounds
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal", xlabel="x (m)")
    n_v = int(viol.sum())
    sub = (f"{n_v} steps inside {D_CLEAR:g} m, min {dist.min():.2f} m" if n_v
           else f"no violations, min {dist.min():.2f} m")
    ax.set_title(f"{LABELS[variant]}\n{len(log.rescue_times)} of {len(site.survivors)} rescued; {sub}",
                 fontsize=9.5, color=VIOL if n_v else OK, loc="left")


def draw_zoom(ax, site, sid, res, half=4.5, window=12.0):
    """Close-up on survivor ``sid`` around the robot's closest approach to it."""
    log, robot, surv, dist = res
    style(ax)
    d = np.linalg.norm(surv[:, sid] - robot, axis=1)
    i = int(d.argmin())
    c = surv[i, sid]
    lo, hi = max(0, i - int(window * 10)), min(len(robot), i + int(window * 10))
    ax.add_patch(Circle(c, D_CLEAR, facecolor=VIOL, alpha=0.12, edgecolor=VIOL, lw=1.4, zorder=2))
    ax.plot(robot[lo:hi, 0], robot[lo:hi, 1], color=INK, lw=1.4, zorder=5)
    inside = d[lo:hi] < D_CLEAR
    if inside.any():
        seg = robot[lo:hi].copy()
        seg[~inside] = np.nan
        ax.plot(seg[:, 0], seg[:, 1], color=VIOL, lw=3.4, zorder=6, solid_capstyle="round")
    ax.plot(surv[lo:hi, sid, 0], surv[lo:hi, sid, 1], color=MUTED, lw=0.9, ls=":", zorder=3)
    color, marker, _ = CLASS_STYLE[site.survivors[sid].priority]
    ax.plot(*c, marker=marker, ms=10, color=color, mec=SURFACE, mew=1.2, zorder=8)
    ax.plot(*robot[i], marker="o", ms=6, color=INK, mec=SURFACE, zorder=9)
    ax.set(xlim=(c[0] - half, c[0] + half), ylim=(c[1] - half, c[1] + half), aspect="equal", xlabel="x (m)")
    ax.set_title(f"Zoom on survivor {sid}, t = {i * 0.1:.0f} s: closest approach {d[i]:.2f} m", fontsize=8, color=MUTED, loc="left")
    ax.tick_params(labelsize=7)


def render(seed, sigma_meas, out):
    site = make_site(seed)
    res = {v: rollout(site, v, seed, sigma_meas) for v in ("no_cbf", "full")}
    fig = plt.figure(figsize=(10.4, 10.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[3.1, 2.3, 1.15], hspace=0.36, wspace=0.12)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, v in zip(axes, ("no_cbf", "full")):
        draw_map(ax, site, v, res[v])
    axes[0].set_ylabel("y (m)", color=MUTED)
    axes[1].tick_params(labelleft=False)

    d0 = res["no_cbf"][3]
    i0 = int(d0.argmin())
    sid = int(np.linalg.norm(res["no_cbf"][2][i0] - res["no_cbf"][1][i0], axis=1).argmin())
    zax = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for z, v in zip(zax, ("no_cbf", "full")):
        draw_zoom(z, site, sid, res[v])
    zax[0].set_ylabel("y (m)", color=MUTED)
    zax[1].tick_params(labelleft=False)

    ax = fig.add_subplot(gs[2, :])
    style(ax)
    for v, color in (("no_cbf", VIOL), ("full", OK)):
        d = res[v][3]
        ax.plot(np.arange(len(d)) * 0.1, d, color=color, lw=1.2, label=LABELS[v])
    ax.axhline(D_CLEAR, color=INK, lw=1.0, ls="--")
    ax.text(0.005, D_CLEAR, f"required clearance {D_CLEAR:g} m", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=8, color=INK)
    ax.set(xlabel="time (s)", ylabel="distance to nearest\nsurvivor (m)", ylim=(0, 8), xlim=(0, max(len(res[v][3]) for v in res) * 0.1))
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

    handles = [Line2D([], [], ls="", marker=m, ms=7, color=c, mec=SURFACE, label=f"survivor: {n}")
               for c, m, n in CLASS_STYLE.values()]
    handles += [Line2D([], [], color=INK, lw=1.1, label="robot path (maps, zooms)"),
                Line2D([], [], color=VIOL, lw=1.2, label="distance, no filter"),
                Line2D([], [], color=OK, lw=1.2, label="distance, with filter"),
                Patch(facecolor=VIOL, alpha=0.15, edgecolor=VIOL, label=f"{D_CLEAR:g} m clearance disc"),
                Line2D([], [], color=VIOL, lw=3.2, label=f"robot within {D_CLEAR:g} m of a survivor"),
                Line2D([], [], color=MUTED, lw=0.7, ls=":", label="survivor path"),
                Line2D([], [], ls="", marker="^", ms=7, color=INK, label="robot start")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(f"Site {seed}, closed loop, localization noise {sigma_meas:g} m: same site and drone "
                 "detections, with and without the safety filter", fontsize=10.5, color=INK, x=0.125,
                 ha="left", y=0.94)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"rollout_site_{seed:03d}.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(path.replace(".png", ".pdf"), facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    for v in res:
        print(v, "min clearance %.3f" % res[v][3].min(), "viol steps", res[v][0].clear_viol_steps,
              "rescued", len(res[v][0].rescue_times))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seed", nargs="?", type=int)
    ap.add_argument("--sigma-meas", type=float, default=0.5)
    ap.add_argument("--out", default="figures/closed_loop")
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else pick_seed(a.sigma_meas)
    print("seed", seed)
    print(render(seed, a.sigma_meas, a.out))
