"""All MuJoCo Go2 figures for the README and the writeup -> figures/mujoco/.

Usage (repo root):  PYTHONPATH=. python scripts/figures/render_mj.py [workers]       (default 4 workers)

Five 600 s Go2 episodes (sites 0, 6 and 74, plus site 6 with the old planner) run in parallel while the
quick figures are drawn. Episodes use the same arguments as rescue.ctrl_experiments.run_ctrl_job, so
they are the episodes in runs/mj_q3.csv.

Screenshots (MuJoCo renders)
  go2_stand.png           the robot standing
  go2_walk.png            five frames of one gait cycle at 1.0 m/s
  go2_cbf_headon.png      head-on stress test: walked at a survivor at 1 m/s, the CBF filter stops it
  go2_site_074.png        site 74, full stack: top-down overview with the 1 Hz trail, and the closest approach
  go2_rescue_074.png      site 74: the robot walking up to its first rescue (3.5 m rescue ring, 1 m clearance ring)
  go2_q3_site_000.png     site 0, the same detections without and with the safety filter, each at its closest approach
  go2_site_006.png        site 6 on the Go2, before and after the planner fix: the old planner (old router, confirmed
                          tracks never fade, 1.5 rad/s planned turn rate) stops for good after 37 s; the fixed one
                          rescues everyone. Overview of both paths, and the old planner's stop.
Charts (matplotlib)
  go2_tracking.png        commanded vs achieved forward speed and yaw rate, Go2 vs the ideal unicycle
  go2_headon_paths.png    top-down paths of the head-on stress test (pad on, 1 m/s), both plants
  go2_conf_clearance.png  closest approach in the smaller-region sweep (needs runs/mj_conf.csv)

Diagnosis: the old-planner site 6 episode also records the stack's internal state at every control step
(runs/mj_site006_diag.npz) and prints a summary of its longest stop.

Site overlays (rubble, hazards, survivors, rings, trails) are drawn into the rendered scene only; they have
no physics. The ground is a flat plane; rubble is a speed cap in the stack.
"""
import csv
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from rescue import ctrl_sim
from rescue.ctrl_behavior import BehaviorConfig, make_site
from rescue.ctrl_mpc import lookahead
from rescue.drone_prior import PriorConfig
from rescue.interfaces import REWARDS, Priority
from rescue.mj_plant import Go2MujocoPlant
from rescue.safety import CBFFilter, SafetyParams
from rescue.sim import Go2Params, step_unicycle
from rescue.stack import R_RESCUE, VARIANTS

OUT = "figures/mujoco"
W, H = 1280, 720
DT = 0.1
G = mujoco.mjtGeom
BIG = (-1e3, -1e3, 1e3, 1e3)

# scene overlay colours (rgba)
PRIORITY_RGBA = {Priority.CRITICAL: (0.85, 0.15, 0.15, 1), Priority.URGENT: (0.95, 0.6, 0.1, 1),
                 Priority.STABLE: (0.2, 0.65, 0.3, 1)}
RUBBLE = (0.72, 0.58, 0.40, 1)
HAZARD = (0.90, 0.20, 0.20, 1)
TRAIL = (0.15, 0.45, 0.90, 0.9)
TRAIL_UNI = (0.92, 0.41, 0.20, 0.95)
RING_CLEAR = (0.85, 0.15, 0.15, 0.30)
RING_BOUND = (0.95, 0.60, 0.10, 0.25)
RING_RESCUE = (0.10, 0.65, 0.45, 0.22)

# chart colours: categorical slots 1 and 2, text and grid inks
UNI, GO2 = "#2a78d6", "#eb6834"
INK, INK2, GRID, CRIT = "#0b0b0b", "#52514e", "#e4e3df", "#d03b3b"
CHART_RC = {"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
            "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
            "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False}


# ================================================================ episodes (worker processes)

class RecordingPlant(Go2MujocoPlant):
    """Records the robot state, the survivor positions and the command after every control step."""

    def __init__(self, *a, **k):
        self.rec, self.spos = [], {}
        super().__init__(*a, **k)

    def step(self, v_cmd, omega_cmd, dt=DT):
        out = super().step(v_cmd, omega_cmd, dt)
        self.rec.append((self.data.qpos.copy(), {s: p.copy() for s, p in self.spos.items()}, (v_cmd, omega_cmd)))
        return out


def diag_stack_class(base, out):
    """A RescueStack that logs its internal state at every control call into ``out``."""

    class DiagStack(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            solve = self.mpc.solve
            self._u_nom = (np.nan, np.nan)

            def logged_solve(*a_, **k_):
                u = solve(*a_, **k_)
                self._u_nom = tuple(float(x) for x in u)
                return u
            self.mpc.solve = logged_solve

        def control(self, t, pos, heading, v_cap, v=0.0):
            self._u_nom = (np.nan, np.nan)
            u = super().control(t, pos, heading, v_cap, v)
            pos = np.asarray(pos, float)
            tgt = self.target
            tracks = [tr for tr in self.search.belief.tracks if tr.r >= self.cfg.r_safe]
            others = [float(np.linalg.norm(pos - tr.mean) - self.boundary(tr)) for tr in tracks if tr is not tgt]
            ko_r = self.cbf.region_radius(self.cfg.sigma_floor)
            ko = [float(np.linalg.norm(pos - np.asarray(q, float)) - ko_r) for q in self.keepout]
            goal = self.goal(pos)
            wp = lookahead(self.path, pos)
            out.append(dict(
                wp_x=float(wp[0]), wp_y=float(wp[1]), path_len=float(len(self.path)),
                target_id=float(id(tgt) % 100003) if tgt is not None else np.nan,
                t=t, x=pos[0], y=pos[1], heading=heading, v=v, v_cap=v_cap, u_v=float(u[0]), u_w=float(u[1]),
                nom_v=self._u_nom[0], nom_w=self._u_nom[1],
                feasible=float(self.last_info.get("feasible", True)), modified=float(self.last_info.get("modified", False)),
                has_target=float(tgt is not None),
                target_confirmed=float(tgt is not None and bool(getattr(tgt, "confirmed", False))),
                target_x=float(tgt.mean[0]) if tgt is not None else np.nan,
                target_y=float(tgt.mean[1]) if tgt is not None else np.nan,
                target_sigma=float(tgt.sigma) if tgt is not None else np.nan,
                goal_x=float(goal[0]), goal_y=float(goal[1]), goal_dist=float(np.linalg.norm(goal - pos)),
                n_tracks=float(len(tracks)), n_keepout=float(len(self.keepout)),
                keepout_margin=min(ko) if ko else np.nan,          # < 0: inside a rescued survivor's barrier
                other_margin=min(others) if others else np.nan,    # < 0: inside another survivor's barrier
            ))
            return u

    return DiagStack


def episode(args):
    """One episode as run_ctrl_job runs it. 'mujoco' returns the recording; 'unicycle' only the 1 Hz path.
    ``old``: the stack before the site 6 fix (route through safety regions, confirmed tracks never fade,
    planning with the unicycle's 1.5 rad/s turn rate)."""
    seed, variant, plant_name, diagnose, old = args
    site = make_site(seed)
    plant = RecordingPlant() if plant_name == "mujoco" else None
    real_step, real_stack = ctrl_sim.step_survivors, ctrl_sim.RescueStack
    diag = []

    def step_survivors(site_, spos, *a, **k):
        if plant is not None:
            plant.spos = spos
        return real_step(site_, spos, *a, **k)

    ctrl_sim.step_survivors = step_survivors
    if diagnose:
        ctrl_sim.RescueStack = diag_stack_class(real_stack, diag)
    try:
        cfg, params = VARIANTS[variant], None
        if old:
            cfg, params = replace(cfg, route_around_regions=False, forget_stale_tracks=False), Go2Params()
        log = ctrl_sim.run_episode_ctrl(site, PriorConfig(), cfg, seed, variant, params=params,
                                        behavior=BehaviorConfig(), dwell=REWARDS["base"].dwell, plant=plant)
    finally:
        ctrl_sim.step_survivors, ctrl_sim.RescueStack = real_step, real_stack
    ep = dict(seed=seed, variant=variant, plant=plant_name, old=old, rescued=len(log.rescue_times), n=len(site.survivors),
              rescue_times=dict(log.rescue_times), min_clear=log.min_clearance, viol_steps=log.clear_viol_steps,
              infeasible=log.infeasible_steps, fell=log.fell_at, traj=np.array(log.trajectory),
              diag={k: np.array([d[k] for d in diag]) for k in diag[0]} if diag else None)
    if plant is not None:
        ep["rec"] = plant.rec
    print(f"   done: site {seed} {variant:6s} {plant_name:8s}{' old planner' if old else ''} rescued {ep['rescued']}/{ep['n']}, "
          f"min clearance {log.min_clearance:.2f} m, violation steps {log.clear_viol_steps}, "
          f"fell={log.fell_at is not None}", flush=True)
    return ep


# ================================================================ rendering helpers

def make_renderer(plant, w=W, h=H):
    m = plant.model
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    m.vis.map.zfar = 400.0 / m.stat.extent          # far clip is zfar * extent; the robot alone is < 1 m
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.6
    m.vis.headlight.specular[:] = 0.0
    m.geom("floor").rgba[:] = (0.83, 0.83, 0.81, 1)
    return mujoco.Renderer(m, h, w, max_geom=20000)


def camera(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    return c


def pair_camera(a, b, elev=-45, distance=None):
    """Looks side-on at the segment a-b so both ends are in frame."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mid = (a + b) / 2
    az = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) + 90
    return camera([mid[0], mid[1], 0.1], distance or max(5.0, 2.2 * np.linalg.norm(b - a)), az, elev)


def topdown_camera(site):
    xmin, ymin, xmax, ymax = site.bounds
    return camera([(xmin + xmax) / 2, (ymin + ymax) / 2, 0], 1.45 * max(xmax - xmin, ymax - ymin), 90, -89)


def add(scene, gtype, size, pos, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def shot(r, plant, cam, extras=()):
    r.update_scene(plant.data, camera=cam)
    for g in extras:
        add(r.scene, *g)
    return r.render().copy()


def set_state(plant, q):
    plant.data.qpos[:] = q
    mujoco.mj_forward(plant.model, plant.data)


def survivor(p, rgba, rings=(), radius=0.3):
    out = [(G.mjGEOM_CYLINDER, [radius, 0.2, 0], [p[0], p[1], 0.2], rgba)]
    for rad, col in rings:
        out.append((G.mjGEOM_CYLINDER, [rad, 0.003, 0], [p[0], p[1], 0.012], col))
    return out


def site_geoms(site, spos, trail=(), rings=False, dot=0.12, srad=0.3, trail_rgba=TRAIL):
    out = []
    for r in site.regions:
        c = ((r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2)
        hz = r.kind == "hazard"
        out.append((G.mjGEOM_BOX, [(r.xmax - r.xmin) / 2, (r.ymax - r.ymin) / 2, 0.004],
                    [c[0], c[1], 0.03 if hz else 0.01], HAZARD if hz else RUBBLE))   # hazards on top
    for s in site.survivors:
        out += survivor(spos[s.id], PRIORITY_RGBA[s.priority], [(1.0, RING_CLEAR)] if rings else [], srad)
    for x, y in trail:
        out.append((G.mjGEOM_SPHERE, [dot, 0, 0], [x, y, 0.05 + dot], trail_rgba))
    return out


def save_panels(images, titles, path, cols=None, size=4.2):
    cols = cols or len(images)
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * images[0].shape[0] / images[0].shape[1]))
    for ax, img, t in zip(np.ravel(axes), images, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    for ax in np.ravel(axes)[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def save_chart(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{OUT}/{name}")


def nearest(q, spos):
    return min(spos.values(), key=lambda p: np.linalg.norm(q[:2] - p))


def closest_step(rec):
    d = [min(np.linalg.norm(q[:2] - p) for p in sp.values()) for q, sp, _ in rec]
    k = int(np.argmin(d))
    return k, d[k]


def headon_run(plant, offset, pad=0.25, v_nom=1.0, steps=150, on_step=None):
    """Walk from (-6, 0) at a survivor at the origin; the CBF filter is the only thing that stops the robot.
    plant None = the ideal unicycle. Returns the path."""
    filt = CBFFilter(SafetyParams(d_clear=1.0, eps_track=pad, w_max=1.0))
    pos, hd, v = np.array([-6.0, 0.0]), float(offset), 0.0
    if plant is not None:
        plant.reset(pos, hd)
    path = [pos.copy()]
    for i in range(steps):
        if on_step:
            on_step(i, pos)
        err = (np.arctan2(-pos[1], -pos[0]) - hd + np.pi) % (2 * np.pi) - np.pi
        u, _ = filt.filter(pos, hd, np.array([v_nom, np.clip(2 * err, -1, 1)]), v_nom, [],
                           [(np.zeros(2), np.eye(2) * 1e-6)])
        if plant is None:
            pos, hd, v = step_unicycle(pos, hd, v, float(u[0]), float(u[1]), DT, Go2Params(), BIG)
        else:
            pos, hd, v = plant.step(float(u[0]), float(u[1]), DT)
        path.append(pos.copy())
    return np.array(path)


# ================================================================ quick figures (main process)

def render_stand_and_walk(plant):
    r = make_renderer(plant)
    plant.reset()
    p = plant.data.qpos[:3]
    plt.imsave(f"{OUT}/go2_stand.png", shot(r, plant, camera([p[0], p[1], 0.2], 1.5, 135, -25)))
    print("wrote", f"{OUT}/go2_stand.png")

    rw = make_renderer(plant, 640, 480)
    plant.reset()
    for _ in range(20):
        plant.step(1.0, 0.0, DT)                   # 2 s to reach a steady gait
    frames, titles = [], []
    for k in range(5):                             # one 0.6 s gait cycle
        p = plant.data.qpos[:3]
        frames.append(shot(rw, plant, camera([p[0], p[1], 0.18], 1.4, 90, -24)))
        titles.append(f"t = {0.12 * k:.2f} s")
        plant.step(1.0, 0.0, 0.12)
    save_panels(frames, titles, f"{OUT}/go2_walk.png", size=3.2)


def render_headon(plant):
    r = make_renderer(plant)
    marks = {0: None, 25: None, 50: None, 150: None}
    extras = survivor(np.zeros(2), PRIORITY_RGBA[Priority.CRITICAL], [(1.0, RING_CLEAR), (1.25, RING_BOUND)])
    cam = camera([-2.0, 0.0, 0.2], 6.3, 105, -38)

    def grab(i, pos):
        if i in marks:
            marks[i] = (shot(r, plant, cam, extras), float(np.linalg.norm(pos)))

    path = headon_run(plant, 0.0, steps=151, on_step=grab)
    save_panels([m[0] for m in marks.values()],
                [f"t = {DT * i:.1f} s, distance {m[1]:.2f} m" for i, m in marks.items()],
                f"{OUT}/go2_cbf_headon.png", cols=2, size=5.0)
    print(f"   head-on: closest approach {np.linalg.norm(path, axis=1).min():.2f} m "
          f"(bound 1.25 m, required 1.0 m), fallen={plant.fallen}")


def chart_tracking():
    """A command schedule at 10 Hz (the stack's rate) on both plants."""
    sched = [(3, 0.3, 0.0), (3, 0.6, 0.0), (3, 1.0, 0.0), (3, 1.0, 0.5), (3, 0.0, 1.0), (3, 0.5, -1.0)]
    cmds = [(v, w) for dur, v, w in sched for _ in range(int(round(dur / DT)))]
    plant = Go2MujocoPlant()
    plant.reset()
    go2, uni = [], []
    pos, hd, v = np.zeros(2), 0.0, 0.0
    h_prev = plant.pose()[1]
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    for vc, wc in cmds:
        _, h, vf = plant.step(vc, wc, DT)
        go2.append((vf, wrap(h - h_prev) / DT))
        h_prev, hd0 = h, hd
        pos, hd, v = step_unicycle(pos, hd, v, vc, wc, DT, Go2Params(), BIG)
        uni.append((v, wrap(hd - hd0) / DT))
    t = DT * np.arange(1, len(cmds) + 1)
    c, g, u = np.array(cmds), np.array(go2), np.array(uni)
    with plt.rc_context(CHART_RC):
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
        for ax, k, lab in ((axes[0], 0, "forward speed (m/s)"), (axes[1], 1, "yaw rate (rad/s)")):
            ax.step(t, c[:, k], where="pre", color=INK2, lw=1.2, ls="--", label="command")
            ax.plot(t, u[:, k], color=UNI, lw=2, label="ideal unicycle")
            ax.plot(t, g[:, k], color=GO2, lw=2, label="MuJoCo Go2 (learned policy)")
            ax.set_ylabel(lab)
            for s in np.cumsum([d for d, _, _ in sched])[:-1]:
                ax.axvline(s, color=GRID, lw=1, zorder=0)
        axes[1].set_xlabel("time (s); sampled at 10 Hz, so the Go2 speed shows the gait's ripple")
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9)
        fig.suptitle("Command tracking: learned Go2 policy vs the reduced-order plant", x=0.01, ha="left",
                     color=INK)
        fig.tight_layout()
        save_chart(fig, "go2_tracking.png")


def chart_headon_paths():
    offsets = np.radians([0, 10, 20, 30, 45, -20])
    mj = Go2MujocoPlant()
    with plt.rc_context(CHART_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
        for ax, name, plant, col in ((axes[0], "Ideal unicycle", None, UNI), (axes[1], "MuJoCo Go2", mj, GO2)):
            paths = [headon_run(plant, off) for off in offsets]
            for p in paths:
                ax.plot(p[:, 0], p[:, 1], color=col, lw=1.6)
                ax.plot(*p[-1], "o", color=col, ms=5)
            for rad, ls, lab in ((1.0, "-", "1.0 m required clearance"), (1.25, "--", "1.25 m filter bound (pad on)")):
                ax.add_patch(plt.Circle((0, 0), rad, fill=False, color=INK2, ls=ls, lw=1.2, label=lab))
            ax.plot(0, 0, "s", color=CRIT, ms=7, label="survivor")
            ax.set_aspect("equal")
            ax.set_xlim(-6.5, 2)
            ax.set_ylim(-2, 2)
            ax.set_xlabel("x (m)")
            dmin = min(np.linalg.norm(p, axis=1).min() for p in paths)
            ax.set_title(f"{name}: closest approach {dmin:.2f} m", loc="left", color=INK, fontsize=10)
        axes[0].set_ylabel("y (m)")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        fig.tight_layout()
        save_chart(fig, "go2_headon_paths.png")


def chart_conf():
    path = "runs/mj_conf.csv"
    if not os.path.exists(path):
        print("skip go2_conf_clearance.png:", path, "missing (run scripts/mujoco/mj_run_conf.py)")
        return
    rows = list(csv.DictReader(open(path)))
    cps = (0.5, 0.7, 0.9)
    w = 0.36
    with plt.rc_context(CHART_RC):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, pad in zip(axes, (0.0, 0.25)):
            for j, (plant, col, lab) in enumerate((("unicycle", UNI, "ideal unicycle"), ("mujoco", GO2, "MuJoCo Go2"))):
                ys = [min(float(r["min_clearance"]) for r in rows if r["plant"] == plant
                          and float(r["conf_p"]) == cp and float(r["eps_track"]) == pad) for cp in cps]
                xs = np.arange(len(cps)) + (j - 0.5) * (w + 0.02)
                ax.bar(xs, ys, w, color=col, label=lab)
                for x, y in zip(xs, ys):
                    ax.text(x, y + 0.04, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
            ax.axhline(1.0, color=CRIT, lw=1.2, ls="--", label="1 m required clearance")
            ax.set_xticks(range(len(cps)), [f"{cp:.1f}" for cp in cps])
            ax.set_xlabel("conf_p (size of the safety region)")
            ax.set_title(f"pad {pad:.2f} m", loc="left", color=INK, fontsize=10)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("closest approach over 20 sites (m)")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        fig.suptitle("Smaller safety regions at sigma_meas 1.0 m: no run gets inside 1 m", x=0.01, ha="left",
                     color=INK)
        fig.tight_layout()
        save_chart(fig, "go2_conf_clearance.png")


# ================================================================ episode figures

def render_site74(plant, r, ep):
    site, rec = make_site(74), ep["rec"]
    xs = np.array([x[0][:2] for x in rec])

    q, spos, _ = rec[-1]
    set_state(plant, q)
    over = shot(r, plant, topdown_camera(site), site_geoms(site, spos, xs[::10], dot=0.25, srad=0.9))
    k, dk = closest_step(rec)
    q, spos, _ = rec[k]
    set_state(plant, q)
    near = shot(r, plant, pair_camera(q[:2], nearest(q, spos)),
                site_geoms(site, spos, xs[max(0, k - 200):k + 1:10], rings=True, dot=0.06))
    save_panels([over, near],
                [f"Site 74, end of episode: {ep['rescued']}/{ep['n']} rescued, trail at 1 Hz (survivors enlarged)",
                 f"Closest approach, t = {DT * (k + 1):.1f} s: {dk:.2f} m (1 m ring = required clearance)"],
                f"{OUT}/go2_site_074.png", size=6.0)

    rt = ep["rescue_times"]
    sid = min(rt, key=rt.get)
    k_r = int(round(rt[sid] / DT)) - 1
    target = rec[k_r][1][sid]                      # spos keeps every survivor, rescued or not
    cam = pair_camera(rec[max(0, k_r - 60)][0][:2], target, -35, distance=11.0)
    imgs, titles = [], []
    for back in (60, 40, 20, 0):
        kk = max(0, k_r - back)
        q, spos, cmd = rec[kk]
        set_state(plant, q)
        ex = site_geoms(site, spos, rings=True)
        ex += [(G.mjGEOM_CYLINDER, [R_RESCUE, 0.002, 0], [target[0], target[1], 0.006], RING_RESCUE)]
        imgs.append(shot(r, plant, cam, ex))
        titles.append(f"t = {DT * (kk + 1):.1f} s, {np.linalg.norm(q[:2] - spos[sid]):.2f} m, "
                      f"command v = {cmd[0]:.2f} m/s" + ("  (rescued)" if back == 0 else ""))
    save_panels(imgs, titles, f"{OUT}/go2_rescue_074.png", cols=2, size=5.0)


def render_q3_site0(plant, r, eps):
    site = make_site(0)
    imgs, titles = [], []
    for var, lab in (("no_cbf", "No safety filter"), ("full", "Full system (CBF filter on)")):
        rec = eps[(0, var, "mujoco", False)]["rec"]
        k, dk = closest_step(rec)
        q, spos, _ = rec[k]
        set_state(plant, q)
        trail = np.array([x[0][:2] for x in rec[max(0, k - 200):k + 1:10]])
        imgs.append(shot(r, plant, pair_camera(q[:2], nearest(q, spos)),
                         site_geoms(site, spos, trail, rings=True, dot=0.06)))
        titles.append(f"{lab}: closest approach {dk:.2f} m at t = {DT * (k + 1):.0f} s")
    save_panels(imgs, titles, f"{OUT}/go2_q3_site_000.png", size=6.0)


def longest_stop(cmd, thresh=0.02):
    """(start, end) step indices of the longest run of near-zero commanded speed and turn rate."""
    still = (np.abs(cmd[:, 0]) < thresh) & (np.abs(cmd[:, 1]) < thresh)
    best, start = (0, 0), None
    for i, s in enumerate(np.r_[still, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            best = max(best, (start, i), key=lambda ab: ab[1] - ab[0])
            start = None
    return best


def diagnose_site6(ep, ep_new):
    """``ep``: the old-planner Go2 episode (with the stack log); ``ep_new``: the fixed planner."""
    rec, d = ep["rec"], ep["diag"]
    cmd = np.array([x[2] for x in rec])
    a, b = longest_stop(cmd)
    np.savez("runs/mj_site006_diag.npz", qpos=np.array([x[0] for x in rec]), cmd=cmd,
             qpos_fixed=np.array([x[0] for x in ep_new["rec"]]), stop=np.array([a, b]),
             **{f"diag_{k}": v for k, v in d.items()})
    print("\n   site 6 diagnosis, old planner on the Go2 (wrote runs/mj_site006_diag.npz)")
    print(f"   old planner rescues   {ep['rescue_times']}")
    print(f"   fixed planner rescues {ep_new['rescue_times']}")
    print(f"   longest stop: t = {DT * (a + 1):.1f} to {DT * b:.1f} s ({DT * (b - a):.0f} s)")
    if b > a:
        sl = slice(a, b)
        for k in ("nom_v", "nom_w", "u_v", "u_w", "v_cap", "feasible", "modified", "has_target",
                  "target_confirmed", "target_sigma", "goal_dist", "n_tracks", "n_keepout", "keepout_margin",
                  "other_margin"):
            x = d[k][sl]
            print(f"     {k:17s} first {x[0]:8.3f}  median {np.nanmedian(x):8.3f}  min {np.nanmin(x):8.3f}  "
                  f"max {np.nanmax(x):8.3f}")
        i = a
        print(f"     at the start: robot ({d['x'][i]:.2f}, {d['y'][i]:.2f}) heading {d['heading'][i]:.2f}, "
              f"target ({d['target_x'][i]:.2f}, {d['target_y'][i]:.2f}), goal ({d['goal_x'][i]:.2f}, "
              f"{d['goal_y'][i]:.2f})")
        lo = max(0, a - 50)
        print("     the 5 s before the stop:  t  x  y  nom_v  u_v  goal_dist  keepout_margin  other_margin")
        for j in range(lo, a + 1, 5):
            print(f"       {d['t'][j]:6.1f} {d['x'][j]:6.2f} {d['y'][j]:6.2f} {d['nom_v'][j]:6.2f} {d['u_v'][j]:6.2f} "
                  f"{d['goal_dist'][j]:7.2f} {d['keepout_margin'][j]:8.2f} {d['other_margin'][j]:8.2f}")
    return a, b


def render_site6(plant, r, ep, ep_new):
    """Before/after: ``ep`` is the old planner (it stops), ``ep_new`` the fixed one."""
    site, rec = make_site(6), ep["rec"]
    a, b = diagnose_site6(ep, ep_new)
    xs = np.array([x[0][:2] for x in rec])
    xs_new = np.array([x[0][:2] for x in ep_new["rec"]])

    q, spos, _ = ep_new["rec"][-1]
    set_state(plant, q)
    ex = site_geoms(site, spos, xs_new[::10], dot=0.25, srad=0.9)
    ex += [(G.mjGEOM_SPHERE, [0.25, 0, 0], [x, y, 0.3], TRAIL_UNI) for x, y in xs[:b + 1:10]]
    over = shot(r, plant, topdown_camera(site), ex)

    k = (a + b) // 2
    q, spos, _ = rec[k]
    set_state(plant, q)
    near = shot(r, plant, pair_camera(q[:2], nearest(q, spos)),
                site_geoms(site, spos, xs[max(0, a - 300):k + 1:10], rings=True, dot=0.06))
    save_panels([over, near],
                [f"Site 6: fixed planner (blue, {ep_new['rescued']}/{ep_new['n']}) vs old (orange, "
                 f"{ep['rescued']}/{ep['n']})",
                 f"Old planner stopped, t = {DT * (a + 1):.0f}-{DT * b:.0f} s, "
                 f"{np.linalg.norm(q[:2] - nearest(q, spos)):.1f} m from a rescued survivor"],
                f"{OUT}/go2_site_006.png", size=6.0)


# ================================================================ main

def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    os.makedirs(OUT, exist_ok=True)
    jobs = [(0, "no_cbf", "mujoco", False, False), (0, "full", "mujoco", False, False),
            (74, "full", "mujoco", False, False), (6, "full", "mujoco", False, False),
            (6, "full", "mujoco", True, True)]
    print(f"running {len(jobs)} episodes on {workers} workers in the background (several minutes) ...", flush=True)
    with Pool(workers) as pool:
        pending = pool.map_async(episode, jobs)
        plant = Go2MujocoPlant()
        render_stand_and_walk(plant)
        render_headon(plant)
        chart_tracking()
        chart_headon_paths()
        chart_conf()
        eps = {(e["seed"], e["variant"], e["plant"], e["old"]): e for e in pending.get()}

    r = make_renderer(plant)
    render_site74(plant, r, eps[(74, "full", "mujoco", False)])
    render_q3_site0(plant, r, eps)
    render_site6(plant, r, eps[(6, "full", "mujoco", True)], eps[(6, "full", "mujoco", False)])


if __name__ == "__main__":
    main()
