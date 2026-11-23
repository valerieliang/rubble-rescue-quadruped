"""Screenshots of the MuJoCo Go2 simulation for the README and the writeup -> figures/mujoco/.

Usage (repo root):  PYTHONPATH=. python scripts/render_mj.py

  go2_stand.png        the robot standing (training model, flat plane)
  go2_walk.png         five frames of one gait cycle at 1.0 m/s
  go2_cbf_headon.png   head-on stress test: walked at a survivor at 1 m/s, the CBF filter stops it
                       (rings: 1.0 m required clearance, 1.25 m the filter's bound with the pad on)
  go2_site_074.png     site 74 (the site of the unicycle rollout figure), full stack on the MuJoCo Go2:
                       overview at the end of the episode, and a close-up at the closest approach

Site overlays (rubble, hazards, survivors, clearance rings, the robot's 1 Hz trail) are drawn into the
rendered scene only; they have no physics. The ground is a flat plane; rubble is a speed cap in the stack.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from rescue import ctrl_sim
from rescue.ctrl_behavior import make_site
from rescue.drone_prior import PriorConfig
from rescue.interfaces import Priority
from rescue.mj_plant import Go2MujocoPlant
from rescue.safety import CBFFilter, SafetyParams
from rescue.stack import VARIANTS

OUT = "figures/mujoco"
W, H = 1280, 720
G = mujoco.mjtGeom
PRIORITY_RGBA = {Priority.CRITICAL: (0.85, 0.15, 0.15, 1), Priority.URGENT: (0.95, 0.6, 0.1, 1),
                 Priority.STABLE: (0.2, 0.65, 0.3, 1)}
RUBBLE = (0.72, 0.58, 0.40, 1)
HAZARD = (0.90, 0.20, 0.20, 1)
TRAIL = (0.15, 0.45, 0.90, 0.9)
RING_CLEAR = (0.85, 0.15, 0.15, 0.30)
RING_BOUND = (0.95, 0.60, 0.10, 0.25)


class RecordingPlant(Go2MujocoPlant):
    """Records the robot state and the current survivor positions after every control step."""

    def __init__(self, *a, **k):
        self.rec, self.spos = [], {}
        super().__init__(*a, **k)

    def step(self, v_cmd, omega_cmd, dt=0.1):
        out = super().step(v_cmd, omega_cmd, dt)
        self.rec.append((self.data.qpos.copy(), {s: p.copy() for s, p in self.spos.items()}))
        return out


def make_renderer(plant, w=W, h=H):
    m = plant.model
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    m.vis.map.zfar = 400.0 / m.stat.extent          # far clip is zfar * extent; the robot alone is < 1 m
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.6
    m.geom("floor").rgba[:] = (0.83, 0.83, 0.81, 1)
    return mujoco.Renderer(m, h, w, max_geom=20000)


def camera(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    return c


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


def survivor(p, rgba, rings=(), radius=0.3):
    out = [(G.mjGEOM_CYLINDER, [radius, 0.2, 0], [p[0], p[1], 0.2], rgba)]
    for rad, col in rings:
        out.append((G.mjGEOM_CYLINDER, [rad, 0.003, 0], [p[0], p[1], 0.012], col))
    return out


def site_geoms(site, spos, trail=(), rings=False, dot=0.12, srad=0.3):
    out = []
    for r in site.regions:
        c = ((r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2)
        hz = r.kind == "hazard"
        out.append((G.mjGEOM_BOX, [(r.xmax - r.xmin) / 2, (r.ymax - r.ymin) / 2, 0.004],
                    [c[0], c[1], 0.03 if hz else 0.01], HAZARD if hz else RUBBLE))   # hazards on top
    for s in site.survivors:
        out += survivor(spos[s.id], PRIORITY_RGBA[s.priority], [(1.0, RING_CLEAR)] if rings else [], srad)
    for x, y in trail:
        out.append((G.mjGEOM_SPHERE, [dot, 0, 0], [x, y, 0.05 + dot], TRAIL))
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


def render_stand_and_walk(plant):
    r = make_renderer(plant)
    plant.reset()
    p = plant.data.qpos[:3]
    img = shot(r, plant, camera([p[0], p[1], 0.2], 1.5, 135, -25))
    plt.imsave(f"{OUT}/go2_stand.png", img)
    print("wrote", f"{OUT}/go2_stand.png")

    rw = make_renderer(plant, 640, 480)
    plant.reset()
    for _ in range(20):
        plant.step(1.0, 0.0, 0.1)                  # 2 s to reach a steady gait
    frames, titles = [], []
    for k in range(5):                             # one 0.6 s gait cycle
        p = plant.data.qpos[:3]
        frames.append(shot(rw, plant, camera([p[0], p[1], 0.18], 1.4, 90, -24)))
        titles.append(f"t = {0.12 * k:.2f} s")
        plant.step(1.0, 0.0, 0.12)
    save_panels(frames, titles, f"{OUT}/go2_walk.png", size=3.2)


def render_headon(plant):
    r = make_renderer(plant)
    filt = CBFFilter(SafetyParams(d_clear=1.0, eps_track=0.25, w_max=1.0))
    target = np.zeros(2)
    plant.reset((-6.0, 0.0), 0.0)
    pos, heading = plant.pose()[:2]
    marks = {0.0: None, 2.5: None, 5.0: None, 15.0: None}
    extras = survivor(target, PRIORITY_RGBA[Priority.CRITICAL], [(1.0, RING_CLEAR), (1.25, RING_BOUND)])
    cam = camera([-2.0, 0.0, 0.2], 6.3, 105, -38)
    dmin = np.inf
    for i in range(151):
        t = round(i * 0.1, 1)
        if t in marks:
            marks[t] = (shot(r, plant, cam, extras), float(np.linalg.norm(pos - target)))
        d = target - pos
        err = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
        u, _ = filt.filter(pos, heading, np.array([1.0, np.clip(2 * err, -1, 1)]), 1.0, [], [(target, np.eye(2) * 1e-6)])
        pos, heading, _ = plant.step(float(u[0]), float(u[1]), 0.1)
        dmin = min(dmin, float(np.linalg.norm(pos - target)))
    save_panels([m[0] for m in marks.values()],
                [f"t = {t:.1f} s, distance {m[1]:.2f} m" for t, m in marks.items()],
                f"{OUT}/go2_cbf_headon.png", cols=2, size=5.0)
    print(f"   head-on: closest approach {dmin:.2f} m (bound 1.25 m, required 1.0 m), fallen={plant.fallen}")


def render_site(seed=74):
    site = make_site(seed)
    plant = RecordingPlant()
    real = ctrl_sim.step_survivors

    def step_survivors(site_, spos, *a, **k):
        plant.spos = spos
        return real(site_, spos, *a, **k)

    ctrl_sim.step_survivors = step_survivors
    try:
        log = ctrl_sim.run_episode_ctrl(site, PriorConfig(), VARIANTS["full"], seed, "full",
                                        sigma_meas=0.5, plant=plant)
    finally:
        ctrl_sim.step_survivors = real
    rescued = len(log.rescue_times)
    print(f"   site {seed}: rescued {rescued}/{len(site.survivors)}, min clearance {log.min_clearance:.2f} m, "
          f"clearance-violation steps {log.clear_viol_steps}, fell={log.fell_at is not None}")

    r = make_renderer(plant)
    xs = np.array([q[:2] for q, _ in plant.rec])
    trail = xs[::10]

    q, spos = plant.rec[-1]
    plant.data.qpos[:] = q
    mujoco.mj_forward(plant.model, plant.data)
    xmin, ymin, xmax, ymax = site.bounds
    span = max(xmax - xmin, ymax - ymin)
    over = shot(r, plant, camera([(xmin + xmax) / 2, (ymin + ymax) / 2, 0], 1.45 * span, 90, -89),   # top-down
                site_geoms(site, spos, trail, dot=0.25, srad=0.9))

    dist = [min(np.linalg.norm(q_[:2] - p) for p in sp.values()) for q_, sp in plant.rec]
    k = int(np.argmin(dist))
    q, spos = plant.rec[k]
    plant.data.qpos[:] = q
    mujoco.mj_forward(plant.model, plant.data)
    sv = min(spos.values(), key=lambda p_: np.linalg.norm(q[:2] - p_))
    mid = (q[:2] + sv) / 2                          # frame the robot and its nearest survivor together
    ax_ = np.degrees(np.arctan2(sv[1] - q[1], sv[0] - q[0]))
    near = shot(r, plant, camera([mid[0], mid[1], 0.1], max(5.0, 2.2 * dist[k]), ax_ + 90, -45),
                site_geoms(site, spos, xs[max(0, k - 200):k + 1:10], rings=True, dot=0.06))
    save_panels([over, near],
                [f"Site {seed}, end of episode: {rescued}/{len(site.survivors)} rescued, trail at 1 Hz (survivors enlarged)",
                 f"Closest approach, t = {0.1 * (k + 1):.1f} s: {dist[k]:.2f} m (1 m ring = required clearance)"],
                f"{OUT}/go2_site_{seed:03d}.png", size=6.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    plant = Go2MujocoPlant()
    render_stand_and_walk(plant)
    render_headon(plant)
    render_site()


if __name__ == "__main__":
    main()
