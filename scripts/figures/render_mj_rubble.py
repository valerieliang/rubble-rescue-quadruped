"""The MuJoCo Go2 walking a site with physical rubble -> figures/mujoco/go2_rubble_site_<seed>.png

Usage (repo root, MUJOCO_GL=egl on a headless machine):
    PYTHONPATH=. python scripts/render_mj_rubble.py [seed]        (default seed 74)

Runs the full stack on the rubble-heightfield Go2 with the rough-terrain policy (sim/go2/policy_rough.onnx)
and draws three panels: the site from above with the 1 Hz trail (the heightfield is the real ground; the
rubble and hazard regions are drawn as translucent boxes over it), the robot on the roughest ground it
crossed, and the closest approach to a survivor. Overlays have no physics.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import render_mj as R                      # noqa: E402  (reuses its episode runner and drawing helpers)
from rescue.ctrl_behavior import make_site  # noqa: E402

ROUGH = os.path.abspath("sim/go2/policy_rough.onnx")
G = R.G


class RubblePlant(R.RecordingPlant):
    def __init__(self, *a, **k):
        super().__init__(*a, rubble=True, policy=ROUGH, **k)


def overlay(site, spos, trail=(), rings=False, dot=0.25, srad=0.9):
    out = []
    for r in site.regions:                 # translucent so the heightfield shows through
        c = ((r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2)
        col = (0.90, 0.20, 0.20, 0.22) if r.kind == "hazard" else (0.80, 0.55, 0.20, 0.20)
        out.append((G.mjGEOM_BOX, [(r.xmax - r.xmin) / 2, (r.ymax - r.ymin) / 2, 0.004], [c[0], c[1], 0.14], col))
    for s in site.survivors:
        out += R.survivor(spos[s.id], R.PRIORITY_RGBA[s.priority], [(1.0, R.RING_CLEAR)] if rings else [], srad)
    for x, y in trail:
        out.append((G.mjGEOM_SPHERE, [dot, 0, 0], [x, y, 0.15 + dot], R.TRAIL))
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 74
    os.makedirs(R.OUT, exist_ok=True)
    R.RecordingPlant = RubblePlant         # episode() builds its plant by this name
    ep = R.episode((seed, "full", "mujoco", False, False))
    rec, site = ep["rec"], make_site(seed)
    xs = np.array([x[0][:2] for x in rec])

    plant = RubblePlant()
    plant.set_site(site)
    r = R.make_renderer(plant)
    r.model.geom("floor").rgba[:] = (0.62, 0.55, 0.45, 1)          # sand-coloured ground so the relief reads

    q, spos, _ = rec[-1]
    R.set_state(plant, q)
    over = R.shot(r, plant, R.topdown_camera(site), overlay(site, spos, xs[::10]))

    h = np.array([plant.ground_height(p) for p in xs])             # roughest ground the robot stood on
    k_rough = int(np.argmax(np.convolve(h, np.ones(21) / 21, mode="same")))
    q, spos, _ = rec[k_rough]
    R.set_state(plant, q)
    close = R.shot(r, plant, R.camera([q[0], q[1], 0.1], 2.4, 135, -28), [])

    k, dk = R.closest_step(rec)
    q, spos, _ = rec[k]
    R.set_state(plant, q)
    near = R.shot(r, plant, R.pair_camera(q[:2], R.nearest(q, spos)),
                  overlay(site, spos, xs[max(0, k - 200):k + 1:10], rings=True, dot=0.06, srad=0.3))

    fell = "fell at %.0f s" % ep["fell"] if ep["fell"] is not None else "no fall"
    R.save_panels([over, close, near],
                  [f"Site {seed} on rubble: {ep['rescued']}/{ep['n']} rescued, {fell} (survivors enlarged)",
                   f"On rubble, t = {R.DT * (k_rough + 1):.0f} s (ground {h[k_rough] * 100:.0f} cm here)",
                   f"Closest approach: {dk:.2f} m (1 m ring = required clearance)"],
                  f"{R.OUT}/go2_rubble_site_{seed:03d}.png", size=5.0)
    print(f"closest approach {dk:.2f} m, violation steps {ep['viol_steps']}, rescued {ep['rescued']}/{ep['n']}, {fell}")


if __name__ == "__main__":
    main()
