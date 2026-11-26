"""Tracking and falls of the MuJoCo Go2 on physical rubble, and the tracking-error pad measured there.

Usage (repo root):  PYTHONPATH=. python scripts/mujoco/mj_rubble_calibrate.py [--policy path.onnx] [--n 200] [--label name]

Builds a 40 m x 40 m patch of each ground type (flat plane, rubble heightfield, hazard heightfield from
``mj_terrain``), starts the robot at random places inside it with random headings, and measures the same thing
as ``mj_calibrate.py``: where the robot is, against the planner's ideal unicycle, after 1 s of a random
constant command (the 95% quantile is the pad ``eps_track``). Also counts falls and the achieved speed.
Run it once per policy (flat-trained ``sim/go2/policy.onnx``, rough-trained one) to compare.
"""
import argparse
import time

import numpy as np

from rescue.interfaces import Region, Site
from rescue.mj_plant import Go2MujocoPlant

PATCH = 40.0


def patch_site(kind: str | None) -> Site:
    regions = () if kind is None else (Region(kind, 0.0, 0.0, PATCH, PATCH),)
    return Site(seed=1, bounds=(0.0, 0.0, PATCH, PATCH), regions=regions, survivors=(),
                start=np.array([-50.0, -50.0]))          # start far outside: no flat disc inside the patch


def measure(plant, n, seed=0, dt=0.1, horizon=1.0):
    rng = np.random.default_rng(seed)
    errs, ratios, falls = [], [], 0
    for _ in range(n):
        xy = rng.uniform(8.0, PATCH - 8.0, 2)
        yaw = rng.uniform(-np.pi, np.pi)
        v0, v_cmd, w_cmd = rng.uniform(0, 1.0), rng.uniform(0.2, 1.0), rng.uniform(-plant.w_max, plant.w_max)
        plant.reset(xy, yaw, settle=0.5)
        for _k in range(20):                          # 2 s run-up at the initial speed
            plant.step(v0, 0.0, dt)
        if plant.fallen:
            falls += 1
            continue
        p0, th, _ = plant.pose()
        ideal = p0.copy()
        for _k in range(int(round(horizon / dt))):
            pos, _, _ = plant.step(v_cmd, w_cmd, dt)
            th += w_cmd * dt
            ideal = ideal + v_cmd * dt * np.array([np.cos(th), np.sin(th)])
        if plant.fallen:
            falls += 1
            continue
        errs.append(float(np.linalg.norm(pos - ideal)))
        ratios.append(float(np.linalg.norm(pos - p0) / max(np.linalg.norm(ideal - p0), 1e-6)))
    return np.array(errs), np.array(ratios), falls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    print(f"policy: {a.policy or 'sim/go2/policy.onnx (flat-trained)'} {a.label}")
    print(f"{'ground':8s} {'n':>4s} {'falls':>6s} {'median':>7s} {'95%':>7s} {'99%':>7s} {'max':>7s} {'dist ratio':>11s}")
    for kind in (None, "rubble", "hazard"):
        plant = Go2MujocoPlant(policy=a.policy, rubble=kind is not None)
        plant.set_site(patch_site(kind))
        t0 = time.time()
        errs, ratios, falls = measure(plant, a.n)
        name = kind or "flat"
        if len(errs):
            print(f"{name:8s} {len(errs):4d} {falls:6d} {np.median(errs):7.3f} {np.quantile(errs, .95):7.3f} "
                  f"{np.quantile(errs, .99):7.3f} {errs.max():7.3f} {np.mean(ratios):11.2f}   ({time.time() - t0:.0f} s)")
        else:
            print(f"{name:8s} all {falls} fell")


if __name__ == "__main__":
    main()
