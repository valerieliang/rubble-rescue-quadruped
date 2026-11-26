"""Check the MuJoCo Go2 plant, then measure the tracking-error pad against it.

Usage (repo root):  PYTHONPATH=. python scripts/mj_calibrate.py [--quick] [--n 400] [--noise]

1. stand:    zero command for 5 s; the robot must stay up (this also validates the joint order and gains).
2. walk:     constant forward command; achieved speed, sideways and heading drift.
3. turn:     in-place yaw command; achieved yaw rate and drift of the position.
4. pad:      the same measurement as ``safety.calibrate_tracking_error`` but against the MuJoCo Go2:
             where the ideal unicycle (position integrates the commanded velocity) and the plant end up
             after 1 s of random piecewise-constant commands. The 95% quantile is the pad ``eps_track``.
             The old number (rate-limited unicycle as the "plant") is printed for comparison.
"""
import argparse
import dataclasses
import time

import numpy as np

from rescue.mj_plant import Go2MujocoPlant
from rescue.safety import calibrate_tracking_error
from rescue.sim import Go2Params


def run(plant, v, w, seconds, dt=0.1):
    for _ in range(int(round(seconds / dt))):
        out = plant.step(v, w, dt)
    return out


def check_stand(plant):
    plant.reset()
    run(plant, 0.0, 0.0, 5.0)
    xy, yaw, _ = plant.pose()
    print(f"[stand] 5 s at zero command: fallen={plant.fallen}  height={plant.data.qpos[2]:.3f} m  "
          f"drift={np.linalg.norm(xy):.3f} m")


def check_walk(plant):
    for v in (0.3, 0.6, 1.0):
        plant.reset()
        run(plant, v, 0.0, 3.0)                       # reach steady gait
        p0, y0, _ = plant.pose()
        speeds = []
        for _ in range(30):
            _, _, vf = plant.step(v, 0.0, 0.1)
            speeds.append(vf)
        p1, y1, _ = plant.pose()
        d = p1 - p0
        fwd = np.array([np.cos(y0), np.sin(y0)])
        along = float(d @ fwd)
        lat = float(d @ np.array([-fwd[1], fwd[0]]))
        dyaw = float((y1 - y0 + np.pi) % (2 * np.pi) - np.pi)
        print(f"[walk {v:.1f} m/s] fallen={plant.fallen}  mean speed={np.mean(speeds):.2f}  "
              f"along={along:.2f} m / 3 s (ideal {v * 3:.2f})  sideways={lat:+.2f} m  yaw drift={np.degrees(dyaw):+.1f} deg")


def check_turn(plant):
    for w in (0.5, 1.0, -1.0):
        plant.reset()
        run(plant, 0.0, w, 2.0)
        p0, y0, _ = plant.pose()
        run(plant, 0.0, w, 3.0)
        p1, y1, _ = plant.pose()
        dyaw = (y1 - y0 + np.pi) % (2 * np.pi) - np.pi
        print(f"[turn {w:+.1f} rad/s] fallen={plant.fallen}  achieved={dyaw / 3.0:+.2f} rad/s  "
              f"position drift={np.linalg.norm(p1 - p0):.2f} m / 3 s")


def mj_pad(plant, n, horizon=1.0, quantile=0.95, seed=0, dt=0.1):
    rng = np.random.default_rng(seed)
    errs, falls = [], 0
    t0 = time.time()
    for i in range(n):
        v0, v_cmd, w_cmd = rng.uniform(0, 1.0), rng.uniform(0, 1.0), rng.uniform(-plant.w_max, plant.w_max)
        plant.reset(settle=0.5)
        run(plant, v0, 0.0, 2.0)                      # reach the initial speed
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
        if (i + 1) % 50 == 0:
            print(f"   {i + 1}/{n} samples, {time.time() - t0:.0f} s")
    return np.array(errs), falls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--quick", action="store_true", help="n = 40")
    ap.add_argument("--noise", action="store_true", help="apply the training observation noise")
    ap.add_argument("--policy", default=None)
    a = ap.parse_args()
    n = 40 if a.quick else a.n

    t0 = time.time()
    plant = Go2MujocoPlant(policy=a.policy, obs_noise=a.noise)
    print(f"plant built in {time.time() - t0:.1f} s; joints: {', '.join(plant.joints)}")
    check_stand(plant)
    if plant.fallen:
        print("The robot fell while standing: stop here and check the joint order / model before anything else.")
        return
    check_walk(plant)
    check_turn(plant)

    print(f"\npad measurement, {n} samples ...")
    errs, falls = mj_pad(plant, n)
    print(f"MuJoCo Go2 vs ideal unicycle after 1 s: n={len(errs)} (fell: {falls})  "
          f"median={np.median(errs):.3f}  95%={np.quantile(errs, 0.95):.3f}  99%={np.quantile(errs, 0.99):.3f}  max={errs.max():.3f} m")
    old = calibrate_tracking_error()
    old_clip = calibrate_tracking_error(dataclasses.replace(Go2Params(), max_turn_rate=plant.w_max))
    print(f"previous pad (rate-limited unicycle as the plant): {old:.3f} m  "
          f"(with the turn rate clipped to {plant.w_max} rad/s: {old_clip:.3f} m); StackConfig.eps_track = 0.25 m")


if __name__ == "__main__":
    main()
