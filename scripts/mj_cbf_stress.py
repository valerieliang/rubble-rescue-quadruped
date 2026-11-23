"""Head-on stress test of the CBF safety filter on the unicycle and on the MuJoCo Go2.

Usage (repo root):  PYTHONPATH=. python scripts/mj_cbf_stress.py

A nominal controller drives the robot straight at a survivor whose position is known exactly, at constant
speed, and keeps steering toward it; the CBF filter (safety.CBFFilter, the one in the closed-loop stack) is
the only thing that stops it. The barrier sits on a control point 0.25 m ahead of the robot with margin
d_clear + eps_track + 0.25, so for the model the filter assumes, the robot centre stays at least
d_clear + eps_track from the survivor (1.0 m with the pad off, 1.25 m with it on), and head-on it stops
about 0.25 m beyond that.

Reported per (plant, pad, speed): the closest approach over all starting headings, how many runs broke the
filter's bound (closer than d_clear + pad), and how many got inside 1 m (a safety violation). Broken bounds
on the MuJoCo plant but not on the unicycle measure the legged robot's tracking error; a violation with the
pad off but not with it on means the pad is needed.
Both plants are limited to the policy's yaw-rate range (1 rad/s) so they get the same filter.
"""
import numpy as np

from rescue.mj_plant import Go2MujocoPlant
from rescue.safety import CBFFilter, SafetyParams
from rescue.sim import Go2Params, step_unicycle

D_CLEAR = 1.0
DT = 0.1
T = 15.0
START_DIST = 6.0
SPEEDS = (0.5, 1.0)
HEADING_OFFSETS = np.radians([0, 10, 20, 30, 45, -20])
PADS = (0.0, 0.25)
W_MAX = 1.0
SURVIVOR = np.zeros(2)
S_EXACT = np.eye(2) * 1e-6            # position known exactly: the region collapses to a point


def nominal(pos, heading, v_nom):
    """Walk straight at the survivor and keep steering toward it."""
    d = SURVIVOR - pos
    err = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
    return np.array([v_nom, float(np.clip(2.0 * err, -W_MAX, W_MAX))])


def run(plant, pad, v_nom, offset, seed):
    filt = CBFFilter(SafetyParams(d_clear=D_CLEAR, eps_track=pad, w_max=W_MAX))
    rng = np.random.default_rng(seed)
    ang = rng.uniform(-np.pi, np.pi)                       # direction the robot comes from
    pos = SURVIVOR + START_DIST * np.array([np.cos(ang), np.sin(ang)])
    heading = float(ang + np.pi + offset)                  # facing the survivor, plus an offset
    v = 0.0
    if plant is not None:
        plant.reset(pos, heading)
    dmin = np.inf
    for _ in range(int(round(T / DT))):
        u, _info = filt.filter(pos, heading, nominal(pos, heading, v_nom), v_nom, [], [(SURVIVOR, S_EXACT)])
        if plant is None:
            pos, heading, v = step_unicycle(pos, heading, v, float(u[0]), float(u[1]), DT, Go2Params(),
                                            (-1e3, -1e3, 1e3, 1e3))
        else:
            pos, heading, v = plant.step(float(u[0]), float(u[1]), DT)
            if plant.fallen:
                return dmin, True
        dmin = min(dmin, float(np.linalg.norm(pos - SURVIVOR)))
    return dmin, False


def main():
    mj = Go2MujocoPlant()
    print(f"{'plant':9s} {'pad':>5s} {'speed':>6s} {'runs':>5s} {'closest (m)':>12s} {'mean closest':>13s} "
          f"{'broke bound':>12s} {'inside 1 m':>11s} {'falls':>5s}")
    for name, plant in (("unicycle", None), ("mujoco", mj)):
        for pad in PADS:
            for v_nom in SPEEDS:
                res = [run(plant, pad, v_nom, off, seed=k) for k, off in enumerate(HEADING_OFFSETS)]
                d = np.array([r[0] for r in res])
                falls = sum(r[1] for r in res)
                print(f"{name:9s} {pad:5.2f} {v_nom:6.1f} {len(res):5d} {d.min():12.3f} {d.mean():13.3f} "
                      f"{int((d < D_CLEAR + pad - 1e-3).sum()):12d} {int((d < D_CLEAR).sum()):11d} {falls:5d}")


if __name__ == "__main__":
    main()
