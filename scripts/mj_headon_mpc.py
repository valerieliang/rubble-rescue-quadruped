"""Head-on test: CBF filter vs a disc-only MPC, on the unicycle and on the MuJoCo Go2 (flat).

Usage (repo root):  PYTHONPATH=. python scripts/mj_headon_mpc.py [n_seeds] [workers]      -> runs/mj_headon_mpc.csv

Same set-up as mj_cbf_stress.py (survivor known exactly at the origin, robot starts 6 m away facing it, plus a
heading offset; both plants limited to the policy's yaw rate). Two controllers:
  filter    the stress-test controller: constant-speed nominal aimed at the survivor, CBF filter on top.
  mpc_disc  ApproachMPC (the stack's planner) with the survivor as a soft keep-out disc of the SAME radius the filter
            keeps (d_clear + pad + cp_offset), and no CBF-QP. Its goal is the point 6 m beyond the survivor, so the
            straight path runs through the survivor and only the disc cost can steer the robot around it.
A violation is the robot centre inside 1 m of the survivor; "broke bound" is closer than d_clear + pad.
"""
import csv
import sys
from multiprocessing import Pool

import numpy as np

from rescue.ctrl_mpc import ApproachMPC
from rescue.safety import CBFFilter, SafetyParams
from rescue.sim import Go2Params, step_unicycle

D_CLEAR = 1.0
DT = 0.1
T = 15.0
START_DIST = 6.0
SPEEDS = (0.5, 1.0)
HEADING_OFFSETS = np.radians([0, 10, 20, 30, 45, -20, -45, 60])
PADS = (0.0, 0.25)
W_MAX = 1.0
SURVIVOR = np.zeros(2)
S_EXACT = np.eye(2) * 1e-6
CONTROLLERS = ("filter", "mpc_disc")
_PLANT = {}


def work(job):
    name, ctrl, pad, v_nom, offset, seed = job
    plant = None
    if name == "mujoco":
        if "p" not in _PLANT:
            from rescue.mj_plant import Go2MujocoPlant
            _PLANT["p"] = Go2MujocoPlant()
        plant = _PLANT["p"]
    filt = CBFFilter(SafetyParams(d_clear=D_CLEAR, eps_track=pad, w_max=W_MAX))
    mpc = ApproachMPC(w_max=W_MAX)
    rng = np.random.default_rng(seed)
    ang = rng.uniform(-np.pi, np.pi)
    pos = SURVIVOR + START_DIST * np.array([np.cos(ang), np.sin(ang)])
    heading = float(ang + np.pi + offset)
    goal = SURVIVOR - START_DIST * np.array([np.cos(ang), np.sin(ang)])      # beyond the survivor
    radius = filt.region_radius(1e-3)                                         # the keep-out radius the filter holds
    v, fell, dmin = 0.0, False, np.inf
    if plant is not None:
        plant.reset(pos, heading)
    for _ in range(int(round(T / DT))):
        if ctrl == "filter":
            d = SURVIVOR - pos
            err = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
            u, _ = filt.filter(pos, heading, np.array([v_nom, float(np.clip(2.0 * err, -W_MAX, W_MAX))]), v_nom, [],
                               [(SURVIVOR, S_EXACT)])
        else:
            u = np.array(mpc.solve(pos, heading, v_nom, goal, (), [(SURVIVOR, radius)]))
        if plant is None:
            pos, heading, v = step_unicycle(pos, heading, v, float(u[0]), float(u[1]), DT, Go2Params(),
                                            (-1e3, -1e3, 1e3, 1e3))
        else:
            pos, heading, v = plant.step(float(u[0]), float(u[1]), DT)
            if plant.fallen:
                fell = True
                break
        dmin = min(dmin, float(np.linalg.norm(pos - SURVIVOR)))
    return dict(plant=name, ctrl=ctrl, pad=pad, speed=v_nom, offset_deg=round(float(np.degrees(offset)), 1), seed=seed,
                dmin=dmin, fell=int(fell))


def summarize(rows):
    print(f"\n{'plant':9s} {'ctrl':9s} {'pad':>5s} {'speed':>5s} {'runs':>5s} {'closest':>8s} {'mean':>6s} "
          f"{'broke bound':>11s} {'inside 1 m':>10s} {'falls':>5s}")
    for name in ("unicycle", "mujoco"):
        for pad in PADS:
            for v in SPEEDS:
                for c in CONTROLLERS:
                    r = [x for x in rows if (x["plant"], x["ctrl"], x["pad"], x["speed"]) == (name, c, pad, v)]
                    if not r:
                        continue
                    d = np.array([x["dmin"] for x in r])
                    print(f"{name:9s} {c:9s} {pad:5.2f} {v:5.1f} {len(r):5d} {d.min():8.3f} {d.mean():6.2f} "
                          f"{int((d < D_CLEAR + pad - 1e-3).sum()):11d} {int((d < D_CLEAR).sum()):10d} "
                          f"{sum(x['fell'] for x in r):5d}")


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    jobs = [(n, c, pad, v, off, 1000 * k + i) for n in ("unicycle", "mujoco") for c in CONTROLLERS for pad in PADS
            for v in SPEEDS for k in range(n_seeds) for i, off in enumerate(HEADING_OFFSETS)]
    print(len(jobs), "episodes ...", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(work, jobs, chunksize=4)
    with open("runs/mj_headon_mpc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
