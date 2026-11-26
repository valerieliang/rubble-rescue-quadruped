"""Sudden close-range threat: a survivor estimate appears ahead of a robot already walking at speed.

Usage (repo root):  PYTHONPATH=. python scripts/mujoco/mj_sudden_threat.py [n_seeds]

The head-on test (mj_cbf_stress.py) starts the robot 6 m from a survivor the filter has known from the start,
so the barrier is never violated at the first step. Here the robot walks toward a far goal at constant speed
with no survivor known; at t0 = 3 s a survivor estimate appears ``d_appear`` metres ahead (the true survivor is
there, the estimate has the robot's measurement noise sigma_meas), exactly what happens when the drone missed
someone and the onboard detector sees them late, or a track jumps close. Only the filter (the one in the
closed-loop stack, confidence 0.7, pad 0.25 m) can stop the robot; the nominal controller never slows down.

Two versions: "single" keeps the first noisy estimate for the whole run (worst case: one bad sighting), and
"fused" re-detects at 10 Hz (90% detection probability) and Kalman-fuses each detection into the estimate (the
region keeps the stack's 0.5 m sigma floor). The stack's belief does NOT fuse at 10 Hz (that is the optional avg_meas),
and it also delays a first sighting until the next 1 Hz plan tick; this stand-alone filter test has neither delay,
so it is a best case. The stack itself is tested in scripts/mujoco/mj_sudden_stack.py.

Reported per (plant, speed, filter, d_appear): runs closer than 1 m to the true survivor (violations), the
closest approach, and the filter's infeasible steps. Below some d_appear no filter can stop the robot in time
(stopping distance + the region around a noisy estimate), so the interesting number is where violations start.
Both plants use the policy's yaw-rate limit (1 rad/s). Writes runs/mj_sudden.csv.
"""
import csv
import sys

import numpy as np

from rescue.mj_plant import Go2MujocoPlant
from rescue.safety import CBFFilter, SafetyParams
from rescue.sim import Go2Params, step_unicycle

DT, T0, T_END = 0.1, 3.0, 11.0
D_CLEAR, SIGMA_MEAS, W_MAX = 1.0, 0.5, 1.0
SPEEDS = (0.5, 1.0)
D_APPEAR = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
LATERAL = (0.0, 0.7)
GOAL = np.array([200.0, 0.0])


def run(plant, v_nom, use_filter, d_appear, lateral, seed, fuse=False, conf_fresh=None, n_fresh=0, fuse_every=1):
    """conf_fresh / n_fresh: while fewer than n_fresh detections have been fused into the estimate, the filter uses
    confidence conf_fresh instead of 0.7 (None = off, the original behaviour). fuse_every: control steps between
    detections (1 = 10 Hz, 10 = the stack's 1 Hz belief update)."""
    rng = np.random.default_rng(seed)
    filt = CBFFilter(SafetyParams(conf_p=0.7, d_clear=D_CLEAR, eps_track=0.25, w_max=W_MAX))
    filt_fresh = CBFFilter(SafetyParams(conf_p=conf_fresh, d_clear=D_CLEAR, eps_track=0.25, w_max=W_MAX)) \
        if conf_fresh else filt
    n_det = 0
    pos, heading, v = np.zeros(2), 0.0, 0.0
    if plant is not None:
        plant.reset(pos, heading)
    truth = est = None
    dmin, infeasible, err0 = np.inf, 0, np.nan
    for i in range(int(round(T_END / DT))):
        t = i * DT
        if truth is None and t >= T0 - 1e-9:
            ahead = np.array([np.cos(heading), np.sin(heading)])
            truth = pos + d_appear * ahead + lateral * np.array([-ahead[1], ahead[0]])
            est = truth + rng.normal(0, SIGMA_MEAS, 2)
            err0 = float(np.linalg.norm(est - truth))
            sig = SIGMA_MEAS
            n_det = 1
        elif fuse and truth is not None and i % fuse_every == 0 and rng.random() < 0.9:
            k = sig**2 / (sig**2 + SIGMA_MEAS**2)
            est = est + k * (truth + rng.normal(0, SIGMA_MEAS, 2) - est)
            n_det += 1
            sig = max(np.sqrt((1 - k) * sig**2), 0.5 * SIGMA_MEAS)
        d = GOAL - pos
        err = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
        u = np.array([v_nom, float(np.clip(2.0 * err, -W_MAX, W_MAX))])
        if use_filter and truth is not None:
            u, info = (filt_fresh if n_det < n_fresh else filt).filter(pos, heading, u, v_nom, [], [(est, np.eye(2) * SIGMA_MEAS**2)])
            infeasible += not info["feasible"]
        if plant is None:
            pos, heading, v = step_unicycle(pos, heading, v, float(u[0]), float(u[1]), DT,
                                            Go2Params(max_turn_rate=W_MAX), (-1e3, -1e3, 1e3, 1e3))
        else:
            pos, heading, v = plant.step(float(u[0]), float(u[1]), DT)
            if plant.fallen:
                return dict(dmin=dmin, fell=1, infeasible=infeasible, est_error=err0)
        if truth is not None:
            dmin = min(dmin, float(np.linalg.norm(pos - truth)))
    return dict(dmin=dmin, fell=0, infeasible=infeasible, est_error=err0)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    mj = Go2MujocoPlant()
    rows = []
    for name, plant in (("unicycle", None), ("mujoco", mj)):
        for v_nom in SPEEDS:
            for use_filter in (True, False):
                for fuse in (False, True):
                    for d_app in D_APPEAR:
                        for lat in LATERAL:
                            for seed in range(n_seeds):
                                r = run(plant, v_nom, use_filter, d_app, lat, seed, fuse)
                                rows.append(dict(plant=name, speed=v_nom, filter=int(use_filter), fused=int(fuse),
                                                 d_appear=d_app, lateral=lat, seed=seed, **r))
        print(name, "done", flush=True)
    with open("runs/mj_sudden.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for fused in (0, 1):
        print(f"\n{'fused' if fused else 'single estimate'}: violations / runs (closest m) by d_appear (m)")
        print(f"{'plant':9s} {'speed':>5s} {'filter':>6s} | " + "  ".join(f"{d:>13.1f}" for d in D_APPEAR))
        for name in ("unicycle", "mujoco"):
            for v_nom in SPEEDS:
                for flt in (1, 0):
                    cells = []
                    for d_app in D_APPEAR:
                        r = [x for x in rows if x["plant"] == name and x["speed"] == v_nom and x["filter"] == flt
                             and x["d_appear"] == d_app and x["fused"] == fused]
                        viol = sum(x["dmin"] < D_CLEAR for x in r)
                        cells.append(f"{viol:2d}/{len(r):2d} ({min(x['dmin'] for x in r):4.2f})".rjust(13))
                    print(f"{name:9s} {v_nom:5.1f} {'on' if flt else 'off':>6s} | " + "  ".join(cells))
    bad = [x for x in rows if x["filter"] and x["dmin"] < D_CLEAR]
    print("\nfiltered runs closer than 1 m:", len(bad), "of", sum(x["filter"] for x in rows),
          "; their first-sighting position errors (m):", sorted({round(x["est_error"], 2) for x in bad}),
          "; median first-sighting error over all runs:", round(float(np.median([x["est_error"] for x in rows])), 2))
    print("falls:", sum(x["fell"] for x in rows))


if __name__ == "__main__":
    main()
