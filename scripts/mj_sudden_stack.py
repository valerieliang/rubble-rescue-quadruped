"""Sudden close-range threat through the real stack (ctrl_sim + the real belief), not the stand-alone filter.

Usage (repo root):  PYTHONPATH=. python scripts/mj_sudden_stack.py [n_seeds] [workers]

mj_sudden_threat.py hands the CBF filter the new estimate at t0 = 3.0 s, which is a plan-tick boundary, and then
re-detects every 10th step. The stack differs in three ways that could make that test optimistic:
  * a first sighting reaches the belief only at the next 1 Hz plan tick, so it is delayed by 0-1 s (the stand-alone
    test has zero delay);
  * the belief receives one detection per second and only if that single 10 Hz sample was a hit (p = 0.9);
  * the MPC, not a constant-speed nominal, drives, and it also reacts to the new track.
Here run_episode_ctrl runs the whole stack. Site: a survivor the drone found at 25 m on the +x axis (the robot's
goal, exact prior), and a second survivor the drone MISSED, on the robot's path (x = 10 m, lateral 0 or 0.7 m).
The onboard sensor radius is set to ``d_appear`` (Euclidean), so the missed survivor is first seen d_appear metres
away, as in the stand-alone test. Rescue radius is 0 so the robot never "rescues" (and so never keeps out of)
the survivor it is not supposed to know about. A patched ``perceive``/``plan`` records the distance at which the
survivor was first sensed and the distance when the belief first held it, so the latency is measured, not assumed.
Violation = robot centre within 1 m of the missed survivor at any control step (10 Hz, stricter than 1 Hz samples).
Variants: fast / fast99 (full with StackConfig.fast_detect: first onboard detection goes to the filter at 10 Hz; fast99 also conf_fresh 0.99), full stack (MPC + CBF filter), no_cbf (MPC alone) and fresh99 (full with StackConfig.conf_fresh=0.99,
n_fresh=5, to see what that option does when wired through the stack). Plants: unicycle and MuJoCo Go2 (flat).
Writes runs/mj_sudden_stack.csv.
"""
import csv
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

import rescue.ctrl_sim as cs
from rescue.ctrl_behavior import BehaviorConfig
from rescue.drone_prior import PriorConfig
from rescue.interfaces import Detection, Priority, ResponseState, Site, Survivor
from rescue.sim import Go2Params
from rescue.stack import RescueStack, StackConfig

SPEEDS = (0.5, 1.0)
D_APPEAR = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
LATERAL = (0.0, 0.7)
X_MISSED, X_GOAL = 10.0, 25.0
T_RUN = 60.0
PROBE: dict = {}
_PLANT = {}


def _detections(site, cfg, rng):
    """The drone found the goal survivor exactly and missed the other one."""
    return [Detection(site.survivors[0].pos.copy(), Priority.CRITICAL, 0)]


_orig_perceive, _orig_plan = RescueStack.perceive, RescueStack.plan


def _perceive(self, t, pos, detected):
    if any(d.source_id == 1 for d in detected) and "t_sense" not in PROBE:
        PROBE.update(t_sense=t, d_sense=float(np.linalg.norm(np.asarray(pos) - PROBE["truth"])))
    return _orig_perceive(self, t, pos, detected)


def _plan(self, obs):
    _orig_plan(self, obs)
    if "t_belief" not in PROBE and any(tr.confirmed and np.linalg.norm(tr.mean - PROBE["truth"]) < 3.0
                                       for tr in self.search.belief.tracks):
        PROBE.update(t_belief=obs.t, d_belief=float(np.linalg.norm(np.asarray(obs.robot_pos) - PROBE["truth"])))


cs.generate_detections = _detections
RescueStack.perceive, RescueStack.plan = _perceive, _plan


def work(job):
    name, v, d_app, lat, seed, variant = job
    plant = None
    if name == "mujoco":
        if "p" not in _PLANT:
            from rescue.mj_plant import Go2MujocoPlant
            _PLANT["p"] = Go2MujocoPlant()
        plant = _PLANT["p"]
    truth = np.array([X_MISSED, lat])
    PROBE.clear()
    PROBE["truth"] = truth
    site = Site(seed, (-10.0, -30.0, 60.0, 30.0), (),
                (Survivor(0, np.array([X_GOAL, 0.0]), Priority.CRITICAL, ResponseState.TRAPPED),
                 Survivor(1, truth, Priority.CRITICAL, ResponseState.TRAPPED)), np.zeros(2))
    cfg = replace(StackConfig(), use_cbf=variant != "no_cbf", infer_state=False,
                  **(dict(conf_fresh=0.99, n_fresh=5) if variant == "fresh99" else {}),
                  **(dict(fast_detect=True) if variant.startswith("fast") else {}),
                  **(dict(conf_fresh=0.99, n_fresh=5) if variant == "fast99" else {}))
    sensor = float(np.hypot(d_app, lat))
    log = cs.run_episode_ctrl(site, PriorConfig(n_false_pos=0.0), cfg, seed, variant, T=T_RUN, sensor_radius=sensor,
                              params=Go2Params(speed_scale=v, max_turn_rate=1.0), rescue_radius=0.0, plant=plant,
                              behavior=BehaviorConfig())
    traj = np.array([(x, y) for _, x, y in log.trajectory])
    dmin_1hz = float(np.min(np.linalg.norm(traj - truth, axis=1))) if len(traj) else np.nan
    return dict(plant=name, speed=v, variant=variant, d_appear=d_app, lateral=lat, seed=seed,
                fell=int(log.fell_at is not None), infeasible=log.infeasible_steps,
                viol_steps=log.clear_viol_steps, dmin_1hz=dmin_1hz,
                t_sense=PROBE.get("t_sense", np.nan), d_sense=PROBE.get("d_sense", np.nan),
                t_belief=PROBE.get("t_belief", np.nan), d_belief=PROBE.get("d_belief", np.nan))


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    jobs = [(n, v, d, lat, s, var) for n in ("unicycle", "mujoco") for v in SPEEDS for d in D_APPEAR
            for lat in LATERAL for s in range(n_seeds) for var in ("full", "no_cbf", "fresh99", "fast", "fast99")]
    print(len(jobs), "episodes ...", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(work, jobs, chunksize=4)
    with open("runs/mj_sudden_stack.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)


def summarize(rows):
    print("\nviolations / runs (10 Hz samples) by d_appear")
    print(f"{'plant':9s} {'speed':>5s} {'variant':>7s} | " + "  ".join(f"{d:>9.1f}" for d in D_APPEAR))
    for name in ("unicycle", "mujoco"):
        for v in SPEEDS:
            for var in ("full", "no_cbf", "fresh99", "fast", "fast99"):
                cells = []
                for d in D_APPEAR:
                    r = [x for x in rows if (x["plant"], x["speed"], x["variant"], x["d_appear"]) == (name, v, var, d)]
                    cells.append(f"{sum(x['viol_steps'] > 0 for x in r):2d}/{len(r):2d}".rjust(9))
                print(f"{name:9s} {v:5.1f} {var:>7s} | " + "  ".join(cells))
    full = [x for x in rows if x["variant"] == "full"]
    lat = [x["t_belief"] - x["t_sense"] for x in full if np.isfinite(x["t_belief"]) and np.isfinite(x["t_sense"])]
    print(f"\nnever sensed: {sum(not np.isfinite(x['t_sense']) for x in full)}; sensed but never in belief: "
          f"{sum(np.isfinite(x['t_sense']) and not np.isfinite(x['t_belief']) for x in full)} of {len(full)} (full)")
    if lat:
        print("sense -> belief latency (s): mean %.2f, median %.2f, max %.2f" % (np.mean(lat), np.median(lat), np.max(lat)))
    print("falls:", sum(x["fell"] for x in rows))


if __name__ == "__main__":
    main()
