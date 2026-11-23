"""Control-rate episode runner for Q3-Q5 (same reduced-order Go2 plant as ``sim.py``).

``sim.run_episode`` calls the policy at 1 Hz and lets a waypoint tracker drive in between; it is left
untouched so the Q1-Q2 results stay reproducible. Here the robot is driven by the approach MPC and
the CBF filter at 10 Hz (the velocity-command level the real Go2 accepts), survivors move according
to their hidden behavior, and safety is scored against the *true* survivor positions and terrain.

Per control step (0.1 s):  move survivors -> sense (disc, noisy positions) -> prompts answered ->
[every 10th step: 1 Hz search/belief update] -> MPC (+ CBF) -> unicycle plant step -> rescue and
violation checks.

Differences from ``sim.py`` (all deliberate; see docs/INTERFACES.md):
  * sensor radius 10 m (an onboard person detector sees farther than the 5 m of Q1-Q2);
  * rescue radius 3.5 m (``R_RESCUE``): the robot confirms from a standoff outside the 1 m clearance;
  * survivors can move; a rescued survivor stops and still counts for clearance violations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ctrl_behavior import BehaviorConfig, assign_modes, sample_response, step_survivors
from .drone_prior import PriorConfig, generate_detections
from .interfaces import BASE_REWARD, Detection, Observation, RunLog, Site
from .sim import Go2Params, step_unicycle
from .stack import R_RESCUE, RescueStack, StackConfig
from .terrain import speed_at

D_CLEAR_EVAL = 1.0   # m; a "clearance violation" is the robot centre within this of any survivor


@dataclass
class CtrlLog(RunLog):
    clear_viol_steps: int = 0
    haz_viol_steps: int = 0
    min_clearance: float = float("inf")
    infeasible_steps: int = 0
    prompts: int = 0
    state_trace: dict = field(default_factory=dict)    # survivor id -> [(t, p over 4 hypotheses)]
    first_seen: dict = field(default_factory=dict)     # survivor id -> time its track was first logged
    viol_events: list = field(default_factory=list)    # first few (t, id, dist, tracked?, pending?) for debugging
    fell_at: float | None = None                       # time the MuJoCo Go2 fell (plant runs only)


def _in_hazard(site: Site, p) -> bool:
    return any(r.kind == "hazard" and r.xmin <= p[0] <= r.xmax and r.ymin <= p[1] <= r.ymax
               for r in site.regions)


def run_episode_ctrl(site: Site, prior_cfg: PriorConfig, stack_cfg: StackConfig, seed: int, method: str = "",
                     T: float = 600.0, sensor_radius: float = 10.0, p_detect: float = 0.9,
                     sigma_meas: float = 0.5, params: Go2Params | None = None,
                     behavior: BehaviorConfig | None = None, dt: float = 0.1,
                     rescue_radius: float = R_RESCUE, dwell: dict | None = None, plant=None) -> CtrlLog:
    """``plant`` (default None = the unicycle) can be a ``mj_plant.Go2MujocoPlant``: the commanded
    (v, omega) is then tracked by the learned Go2 policy in MuJoCo, the pose comes back from it, and the
    episode ends (``log.fell_at``) if the robot falls. Terrain stays a speed cap in the stack. With a plant
    and no ``params``, the stack plans and filters with the plant's turn-rate limit (``plant.w_max``)."""
    p = params or Go2Params()
    if plant is not None and params is None:
        # plan and filter with the turn rate the plant can actually follow (the policy's +-1 rad/s range)
        p = Go2Params(max_turn_rate=min(p.max_turn_rate, plant.w_max))
    beh = behavior or BehaviorConfig()
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(seed + 1_000_003)
    modes = assign_modes(site, beh, np.random.default_rng(site.seed + 7919))
    truth = {s.id: s for s in site.survivors}
    spos = {s.id: s.pos.astype(float).copy() for s in site.survivors}
    sheading = {s.id: float(brng.uniform(-np.pi, np.pi)) for s in site.survivors}
    pending = set(spos)
    dwell = dwell or BASE_REWARD.dwell      # s the robot must stay within rescue_radius, by true state
    held = {s: 0.0 for s in spos}

    stack = RescueStack(site, generate_detections(site, prior_cfg, rng), prior_cfg, stack_cfg, p, T,
                        sensor_radius, sigma_meas)
    log = CtrlLog(site.seed, method)
    pos, heading, v = site.start.astype(float).copy(), None, 0.0
    rescued_now: list[np.ndarray] = []
    steps_per_plan = int(round(1.0 / dt))

    for i in range(int(round(T / dt))):
        if not pending:
            break
        t = i * dt
        step_survivors(site, spos, modes, sheading, pending, pos, dt, beh, brng)
        detected = tuple(
            Detection(spos[s] + rng.normal(0, sigma_meas, 2), truth[s].priority, s)
            for s in sorted(pending)
            if np.linalg.norm(spos[s] - pos) <= sensor_radius and rng.random() < p_detect)

        for req in stack.perceive(t, pos, detected):
            z = truth[req.source_id].state
            if req.kind == "prompt":
                log.prompts += 1
                stack.apply_response(req.key, sample_response(z, beh.reliability, brng))
            else:
                stack.apply_reveal(req.key, z)

        if i % steps_per_plan == 0:
            stack.plan(Observation(t, pos.copy(), sensor_radius, detected, tuple(rescued_now)))
            rescued_now = []
            if heading is None:
                d = stack.goal(pos) - pos
                heading = float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 1e-9 else 0.0
                if plant is not None:
                    plant.reset(pos, heading)
            for tr_mean, ph in stack.snapshot():
                sid = min(pending | set(log.rescue_times), key=lambda s: np.linalg.norm(spos[s] - tr_mean))
                if np.linalg.norm(spos[sid] - tr_mean) < 3.0:
                    log.state_trace.setdefault(sid, []).append((t, ph))
                    log.first_seen.setdefault(sid, t)
            log.trajectory.append((t, float(pos[0]), float(pos[1])))

        v_cap = speed_at(site, pos) * p.speed_scale
        u = stack.control(t, pos, heading, v_cap, v)
        if not stack.last_info.get("feasible", True):
            log.infeasible_steps += 1
        if plant is None:
            pos, heading, v = step_unicycle(pos, heading, v, float(u[0]), float(u[1]), dt, p, site.bounds)
            v = min(v, speed_at(site, pos) * p.speed_scale)
        else:
            pos, heading, v = plant.step(float(u[0]), float(u[1]), dt)
            if plant.fallen:
                log.fell_at = t + dt
                break

        for s in sorted(pending):
            if np.linalg.norm(spos[s] - pos) > rescue_radius:
                held[s] = 0.0
                continue
            held[s] += dt
            if held[s] >= dwell[truth[s].state] - 1e-9:
                log.rescue_times[s] = t + dt
                rescued_now.append(spos[s].copy())
                pending.discard(s)
        dmin = min(float(np.linalg.norm(q - pos)) for q in spos.values())
        log.min_clearance = min(log.min_clearance, dmin)
        if dmin < D_CLEAR_EVAL:
            log.clear_viol_steps += 1
            if len(log.viol_events) < 6:
                sid = min(spos, key=lambda s: np.linalg.norm(spos[s] - pos))
                known = any(np.linalg.norm(tr.mean - spos[sid]) < 3 * tr.sigma + 1
                            for tr in stack.search.belief.tracks if tr.r >= stack_cfg.r_safe)
                log.viol_events.append((round(t, 1), sid, round(dmin, 2), bool(known), sid in pending))
        if _in_hazard(site, pos):
            log.haz_viol_steps += 1
    return log
