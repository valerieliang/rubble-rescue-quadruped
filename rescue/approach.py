"""Q5 harness: one approach to one survivor among distractors, evaluated by simulated observers.

Open ground (no rubble, no hazards) so the only thing that varies is how the robot moves. The robot
starts D m from the target; one or two other survivors sit at a similar range, 15-45 degrees to the
same side, so a straight approach is ambiguous for a long time. The approach runs through the same
MPC + CBF as the full system (``RescueStack.control``) on the Go2 plant, to the standoff pose.

Legibility weight by state: a survivor who cannot watch (unresponsive) gets no legibility term, so
the planner does not pay for it. ``state_weight`` below is the weight the planner applies.

Metrics per approach: time to reach the standoff pose and path length (the cost), and for each
observer the time after which P(true goal) stays >= 0.9, that time as a fraction of the approach, and
the mean P(true goal) over the approach.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .drone_prior import PriorConfig
from .interfaces import Priority, ResponseState, Site, Survivor
from .legibility import CostObserver, HeadingObserver, LearnedObserver, identification_time
from .sim import Go2Params, step_unicycle
from .stack import R_RESCUE, RescueStack, StackConfig

STATE_LEG_WEIGHT = {ResponseState.UNRESPONSIVE: 0.0, ResponseState.TRAPPED: 0.6, ResponseState.MOBILE: 1.0}


@dataclass
class ApproachResult:
    t_total: float
    path_len: float
    reached: bool
    ident: dict          # observer name -> identification time (s, inf if never)
    mean_p: dict         # observer name -> mean P(true goal) over the approach
    path: np.ndarray
    goals: np.ndarray


def make_scenario(seed: int, n_distractors: int = 1, dist: float = 22.0, size: float = 100.0):
    """(site, goals (K,2) with the target at index 0, start (2,))."""
    rng = np.random.default_rng(seed)
    start = np.array([size * 0.2, size * 0.5]) + rng.normal(0, 2, 2)
    bearing = rng.uniform(-0.4, 0.4)
    d0 = dist * rng.uniform(0.9, 1.1)
    target = start + d0 * np.array([np.cos(bearing), np.sin(bearing)])
    goals = [target]
    side = 1 if rng.random() < 0.5 else -1       # distractors share a side, so there is room to exaggerate away
    for k in range(n_distractors):
        off = np.deg2rad(rng.uniform(15, 45)) * side
        d = dist * rng.uniform(0.85, 1.15)
        goals.append(start + d * np.array([np.cos(bearing + off), np.sin(bearing + off)]))
    goals = np.array(goals)
    survivors = tuple(Survivor(i, g, Priority.URGENT, ResponseState.MOBILE) for i, g in enumerate(goals))
    return Site(seed, (0.0, 0.0, size, size), (), survivors, start), goals, start


def run_approach(seed: int, stack_cfg: StackConfig, state: ResponseState = ResponseState.MOBILE,
                 state_blind: bool = False, n_distractors: int = 1, T: float = 120.0, dt: float = 0.1,
                 params: Go2Params | None = None) -> ApproachResult:
    """``state_blind=True`` applies the full legibility weight whatever the survivor's state."""
    p = params or Go2Params()
    site, goals, start = make_scenario(seed, n_distractors)
    w = 1.0 if state_blind else STATE_LEG_WEIGHT[state]
    cfg = replace(stack_cfg, infer_state=False, comm_oracle=False, legibility=stack_cfg.legibility * w)
    stack = RescueStack(site, [], PriorConfig(), cfg, p)
    # Hand the stack the (accurately localized) tracks directly: this experiment is about motion, not search.
    from .belief import Track
    stack.search.belief.tracks = [Track(g.copy(), 0.5, 1.0, Priority.URGENT, confirmed=True) for g in goals]
    pos, heading, v = start.copy(), None, 0.0
    path = [pos.copy()]
    reached = False
    n = int(round(T / dt))
    for i in range(n):
        t = i * dt
        if i % 10 == 0:
            tgt = stack.search.belief.tracks[0]
            stack.target, stack.target_key = tgt, id(tgt)
            stack.approach_start = start
            stack.path = np.array([pos, stack.goal(pos)])
            if heading is None:
                d = goals[0] - pos
                heading = float(np.arctan2(d[1], d[0]))
        u = stack.control(t, pos, heading, 1.0)
        pos, heading, v = step_unicycle(pos, heading, v, float(u[0]), float(u[1]), dt, p, site.bounds)
        path.append(pos.copy())
        if np.linalg.norm(pos - goals[0]) <= R_RESCUE:
            reached = True
            break
    path = np.array(path)
    rng = np.random.default_rng(seed + 99)
    obs = {"cost": CostObserver(), "heading": HeadingObserver(), "learned": LearnedObserver()}
    ident, mean_p = {}, {}
    for name, o in obs.items():
        post = o.posterior(path, start, goals, rng)
        ident[name] = identification_time(post, 0, dt)
        mean_p[name] = float(post[:, 0].mean())
    t_total = (len(path) - 1) * dt
    return ApproachResult(t_total, float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))), reached,
                          ident, mean_p, path, goals)
