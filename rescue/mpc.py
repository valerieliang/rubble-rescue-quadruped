"""Approach planner (~10 Hz): sampling-based MPC on the unicycle model, with a legibility term.

Each call samples K control sequences (v_n, omega_n), n = 0..N-1, around the previous solution,
rolls them through the unicycle (x' = v cos th, y' = v sin th, th' = omega) with forward speed capped
by the terrain, scores them, and returns the first control of the cost-weighted average (MPPI).
Sampling keeps the solve at about a millisecond in numpy and handles the non-smooth barrier and
legibility costs without gradients; there is no QP solver in the loop. The CBF filter (``cbf.py``)
sits after this and makes the final command safe.

Cost of a rollout:
    sum_n  w_goal |p_n - g|          distance to the standoff pose g (and 5x at the end)
         + w_head  angle error at the end, so the robot arrives facing the survivor
         + w_dv (v_n - v_{n-1})^2 + w_om omega_n^2
         + w_bar  sum_n max(0, -h(p_n))      soft barrier penalties (hazards, clearance regions)
         - w_leg  sum_n log P(true goal | prefix + rollout up to n)      legibility

Legibility follows the observer model (``observer.py``): the observer's posterior over candidate
goals is a softmax of -beta * detour, where detour_g is the travel the robot has done plus the
rollout so far plus the straight distance remaining to g, minus the direct distance from the start.
Costs here are open-ground distances (speed 1 m/s), which matches the observer on open ground.
The legibility weight is scaled by the inferred state of the target (``legibility_weight``): the
robot is legible for survivors who can watch it, not for those who cannot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .cbf import SafetyFilter, rect_sd
from .clearance import ClearanceRegion
from .interfaces import STANDOFF_RADIUS, Region, ResponseState

SpeedFn = Callable[[np.ndarray], np.ndarray]    # points (..., 2) -> terrain speed cap (...,)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def standoff_pose(pos: np.ndarray, target: np.ndarray, standoff: float = 0.8 * STANDOFF_RADIUS) -> np.ndarray:
    """Point ``standoff`` metres from the target on the robot's side: where the approach ends."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    d = pos - target
    n = float(np.linalg.norm(d))
    return target if n <= standoff else target + d / n * standoff


def legibility_weight(state_posterior: np.ndarray | None, base: float,
                      floor: float = 0.2) -> float:
    """Scale the legibility weight by the chance the target can observe the robot (is mobile)."""
    if state_posterior is None:
        return base
    return base * (floor + (1.0 - floor) * float(state_posterior[ResponseState.MOBILE]))


def _seg_hits_rect(a: np.ndarray, b: np.ndarray, r: Region, pad: float) -> bool:
    """Slab test: does segment a-b intersect the rectangle grown by ``pad``?"""
    lo = np.array([r.xmin - pad, r.ymin - pad])
    hi = np.array([r.xmax + pad, r.ymax + pad])
    d = b - a
    t0, t1 = 0.0, 1.0
    for k in range(2):
        if abs(d[k]) < 1e-12:
            if a[k] < lo[k] or a[k] > hi[k]:
                return False
        else:
            ta, tb = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
            if t0 > t1:
                return False
    return True


def _seg_hits_disc(a: np.ndarray, b: np.ndarray, c: np.ndarray, R: float) -> bool:
    d = b - a
    L2 = float(d @ d)
    t = 0.0 if L2 < 1e-12 else float(np.clip((c - a) @ d / L2, 0.0, 1.0))
    return float(np.linalg.norm(a + t * d - c)) < R


def route_goal(pos: np.ndarray, goal: np.ndarray, hazards: Sequence[Region],
               discs: Sequence[tuple[np.ndarray, float]], pad: float = 1.0,
               prefer: np.ndarray | None = None) -> np.ndarray:
    """Next intermediate point on a one-detour route around hazards and keep-out discs.

    The MPC horizon (~1.5 s) is too short to see a large obstacle, so a straight goal behind one is a
    local minimum. If the straight line to ``goal`` is blocked, return the corner of a grown hazard
    (or tangent-ish point of a disc) that gives the shortest visible two-leg route; otherwise ``goal``.
    ``prefer`` (the previous choice) breaks ties so the route does not flip between symmetric sides.
    """
    pos, goal = np.asarray(pos, float), np.asarray(goal, float)

    def blocked(a, b):
        return (any(_seg_hits_rect(a, b, r, pad - 0.2) for r in hazards)
                or any(_seg_hits_disc(a, b, c, R + pad - 0.2) for c, R in discs))

    if not blocked(pos, goal):
        return goal
    cands = []
    for r in hazards:
        for x in (r.xmin - pad, r.xmax + pad):
            for y in (r.ymin - pad, r.ymax + pad):
                cands.append(np.array([x, y]))
    for c, R in discs:
        d = goal - pos
        n = float(np.linalg.norm(d))
        perp = np.array([-d[1], d[0]]) / n if n > 1e-9 else np.array([0.0, 1.0])
        for sgn in (1.0, -1.0):
            cands.append(c + sgn * perp * (R + pad) * 1.15)
    best, best_cost = None, np.inf
    for c in cands:
        if blocked(pos, c) and np.linalg.norm(c - pos) > 0.5:
            continue
        cost = float(np.linalg.norm(c - pos) + np.linalg.norm(goal - c))
        if blocked(c, goal):
            cost += 1000.0                          # needs a second detour: only if nothing better
        if prefer is not None and np.linalg.norm(c - prefer) < 1e-6:
            cost -= 2.0
        if cost < best_cost:
            best, best_cost = c, cost
    return goal if best is None else best


@dataclass
class Legibility:
    """Observer-facing context for one approach."""
    goals: np.ndarray                 # (G, 2) candidate goals the observer considers (true goal included)
    true_idx: int
    start: np.ndarray                 # (2,) where the approach began
    beta: float = 0.5                 # observer rationality
    done: float = 0.0                 # distance travelled so far in this approach


@dataclass
class MPCParams:
    dt: float = 0.1
    horizon: int = 15
    n_samples: int = 256
    v_max: float = 1.0
    omega_max: float = 1.5
    w_goal: float = 1.0
    w_head: float = 0.5
    w_dv: float = 0.5
    w_omega: float = 0.05
    w_barrier: float = 50.0
    barrier_margin: float = 0.5       # m of clearance the soft barrier asks for beyond h = 0
    temperature: float = 0.5          # MPPI temperature, relative to the cost spread
    v_noise: float = 0.3
    omega_noise: float = 0.6


class MPCPlanner:
    def __init__(self, params: MPCParams | None = None, seed: int = 0):
        self.p = params or MPCParams()
        self.rng = np.random.default_rng(seed)
        self.U = np.zeros((self.p.horizon, 2))     # warm start (v, omega)
        self.solve_count = 0

    def reset(self) -> None:
        self.U[:] = 0.0

    def plan(self, pos: np.ndarray, heading: float, v: float, goal: np.ndarray,
             speed_fn: SpeedFn | None = None, hazards: Sequence[Region] = (),
             regions: Sequence[ClearanceRegion] = (), target: np.ndarray | None = None,
             target_keepout: float = 1.0, legibility: Legibility | None = None,
             w_leg: float = 0.0) -> tuple[float, float]:
        """First control (v_cmd, omega_cmd) of the best rollout toward ``goal`` (the standoff pose)."""
        P = self.p
        K, N = P.n_samples, P.horizon
        pos, goal = np.asarray(pos, float), np.asarray(goal, float)
        noise = self.rng.normal(size=(K, N, 2)) * np.array([P.v_noise, P.omega_noise])
        noise[0] = 0.0                                              # keep the warm start itself
        U = self.U[None] + noise
        # Also include a direct-to-goal candidate so a cold start is not blind.
        to_goal = goal - pos
        err0 = _wrap(np.arctan2(to_goal[1], to_goal[0]) - heading)
        U[1, :, 0] = P.v_max * max(0.0, np.cos(err0))
        U[1, :, 1] = np.clip(2.0 * err0, -P.omega_max, P.omega_max)
        U[..., 0] = np.clip(U[..., 0], 0.0, P.v_max)
        U[..., 1] = np.clip(U[..., 1], -P.omega_max, P.omega_max)

        xy, th = np.empty((K, N, 2)), np.empty((K, N))
        x, y, a = np.full(K, pos[0]), np.full(K, pos[1]), np.full(K, heading)
        v_applied = np.empty((K, N))
        for n in range(N):
            vn = U[:, n, 0]
            if speed_fn is not None:
                vn = np.minimum(vn, speed_fn(np.stack([x, y], axis=-1)))
            v_applied[:, n] = vn
            a = _wrap(a + U[:, n, 1] * P.dt)
            x = x + vn * np.cos(a) * P.dt
            y = y + vn * np.sin(a) * P.dt
            xy[:, n, 0], xy[:, n, 1], th[:, n] = x, y, a

        dist = np.linalg.norm(xy - goal, axis=-1)                   # (K, N)
        cost = P.w_goal * dist.sum(1) + 5.0 * P.w_goal * dist[:, -1]
        head_err = np.abs(_wrap(np.arctan2(goal[1] - xy[:, -1, 1], goal[0] - xy[:, -1, 0]) - th[:, -1]))
        cost += P.w_head * head_err * np.minimum(dist[:, -1], 3.0)
        dv = np.diff(np.concatenate([np.full((K, 1), v), v_applied], axis=1), axis=1)
        cost += P.w_dv * (dv**2).sum(1) + P.w_omega * (U[..., 1] ** 2).sum(1)

        cost += P.w_barrier * self._barrier_penalty(xy, hazards, regions, target, target_keepout).sum(1)
        if legibility is not None and w_leg > 0.0:
            cost += w_leg * self._legibility_cost(xy, pos, legibility)

        J = cost - cost.min()
        scale = max(float(J.mean()), 1e-6)
        w = np.exp(-J / (P.temperature * scale))
        w /= w.sum()
        U_new = np.tensordot(w, U, axes=(0, 0))
        self.U = np.vstack([U_new[1:], U_new[-1:]])                 # shift the warm start
        self.solve_count += 1
        return float(U_new[0, 0]), float(U_new[0, 1])

    def _barrier_penalty(self, xy, hazards, regions, target, target_keepout) -> np.ndarray:
        pen = np.zeros(xy.shape[:2])
        m = self.p.barrier_margin
        for r in hazards:
            pen += np.maximum(0.0, m - rect_sd(xy, r))
        for reg in regions:
            R = target_keepout if (target is not None and np.allclose(reg.center, target)) else reg.radius
            pen += np.maximum(0.0, m - (np.linalg.norm(xy - reg.center, axis=-1) - R))
        return pen

    def _legibility_cost(self, xy: np.ndarray, pos: np.ndarray, leg: Legibility) -> np.ndarray:
        """-sum_n log P(true goal | prefix + rollout to n) under the observer model; (K,)."""
        K, N = xy.shape[:2]
        steps = np.linalg.norm(np.diff(np.concatenate([np.broadcast_to(pos, (K, 1, 2)), xy], axis=1), axis=1),
                               axis=-1)
        travelled = leg.done + np.cumsum(steps, axis=1)                      # (K, N)
        direct = np.linalg.norm(leg.goals - leg.start, axis=-1)              # (G,)
        remaining = np.linalg.norm(xy[:, :, None, :] - leg.goals[None, None], axis=-1)   # (K, N, G)
        logit = -leg.beta * (travelled[..., None] + remaining - direct)
        logit -= logit.max(-1, keepdims=True)
        logp = logit - np.log(np.exp(logit).sum(-1, keepdims=True))
        return -logp[..., leg.true_idx].sum(1)


class ApproachController:
    """Planner + safety filter, called every control step (10 Hz).

    ``command`` returns the (v, omega) to send to the robot; keeps the approach bookkeeping the
    legibility term needs (start point and distance travelled) and resets it when the target changes.
    """

    def __init__(self, planner: MPCPlanner | None = None, safety: SafetyFilter | None = None,
                 legibility_base: float = 0.0, beta: float = 0.5):
        self.planner = planner or MPCPlanner()
        self.safety = safety
        self.legibility_base, self.beta = legibility_base, beta
        self._target: np.ndarray | None = None
        self._start = np.zeros(2)
        self._last = np.zeros(2)
        self._done = 0.0
        self._via: np.ndarray | None = None

    def command(self, pos: np.ndarray, heading: float, v: float, target: np.ndarray,
                speed_fn: SpeedFn | None = None, hazards: Sequence[Region] = (),
                regions: Sequence[ClearanceRegion] = (), other_goals: Sequence[np.ndarray] = (),
                state_posterior: np.ndarray | None = None) -> tuple[float, float]:
        pos, target = np.asarray(pos, float), np.asarray(target, float)
        if self._target is None or not np.allclose(target, self._target, atol=1e-6):
            self._target, self._start, self._last, self._done = target.copy(), pos.copy(), pos.copy(), 0.0
        else:
            self._done += float(np.linalg.norm(pos - self._last))
            self._last = pos.copy()
        goal = standoff_pose(pos, target)
        discs = [(reg.center, reg.radius) for reg in regions if not np.allclose(reg.center, target)]
        via = route_goal(pos, goal, hazards, discs, prefer=self._via)
        self._via = via if via is not goal else None
        leg, w_leg = None, 0.0
        if self.legibility_base > 0.0 and len(other_goals):
            goals = np.vstack([target[None]] + [np.asarray(g, float)[None] for g in other_goals])
            leg = Legibility(goals, 0, self._start, self.beta, self._done)
            w_leg = legibility_weight(state_posterior, self.legibility_base)
        v_cmd, omega = self.planner.plan(pos, heading, v, via, speed_fn, hazards, regions, target,
                                         self.safety.target_keepout if self.safety else 1.0, leg, w_leg)
        if self.safety is not None:
            v_cmd, omega = self.safety.filter(pos, heading, v_cmd, omega, hazards, regions, target, via)
        return v_cmd, omega
