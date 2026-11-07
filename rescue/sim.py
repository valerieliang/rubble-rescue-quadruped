"""Reduced-order Go2 simulator: unicycle robot, waypoint tracker, disc sensor.

Model
-----
State (x, y, heading, v). Inputs come from the Go2's high-level velocity interface:
forward speed and turn rate. This is deliberately what the real Go2's built-in controller
accepts, so everything above the tracker transfers to hardware unchanged.

    x'       = v cos(heading)
    y'       = v sin(heading)
    heading' = omega,                 |omega| <= max_turn_rate
    v'       = (v_cmd - v) rate-limited by max_accel  (first-order velocity response)

Forward speed is capped by the terrain under the robot (``terrain.speed_at``: open / rubble /
hazard). Hazards are slow but traversable (no barrier constraints in this model).

Timing: dynamics step at ``sim_dt`` (0.1 s). The policy is called every ``dt`` (1 s) with an
Observation; between calls the tracker drives toward the last waypoint. The sim owns the clock.

Randomness: the episode RNG is consumed in the same order as ``toy_sim`` (drone detections
first, then robot sensing each tick), so the drone prior for a given seed is identical across
the two simulators.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .drone_prior import PriorConfig, generate_detections
from .interfaces import STANDOFF_RADIUS, Detection, Observation, RunLog, Site, TravelTime
from .terrain import speed_at
from .toy_sim import generate_site  # re-exported: same seeded site generator

__all__ = ["Go2Params", "generate_site", "run_episode", "travel_time", "make_travel_time",
           "track_waypoint", "step_unicycle"]


@dataclass(frozen=True)
class Go2Params:
    """Reduced-order Go2 parameters (conservative relative to the real robot; not identified from hardware)."""
    speed_scale: float = 1.0      # multiplies terrain.speed_at (1.0 keeps the 1 m/s open-ground cap)
    max_turn_rate: float = 1.5    # rad/s
    max_accel: float = 1.0        # m/s^2, forward speed rate limit
    k_heading: float = 2.0        # turn-rate gain on heading error (1/s)
    k_arrive: float = 1.0         # forward speed <= k_arrive * distance, so it slows to stop
    arrive_tol: float = 0.05      # m; closer than this counts as at the waypoint


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def track_waypoint(pos: np.ndarray, heading: float, wp: np.ndarray, v_cap: float,
                   p: Go2Params) -> tuple[float, float]:
    """Waypoint tracker -> (v_cmd, omega_cmd).

    Turns toward the waypoint and walks; forward speed drops with cos(heading error), so with
    the waypoint behind it the robot turns in place instead of walking away.
    """
    d = wp - pos
    dist = float(np.hypot(d[0], d[1]))
    if dist < p.arrive_tol:
        return 0.0, 0.0
    err = _wrap(np.arctan2(d[1], d[0]) - heading)
    omega = float(np.clip(p.k_heading * err, -p.max_turn_rate, p.max_turn_rate))
    v = v_cap * max(0.0, np.cos(err))
    return min(v, p.k_arrive * dist), omega


def step_unicycle(pos: np.ndarray, heading: float, v: float, v_cmd: float, omega: float,
                  dt: float, p: Go2Params, bounds) -> tuple[np.ndarray, float, float]:
    """One Euler step of the unicycle with a rate-limited forward speed; clamps to the site."""
    v = v + float(np.clip(v_cmd - v, -p.max_accel * dt, p.max_accel * dt))
    heading = _wrap(heading + omega * dt)
    pos = pos + v * dt * np.array([np.cos(heading), np.sin(heading)])
    xmin, ymin, xmax, ymax = bounds
    pos = np.array([np.clip(pos[0], xmin, xmax), np.clip(pos[1], ymin, ymax)])
    return pos, heading, v


def travel_time(site: Site, a, b, params: Go2Params | None = None, ds: float = 0.5,
                heading: float | None = None) -> float:
    """Terrain-aware travel time (s) along the straight segment a -> b.

    Integrates 1/speed over the segment, so crossing rubble or a hazard costs its real time
    instead of being charged at the start point's speed. If ``heading`` is given, adds the time
    to turn toward b first. Ignores acceleration (it's an estimate for scoring targets).
    """
    p = params or Go2Params()
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    t = 0.0
    if heading is not None and L > 0:
        t += abs(_wrap(np.arctan2(*(b - a)[::-1]) - heading)) / p.max_turn_rate
    if L == 0:
        return t
    n = max(1, int(np.ceil(L / ds)))
    for i in range(n):
        mid = a + (b - a) * (i + 0.5) / n
        t += (L / n) / (speed_at(site, mid) * p.speed_scale)
    return t


def make_travel_time(site: Site, params: Go2Params | None = None, ds: float = 0.5) -> TravelTime:
    """Bind a site so the search layer gets the ``TravelTime`` callable from interfaces.py."""
    return lambda a, b: travel_time(site, a, b, params, ds)


def run_episode(site: Site, make_policy: Callable, prior_cfg: PriorConfig, seed: int,
                method: str = "", T: float = 600.0, dt: float = 1.0, sensor_radius: float = 5.0,
                p_detect: float = 0.9, sigma_meas: float = 0.5, standoff: float = STANDOFF_RADIUS,
                sim_dt: float = 0.1, params: Go2Params | None = None,
                heading0: float | None = None) -> RunLog:
    """Run one episode. ``make_policy(site, detections) -> Policy``; the policy only sees
    Observations (the robot pose in an Observation is the true pose).

    ``heading0=None`` starts the robot facing its first waypoint, so it isn't charged a turn
    the toy sim never had. Pass a value to start from a fixed heading instead.
    """
    p = params or Go2Params()
    rng = np.random.default_rng(seed)
    policy = make_policy(site, generate_detections(site, prior_cfg, rng))
    log = RunLog(site.seed, method)

    pos = site.start.astype(float).copy()
    heading, v, t = heading0, 0.0, 0.0
    rescued_now: list[np.ndarray] = []
    pending = {s.id: s for s in site.survivors}
    n_sub = max(1, int(round(dt / sim_dt)))
    h = dt / n_sub

    while t < T - 1e-9 and pending:
        detected = tuple(
            Detection(s.pos + rng.normal(0, sigma_meas, 2), s.priority, s.id)
            for s in pending.values()
            if np.linalg.norm(s.pos - pos) <= sensor_radius and rng.random() < p_detect)
        obs = Observation(t, pos.copy(), sensor_radius, detected, tuple(rescued_now))
        wp = np.asarray(policy.act(obs), float)
        if heading is None:
            d = wp - pos
            heading = float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 1e-9 else 0.0

        rescued_now = []
        for k in range(n_sub):
            v_cap = speed_at(site, pos) * p.speed_scale
            v_cmd, omega = track_waypoint(pos, heading, wp, v_cap, p)
            pos, heading, v = step_unicycle(pos, heading, v, v_cmd, omega, h, p, site.bounds)
            v = min(v, v_cap)  # entering rubble/hazard caps speed immediately
            for sid in [i for i, s in pending.items() if np.linalg.norm(s.pos - pos) <= standoff]:
                log.rescue_times[sid] = t + (k + 1) * h
                rescued_now.append(pending.pop(sid).pos)
        t += dt
        log.trajectory.append((t, float(pos[0]), float(pos[1])))
    return log