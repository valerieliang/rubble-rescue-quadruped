"""Unicycle plant with injected tracking error, to test whether the tracking-error pad ever binds.

The reduced-order unicycle (``sim.step_unicycle``) plus an unmodelled position disturbance: a 2-D
Ornstein-Uhlenbeck velocity w (per-axis std ``sigma_d`` m/s, time constant ``tau`` s) is integrated into the
position. The planner and the safety filter assume the ideal unicycle, so w is exactly the kind of error the pad
``eps_track`` is meant to cover. ``sigma_d`` is reported through ``measured_tracking_error``: the same 1 s gap
statistic as ``safety.calibrate_tracking_error`` (95% quantile of the distance between this plant and the ideal
unicycle after 1 s of random constant commands), so levels can be compared with the Go2's measured values
(Table 19: 0.2 m flat, 0.36 m rubble with the rough policy, 0.59 m rubble with the flat policy, 0.8-0.95 m hazard).

Same interface as ``mj_plant.Go2MujocoPlant`` (reset, step, fallen, w_max), so ``ctrl_sim`` can use it.
"""
from __future__ import annotations

import numpy as np

from .sim import Go2Params, step_unicycle

_BOUNDS = (-1e3, -1e3, 1e3, 1e3)


class DisturbedUnicycle:
    def __init__(self, sigma_d: float = 0.0, tau: float = 1.0, w_max: float = 1.0, seed: int = 0):
        self.sigma_d, self.tau, self.w_max = sigma_d, tau, w_max
        self.p = Go2Params(max_turn_rate=w_max)
        self.rng = np.random.default_rng(seed)
        self.fallen = False
        self.reset(np.zeros(2), 0.0)

    def reset(self, xy=(0.0, 0.0), yaw: float = 0.0, settle: float = 1.0) -> None:
        self.pos, self.heading, self.v = np.asarray(xy, float).copy(), float(yaw), 0.0
        self.w = self.rng.normal(0.0, self.sigma_d, 2)    # start in the stationary distribution
        self.fallen = False

    def step(self, v_cmd: float, omega_cmd: float, dt: float = 0.1):
        self.pos, self.heading, self.v = step_unicycle(self.pos, self.heading, self.v, float(v_cmd),
                                                       float(omega_cmd), dt, self.p, _BOUNDS)
        a = np.exp(-dt / self.tau)
        self.w = a * self.w + np.sqrt(1 - a * a) * self.rng.normal(0.0, self.sigma_d, 2)
        self.pos = self.pos + self.w * dt
        return self.pos.copy(), self.heading, self.v


def measured_tracking_error(sigma_d: float, tau: float = 1.0, horizon: float = 1.0, n: int = 2000,
                            quantile: float = 0.95, seed: int = 0, dt: float = 0.1) -> float:
    """95% gap (m) after ``horizon`` s between this plant and the ideal unicycle, as in calibrate_tracking_error."""
    rng = np.random.default_rng(seed)
    plant = DisturbedUnicycle(sigma_d, tau, seed=seed + 1)
    p = plant.p
    errs = []
    for _ in range(n):
        th, v0 = float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(0, 1.0))
        plant.reset(np.zeros(2), th)
        plant.v = v0
        ideal, ith = np.zeros(2), th
        v_cmd, w_cmd = float(rng.uniform(0, 1.0)), float(rng.uniform(-p.max_turn_rate, p.max_turn_rate))
        for _k in range(int(round(horizon / dt))):
            pos, _, _ = plant.step(v_cmd, w_cmd, dt)
            ith = ith + w_cmd * dt
            ideal = ideal + v_cmd * dt * np.array([np.cos(ith), np.sin(ith)])
        errs.append(float(np.linalg.norm(pos - ideal)))
    return float(np.quantile(errs, quantile))
