"""Point-robot simulator with a disc sensor, plus the seeded site generator.

Kinematic reference simulator (no turning or acceleration limits). ``sim.py`` adds the Go2
reduced-order dynamics; both share the Observation/RunLog API and the site generator.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .drone_prior import PriorConfig, generate_detections
from .interfaces import (Detection, Observation, Priority, Region, ResponseState, RunLog, Site,
                         Survivor)
from .terrain import speed_at


def generate_site(seed: int, size: float = 100.0) -> Site:
    rng = np.random.default_rng(seed)

    def rect(kind, lo, hi):
        w, h = rng.uniform(lo, hi, 2)
        x, y = rng.uniform(0, size - w), rng.uniform(0, size - h)
        return Region(kind, x, y, x + w, y + h)

    regions = tuple([rect("rubble", 10, 30) for _ in range(rng.integers(4, 7))]
                    + [rect("hazard", 8, 15) for _ in range(rng.integers(1, 4))])
    hazards = [r for r in regions if r.kind == "hazard"]
    survivors = []
    for i in range(int(rng.integers(3, 9))):
        while True:
            p = rng.uniform(5, size - 5, 2)
            if not any(h.xmin <= p[0] <= h.xmax and h.ymin <= p[1] <= h.ymax for h in hazards):
                break
        survivors.append(Survivor(i, p, Priority(rng.choice(3, p=[0.4, 0.35, 0.25])),
                                  ResponseState(rng.integers(3))))
    return Site(seed, (0.0, 0.0, size, size), regions, tuple(survivors), np.array([3.0, 3.0]))


def run_episode(site: Site, make_policy: Callable, prior_cfg: PriorConfig, seed: int,
                method: str = "", T: float = 600.0, dt: float = 1.0, sensor_radius: float = 5.0,
                p_detect: float = 0.9, sigma_meas: float = 0.5, standoff: float = 1.5) -> RunLog:
    """make_policy(site, detections) -> Policy. The policy only gets Observations."""
    rng = np.random.default_rng(seed)
    policy = make_policy(site, generate_detections(site, prior_cfg, rng))
    log = RunLog(site.seed, method)
    pos, rescued_now, t = site.start.astype(float).copy(), [], 0.0
    pending = {s.id: s for s in site.survivors}
    while t < T and pending:
        detected = tuple(
            Detection(s.pos + rng.normal(0, sigma_meas, 2), s.priority, s.id)
            for s in pending.values()
            if np.linalg.norm(s.pos - pos) <= sensor_radius and rng.random() < p_detect)
        obs = Observation(t, pos.copy(), sensor_radius, detected, tuple(rescued_now))
        wp = np.asarray(policy.act(obs), float)
        rescued_now, sub = [], 0.1
        for k in range(int(round(dt / sub))):
            d = wp - pos
            n = np.linalg.norm(d)
            if n > 1e-9:
                pos = pos + d / n * min(n, speed_at(site, pos) * sub)
            for sid in [i for i, s in pending.items() if np.linalg.norm(s.pos - pos) <= standoff]:
                log.rescue_times[sid] = t + (k + 1) * sub
                rescued_now.append(pending.pop(sid).pos)
        t += dt
        log.trajectory.append((t, float(pos[0]), float(pos[1])))
    return log
