"""Terrain speed model and straight-line travel-time estimate over rubble and hazard regions."""
from __future__ import annotations

import math

import numpy as np

from .interfaces import Site

# Fraction of nominal speed inside each region kind.
REGION_SPEED = {"rubble": 0.5, "hazard": 0.25}
BASE_SPEED = 1.0   # m/s


def speed_at(site: Site, p: np.ndarray) -> float:
    f = 1.0
    for r in site.regions:
        if r.xmin <= p[0] <= r.xmax and r.ymin <= p[1] <= r.ymax:
            f = min(f, REGION_SPEED[r.kind])
    return BASE_SPEED * f


def travel_time(site: Site, a: np.ndarray, b: np.ndarray, step: float = 1.0) -> float:
    """Straight-line travel time, sampling terrain speed every `step` metres."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.linalg.norm(b - a))
    if d == 0.0:
        return 0.0
    n = max(1, math.ceil(d / step))
    mids = a + (np.arange(n) + 0.5)[:, None] / n * (b - a)
    return float(sum((d / n) / speed_at(site, m) for m in mids))
