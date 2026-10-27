"""Policy factories for the compared methods. All take the drone detections and the site map."""
from __future__ import annotations

from functools import partial

from .drone_prior import PriorConfig
from .interfaces import Site
from .search import PriorSearch
from .terrain import travel_time


def _make(mode: str, use_prior: bool, site: Site, detections, cfg: PriorConfig,
          sensor_radius: float, horizon: float) -> PriorSearch:
    return PriorSearch(detections if use_prior else [], cfg, site.bounds,
                       partial(travel_time, site), sensor_radius, site.start, horizon, mode)


def priority_policy(site, detections, cfg, sensor_radius=5.0, horizon=600.0):
    """Ours: prior + priority-weighted, urgency-discounted greedy."""
    return _make("priority", True, site, detections, cfg, sensor_radius, horizon)


def nearest_first_policy(site, detections, cfg, sensor_radius=5.0, horizon=600.0):
    """Baseline 2: prior + nearest-first."""
    return _make("nearest", True, site, detections, cfg, sensor_radius, horizon)


def coverage_policy(site, detections, cfg, sensor_radius=5.0, horizon=600.0):
    """Baseline 1: no prior; lawnmower coverage, nearest-first on anything sensed."""
    return _make("nearest", False, site, detections, cfg, sensor_radius, horizon)
