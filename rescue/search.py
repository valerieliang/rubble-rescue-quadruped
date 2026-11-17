"""Greedy search layer (~1 Hz): picks the next waypoint from the belief.

mode="priority": score = r * w * exp(-lambda * t_arrive) / (tau + 1)  (priority-weighted, urgency-discounted)
mode="nearest":  score = -tau                                         (baseline 2)
With no viable track left (or no prior), falls back to lawnmower coverage (baseline 1).
"""
from __future__ import annotations

import numpy as np

from .belief import Belief
from .drone_prior import PriorConfig
from .interfaces import (PRIORITY_DECAY, PRIORITY_WEIGHT, Detection, Observation,
                         TravelTime)


class Coverage:
    """Lawnmower sweep over the site bounds with row spacing just under the sensor diameter."""

    def __init__(self, bounds, sensor_radius: float, start: np.ndarray):
        xmin, ymin, xmax, ymax = bounds
        xa = min(xmin + sensor_radius, (xmin + xmax) / 2)
        xb = max(xmax - sensor_radius, (xmin + xmax) / 2)
        ys = np.arange(ymin + sensor_radius, ymax, 1.8 * sensor_radius)
        if len(ys) == 0:
            ys = np.array([(ymin + ymax) / 2])
        self.wps = []
        for k, y in enumerate(ys):
            row = [(xa, y), (xb, y)]
            self.wps += row[::-1] if k % 2 else row
        self.wps = np.array(self.wps)
        self.idx = int(np.argmin(np.linalg.norm(self.wps - np.asarray(start), axis=1)))

    def waypoint(self, pos: np.ndarray) -> np.ndarray:
        if np.linalg.norm(self.wps[self.idx] - pos) < 1.0:
            self.idx = (self.idx + 1) % len(self.wps)
        return self.wps[self.idx]


class PriorSearch:
    def __init__(self, detections: list[Detection], cfg: PriorConfig, bounds,
                 travel_time: TravelTime, sensor_radius: float, start: np.ndarray,
                 horizon: float, mode: str = "priority", r_min: float = 0.1,
                 dist_exp: float = 1.0):
        assert mode in ("priority", "nearest")
        self.belief = Belief(detections, cfg)
        self.coverage = Coverage(bounds, sensor_radius, start)
        self.travel_time, self.horizon, self.mode, self.r_min = travel_time, horizon, mode, r_min
        self.dist_exp = dist_exp   # exponent on (tau+1) in the priority score; 1 = reward per second

    def clearance_regions(self, confidence: float = 0.95, keepout: float | None = None):
        """Keep-out regions for the current belief, passed to the approach planner / safety filter.
        Uses the search layer's own existence threshold so both layers see the same tracks."""
        return self.belief.clearance_regions(confidence, keepout, self.r_min)

    def act(self, obs: Observation) -> np.ndarray:
        pos = np.asarray(obs.robot_pos, float)
        for p in obs.rescued:
            self.belief.remove_near(p)
        self.belief.update(pos, obs.sensor_radius, list(obs.detected))
        best, best_score = None, -np.inf
        for t in self.belief.tracks:
            if t.r < self.r_min:
                continue
            tau = self.travel_time(pos, t.mean)
            arrive = obs.t + tau
            if arrive > self.horizon:
                continue
            if self.mode == "priority":
                score = (t.r * PRIORITY_WEIGHT[t.priority]
                         * np.exp(-PRIORITY_DECAY[t.priority] * arrive) / (tau + 1.0) ** self.dist_exp)
            else:
                score = -tau
            if score > best_score:
                best, best_score = t.mean, score
        return best if best is not None else self.coverage.waypoint(pos)
