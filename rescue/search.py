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


class ObservedMap:
    """Which parts of the site the robot's sensor has covered so far (boolean grid, 1 m cells)."""

    def __init__(self, bounds, cell: float = 1.0, block: float = 5.0):
        self.xmin, self.ymin, xmax, ymax = bounds
        self.cell = cell
        self.nx, self.ny = int(np.ceil((xmax - self.xmin) / cell)), int(np.ceil((ymax - self.ymin) / cell))
        self.gx = self.xmin + cell * (np.arange(self.nx) + 0.5)[:, None] * np.ones((1, self.ny))
        self.gy = self.ymin + cell * (np.arange(self.ny) + 0.5)[None, :] * np.ones((self.nx, 1))
        self.seen = np.zeros((self.nx, self.ny), bool)
        self.bs = max(1, int(round(block / cell)))     # block size in cells, for the unobserved-region search

    def mark(self, pos, radius: float) -> None:
        self.seen |= (self.gx - pos[0]) ** 2 + (self.gy - pos[1]) ** 2 <= radius**2

    def disc(self, center, radius: float) -> np.ndarray:
        return (self.gx - center[0]) ** 2 + (self.gy - center[1]) ** 2 <= radius**2

    def frac_seen(self, center, radius: float) -> float:
        m = self.disc(center, radius)
        n = int(m.sum())
        return float(self.seen[m].mean()) if n else 1.0

    def nearest_unseen(self, center, radius: float, pos):
        """Centre of the unobserved cell inside the disc that is nearest to ``pos`` (None if all seen)."""
        m = self.disc(center, radius) & ~self.seen
        if not m.any():
            return None
        d2 = (self.gx[m] - pos[0]) ** 2 + (self.gy[m] - pos[1]) ** 2
        k = int(np.argmin(d2))
        return np.array([self.gx[m][k], self.gy[m][k]])

    def best_block(self, pos, min_unseen: float = 0.2):
        """Centre of the block that offers the most unobserved cells per metre of travel, or None."""
        bs = self.bs
        nbx, nby = int(np.ceil(self.nx / bs)), int(np.ceil(self.ny / bs))
        pad = np.zeros((nbx * bs, nby * bs))
        pad[:self.nx, :self.ny] = ~self.seen
        cnt = pad.reshape(nbx, bs, nby, bs).sum(axis=(1, 3))
        cx = self.xmin + self.cell * bs * (np.arange(nbx) + 0.5)
        cy = self.ymin + self.cell * bs * (np.arange(nby) + 0.5)
        d = np.hypot(cx[:, None] - pos[0], cy[None, :] - pos[1])
        val = np.where(cnt >= min_unseen * bs * bs, cnt / (d + 10.0), -1.0)
        if val.max() <= 0:
            return None
        i, j = np.unravel_index(int(np.argmax(val)), val.shape)
        return np.array([cx[i], cy[j]])


class PriorSearch:
    """``retention``: comma-separated remedies for the loss of reported survivors (all off by default).
      resume  when the policy falls back to the lawnmower sweep, restart it at the waypoint nearest the
              robot and skip waypoints whose surroundings are already observed
      region  when no track is viable, go to the block with the most unobserved area per metre of travel
              instead of following the fixed sweep
      keep    keep an unconfirmed track viable until its ``keep_k``-sigma region has been searched, and
              search that region (nearest unobserved point) rather than only its mean
    """

    def __init__(self, detections: list[Detection], cfg: PriorConfig, bounds,
                 travel_time: TravelTime, sensor_radius: float, start: np.ndarray,
                 horizon: float, mode: str = "priority", r_min: float = 0.1,
                 dist_exp: float = 1.0, retention: str = "", keep_k: float = 2.0):
        assert mode in ("priority", "nearest")
        self.flags = {f for f in retention.replace("+", ",").split(",") if f}
        assert self.flags <= {"resume", "region", "keep"}, retention
        self.keep_k, self.sensor_radius = keep_k, sensor_radius
        self.obs_map = ObservedMap(bounds) if self.flags else None
        self._in_fallback, self._region_wp = False, None
        self.belief = Belief(detections, cfg)
        self.coverage = Coverage(bounds, sensor_radius, start)
        self.travel_time, self.horizon, self.mode, self.r_min = travel_time, horizon, mode, r_min
        self.dist_exp = dist_exp   # exponent on (tau+1) in the priority score; 1 = reward per second

    def clearance_regions(self, confidence: float = 0.95, keepout: float | None = None):
        """Keep-out regions for the current belief, passed to the approach planner / safety filter.
        Uses the search layer's own existence threshold so both layers see the same tracks."""
        return self.belief.clearance_regions(confidence, keepout, self.r_min)

    # ---- retention remedies ------------------------------------------------------------------
    def _retain(self) -> None:
        """keep: an unconfirmed track stays viable (r held at r_min) until its keep_k-sigma region is searched."""
        for t in self.belief.tracks:
            if t.confirmed or t.r >= self.r_min:
                continue
            if self.obs_map.frac_seen(t.mean, self.keep_k * t.sigma) < 0.9:
                t.r = self.r_min
            else:
                t.r = min(t.r, 0.99 * self.r_min)

    def _target_point(self, t, pos) -> np.ndarray:
        if "keep" in self.flags and not t.confirmed:
            R = self.keep_k * t.sigma
            if self.obs_map.frac_seen(t.mean, 1.0) >= 1.0 and self.obs_map.frac_seen(t.mean, R) < 0.9:
                q = self.obs_map.nearest_unseen(t.mean, R, pos)
                if q is not None:
                    return q
        return t.mean

    def _fallback(self, pos) -> np.ndarray:
        om = self.obs_map
        if "region" in self.flags:
            wp = self._region_wp
            if (wp is None or np.linalg.norm(wp - pos) < 1.0 or om.frac_seen(wp, om.bs * om.cell / 2) > 0.9
                    or not self._in_fallback):
                wp = self._region_wp = om.best_block(pos)
            if wp is not None:
                return wp
        cov = self.coverage
        if "resume" in self.flags:
            if not self._in_fallback:
                cov.idx = int(np.argmin(np.linalg.norm(cov.wps - pos, axis=1)))
            wp = cov.waypoint(pos)
            for _ in range(len(cov.wps)):
                if om.frac_seen(wp, self.sensor_radius) <= 0.9:
                    break
                cov.idx = (cov.idx + 1) % len(cov.wps)
                wp = cov.wps[cov.idx]
            return wp
        return cov.waypoint(pos)

    def act(self, obs: Observation) -> np.ndarray:
        pos = np.asarray(obs.robot_pos, float)
        if self.obs_map is not None:
            self.obs_map.mark(pos, obs.sensor_radius)
        for p in obs.rescued:
            self.belief.remove_near(p)
        self.belief.update(pos, obs.sensor_radius, list(obs.detected))
        if "keep" in self.flags:
            self._retain()
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
                best, best_score = t, score
        if best is not None:
            self._in_fallback = False
            return self._target_point(best, pos)
        if self.obs_map is None:
            return self.coverage.waypoint(pos)
        wp = self._fallback(pos)
        self._in_fallback = True
        return wp
