"""Belief over survivor locations, built from the drone prior and updated by Bayes.

Each drone detection becomes a track: an isotropic Gaussian over position plus an
existence probability r (it may be a false positive). Robot observations update tracks:
  * a detection inside the footprint -> Kalman position update, r raised;
  * no detection while the footprint covers mass of the track -> r lowered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ncx2

from .drone_prior import PriorConfig
from .interfaces import Detection, Priority


@dataclass
class Track:
    mean: np.ndarray
    sigma: float               # isotropic std (m)
    r: float                   # probability this track is a real survivor
    priority: Priority
    confirmed: bool = False    # a robot detection has matched it


def disc_mass(mean: np.ndarray, sigma: float, center: np.ndarray, radius: float) -> float:
    """P(x in disc) for x ~ N(mean, sigma^2 I); exact via noncentral chi-square."""
    d2 = float(np.sum((mean - center) ** 2))
    return float(ncx2.cdf((radius / sigma) ** 2, 2, d2 / sigma**2))


class Belief:
    def __init__(self, detections: list[Detection], cfg: PriorConfig,
                 p_detect: float = 0.9, sigma_meas: float = 0.5):
        self.cfg = cfg
        self.p_detect = p_detect       # robot sensor detection probability inside footprint
        self.sigma_meas = sigma_meas   # robot sensor position noise (m)
        # Initial existence: share of detections expected to be real.
        n = max(len(detections), 1)
        r0 = float(np.clip(1.0 - cfg.n_false_pos / n, 0.05, 0.99))
        self.tracks = [Track(np.array(d.pos, float), cfg.sigma_pos, r0, d.priority)
                       for d in detections]

    def update(self, robot_pos: np.ndarray, sensor_radius: float,
               detected: list[Detection], sigmas: list[float] | None = None) -> None:
        """Bayes update with one observation (footprint + any detections inside it).
        ``sigmas``: optional per-detection measurement std (e.g. a mean of several raw detections)."""
        robot_pos = np.asarray(robot_pos, float)
        matched: set[int] = set()
        for k, d in enumerate(detected):
            sm = self.sigma_meas if sigmas is None else sigmas[k]
            i = self._nearest_track(d.pos, robot_pos, sensor_radius, matched)
            if i is None:   # drone missed this survivor: start a confirmed track
                self.tracks.append(Track(np.array(d.pos, float), sm, 1.0,
                                         d.priority, confirmed=True))
                matched.add(len(self.tracks) - 1)
                continue
            matched.add(i)
            self._positive_update(self.tracks[i], d, sm)
        for i, t in enumerate(self.tracks):
            if i not in matched:
                self._negative_update(t, robot_pos, sensor_radius)

    def remove_near(self, pos: np.ndarray) -> None:
        """Drop the track of a survivor that has just been rescued at pos."""
        pos = np.asarray(pos, float)
        dists = [float(np.linalg.norm(t.mean - pos)) for t in self.tracks]
        if dists:
            i = int(np.argmin(dists))
            if dists[i] < 3 * self.tracks[i].sigma + 2.0:
                del self.tracks[i]

    def _nearest_track(self, pos, center, radius, exclude=()):
        best, best_d = None, np.inf
        for i, t in enumerate(self.tracks):
            if i in exclude:
                continue
            d = float(np.linalg.norm(t.mean - pos))
            if d < best_d and d < 3 * t.sigma + radius:
                best, best_d = i, d
        return best

    def _positive_update(self, t: Track, d: Detection, sigma_meas: float | None = None) -> None:
        # Kalman fuse (isotropic): prior N(mean, s^2) with measurement N(d.pos, sm^2).
        s2, m2 = t.sigma**2, (self.sigma_meas if sigma_meas is None else sigma_meas)**2
        k = s2 / (s2 + m2)
        t.mean = t.mean + k * (np.asarray(d.pos, float) - t.mean)
        t.sigma = float(np.sqrt((1 - k) * s2))
        t.r, t.confirmed, t.priority = 1.0, True, d.priority

    def _negative_update(self, t: Track, center: np.ndarray, radius: float) -> None:
        if t.confirmed:
            return
        # P(no detection | exists) = 1 - p_detect * mass in footprint; P(no det | not exists) = 1.
        miss = 1.0 - self.p_detect * disc_mass(t.mean, t.sigma, center, radius)
        t.r = t.r * miss / (t.r * miss + (1.0 - t.r))

    def clearance_regions(self, confidence: float = 0.95, keepout: float | None = None,
                          r_min: float = 0.1):
        """Confidence-ellipse keep-out region per track, for the safety filter (see clearance.py)."""
        from .clearance import clearance_regions
        kw = {} if keepout is None else {"keepout": keepout}
        return clearance_regions(self.tracks, confidence, r_min=r_min, **kw)

    def expected_survivors(self) -> float:
        return sum(t.r for t in self.tracks)
