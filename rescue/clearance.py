"""High-probability clearance regions from the belief (input to the CBF safety filter, Q3).

Each belief track is a Gaussian over the survivor's position, N(mean, cov). Its confidence
ellipse {x : (x - mean)^T cov^-1 (x - mean) <= k2}, with k2 the chi-square quantile for
``confidence`` at 2 dof, contains the survivor with that probability. The robot should stay out
of the ellipse grown by a physical keep-out margin, so the region widens as localization gets
worse. That is the quantity Q3 sweeps.

The safety filter wants a smooth barrier, so ``ClearanceRegion.circle`` returns the conservative
enclosing disc (centre, radius R) with ``h(x) = |x - c|^2 - R^2 >= 0`` safe; for the isotropic
tracks the belief currently produces, the disc and the ellipse coincide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .interfaces import SURVIVOR_KEEPOUT, Priority


def chi2_level(confidence: float) -> float:
    """k2 such that a 2-D Gaussian puts probability ``confidence`` inside its k2-ellipse."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return float(chi2.ppf(confidence, 2))


@dataclass(frozen=True)
class ClearanceRegion:
    center: np.ndarray        # (2,) belief mean
    cov: np.ndarray           # (2, 2) belief covariance
    confidence: float         # probability mass the ellipse covers
    keepout: float            # physical margin added around the ellipse (m)
    priority: Priority = Priority.STABLE
    existence: float = 1.0    # track existence probability r

    @property
    def k2(self) -> float:
        return chi2_level(self.confidence)

    @property
    def semi_axes(self) -> np.ndarray:
        """Confidence-ellipse semi-axes (m), largest first (no keep-out margin)."""
        return np.sqrt(self.k2 * np.sort(np.linalg.eigvalsh(self.cov))[::-1])

    @property
    def radius(self) -> float:
        """Conservative enclosing radius (m): largest semi-axis plus the keep-out margin."""
        return float(self.semi_axes[0] + self.keepout)

    def mahalanobis2(self, p: np.ndarray) -> float:
        d = np.asarray(p, float) - self.center
        return float(d @ np.linalg.solve(self.cov, d))

    def contains(self, p: np.ndarray) -> bool:
        """True if p is inside the confidence ellipse (the survivor may be there)."""
        return self.mahalanobis2(p) <= self.k2

    def circle(self) -> tuple[np.ndarray, float]:
        """(centre, radius) of the enclosing keep-out disc used for the barrier."""
        return self.center, self.radius

    def barrier(self, p: np.ndarray) -> float:
        """h(p) = |p - c|^2 - R^2; h >= 0 is safe, h < 0 is inside the keep-out disc."""
        d = np.asarray(p, float) - self.center
        return float(d @ d - self.radius**2)

    def barrier_grad(self, p: np.ndarray) -> np.ndarray:
        """Gradient of h at p (for the CBF constraint)."""
        return 2.0 * (np.asarray(p, float) - self.center)


def is_fresh(t, fresh_n: int) -> bool:
    """A track the robot has detected, but fewer than ``fresh_n`` times: its position is one or two noisy sightings."""
    return bool(getattr(t, "confirmed", False)) and 1 <= getattr(t, "n_det", 0) < fresh_n


def clearance_regions(tracks, confidence: float = 0.95, keepout: float = SURVIVOR_KEEPOUT,
                      r_min: float = 0.1, fresh_confidence: float = 0.0, fresh_n: int = 0) -> list[ClearanceRegion]:
    """One region per belief track with existence probability >= r_min. With ``fresh_confidence`` > 0, tracks
    the robot has detected fewer than ``fresh_n`` times use that (higher) confidence instead."""
    return [ClearanceRegion(np.array(t.mean, float), t.sigma**2 * np.eye(2),
                            fresh_confidence if fresh_confidence and is_fresh(t, fresh_n) else confidence, keepout,
                            t.priority, t.r)
            for t in tracks if t.r >= r_min]
