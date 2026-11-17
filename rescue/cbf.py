"""Control-barrier-function safety filter (Q3), applied to the planner's command every control step.

Safe set: outside every hazard rectangle (grown by ``hazard_margin``) and outside the keep-out disc
of every survivor clearance region (``clearance.ClearanceRegion``), except regions the caller
marks as the current approach target.

For the unicycle (x' = v cos(th), y' = v sin(th), th' = omega) every barrier h depends on position
only, so h' = (grad h . d) v with d = (cos th, sin th): relative degree one in v and no dependence on
omega. The CBF condition h' >= -alpha h is therefore one linear constraint on v per barrier, and
the QP

    min (v - v_nom)^2 + (omega - omega_nom)^2   s.t.   a_i v >= -alpha h_i,  0 <= v <= v_max,
                                                          |omega| <= omega_max

separates: omega is untouched and v is v_nom clipped to the interval the constraints define. That
is its exact solution, so no numerical QP solver is needed. A barrier pointing the robot straight
at an obstacle forces v to zero at the boundary; with ``slide=True`` the filter then turns the robot
along the boundary toward the goal instead of leaving it stuck (a standard CBF deadlock remedy).

Barriers are evaluated at the robot's estimated position and ignore braking distance beyond the
``brake_margin`` added to every barrier, so the guarantee holds for the commanded velocity, not
for the rate-limited velocity the real robot follows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clearance import ClearanceRegion
from .interfaces import Region


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def rect_sd(points: np.ndarray, r: Region) -> np.ndarray:
    """Signed distance from points (..., 2) to a rectangle: > 0 outside, < 0 inside."""
    c = np.array([(r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2])
    half = np.array([(r.xmax - r.xmin) / 2, (r.ymax - r.ymin) / 2])
    q = np.abs(np.asarray(points, float) - c) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)


def rect_sd_grad(p: np.ndarray, r: Region) -> np.ndarray:
    """Gradient of the signed distance at p (unit vector pointing out of the rectangle)."""
    c = np.array([(r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2])
    half = np.array([(r.xmax - r.xmin) / 2, (r.ymax - r.ymin) / 2])
    d = np.asarray(p, float) - c
    sign = np.where(d >= 0, 1.0, -1.0)
    q = np.abs(d) - half
    if q[0] > 0 and q[1] > 0:                     # nearest feature is a corner
        return sign * q / np.linalg.norm(q)
    g = np.zeros(2)
    k = 0 if q[0] > q[1] else 1                    # nearest edge (outside one side, or inside)
    g[k] = sign[k]
    return g


@dataclass(frozen=True)
class Barrier:
    h: float                  # barrier value at the robot (>= 0 safe)
    grad: np.ndarray          # (2,) gradient of h at the robot


def barriers(pos: np.ndarray, hazards: list[Region], regions: list[ClearanceRegion],
             hazard_margin: float, brake_margin: float, target: np.ndarray | None = None,
             target_keepout: float = 0.0) -> list[Barrier]:
    """All active barriers at pos. A region whose centre is the approach ``target`` is replaced by a
    small physical keep-out disc (``target_keepout``), since the robot must get close to it."""
    out = []
    for r in hazards:
        out.append(Barrier(float(rect_sd(pos, r)) - hazard_margin - brake_margin, rect_sd_grad(pos, r)))
    for reg in regions:
        R = reg.radius
        if target is not None and np.allclose(reg.center, target):
            R = target_keepout
        d = np.asarray(pos, float) - reg.center
        dist = float(np.linalg.norm(d))
        # Distance-form barrier h = |p - c| - R (same zero set as the squared form, unit gradient).
        out.append(Barrier(dist - R - brake_margin, d / dist if dist > 1e-9 else np.array([1.0, 0.0])))
    return out


class SafetyFilter:
    def __init__(self, alpha: float = 1.0, hazard_margin: float = 0.3, brake_margin: float = 0.3,
                 v_max: float = 1.0, omega_max: float = 1.5, slide: bool = True,
                 target_keepout: float = 1.0):
        self.alpha, self.hazard_margin, self.brake_margin = alpha, hazard_margin, brake_margin
        self.v_max, self.omega_max, self.slide = v_max, omega_max, slide
        self.target_keepout = target_keepout
        self.n_interventions = 0      # control steps where the filter changed the command

    def filter(self, pos: np.ndarray, heading: float, v_nom: float, omega_nom: float,
               hazards: list[Region] = (), regions: list[ClearanceRegion] = (),
               target: np.ndarray | None = None, goal: np.ndarray | None = None) -> tuple[float, float]:
        """Return the safe (v, omega) closest to the nominal command."""
        d = np.array([np.cos(heading), np.sin(heading)])
        lo, hi, binding = 0.0, self.v_max, None
        for b in barriers(pos, list(hazards), list(regions), self.hazard_margin, self.brake_margin,
                          target, self.target_keepout):
            a = float(b.grad @ d)
            rhs = -self.alpha * b.h                 # need a * v >= rhs
            if abs(a) < 1e-9:                       # moving tangentially: speed does not change h
                continue
            if a > 0:
                lo = max(lo, rhs / a)               # inside the barrier: must move out at least this fast
            else:
                cap = rhs / a                       # = alpha h / |a|; negative when h < 0
                if cap < hi:
                    hi, binding = cap, b
        v = float(np.clip(v_nom, lo, hi)) if lo <= hi else max(0.0, hi)   # conflict: safety wins
        v = float(np.clip(v, 0.0, self.v_max))
        omega = float(np.clip(omega_nom, -self.omega_max, self.omega_max))
        if self.slide and binding is not None and v < 0.25 * max(v_nom, 1e-9):
            omega = self._slide_turn(heading, binding, pos, goal, omega)
        if abs(v - v_nom) > 1e-9 or abs(omega - omega_nom) > 1e-9:
            self.n_interventions += 1
        return v, omega

    def _slide_turn(self, heading: float, b: Barrier, pos: np.ndarray, goal: np.ndarray | None,
                    omega_nom: float) -> float:
        """Turn toward the boundary tangent that points closer to the goal."""
        t1 = np.array([-b.grad[1], b.grad[0]])
        choice = t1
        if goal is not None:
            choice = t1 if float(t1 @ (np.asarray(goal, float) - pos)) >= 0 else -t1
        err = _wrap(np.arctan2(choice[1], choice[0]) - heading)
        return float(np.clip(2.0 * err, -self.omega_max, self.omega_max))
