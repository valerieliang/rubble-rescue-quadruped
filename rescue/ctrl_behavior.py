"""Survivor behavior model and online state inference (Q4).

Hidden state z in {UNRESPONSIVE, TRAPPED, MOBILE}. Only MOBILE survivors move. A mobile survivor is
APPROACHING the robot or RETREATING from it (frightened) once the robot is within ``aware_radius``.
``Mode.ERRATIC`` is an out-of-model behavior (a random walk independent of the robot) used only to
test robustness: the inference model below does not contain it.

The robot's belief is one categorical distribution over four joint hypotheses (z, mode):

    (UNRESPONSIVE, STILL), (TRAPPED, STILL), (MOBILE, APPROACHING), (MOBILE, RETREATING)

updated from two noisy cues:
  * prompt responses (answer / wave / nothing), whose reliability is ``q``;
  * survivor velocity over a few seconds, resolved in the robot-survivor frame
    (radial = toward the robot, tangential = across), with Gaussian likelihoods.

Everything here is a crude stand-in for human behavior (see the README's limitations).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

from .interfaces import STATE_FACTOR, Priority, ResponseState, Site, Survivor
from .toy_sim import generate_site


class Mode(IntEnum):
    STILL = 0
    APPROACHING = 1
    RETREATING = 2
    ERRATIC = 3   # out-of-model


HYPOTHESES = ((ResponseState.UNRESPONSIVE, Mode.STILL), (ResponseState.TRAPPED, Mode.STILL),
              (ResponseState.MOBILE, Mode.APPROACHING), (ResponseState.MOBILE, Mode.RETREATING))
RESPONSES = ("answer", "wave", "nothing")


@dataclass(frozen=True)
class BehaviorConfig:
    walk_speed: float = 0.3        # m/s for mobile survivors
    aware_radius: float = 12.0     # m; mobile survivors react to the robot inside this range
    approach_stop: float = 2.0     # m; approaching survivors stop this far from the robot
    p_oom: float = 0.0             # fraction of mobile survivors that move erratically (out of model)
    reliability: float = 0.8       # q: probability a responsive survivor answers/waves when prompted


def assign_modes(site: Site, cfg: BehaviorConfig, rng: np.random.Generator) -> dict[int, Mode]:
    modes = {}
    for s in site.survivors:
        if s.state != ResponseState.MOBILE:
            modes[s.id] = Mode.STILL
        elif rng.random() < cfg.p_oom:
            modes[s.id] = Mode.ERRATIC
        else:
            modes[s.id] = Mode.APPROACHING if rng.random() < 0.5 else Mode.RETREATING
    return modes


def _in_hazard(site: Site, p) -> bool:
    return any(r.kind == "hazard" and r.xmin <= p[0] <= r.xmax and r.ymin <= p[1] <= r.ymax
               for r in site.regions)


def step_survivors(site: Site, pos: dict, modes: dict, heading: dict, active: set, robot_pos: np.ndarray,
                   dt: float, cfg: BehaviorConfig, rng: np.random.Generator) -> None:
    """Advance every survivor in ``active`` (not yet rescued) by dt, in place.

    The erratic heading noise is drawn for every erratic survivor whether or not it is active, so the
    random stream does not depend on what the robot does.
    """
    xmin, ymin, xmax, ymax = site.bounds
    for s in site.survivors:
        m = modes[s.id]
        noise = rng.normal() if m == Mode.ERRATIC else 0.0
        if s.id not in active or m == Mode.STILL:
            continue
        p = pos[s.id]
        if m == Mode.ERRATIC:
            heading[s.id] += noise * np.sqrt(dt)
            new = p + cfg.walk_speed * dt * np.array([np.cos(heading[s.id]), np.sin(heading[s.id])])
        else:
            d = robot_pos - p
            dist = float(np.hypot(d[0], d[1]))
            if dist > cfg.aware_radius or dist < 1e-9:
                continue
            u = d / dist
            if m == Mode.APPROACHING:
                step = min(cfg.walk_speed * dt, dist - cfg.approach_stop)
                if step <= 0:
                    continue
                new = p + u * step
            else:
                new = p - u * cfg.walk_speed * dt
        new = np.array([np.clip(new[0], xmin, xmax), np.clip(new[1], ymin, ymax)])
        if not _in_hazard(site, new):
            pos[s.id] = new


def response_probs(z: ResponseState, q: float) -> np.ndarray:
    """P(answer, wave, nothing | z). Silence is evidence: unresponsive survivors almost never respond."""
    if z == ResponseState.UNRESPONSIVE:
        return np.array([0.02, 0.02, 0.96])
    if z == ResponseState.TRAPPED:
        return np.array([0.3 * q, 0.7 * q, 1.0 - q])
    return np.array([0.7 * q, 0.3 * q, 1.0 - q])


def sample_response(z: ResponseState, q: float, rng: np.random.Generator) -> int:
    return int(rng.choice(3, p=response_probs(z, q)))


class StateBelief:
    """Categorical belief over the four (z, mode) hypotheses."""

    PRIOR = np.array([1 / 3, 1 / 3, 1 / 6, 1 / 6])

    def __init__(self):
        self.p = self.PRIOR.copy()

    def _mul(self, lik: np.ndarray) -> None:
        p = self.p * lik
        s = p.sum()
        p = p / s if s > 0 else self.PRIOR.copy()
        p = np.maximum(p, 1e-6)          # never absorbing: a later cue can still overturn it
        self.p = p / p.sum()

    def update_prompt(self, response: int, q: float) -> None:
        self._mul(np.array([response_probs(z, q)[response] for z, _ in HYPOTHESES]))

    def update_motion(self, vel_radial: np.ndarray, sigma_v: float, v0: float) -> None:
        """vel_radial = (toward-robot, across) velocity estimate in m/s."""
        means = np.array([[0.0, 0.0], [0.0, 0.0], [v0, 0.0], [-v0, 0.0]])
        ll = -0.5 * np.sum((np.asarray(vel_radial, float) - means) ** 2, axis=1) / sigma_v**2
        self._mul(np.exp(ll - ll.max()))

    def reveal(self, z: ResponseState) -> None:
        """Communication oracle (baseline 5): the survivor's z becomes known."""
        self._mul(np.array([1.0 if h[0] == z else 0.0 for h in HYPOTHESES]))

    def p_state(self) -> np.ndarray:
        return np.array([self.p[0], self.p[1], self.p[2] + self.p[3]])

    def p_mode(self) -> np.ndarray:
        return np.array([self.p[0] + self.p[1], self.p[2], self.p[3]])

    def map_state(self) -> ResponseState:
        return ResponseState(int(np.argmax(self.p_state())))

    def map_mode(self) -> Mode:
        return Mode(int(np.argmax(self.p_mode())))

    def expected_factor(self) -> float:
        ps = self.p_state()
        return float(sum(ps[int(z)] * f for z, f in STATE_FACTOR.items()))


class MotionTracker:
    """Rolling buffer of a track's measured positions; least-squares velocity over the window."""

    def __init__(self):
        self.buf: list[tuple[float, float, float]] = []

    def add(self, t: float, p) -> None:
        self.buf.append((t, float(p[0]), float(p[1])))

    def clear(self) -> None:
        self.buf = []

    def ready(self, window: float, min_n: int = 20) -> bool:
        return len(self.buf) >= min_n and self.buf[-1][0] - self.buf[0][0] >= 0.9 * window

    def velocity(self, sigma_meas: float) -> tuple[np.ndarray, float]:
        a = np.array(self.buf)
        t = a[:, 0] - a[:, 0].mean()
        sxx = float(np.sum(t * t))
        v = np.array([np.sum(t * (a[:, 1] - a[:, 1].mean())), np.sum(t * (a[:, 2] - a[:, 2].mean()))]) / sxx
        return v, sigma_meas / np.sqrt(sxx)


def make_site(seed: int, n_survivors: int | None = None, frac_mobile: float | None = None) -> Site:
    """The standard seeded site, optionally with a fixed survivor count / mobile fraction (ablations)."""
    site = generate_site(seed)
    if n_survivors is None and frac_mobile is None:
        return site
    rng = np.random.default_rng(seed + 10_001)
    n = len(site.survivors) if n_survivors is None else n_survivors
    hazards = [r for r in site.regions if r.kind == "hazard"]
    size = site.bounds[2]
    n_mobile = None if frac_mobile is None else int(round(frac_mobile * n))
    mobile_ids = set(rng.permutation(n)[:n_mobile].tolist()) if n_mobile is not None else None
    survivors = []
    for i in range(n):
        while True:
            p = rng.uniform(5, size - 5, 2)
            if not any(h.xmin <= p[0] <= h.xmax and h.ymin <= p[1] <= h.ymax for h in hazards):
                break
        pri = Priority(rng.choice(3, p=[0.4, 0.35, 0.25]))
        if mobile_ids is None:
            state = ResponseState(rng.integers(3))
        elif i in mobile_ids:
            state = ResponseState.MOBILE
        else:
            state = ResponseState(rng.integers(2))
        survivors.append(Survivor(i, p, pri, state))
    return replace(site, survivors=tuple(survivors))
