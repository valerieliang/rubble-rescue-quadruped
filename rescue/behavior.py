"""Survivor behaviour inference (Q4): belief over still / approaching / retreating from a few
seconds of observed motion, and the induced belief over the hidden response state.

Observation model. The robot sees noisy survivor positions z_k = x_k + N(0, sigma_obs^2 I) at
times t_k. Over a short window the survivor moves at roughly constant velocity, so the
least-squares velocity estimate v_hat is a sufficient statistic: it is Gaussian with variance
sigma_obs^2 / sum_k (t_k - tbar)^2 per axis. Each mode m predicts a mean velocity mu_m along the
line to the robot (u = unit vector survivor -> robot):

    STILL:       mu = 0
    APPROACHING: mu = +speed * u
    RETREATING:  mu = -speed * u

with extra model variance: speed spread s_m along u (s_still when still, s_move when moving) and
sway s_still across it. The mode posterior is prior(m) * N(v_hat; mu_m, diag(var + s_m^2, var + s_still^2)). The posterior is recomputed over a sliding window, not
multiplied over steps, because successive velocity differences share measurement noise.

The response state z (unresponsive / trapped / mobile) is not observed directly; it generates
modes through ``mode_given_state``. Those probabilities, and ``speed`` and the variances, are
modelling assumptions, not fitted to data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .interfaces import ResponseState


class Behavior(IntEnum):
    STILL = 0
    APPROACHING = 1
    RETREATING = 2


@dataclass(frozen=True)
class BehaviorModel:
    speed: float = 0.5        # m/s radial speed of a moving survivor
    s_still: float = 0.05     # m/s model std of residual velocity when still
    s_move: float = 0.2       # m/s model std of speed when moving
    # Rows: ResponseState (UNRESPONSIVE, TRAPPED, MOBILE); columns: Behavior (STILL, APPROACHING, RETREATING).
    mode_given_state: np.ndarray = field(default_factory=lambda: np.array([
        [0.96, 0.02, 0.02],    # unresponsive: essentially motionless
        [0.80, 0.15, 0.05],    # trapped: may reach toward the robot, cannot leave
        [0.20, 0.50, 0.30],    # mobile: moves toward help or away from danger
    ]))
    prior_state: np.ndarray = field(default_factory=lambda: np.full(3, 1 / 3))

    @property
    def prior_mode(self) -> np.ndarray:
        return self.prior_state @ self.mode_given_state


class ModeBelief:
    """Recursive mode / state belief for one survivor from a stream of position observations."""

    def __init__(self, model: BehaviorModel | None = None, sigma_obs: float = 0.5, window: int = 6):
        if window < 2:
            raise ValueError("window must be at least 2 samples")
        self.model, self.sigma_obs, self.window = model or BehaviorModel(), sigma_obs, window
        self._t: list[float] = []
        self._z: list[np.ndarray] = []
        self._r: list[np.ndarray] = []
        self._v_hat = np.zeros(2)
        self._var_v = np.inf
        self._u = np.array([1.0, 0.0])

    def observe(self, t: float, z: np.ndarray, robot_pos: np.ndarray) -> None:
        """Add one survivor position observation z at time t (robot at robot_pos)."""
        self._t = (self._t + [float(t)])[-self.window:]
        self._z = (self._z + [np.asarray(z, float)])[-self.window:]
        self._r = (self._r + [np.asarray(robot_pos, float)])[-self.window:]
        if len(self._t) < 2:
            return
        t_arr, z_arr = np.array(self._t), np.array(self._z)
        dt = t_arr - t_arr.mean()
        stt = float(dt @ dt)
        self._v_hat = (dt[:, None] * (z_arr - z_arr.mean(axis=0))).sum(axis=0) / stt
        self._var_v = self.sigma_obs**2 / stt
        to_robot = np.mean(self._r, axis=0) - z_arr.mean(axis=0)
        n = float(np.linalg.norm(to_robot))
        if n > 1e-9:
            self._u = to_robot / n

    @property
    def n_obs(self) -> int:
        return len(self._t)

    def _mode_terms(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-mode (Mahalanobis^2 of the velocity residual, determinant of its covariance).

        The covariance is diagonal in the (along, perpendicular) frame of the line to the robot:
        speed spread s_m along it, only sway s_still across it, plus the estimation variance."""
        m = self.model
        perp = np.array([-self._u[1], self._u[0]])
        mus = np.array([np.zeros(2), m.speed * self._u, -m.speed * self._u])
        va = self._var_v + np.array([m.s_still, m.s_move, m.s_move]) ** 2
        vp = self._var_v + m.s_still**2
        e = self._v_hat - mus
        return (e @ self._u) ** 2 / va + (e @ perp) ** 2 / vp, va * vp

    @property
    def mode_posterior(self) -> np.ndarray:
        """P(mode | window), ordered as Behavior. The prior until two observations are in."""
        prior = self.model.prior_mode
        if self.n_obs < 2:
            return prior.copy()
        quad, det = self._mode_terms()
        logp = np.log(prior) - quad / 2 - np.log(det) / 2
        p = np.exp(logp - logp.max())
        return p / p.sum()

    @property
    def state_posterior(self) -> np.ndarray:
        """P(state | window) over ResponseState, marginalizing the mode: p(z) * sum_m p(m|z) L_m."""
        m = self.model
        if self.n_obs < 2:
            return m.prior_state.copy()
        quad, det = self._mode_terms()
        like = np.exp(-quad / 2) / np.sqrt(det)
        p = m.prior_state * (m.mode_given_state @ like)
        return p / p.sum()

    @property
    def surprise(self) -> float:
        """Smallest Mahalanobis^2 of the velocity residual over modes: about chi-square with 2
        degrees of freedom when the motion fits a mode (99th percentile 9.2); large values mean
        none of the modelled behaviours explains it (e.g. movement across the line to the robot)."""
        if self.n_obs < 2:
            return 0.0
        return float(np.min(self._mode_terms()[0]))

    def map_mode(self) -> Behavior:
        return Behavior(int(np.argmax(self.mode_posterior)))

    def map_state(self) -> ResponseState:
        return ResponseState(int(np.argmax(self.state_posterior)))


def synthetic_track(mode: Behavior, rng: np.random.Generator, n: int = 8, dt: float = 1.0,
                    sigma_obs: float = 0.5, speed: float = 0.5, start_dist: float = 8.0,
                    lateral: bool = False, speed_std: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stand-in survivor motion for offline Q4 evaluation: (times (n,), positions (n,2), robot (n,2)).

    The survivor starts at the origin with the robot ``start_dist`` away along +x. Approaching and
    retreating move along the line to the robot at ``speed``; ``lateral=True`` instead moves
    perpendicular to it (an out-of-model behaviour for robustness tests). A moving survivor's speed is
    drawn from N(speed, speed_std) (floored at 0.1 m/s) so tracks match the model's speed spread. Positions carry
    N(0, sigma_obs^2) observation noise. The simulator's survivor motion is the real source of tracks.
    """
    t = np.arange(n) * dt
    if speed_std > 0 and (lateral or mode != Behavior.STILL):
        speed = max(0.1, float(rng.normal(speed, speed_std)))
    along = {Behavior.STILL: 0.0, Behavior.APPROACHING: speed, Behavior.RETREATING: -speed}[mode]
    direction = np.array([0.0, 1.0]) if lateral else np.array([1.0, 0.0])
    true = (t[:, None] * (speed if lateral else along)) * direction
    robot = np.tile([start_dist, 0.0], (n, 1))
    return t, true + rng.normal(0.0, sigma_obs, true.shape), robot
