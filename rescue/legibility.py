"""Legibility (Q5): a cost term for the planner and simulated observers to evaluate it.

Following Dragan et al. (legible vs. predictable motion), an observer who assumes the robot moves
efficiently toward its goal infers the goal G from the motion so far, xi_{S->Q}:

    P(G | xi) ~ exp(-b [C(xi) + d(Q,G)]) / exp(-b d(S,G)),   normalised over the candidate goals.

C(xi) is common to all goals, so it cancels and the posterior depends on the current point Q only:

    log P(G* | Q) = -b (d(Q,G*) - d(S,G*)) - logsumexp_g [ -b (d(Q,g) - d(S,g)) ].

The planner's legibility cost is -log P(G* | Q) along a candidate rollout. The evaluation observers
below are *simulated* people: ``CostObserver`` uses the same family of model as the cost (with a
different temperature and noisy position observations), while ``HeadingObserver`` ignores path cost
and just extrapolates the robot's recent heading, so a result that holds for both is not just the
planner optimising its own model.
"""
from __future__ import annotations

import numpy as np


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def log_prob_goal(Q, S, goals, k: int, beta: float) -> np.ndarray:
    """log P(goal k | robot at Q, started at S). Q has shape (..., 2); goals (K, 2)."""
    Q, S, goals = np.asarray(Q, float), np.asarray(S, float), np.asarray(goals, float)
    dQ = np.linalg.norm(Q[..., None, :] - goals, axis=-1)
    dS = np.linalg.norm(S - goals, axis=-1)
    score = -beta * (dQ - dS)
    return score[..., k] - _logsumexp(score)


class CostObserver:
    """Bayesian observer with a cost-based (Dragan-style) model and noisy robot-position observations."""

    def __init__(self, beta: float = 0.7, obs_noise: float = 0.3):
        self.beta, self.obs_noise = beta, obs_noise

    def posterior(self, path: np.ndarray, S, goals, rng: np.random.Generator) -> np.ndarray:
        """(N, K) posterior over goals after each point of ``path`` (N, 2)."""
        goals = np.asarray(goals, float)
        seen = path + rng.normal(0.0, self.obs_noise, path.shape)
        dQ = np.linalg.norm(seen[:, None, :] - goals, axis=-1)
        dS = np.linalg.norm(np.asarray(S, float) - goals, axis=-1)
        score = -self.beta * (dQ - dS)
        return np.exp(score - _logsumexp(score)[:, None])


class HeadingObserver:
    """Observer that extrapolates the robot's heading over the last ``window`` samples."""

    def __init__(self, kappa: float = 4.0, window: int = 10, obs_noise: float = 0.1):
        self.kappa, self.window, self.obs_noise = kappa, window, obs_noise

    def posterior(self, path: np.ndarray, S, goals, rng: np.random.Generator) -> np.ndarray:
        goals = np.asarray(goals, float)
        K, N = len(goals), len(path)
        seen = path + rng.normal(0.0, self.obs_noise, path.shape)
        out = np.full((N, K), 1.0 / K)
        for n in range(self.window, N):
            v = seen[n] - seen[n - self.window]
            sp = float(np.hypot(v[0], v[1]))
            if sp < 0.1 * self.window * 0.1:      # < 0.1 m/s over the window: no heading information
                continue
            to_g = goals - seen[n]
            cos = (to_g @ v) / (np.linalg.norm(to_g, axis=1) * sp + 1e-9)
            w = np.exp(self.kappa * cos)
            out[n] = w / w.sum()
        return out


def identification_time(post: np.ndarray, k: int, dt: float, thresh: float = 0.9) -> float:
    """First time after which P(true goal) stays >= thresh; inf if it never does."""
    ok = post[:, k] >= thresh
    if not ok[-1]:
        return float("inf")
    bad = np.where(~ok)[0]
    return float((bad[-1] + 1 if len(bad) else 0) * dt)
