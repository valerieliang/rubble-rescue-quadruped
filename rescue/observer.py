"""Observer model for legibility (Q5): how quickly can a watcher tell which survivor the robot is
heading to?

The observer holds a belief over the candidate goals g and assumes the robot is roughly
cost-efficient (Dragan and Srinivasa, 2013): the likelihood of a trajectory prefix from S to the
current point Q is proportional to exp(-beta * (C(prefix) + C*(Q, g))) / exp(-beta * C*(S, g)), the
cost of the detour the prefix implies for goal g relative to going straight there. Cost is travel
time, so terrain (rubble, hazards) shapes what looks efficient. beta is the observer's
rationality: large beta means a small detour already rules a goal out.

The metrics for Q5 are the time until the true goal's posterior crosses a threshold, and the
extra travel time the robot spends being legible (path time over the direct travel time).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .interfaces import TravelTime


def path_cost(points: np.ndarray, travel: TravelTime) -> float:
    """Travel time along the polyline through ``points`` (n, 2)."""
    pts = np.asarray(points, float)
    return float(sum(travel(a, b) for a, b in zip(pts[:-1], pts[1:])))


def goal_posterior(prefix: np.ndarray, goals: Sequence[np.ndarray], travel: TravelTime,
                   beta: float = 0.5, prior: np.ndarray | None = None) -> np.ndarray:
    """P(goal | trajectory prefix). ``prefix`` is (n, 2), the first row the start S."""
    prefix = np.asarray(prefix, float)
    k = len(goals)
    logp = np.log(np.full(k, 1.0 / k) if prior is None else np.asarray(prior, float))
    if len(prefix) >= 2:
        done = path_cost(prefix, travel)
        s, q = prefix[0], prefix[-1]
        # Extra time vs. the direct route S -> g, if the robot had been heading to g all along.
        detour = np.array([done + travel(q, g) - travel(s, g) for g in goals])
        logp = logp - beta * detour
    p = np.exp(logp - logp.max())
    return p / p.sum()


def time_to_identify(trajectory: np.ndarray, goals: Sequence[np.ndarray], true_goal: int,
                     travel: TravelTime, beta: float = 0.5, threshold: float = 0.8,
                     prior: np.ndarray | None = None) -> float:
    """First time the observer's posterior on the true goal reaches ``threshold`` and stays
    there for the rest of the trajectory; inf if it never does. ``trajectory`` is (n, 3): t, x, y."""
    traj = np.asarray(trajectory, float)
    hit = np.array([goal_posterior(traj[:k + 1, 1:], goals, travel, beta, prior)[true_goal] >= threshold
                    for k in range(len(traj))])
    for k in range(len(hit)):
        if hit[k:].all():
            return float(traj[k, 0])
    return float("inf")


def legibility_cost(trajectory: np.ndarray, travel: TravelTime) -> float:
    """Path travel time divided by the direct start -> end travel time (1 = no cost of legibility)."""
    pts = np.asarray(trajectory, float)[:, 1:]
    direct = travel(pts[0], pts[-1])
    return path_cost(pts, travel) / direct if direct > 0 else 1.0
