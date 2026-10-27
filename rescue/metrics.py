"""Metrics: priority-weighted rescue score (oracle-normalized), time to critical, rescue rate within T."""
from __future__ import annotations

from functools import partial
from itertools import permutations

import numpy as np

from .interfaces import PRIORITY_DECAY, PRIORITY_WEIGHT, Priority, RunLog, Site
from .terrain import travel_time


def _reward(priority: Priority, tau: float, T: float) -> float:
    return PRIORITY_WEIGHT[priority] * np.exp(-PRIORITY_DECAY[priority] * tau) if tau <= T else 0.0


def rescue_score(site: Site, log: RunLog, T: float) -> float:
    return sum(_reward(s.priority, log.rescue_times[s.id], T)
               for s in site.survivors if s.id in log.rescue_times)


def oracle_score(site: Site, T: float, travel=None) -> float:
    """Best visiting order with perfect knowledge of positions (brute force, <= 8 survivors).
    Visits to the survivor point rather than the standoff radius, so ratios can be slightly pessimistic."""
    travel = travel or partial(travel_time, site)
    pts = [site.start] + [s.pos for s in site.survivors]
    n = len(site.survivors)
    D = np.array([[travel(a, b) for b in pts] for a in pts])
    best = 0.0
    for order in permutations(range(1, n + 1)):
        t, prev, tot = 0.0, 0, 0.0
        for j in order:
            t += D[prev, j]
            tot += _reward(site.survivors[j - 1].priority, t, T)
            prev = j
        best = max(best, tot)
    return best


def time_to_critical(site: Site, log: RunLog, T: float) -> float:
    """Mean over critical survivors of min(time rescued, T); NaN if the site has none."""
    ts = [min(log.rescue_times.get(s.id, T), T) for s in site.survivors
          if s.priority == Priority.CRITICAL]
    return float(np.mean(ts)) if ts else float("nan")


def rescue_rate(site: Site, log: RunLog, T: float) -> float:
    return sum(log.rescue_times.get(s.id, np.inf) <= T for s in site.survivors) / len(site.survivors)


def summarize(site: Site, log: RunLog, T: float, oracle: float | None = None) -> dict:
    oracle = oracle_score(site, T) if oracle is None else oracle
    score = rescue_score(site, log, T)
    return {"score": score, "oracle": oracle, "norm_score": score / oracle if oracle else float("nan"),
            "time_to_critical": time_to_critical(site, log, T), "rescue_rate": rescue_rate(site, log, T)}
