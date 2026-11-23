"""Metrics: priority-weighted rescue score (oracle-normalized), time to critical, rescue rate within T."""
from __future__ import annotations

from functools import partial
from itertools import permutations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .interfaces import (PRIORITY_DECAY, PRIORITY_WEIGHT, STANDOFF_RADIUS, BASE_REWARD, SURVIVOR_KEEPOUT,
                         Priority, RunLog, Site)
from .terrain import BASE_SPEED, REGION_SPEED, travel_time


def _model(state_weighted):
    """False/None: class reward only; True: BASE_REWARD; a RewardModel: that model."""
    return BASE_REWARD if state_weighted is True else (state_weighted or None)


def _reward(s, tau: float, T: float, model=None) -> float:
    """Reward for reaching survivor ``s`` at time tau (0 after T)."""
    if tau > T:
        return 0.0
    if model is None:
        return PRIORITY_WEIGHT[s.priority] * float(np.exp(-PRIORITY_DECAY[s.priority] * tau))
    return model.value(s.priority, s.state, tau)


def rescue_score(site: Site, log: RunLog, T: float, state_weighted: bool = False) -> float:
    """``state_weighted=True`` uses BASE_REWARD (reward scaled by the true state); pass a RewardModel to
    use another one (control-rate Q4-Q5 runs)."""
    m = _model(state_weighted)
    return sum(_reward(s, log.rescue_times[s.id], T, m)
               for s in site.survivors if s.id in log.rescue_times)


def _leg_time(travel, a: np.ndarray, b: np.ndarray, shrink_a: float, shrink_b: float) -> float:
    """Straight-line travel time a -> b where the robot may start `shrink_a` metres closer to b (it is
    already at the previous survivor's standoff circle) and stops `shrink_b` metres short of it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L <= shrink_a + shrink_b:
        return 0.0
    u = d / L
    return travel(a + u * shrink_a, b - u * shrink_b)


_NEIGHBOURS = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (1, -2))


def _grid_times(site: Site, sources: list[np.ndarray], cell: float = 0.5) -> np.ndarray:
    """Terrain-aware shortest travel times (s) from each source to every other source, by Dijkstra
    on a 16-neighbour grid whose edge cost is distance / terrain speed. Returns len(sources)^2."""
    xmin, ymin, xmax, ymax = site.bounds
    nx, ny = int(round((xmax - xmin) / cell)) + 1, int(round((ymax - ymin) / cell)) + 1
    gx, gy = np.meshgrid(xmin + cell * np.arange(nx), ymin + cell * np.arange(ny), indexing="ij")
    speed = np.full(gx.shape, BASE_SPEED)
    for r in site.regions:
        inside = (gx >= r.xmin) & (gx <= r.xmax) & (gy >= r.ymin) & (gy <= r.ymax)
        speed = np.where(inside, np.minimum(speed, BASE_SPEED * REGION_SPEED[r.kind]), speed)
    inv = 1.0 / speed
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, w = [], [], []
    for dx, dy in _NEIGHBOURS:
        xs, xe = slice(0, nx - dx), slice(dx, nx)
        ys, ye = (slice(0, ny - dy), slice(dy, ny)) if dy >= 0 else (slice(-dy, ny), slice(0, ny + dy))
        rows.append(idx[xs, ys].ravel())
        cols.append(idx[xe, ye].ravel())
        w.append((np.hypot(dx, dy) * cell * 0.5 * (inv[xs, ys] + inv[xe, ye])).ravel())
    graph = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(nx * ny,) * 2).tocsr()
    nodes = [int(np.clip(round((p[0] - xmin) / cell), 0, nx - 1)) * ny
             + int(np.clip(round((p[1] - ymin) / cell), 0, ny - 1)) for p in sources]
    dist = dijkstra(graph, directed=False, indices=nodes)
    return dist[:, nodes]


def oracle_score(site: Site, T: float, travel=None, standoff: float = STANDOFF_RADIUS,
                 state_weighted=False, blind_order=None, return_order: bool = False):
    """Best visiting order with perfect knowledge of positions (brute force, <= 8 survivors).

    A survivor counts as rescued at the standoff radius, as in the simulators. Leg times are the
    cheaper of the straight-line estimate and a terrain-aware shortest path (the robot may detour
    around slow terrain), minus the standoff slack at each end, so the oracle is a relaxation:
    it ignores robot dynamics and never does worse than the policies it normalizes.
    Pass ``travel`` to use straight-line legs only. ``blind_order``: a RewardModel used to *choose* the
    visiting order (scored with ``state_weighted``): the ceiling for a planner that lacks the state.
    ``return_order`` also returns the chosen order.
    """
    m = _model(state_weighted)
    pts = [site.start] + [s.pos for s in site.survivors]
    n = len(site.survivors)
    straight = travel or partial(travel_time, site)
    G = None if travel else _grid_times(site, pts)
    slack = 2 * standoff / BASE_SPEED + 1.0   # standoff at both ends, plus grid snapping error
    D = np.zeros((n + 1, n + 1))
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i == j:
                continue
            t = _leg_time(straight, a, b, standoff if i else 0.0, standoff)
            if G is not None:
                t = min(t, max(0.0, G[i, j] - (slack if i else slack / 2)))
            D[i, j] = t
    pick = _model(blind_order) if blind_order is not None else m

    def total(order, model):
        t, prev, tot = 0.0, 0, 0.0
        for j in order:
            sj = site.survivors[j - 1]
            t += D[prev, j] + (model.dwell[sj.state] if model is not None else 0.0)
            tot += _reward(sj, t, T, model)
            prev = j
        return tot

    orders = list(permutations(range(1, n + 1)))
    best = max(orders, key=lambda o: total(o, pick))
    val = total(best, m)
    return (val, best) if return_order else val


def time_to_critical(site: Site, log: RunLog, T: float) -> float:
    """Mean over critical survivors of min(time rescued, T); NaN if the site has none."""
    ts = [min(log.rescue_times.get(s.id, T), T) for s in site.survivors
          if s.priority == Priority.CRITICAL]
    return float(np.mean(ts)) if ts else float("nan")


def rescue_rate(site: Site, log: RunLog, T: float) -> float:
    return sum(log.rescue_times.get(s.id, np.inf) <= T for s in site.survivors) / len(site.survivors)


def safety_violations(site: Site, log: RunLog, keepout: float = SURVIVOR_KEEPOUT) -> dict:
    """Safety violations along the logged trajectory (one sample per policy step, 1 Hz).

    A sample violates safety if the robot is inside a hazard region or within ``keepout`` metres
    of any survivor's true position (rescued or not). Crossings between samples are not seen, so
    counts are a lower bound. Returns per-kind sample counts, the number of violating samples,
    and that number as a fraction of all samples.
    """
    traj = np.asarray(log.trajectory, float).reshape(-1, 3)
    pts = traj[:, 1:]
    hazard = np.zeros(len(pts), bool)
    for r in site.regions:
        if r.kind == "hazard":
            hazard |= ((pts[:, 0] >= r.xmin) & (pts[:, 0] <= r.xmax)
                       & (pts[:, 1] >= r.ymin) & (pts[:, 1] <= r.ymax))
    near = np.zeros(len(pts), bool)
    for s in site.survivors:
        near |= np.linalg.norm(pts - s.pos, axis=1) < keepout
    bad = hazard | near
    return {"hazard_steps": int(hazard.sum()), "survivor_steps": int(near.sum()),
            "violation_steps": int(bad.sum()),
            "violation_rate": float(bad.mean()) if len(bad) else 0.0}


def summarize(site: Site, log: RunLog, T: float, oracle: float | None = None) -> dict:
    oracle = oracle_score(site, T) if oracle is None else oracle
    score = rescue_score(site, log, T)
    return {"score": score, "oracle": oracle, "norm_score": score / oracle if oracle else float("nan"),
            "time_to_critical": time_to_critical(site, log, T), "rescue_rate": rescue_rate(site, log, T),
            **safety_violations(site, log)}
