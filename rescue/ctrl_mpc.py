"""Approach planner: receding-horizon MPC at ~10 Hz on the unicycle model, plus a terrain path planner.

MPC. At each control step it scores a library of 405 candidate command sequences, rolls each out
through the unicycle model (x' = v cos th, y' = v sin th, th' = omega) over a 2 s horizon, and applies
the first command of the cheapest one (sampling-based receding-horizon control; the candidates are
a fixed grid of {speed} x {turn rate for the first second} x {turn rate for the second second}, so it
is deterministic and needs no solver). Cost terms:

    goal distance (stage + terminal) + stop near the goal + effort
    + hazard penalty (rollout inside an inflated hazard rectangle)
    + clearance penalty (rollout inside a survivor's high-probability region)
    + legibility  lambda * sum_k -log P(G* | Q_k)

The goal is a standoff pose, not the survivor: a point at the region boundary on the robot's side.

Path planner. A coarse grid (2 m) over the site; edge cost = distance / terrain speed, with cells
inside (inflated) hazards made very expensive. One Dijkstra from the goal gives a path from any robot
position; the MPC chases a lookahead point on it. This keeps the MPC from stalling in front of a
large hazard rectangle (a local minimum for any purely local planner).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .interfaces import Site
from .legibility import log_prob_goal
from .terrain import BASE_SPEED, REGION_SPEED

_NB8 = ((1, 0), (0, 1), (1, 1), (1, -1))


class TerrainGrid:
    """Static 2 m grid graph of a site; goal-rooted Dijkstra gives paths that avoid hazards."""

    def __init__(self, site: Site, cell: float = 2.0, hazard_margin: float = 1.5,
                 hazard_cost: float = 40.0):
        self.cell = cell
        xmin, ymin, xmax, ymax = site.bounds
        self.xmin, self.ymin = xmin, ymin
        self.nx, self.ny = int(round((xmax - xmin) / cell)) + 1, int(round((ymax - ymin) / cell)) + 1
        gx, gy = np.meshgrid(xmin + cell * np.arange(self.nx), ymin + cell * np.arange(self.ny), indexing="ij")
        self.centers = np.stack([gx, gy], -1)
        speed = np.full(gx.shape, BASE_SPEED)
        cost = np.ones(gx.shape)
        for r in site.regions:
            m = hazard_margin if r.kind == "hazard" else 0.0
            inside = ((gx >= r.xmin - m) & (gx <= r.xmax + m) & (gy >= r.ymin - m) & (gy <= r.ymax + m))
            speed = np.where(inside, np.minimum(speed, BASE_SPEED * REGION_SPEED[r.kind]), speed)
            if r.kind == "hazard":
                cost = np.where(inside, hazard_cost, cost)
        inv = cost / speed
        idx = np.arange(self.nx * self.ny).reshape(self.nx, self.ny)
        rows, cols, w = [], [], []
        for dx, dy in _NB8:
            xs, xe = slice(0, self.nx - dx), slice(dx, self.nx)
            ys, ye = (slice(0, self.ny - dy), slice(dy, self.ny)) if dy >= 0 else (
                slice(-dy, self.ny), slice(0, self.ny + dy))
            rows.append(idx[xs, ys].ravel())
            cols.append(idx[xe, ye].ravel())
            w.append((np.hypot(dx, dy) * cell * 0.5 * (inv[xs, ys] + inv[xe, ye])).ravel())
        self.graph = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(self.nx * self.ny,) * 2).tocsr()

    def node(self, p) -> int:
        i = int(np.clip(round((p[0] - self.xmin) / self.cell), 0, self.nx - 1))
        j = int(np.clip(round((p[1] - self.ymin) / self.cell), 0, self.ny - 1))
        return i * self.ny + j

    def path(self, start, goal) -> np.ndarray:
        """Polyline from start to goal (cell centres, endpoints replaced by the exact points)."""
        g, s = self.node(goal), self.node(start)
        _, pred = dijkstra(self.graph, directed=False, indices=g, return_predecessors=True)
        nodes = [s]
        while nodes[-1] != g and pred[nodes[-1]] >= 0 and len(nodes) < self.nx * self.ny:
            nodes.append(int(pred[nodes[-1]]))
        pts = [self.centers[n // self.ny, n % self.ny] for n in nodes]
        pts[0], pts[-1] = np.asarray(start, float), np.asarray(goal, float)
        return np.array(pts)


def lookahead(path: np.ndarray, pos, L: float = 4.0) -> np.ndarray:
    """First point on the polyline at least L metres from pos (searching from the nearest vertex)."""
    pos = np.asarray(pos, float)
    d = np.linalg.norm(path - pos, axis=1)
    for i in range(int(np.argmin(d)), len(path)):
        if d[i] >= L:
            return path[i]
    return path[-1]


class ApproachMPC:
    def __init__(self, dt: float = 0.2, n_steps: int = 10, w_max: float = 1.5, n_w: int = 9,
                 v_fracs=(0.0, 0.15, 0.4, 0.7, 1.0), w_goal: float = 0.1, w_term: float = 3.0,
                 w_hazard: float = 4.0, w_clear: float = 8.0, w_effort: float = 0.02,
                 hazard_margin: float = 0.8, w_head: float = 1.0):
        self.dt, self.n = dt, n_steps
        self.w_goal, self.w_term, self.w_hazard, self.w_clear, self.w_effort = w_goal, w_term, w_hazard, w_clear, w_effort
        self.hazard_margin = hazard_margin
        self.w_head = w_head
        lv = np.linspace(-w_max, w_max, n_w)
        W1, W2 = np.meshgrid(lv, lv, indexing="ij")
        nv = len(v_fracs)
        self.vf = np.repeat(np.asarray(v_fracs, float), n_w * n_w)
        self.w1 = np.tile(W1.ravel(), nv)
        w2 = np.tile(W2.ravel(), nv)
        half = n_steps // 2
        self.omega = np.concatenate([np.repeat(self.w1[:, None], half, 1),
                                     np.repeat(w2[:, None], n_steps - half, 1)], 1)

    def rollouts(self, pos, heading: float, v_cap: float):
        v = (self.vf * v_cap)[:, None]
        th = heading + np.cumsum(self.omega * self.dt, axis=1)
        x = pos[0] + np.cumsum(v * np.cos(th) * self.dt, axis=1)
        y = pos[1] + np.cumsum(v * np.sin(th) * self.dt, axis=1)
        return x, y, v[:, 0], th

    def solve(self, pos, heading: float, v_cap: float, goal, hazards=(), discs=(), leg=None):
        """Returns (v_cmd, omega_cmd). ``discs``: [(centre, radius)] soft keep-out regions.
        ``leg``: (goals (K,2), k, start (2,), weight, beta) or None."""
        pos = np.asarray(pos, float)
        x, y, v, th = self.rollouts(pos, heading, max(v_cap, 1e-3))
        goal = np.asarray(goal, float)
        dist = np.hypot(x - goal[0], y - goal[1])
        cost = self.w_goal * self.dt * dist.sum(1) + self.w_term * dist[:, -1]
        cost += 2.0 * v**2 * np.exp(-dist[:, -1] / 1.5) * (dist[:, -1] < 2.0)       # arrive slowly
        cost += self.w_effort * np.sum(self.omega**2, 1)
        # Heading error to the goal at the end of the rollout: without it every zero-speed candidate costs
        # the same, so a robot facing away from its goal would never turn in place.
        bearing = np.arctan2(goal[1] - y[:, -1], goal[0] - x[:, -1])
        cost += self.w_head * (1.0 - np.cos(th[:, -1] - bearing)) * (dist[:, -1] > 1.5)
        m = self.hazard_margin
        for r in hazards:
            inside = (x >= r.xmin - m) & (x <= r.xmax + m) & (y >= r.ymin - m) & (y <= r.ymax + m)
            cost += self.w_hazard * inside.sum(1)
        for c, R in discs:
            d = np.hypot(x - c[0], y - c[1])
            cost += self.w_clear * np.sum(np.maximum(0.0, R + 0.3 - d) ** 2, 1)
        if leg is not None:
            goals, k, S, lam, beta, *rest = leg
            lam_h = rest[0] if rest else 0.0
            Q = np.stack([x, y], -1)
            if lam > 0:
                cost += lam * self.dt * np.sum(-log_prob_goal(Q, S, goals, k, beta), 1)
            if lam_h > 0:
                # heading legibility: -log P(G* | motion direction), the HeadingObserver's likelihood form
                to_g = goals[None, None] - Q[:, :, None, :]
                cos = (to_g[..., 0] * np.cos(th)[..., None] + to_g[..., 1] * np.sin(th)[..., None]) / (
                    np.linalg.norm(to_g, axis=-1) + 1e-9)
                sc = 4.0 * cos
                lse = np.log(np.sum(np.exp(sc - sc.max(-1, keepdims=True)), -1)) + sc.max(-1)
                cost += lam_h * self.dt * np.sum(lse - sc[..., k], 1)
        i = int(np.argmin(cost))
        return float(self.vf[i] * v_cap), float(self.w1[i])
