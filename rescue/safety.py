"""Safety filter: control barrier functions (CBF) on the velocity commands, solved as a small QP.

Barriers (all h >= 0 is safe, evaluated at a control point a distance ``cp_offset`` ahead of the
robot, so both forward speed and turn rate affect it: relative degree 1 for the unicycle):

  * Survivor clearance. The high-probability region of the belief over a survivor's position is the
    confidence ellipse E_p = {q : (q-mu)^T S^-1 (q-mu) <= c_p^2},  c_p^2 = -2 ln(1-p).
        h(x) = sqrt(lambda_min(S)) * (m(x) - c_p) - d_clear - eps,   m = Mahalanobis distance.
    sqrt(lambda_min) * (m - c_p) lower-bounds the Euclidean distance to E_p, so h >= 0 keeps the
    robot at least d_clear + eps from every point of the region (exact when S is isotropic).
    The region grows with localization uncertainty, so the filter gets more conservative as the
    belief gets worse; that is the cost measured in Q3.
  * Hazard terrain. h(x) = signed distance to the (axis-aligned) hazard rectangle - eps.

``eps`` pads for tracking error: the unicycle model the planner and filter assume differs from the
rate-limited Go2 plant, and ``calibrate_tracking_error`` measures that gap.

CBF condition:  grad h . pdot >= -alpha * h,  with pdot = v e_theta + l * omega * e_perp, which is
linear in u = (v, omega). The QP  min (u - u_nom)^T W (u - u_nom)  s.t.  A u >= b,  lo <= u <= hi
has two variables, so it is solved exactly by enumerating active sets (no solver dependency).
If the constraints are mutually infeasible (e.g. boxed in), the least-violating candidate is used
and ``info["feasible"]`` is False.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import Region


def chi_radius(p: float) -> float:
    """c_p for a 2-D Gaussian: P(Mahalanobis distance <= c_p) = p."""
    return float(np.sqrt(-2.0 * np.log(1.0 - p)))


@dataclass(frozen=True)
class SafetyParams:
    conf_p: float = 0.9         # confidence level of the survivor region
    d_clear: float = 1.0        # m; required clearance from any survivor
    eps_track: float = 0.25     # m; tracking-error pad (see calibrate_tracking_error)
    cp_offset: float = 0.25     # m; control point ahead of the robot
    alpha: float = 0.8          # 1/s; class-K gain
    v_min: float = -0.3         # m/s (the Go2 can walk backward slowly)
    w_max: float = 1.5          # rad/s
    w_weight: float = 0.2       # QP weight on turn-rate deviation relative to speed deviation


def ellipse_barrier(p: np.ndarray, mu: np.ndarray, S: np.ndarray, c: float, margin: float):
    """(h, grad h) for the survivor-clearance barrier."""
    d = np.asarray(p, float) - mu
    Sinv = np.linalg.inv(S)
    m = float(np.sqrt(d @ Sinv @ d))
    lam = float(np.sqrt(np.linalg.eigvalsh(S)[0]))
    if m < 1e-9:
        return -lam * c - margin, np.array([lam, 0.0])
    return lam * (m - c) - margin, lam * (Sinv @ d) / m


def box_sd(p: np.ndarray, r: Region):
    """Signed distance (negative inside) from p to the rectangle r, and its gradient."""
    c = np.array([(r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2])
    half = np.array([(r.xmax - r.xmin) / 2, (r.ymax - r.ymin) / 2])
    d = np.asarray(p, float) - c
    q = np.abs(d) - half
    sgn = np.where(d >= 0, 1.0, -1.0)
    out = np.maximum(q, 0.0)
    n = float(np.hypot(out[0], out[1]))
    if n > 1e-12:
        return n, sgn * out / n
    i = int(np.argmax(q))           # inside (or on the boundary): nearest face
    g = np.zeros(2)
    g[i] = sgn[i]
    return float(q.max()), g


def solve_qp2(u_nom, W, A, b, lo, hi):
    """min (u-u_nom)^T diag(W) (u-u_nom)  s.t.  A u >= b, lo <= u <= hi.  Returns (u, feasible)."""
    u_nom, W = np.asarray(u_nom, float), np.asarray(W, float)
    rows = [np.asarray(a, float) for a in A] + [np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
                                                 np.array([0.0, 1.0]), np.array([0.0, -1.0])]
    rhs = list(b) + [lo[0], -hi[0], lo[1], -hi[1]]
    A_, b_ = np.array(rows), np.array(rhs, float)

    def viol(u):
        return float(np.maximum(b_ - A_ @ u - 1e-9, 0.0).sum())

    if viol(u_nom) == 0.0:
        return u_nom, True
    s = 1.0 / np.sqrt(W)                       # u = u_nom + s * w  ->  cost = |w|^2
    As = A_ * s
    r = b_ - A_ @ u_nom                        # need As w >= r
    cands = []
    for i in range(len(r)):                    # projection onto each constraint boundary
        n2 = float(As[i] @ As[i])
        if n2 > 1e-12:
            cands.append(r[i] / n2 * As[i])
    for i in range(len(r)):                    # vertices
        for j in range(i + 1, len(r)):
            M = np.array([As[i], As[j]])
            if abs(np.linalg.det(M)) > 1e-9:
                cands.append(np.linalg.solve(M, np.array([r[i], r[j]])))
    cands = [u_nom + s * w for w in cands]
    ok = [u for u in cands if viol(u) == 0.0]
    if ok:
        return min(ok, key=lambda u: float(np.sum(W * (u - u_nom) ** 2))), True
    # Infeasible: take the least-violating candidate (ties broken by closeness to nominal).
    best = min(cands + [np.clip(np.zeros(2), lo, hi)],
               key=lambda u: (viol(u), float(np.sum(W * (u - u_nom) ** 2))))
    return best, False


class CBFFilter:
    def __init__(self, params: SafetyParams | None = None):
        self.p = params or SafetyParams()

    @property
    def margin_survivor(self) -> float:
        return self.p.d_clear + self.p.eps_track + self.p.cp_offset

    @property
    def margin_hazard(self) -> float:
        return self.p.eps_track + self.p.cp_offset

    def region_radius(self, sigma: float, conf_p: float | None = None) -> float:
        """Distance from an (isotropic) belief mean at which the survivor barrier is zero."""
        return chi_radius(self.p.conf_p if conf_p is None else conf_p) * sigma + self.margin_survivor

    def filter(self, pos, heading: float, u_nom, v_cap: float, hazards, ellipses):
        """Minimally modify u_nom = (v, omega). ``ellipses`` is a list of (mu, Sigma)."""
        P = self.p
        e_t = np.array([np.cos(heading), np.sin(heading)])
        e_p = np.array([-np.sin(heading), np.cos(heading)])
        cp = np.asarray(pos, float) + P.cp_offset * e_t
        c = chi_radius(P.conf_p)
        A, b, hs = [], [], []
        for mu, S in ellipses:
            h, g = ellipse_barrier(cp, np.asarray(mu, float), S, c, self.margin_survivor)
            hs.append(h)
            if h < 8.0:           # far barriers cannot bind within the speed limit
                A.append([g @ e_t, P.cp_offset * (g @ e_p)])
                b.append(-P.alpha * h)
        for r in hazards:
            sd, g = box_sd(cp, r)
            h = sd - self.margin_hazard
            hs.append(h)
            if h < 8.0:
                A.append([g @ e_t, P.cp_offset * (g @ e_p)])
                b.append(-P.alpha * h)
        u_nom = np.asarray(u_nom, float)
        if not A:
            return u_nom, {"min_h": min(hs) if hs else np.inf, "modified": False, "feasible": True}
        u, ok = solve_qp2(u_nom, [1.0, P.w_weight], A, b, (P.v_min, -P.w_max), (max(v_cap, 0.0), P.w_max))
        return u, {"min_h": min(hs), "modified": bool(np.linalg.norm(u - u_nom) > 1e-6), "feasible": ok}


def calibrate_tracking_error(params=None, horizon: float = 1.0, n: int = 400, quantile: float = 0.95,
                             seed: int = 0, dt: float = 0.1) -> float:
    """Gap (m) between where the ideal unicycle (what the planner/filter assume: position integrates
    the commanded velocity) and the rate-limited Go2 plant end up after ``horizon`` seconds under
    random piecewise-constant commands. The ``quantile`` of this gap is the tracking-error pad."""
    from .sim import Go2Params, step_unicycle   # local import: sim pulls in the drone prior
    p = params or Go2Params()
    rng = np.random.default_rng(seed)
    bounds = (-1e3, -1e3, 1e3, 1e3)
    errs = []
    for _ in range(n):
        pos, th, v = np.zeros(2), float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(0, 1.0))
        ideal, ith = pos.copy(), th
        v_cmd = float(rng.uniform(0, 1.0))
        w_cmd = float(rng.uniform(-p.max_turn_rate, p.max_turn_rate))
        for _k in range(int(round(horizon / dt))):
            pos, th, v = step_unicycle(pos, th, v, v_cmd, w_cmd, dt, p, bounds)
            ith = ith + w_cmd * dt
            ideal = ideal + v_cmd * dt * np.array([np.cos(ith), np.sin(ith)])
        errs.append(float(np.linalg.norm(pos - ideal)))
    return float(np.quantile(errs, quantile))
