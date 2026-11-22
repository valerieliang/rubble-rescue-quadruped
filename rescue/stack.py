"""The full three-layer system, assembled.

    search layer  (1 Hz)   greedy priority-weighted search over the belief; weights are scaled by the
                           inferred survivor state when ``infer_state`` is on
    approach MPC  (10 Hz)  unicycle MPC toward a standoff pose (+ legibility term)
    safety filter (10 Hz)  CBF-QP on the MPC's command (``use_cbf``)

Variants (see ``VARIANTS``): full system, baseline 3 (no safety filter), baseline 4 (no state
inference, priority fixed by the drone), baseline 5 (communication oracle reveals z in range).

The stack never reads ground truth. ``PromptRequest.source_id`` is only a handle the simulator uses to
answer a prompt for the right survivor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np

from .ctrl_behavior import MotionTracker, StateBelief
from .drone_prior import PriorConfig
from .interfaces import PRIORITY_DECAY, PRIORITY_WEIGHT, Detection, Observation, ResponseState, Site
from .ctrl_mpc import ApproachMPC, TerrainGrid, lookahead
from .safety import CBFFilter, SafetyParams
from .search import PriorSearch
from .terrain import travel_time

R_RESCUE = 3.5   # m; confirm-by-sensing range in the control-rate sim (outside the 1 m clearance)


@dataclass(frozen=True)
class StackConfig:
    use_cbf: bool = True
    infer_state: bool = True
    comm_oracle: bool = False
    controller: str = "closed_loop"  # "closed_loop": ctrl_mpc + safety.py;  "components": mpc.ApproachController + cbf.SafetyFilter
    hazard_aware_planner: bool = True   # False: planner/MPC ignore hazards (stress test for the filter)
    legibility: float = 0.0          # lambda; 0 = predictable (no legibility term)
    obs_beta: float = 0.7            # observer temperature assumed by the legibility term
    # safety
    conf_p: float = 0.9
    d_clear: float = 1.0
    eps_track: float = 0.25
    cp_offset: float = 0.25
    r_stand: float = 2.5             # standoff distance when no safety region is larger
    sigma_floor: float = 0.5         # belief std floor (survivors may move, so never fully certain)
    r_safe: float = 0.2              # tracks below this existence probability get no barrier
    # state inference
    prompt_range: float = 6.0
    prompt_gap: float = 3.0
    max_prompts: int = 3
    motion_window: float = 5.0
    reliability: float = 0.8         # q the robot assumes when interpreting prompt responses
    walk_speed: float = 0.3          # v0 the robot assumes for mobile survivors


VARIANTS = {
    "full": StackConfig(),
    "no_cbf": StackConfig(use_cbf=False),                                # baseline 3
    "no_state": StackConfig(infer_state=False),                          # baseline 4
    "oracle": StackConfig(infer_state=False, comm_oracle=True),          # baseline 5
}


@dataclass
class PromptRequest:
    key: int
    source_id: int | None
    kind: str                         # "prompt" or "reveal"


class StateAwareSearch(PriorSearch):
    """PriorSearch whose score is also scaled by a per-track weight (the inferred state factor)."""

    def __init__(self, *args, weight_fn=None, **kw):
        super().__init__(*args, **kw)
        self.weight_fn = weight_fn or (lambda t: 1.0)

    def select(self, obs: Observation, sigma_floor: float):
        pos = np.asarray(obs.robot_pos, float)
        for p in obs.rescued:
            self.belief.remove_near(p)
        self.belief.update(pos, obs.sensor_radius, list(obs.detected))
        for t in self.belief.tracks:
            t.sigma = max(t.sigma, sigma_floor)
        best, best_score = None, -np.inf
        for t in self.belief.tracks:
            if t.r < self.r_min:
                continue
            tau = self.travel_time(pos, t.mean)
            arrive = obs.t + tau
            if arrive > self.horizon:
                continue
            if self.mode == "priority":
                score = (t.r * PRIORITY_WEIGHT[t.priority] * self.weight_fn(t)
                         * np.exp(-PRIORITY_DECAY[t.priority] * arrive) / (tau + 1.0) ** self.dist_exp)
            else:
                score = -tau
            if score > best_score:
                best, best_score = t, score
        return best, (best.mean if best is not None else self.coverage.waypoint(pos))


class RescueStack:
    def __init__(self, site: Site, detections: list[Detection], prior_cfg: PriorConfig, cfg: StackConfig,
                 params, T: float = 600.0, sensor_radius: float = 10.0, sigma_meas: float = 0.5):
        self.site, self.cfg, self.params, self.sigma_meas = site, cfg, params, sigma_meas
        self.search = StateAwareSearch(detections, prior_cfg, site.bounds, partial(travel_time, site),
                                       sensor_radius, site.start, T, weight_fn=self._state_weight)
        self.search.belief.sigma_meas = sigma_meas
        self.safety = SafetyParams(conf_p=cfg.conf_p, d_clear=cfg.d_clear, eps_track=cfg.eps_track,
                                   cp_offset=cfg.cp_offset, w_max=params.max_turn_rate)
        self.cbf = CBFFilter(self.safety)
        self.mpc = ApproachMPC(w_max=params.max_turn_rate)
        self.grid = TerrainGrid(site)
        self.hazards = [r for r in site.regions if r.kind == "hazard"]
        self.states: dict[int, StateBelief] = {}
        self.motion: dict[int, MotionTracker] = {}
        self.n_prompts: dict[int, int] = {}
        self.last_prompt: dict[int, float] = {}
        self.revealed: set[int] = set()
        self.target = None
        self.target_key: int | None = None
        self.approach_start = site.start.astype(float).copy()
        self.fallback_wp = site.start.astype(float).copy()
        self.path = np.array([site.start, site.start], float)
        self.last_info: dict = {}
        self.orbit = 0.0
        self.components = self._make_components(cfg, params) if cfg.controller == "components" else None
        self.keepout: list[np.ndarray] = []    # where survivors were reached; they stay put and stay protected

    def _make_components(self, cfg, params):
        from .cbf import SafetyFilter
        from .mpc import ApproachController, MPCParams, MPCPlanner
        planner = MPCPlanner(MPCParams(omega_max=params.max_turn_rate))
        safety = SafetyFilter(omega_max=params.max_turn_rate, target_keepout=cfg.d_clear) if cfg.use_cbf else None
        return ApproachController(planner, safety, legibility_base=cfg.legibility, beta=cfg.obs_beta)

    def _control_components(self, pos, heading, v):
        from .clearance import ClearanceRegion
        from .interfaces import Priority
        from .terrain import REGION_SPEED, BASE_SPEED
        site = self.site

        def speed_fn(pts):
            out = np.full(pts.shape[:-1], BASE_SPEED)
            for r in site.regions:
                inside = (pts[..., 0] >= r.xmin) & (pts[..., 0] <= r.xmax) & (pts[..., 1] >= r.ymin) & (pts[..., 1] <= r.ymax)
                out = np.where(inside, np.minimum(out, BASE_SPEED * REGION_SPEED[r.kind]), out)
            return out

        tracks = [tr for tr in self.search.belief.tracks if tr.r >= self.cfg.r_safe]
        regions = self.search.belief.clearance_regions(self.cfg.conf_p, self.cfg.d_clear, self.cfg.r_safe)
        regions += [ClearanceRegion(q, np.eye(2) * self.cfg.sigma_floor**2, self.cfg.conf_p, self.cfg.d_clear,
                                    Priority.STABLE, 1.0) for q in self.keepout]
        target = self.target.mean if self.target is not None else np.asarray(self.fallback_wp, float)
        others = [tr.mean for tr in tracks if tr is not self.target]
        post = self._sb(self.target).p_state() if (self.target is not None and self.cfg.infer_state) else None
        hz = self.hazards if self.cfg.hazard_aware_planner else ()
        return np.array(self.components.command(pos, heading, v, target, speed_fn, hz,
                                            regions if self.cfg.use_cbf else (), others, post))

    # ---- state inference -------------------------------------------------------------------
    def _sb(self, track) -> StateBelief:
        return self.states.setdefault(id(track), StateBelief())

    def _state_weight(self, track) -> float:
        if not (self.cfg.infer_state or self.cfg.comm_oracle):
            return 1.0
        return self._sb(track).expected_factor()

    def _legibility_weight(self, track) -> float:
        """Legibility only matters to a survivor who can watch the robot: mobile fully, trapped partly."""
        if not (self.cfg.infer_state or self.cfg.comm_oracle):
            return 1.0
        ps = self._sb(track).p_state()
        return float(ps[ResponseState.MOBILE] + 0.6 * ps[ResponseState.TRAPPED])

    def perceive(self, t: float, pos, detected) -> list[PromptRequest]:
        """10 Hz: record motion samples, decide which survivors to prompt / reveal."""
        reqs: list[PromptRequest] = []
        if not (self.cfg.infer_state or self.cfg.comm_oracle):
            return reqs
        pos = np.asarray(pos, float)
        for d in detected:
            tr = self._match(d.pos)
            if tr is None:
                continue
            key = id(tr)
            if self.cfg.infer_state:
                self.motion.setdefault(key, MotionTracker()).add(t, d.pos)
                if (np.linalg.norm(tr.mean - pos) <= self.cfg.prompt_range
                        and self.n_prompts.get(key, 0) < self.cfg.max_prompts
                        and t - self.last_prompt.get(key, -1e9) >= self.cfg.prompt_gap):
                    self.n_prompts[key] = self.n_prompts.get(key, 0) + 1
                    self.last_prompt[key] = t
                    reqs.append(PromptRequest(key, d.source_id, "prompt"))
            if self.cfg.comm_oracle and key not in self.revealed:
                self.revealed.add(key)
                reqs.append(PromptRequest(key, d.source_id, "reveal"))
        return reqs

    def _match(self, p):
        best, bd = None, 3.0
        for tr in self.search.belief.tracks:
            if not tr.confirmed:
                continue
            d = float(np.linalg.norm(tr.mean - p))
            if d < bd:
                best, bd = tr, d
        return best

    def _track(self, key: int):
        return next((t for t in self.search.belief.tracks if id(t) == key), None)

    def apply_response(self, key: int, response: int) -> None:
        self.states.setdefault(key, StateBelief()).update_prompt(response, self.cfg.reliability)

    def apply_reveal(self, key: int, z: ResponseState) -> None:
        self.states.setdefault(key, StateBelief()).reveal(z)

    def _motion_updates(self, pos) -> None:
        for key, mt in list(self.motion.items()):
            tr = self._track(key)
            if tr is None:
                del self.motion[key]
                continue
            if not mt.ready(self.cfg.motion_window):
                continue
            v, sv = mt.velocity(self.sigma_meas)
            u = pos - tr.mean
            n = float(np.hypot(u[0], u[1]))
            if n > 1e-6:
                u = u / n
                self.states.setdefault(key, StateBelief()).update_motion(
                    [v @ u, v @ np.array([-u[1], u[0]])], float(np.hypot(sv, 0.05)), self.cfg.walk_speed)
            mt.clear()

    def snapshot(self):
        """[(track mean, p over the 4 hypotheses)] for confirmed tracks (for evaluation/logging)."""
        return [(t.mean.copy(), self.states[id(t)].p.copy()) for t in self.search.belief.tracks
                if t.confirmed and id(t) in self.states]

    # ---- 1 Hz search ------------------------------------------------------------------------
    def plan(self, obs: Observation) -> None:
        pos = np.asarray(obs.robot_pos, float)
        self.keepout += [np.asarray(q, float) for q in obs.rescued]
        if self.cfg.infer_state:
            self._motion_updates(pos)
        target, wp = self.search.select(obs, self.cfg.sigma_floor)
        key = id(target) if target is not None else None
        if key != self.target_key:
            self.approach_start = pos.copy()
            self.target_key = key
            self.orbit = 0.0
        self.target, self.fallback_wp = target, wp
        # Active uncertainty reduction: if the safety region keeps the robot outside sensing range of an
        # unconfirmed target, circle the region boundary until the survivor is seen (then it shrinks).
        if target is None or target.confirmed or not self.cfg.use_cbf:
            self.orbit = 0.0
        elif np.linalg.norm(pos - self.goal(pos)) < 1.5:
            self.orbit += 0.35
        goal = self.goal(pos)
        self.path = self.grid.path(pos, goal) if self.cfg.hazard_aware_planner else np.array([pos, goal])

    def boundary(self, track) -> float:
        return self.cbf.region_radius(track.sigma)

    def goal(self, pos) -> np.ndarray:
        """Standoff pose: on the robot's side of the target, outside its safety region."""
        if self.target is None:
            return np.asarray(self.fallback_wp, float)
        mu = self.target.mean
        u = np.asarray(pos, float) - mu
        n = float(np.hypot(u[0], u[1]))
        u = u / n if n > 1e-6 else np.array([1.0, 0.0])
        if self.orbit:
            c, s = np.cos(self.orbit), np.sin(self.orbit)
            u = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])
        R = self.cfg.r_stand
        if self.cfg.use_cbf:
            R = max(R, self.boundary(self.target) + 0.1)
        return mu + u * R

    # ---- 10 Hz control ----------------------------------------------------------------------
    def control(self, t: float, pos, heading: float, v_cap: float, v: float = 0.0):
        pos = np.asarray(pos, float)
        if self.components is not None:
            u = self._control_components(pos, heading, v)
            self.last_info = {"modified": False, "feasible": True}
            return np.array([min(u[0], max(v_cap, 0.0)), u[1]])
        goal = self.goal(pos)
        wp = lookahead(self.path, pos) if np.linalg.norm(goal - self.path[-1]) < 1.0 else goal
        tracks = [tr for tr in self.search.belief.tracks if tr.r >= self.cfg.r_safe]
        discs = []
        if self.cfg.use_cbf:
            discs = [(tr.mean, self.boundary(tr)) for tr in tracks if tr is not self.target]
            discs += [(q, self.cbf.region_radius(self.cfg.sigma_floor)) for q in self.keepout]
        leg = None
        if self.cfg.legibility > 0 and self.target is not None:
            near = [tr for tr in tracks if np.linalg.norm(tr.mean - pos) < 25.0]
            if any(tr is self.target for tr in near) and len(near) > 1:
                goals = np.array([tr.mean for tr in near])
                k = next(i for i, tr in enumerate(near) if tr is self.target)
                lam = self.cfg.legibility * self._legibility_weight(self.target)
                if lam > 0:
                    leg = (goals, k, self.approach_start, lam, self.cfg.obs_beta)
        mpc_hazards = self.hazards if self.cfg.hazard_aware_planner else ()
        u_nom = np.array(self.mpc.solve(pos, heading, v_cap, wp, mpc_hazards, discs, leg))
        if not self.cfg.use_cbf:
            self.last_info = {"modified": False, "feasible": True}
            return u_nom
        ell = [(tr.mean, np.eye(2) * tr.sigma**2) for tr in tracks]
        ell += [(q, np.eye(2) * self.cfg.sigma_floor**2) for q in self.keepout]
        u, info = self.cbf.filter(pos, heading, u_nom, v_cap, self.hazards, ell)
        self.last_info = info
        return u
