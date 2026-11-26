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
from .interfaces import PRIORITY_DECAY, PRIORITY_WEIGHT, REWARDS, Detection, Observation, ResponseState, Site
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
    oracle_global: bool = False      # diagnostic upper bound: z of every drone track revealed at t=0
    controller: str = "closed_loop"  # "closed_loop": ctrl_mpc + safety.py;  "components": mpc.ApproachController + cbf.SafetyFilter
    hazard_aware_planner: bool = True   # False: planner/MPC ignore hazards (stress test for the filter)
    route_around_regions: bool = True   # route around survivors' keep-out regions (False: the pre-fix router,
                                        # whose route could stall the MPC at a region's edge; site 6 on the Go2)
    forget_stale_tracks: bool = True    # missed detections lower a confirmed track's existence too (False: the
                                        # old belief, where the robot could wait for good at the standoff of a
                                        # survivor who had walked away; sites 16 and 18 at sigma_meas 1 m)
    retention: str = ""              # search.PriorSearch retention remedies ("keep", "region", "resume"; "" = off)
    reward: str = "base"             # key of interfaces.REWARDS: how the hidden state enters the reward
    legibility: float = 0.0          # lambda; 0 = predictable (no legibility term)
    legibility_heading: float = 0.0  # weight of the heading-legibility term (motion direction points at the goal)
    obs_beta: float = 0.7            # observer temperature assumed by the legibility term
    # safety
    conf_p: float = 0.7              # region confidence; 0.7 chosen from the Q3 sweep (0.9 costs 1-5 points, same safety)
    d_clear: float = 1.0
    eps_track: float = 0.25
    cp_offset: float = 0.25
    r_stand: float = 2.5             # standoff distance when no safety region is larger
    sigma_floor: float = 0.5         # belief std floor (survivors may move, so never fully certain)
    r_safe: float = 0.2              # tracks below this existence probability get no barrier
    conf_fresh: float = 0.0          # >0: confidence for a robot-detected track with fewer than n_fresh detections fused
    n_fresh: int = 5                 # (the belief updates at 1 Hz, so 5 detections is about 5 s); off while conf_fresh is 0
    # state inference
    prompt_range: float = 6.0
    prompt_gap: float = 3.0
    max_prompts: int = 3
    motion_window: float = 5.0
    fast_detect: bool = True         # 10 Hz first-detection path: an onboard detection no belief track explains goes to the filter at once (on since the gate, below, removed its clutter cost)
    fast_gate: int = 3               # a provisional track reaches the filter only after this many detections within fast_window
    fast_window: float = 0.5         # s (a real survivor in range is detected on ~90% of 10 Hz steps; a phantom rarely repeats)
    fast_ttl: float = 5.0            # s a provisional track survives without a detection
    apply_filter: bool = True        # False (with use_cbf): the MPC sees the keep-out discs but the CBF-QP does not run (soft-constraint baseline)
    mpc_fast: bool = False           # also give the MPC a keep-out disc for each active provisional (fast_detect) track; with apply_filter=False this is the fair disc-only baseline (same information as the filter)
    avg_meas: bool = False         # fuse all 10 Hz detections of the last second (mean, sigma_meas/sqrt(n)), not one
    hint_acc: float = 0.7            # accuracy the robot assumes for the drone's state cue
    reliability: float = 0.8         # q the robot assumes when interpreting prompt responses
    walk_speed: float = 0.3          # v0 the robot assumes for mobile survivors


VARIANTS = {
    "full": StackConfig(),
    "no_cbf": StackConfig(use_cbf=False),                                # baseline 3
    "full_far": StackConfig(prompt_range=10.0, motion_window=3.0),       # infer earlier: prompt at sensing range
    "no_state": StackConfig(infer_state=False),                          # baseline 4
    "oracle": StackConfig(infer_state=False, comm_oracle=True),          # baseline 5
    "oracle_global": StackConfig(infer_state=False, comm_oracle=True, oracle_global=True),  # upper bound on the value of z
}


@dataclass
class PromptRequest:
    key: int
    source_id: int | None
    kind: str                         # "prompt" or "reveal"


class StateAwareSearch(PriorSearch):
    """PriorSearch whose score is also scaled by a per-track weight (the inferred state factor)."""

    def __init__(self, *args, weight_fn=None, dwell_fn=None, **kw):
        super().__init__(*args, **kw)
        self.dwell_fn = dwell_fn or (lambda t: 0.0)
        self.weight_fn = weight_fn or (lambda t, arrive: 1.0)

    def select(self, obs: Observation, sigma_floor: float, sigmas=None):
        pos = np.asarray(obs.robot_pos, float)
        for p in obs.rescued:
            self.belief.remove_near(p)
        if self.obs_map is not None:
            self.obs_map.mark(pos, obs.sensor_radius)
        self.belief.update(pos, obs.sensor_radius, list(obs.detected), sigmas)
        for t in self.belief.tracks:
            t.sigma = max(t.sigma, sigma_floor)
        if "keep" in self.flags:
            self._retain()
        best, best_score = None, -np.inf
        for t in self.belief.tracks:
            if t.r < self.r_min:
                continue
            tau = self.travel_time(pos, t.mean)
            arrive = obs.t + tau
            if arrive > self.horizon:
                continue
            if self.mode == "priority":
                score = (t.r * PRIORITY_WEIGHT[t.priority] * self.weight_fn(t, arrive)
                         * np.exp(-PRIORITY_DECAY[t.priority] * arrive) / (tau + self.dwell_fn(t) + 1.0) ** self.dist_exp)
            else:
                score = -tau
            if score > best_score:
                best, best_score = t, score
        if best is not None:
            self._in_fallback = False
            return best, self._target_point(best, pos)
        if self.obs_map is None:
            return best, self.coverage.waypoint(pos)
        wp = self._fallback(pos)
        self._in_fallback = True
        return best, wp


class RescueStack:
    def __init__(self, site: Site, detections: list[Detection], prior_cfg: PriorConfig, cfg: StackConfig,
                 params, T: float = 600.0, sensor_radius: float = 10.0, sigma_meas: float = 0.5):
        self.site, self.cfg, self.params, self.sigma_meas = site, cfg, params, sigma_meas
        self.search = StateAwareSearch(detections, prior_cfg, site.bounds, partial(travel_time, site),
                                       sensor_radius, site.start, T, weight_fn=self._state_weight, dwell_fn=self._expected_dwell,
                                       retention=cfg.retention)
        self.search.belief.sigma_meas = sigma_meas
        self.search.belief.forget_confirmed = cfg.forget_stale_tracks
        self.states: dict[int, StateBelief] = {}
        for tr, d in zip(self.search.belief.tracks, detections):
            if d.state_hint is not None and (cfg.infer_state or cfg.comm_oracle):
                self.states.setdefault(id(tr), StateBelief()).update_hint(d.state_hint, cfg.hint_acc)
        self._init_ids = [(id(tr), d.source_id) for tr, d in zip(self.search.belief.tracks, detections)
                          if d.source_id is not None]
        self.safety = SafetyParams(conf_p=cfg.conf_p, d_clear=cfg.d_clear, eps_track=cfg.eps_track,
                                   cp_offset=cfg.cp_offset, w_max=params.max_turn_rate)
        self.cbf = CBFFilter(self.safety)
        self.mpc = ApproachMPC(w_max=params.max_turn_rate)
        self.grid = TerrainGrid(site)
        self.hazards = [r for r in site.regions if r.kind == "hazard"]
        self.motion: dict[int, MotionTracker] = {}
        self.n_prompts: dict[int, int] = {}
        self.last_prompt: dict[int, float] = {}
        self.revealed: set[int] = set()
        self._det_buf: list[Detection] = []
        self.fast: list[list] = []             # [mean, sigma, t_last] provisional tracks from the 10 Hz path (fast_detect)
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
        regions = self.search.belief.clearance_regions(self.cfg.conf_p, self.cfg.d_clear, self.cfg.r_safe,
                                                       self.cfg.conf_fresh, self.cfg.n_fresh)
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

    def _expected_dwell(self, track) -> float:
        return self._sb(track).expected_dwell(REWARDS[self.cfg.reward])

    def _state_weight(self, track, arrive: float) -> float:
        """Expected state-dependent reward factor. Without inference or an oracle the belief stays at its
        prior, so the no-inference baseline plans with the prior-expected reward (the best it can do)."""
        return self._sb(track).expected_ratio(track.priority, arrive, REWARDS[self.cfg.reward])

    def _legibility_weight(self, track) -> float:
        """Legibility only matters to a survivor who can watch the robot: mobile fully, trapped partly."""
        if not (self.cfg.infer_state or self.cfg.comm_oracle):
            return 1.0
        ps = self._sb(track).p_state()
        return float(ps[ResponseState.MOBILE] + 0.6 * ps[ResponseState.TRAPPED])

    def perceive(self, t: float, pos, detected) -> list[PromptRequest]:
        """10 Hz: record motion samples, decide which survivors to prompt / reveal."""
        reqs: list[PromptRequest] = []
        if self.cfg.fast_detect:
            self._fast_update(t, detected)
        if self.cfg.avg_meas:
            self._det_buf += list(detected)
        if not (self.cfg.infer_state or self.cfg.comm_oracle):
            return reqs
        if self.cfg.oracle_global and self._init_ids:
            for key, sid in self._init_ids:
                self.revealed.add(key)
                reqs.append(PromptRequest(key, sid, "reveal"))
            self._init_ids = []
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

    def _fast_update(self, t: float, detected) -> None:
        """Provisional tracks for detections the belief has no confirmed track for; fused at 10 Hz, dropped once the
        1 Hz belief holds the survivor (or after 5 s without a detection). Used by the filter only."""
        confirmed = [tr.mean for tr in self.search.belief.tracks if tr.confirmed]
        self.fast = [f for f in self.fast if t - f[2] < self.cfg.fast_ttl
                     and not any(np.linalg.norm(f[0] - m) < 3.0 for m in confirmed)]
        for d in detected:
            if any(np.linalg.norm(d.pos - m) < 3.0 for m in confirmed):
                continue
            for f in self.fast:
                if np.linalg.norm(f[0] - d.pos) < 2.0:
                    k = f[1] ** 2 / (f[1] ** 2 + self.sigma_meas ** 2)
                    f[0] = f[0] + k * (np.asarray(d.pos, float) - f[0])
                    f[1] = max(float(np.sqrt((1 - k) * f[1] ** 2)), self.cfg.sigma_floor)
                    f[2] = t
                    f[3] = [ti for ti in f[3] if t - ti <= self.cfg.fast_window] + [t]
                    break
            else:
                self.fast.append([np.asarray(d.pos, float).copy(), max(self.sigma_meas, self.cfg.sigma_floor), t, [t]])

    def _fast_active(self, t: float) -> list[list]:
        """Provisional tracks the filter uses: at least ``fast_gate`` detections within the last ``fast_window`` s."""
        g, w = self.cfg.fast_gate, self.cfg.fast_window
        return [f for f in self.fast if sum(t - ti <= w + 1e-9 for ti in f[3]) >= g or g <= 1]

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
        sigmas = None
        if self.cfg.avg_meas and self._det_buf:
            obs, sigmas = self._averaged(obs)
        self._det_buf = []
        prev = self.target
        target, wp = self.search.select(obs, self.cfg.sigma_floor, sigmas)
        # Finish the survivor being handled: leaving mid-dwell would waste the time already spent there.
        if (prev is not None and target is not prev and any(t is prev for t in self.search.belief.tracks)
                and prev.confirmed and np.linalg.norm(pos - prev.mean) <= R_RESCUE + 0.5
                and prev.r >= self.search.r_min):   # not once the survivor is gone (forget_stale_tracks)
            target, wp = prev, prev.mean
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
        if self.cfg.hazard_aware_planner:
            self.path = self.grid.path(pos, goal, self._path_discs())
        else:
            self.path = np.array([pos, goal])

    def _path_discs(self):
        """The keep-out discs the MPC sees (other survivors' regions, rescued survivors), for the route."""
        if not (self.cfg.use_cbf and self.cfg.route_around_regions):
            return []
        tracks = [tr for tr in self.search.belief.tracks if tr.r >= self.cfg.r_safe and tr is not self.target]
        return ([(tr.mean, self.boundary(tr)) for tr in tracks]
                + [(q, self.cbf.region_radius(self.cfg.sigma_floor)) for q in self.keepout])

    def _averaged(self, obs: Observation):
        """Cluster the last second's raw detections; one mean detection per cluster with std sigma_meas/sqrt(n)."""
        gate = max(2.0, 3.0 * self.sigma_meas + 1.0)
        # raw detections of a survivor rescued within this second would re-create the track just removed
        buf = [d for d in self._det_buf
               if all(np.linalg.norm(d.pos - np.asarray(q, float)) > 3.0 + 3.0 * self.sigma_meas for q in obs.rescued)]
        if not buf:
            return Observation(obs.t, obs.robot_pos, obs.sensor_radius, (), obs.rescued), []
        clusters: list[list[Detection]] = []
        for d in buf:
            for c in clusters:
                if np.linalg.norm(np.mean([x.pos for x in c], axis=0) - d.pos) < gate:
                    c.append(d)
                    break
            else:
                clusters.append([d])
        dets = tuple(Detection(np.mean([x.pos for x in c], axis=0), c[-1].priority, c[-1].source_id) for c in clusters)
        sig = [self.sigma_meas / np.sqrt(len(c)) for c in clusters]
        return Observation(obs.t, obs.robot_pos, obs.sensor_radius, dets, obs.rescued), sig

    def boundary(self, track) -> float:
        from .clearance import is_fresh
        fresh = self.cfg.conf_fresh and is_fresh(track, self.cfg.n_fresh)
        return self.cbf.region_radius(track.sigma, self.cfg.conf_fresh if fresh else None)

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
            if self.cfg.mpc_fast:
                discs += [(f[0], self.cbf.region_radius(f[1], self.cfg.conf_fresh or None)) for f in self._fast_active(t)]
        leg = None
        if (self.cfg.legibility > 0 or self.cfg.legibility_heading > 0) and self.target is not None:
            near = [tr for tr in tracks if np.linalg.norm(tr.mean - pos) < 25.0]
            if any(tr is self.target for tr in near) and len(near) > 1:
                goals = np.array([tr.mean for tr in near])
                k = next(i for i, tr in enumerate(near) if tr is self.target)
                lam = self.cfg.legibility * self._legibility_weight(self.target)
                lam_h = self.cfg.legibility_heading * self._legibility_weight(self.target)
                if lam > 0 or lam_h > 0:
                    leg = (goals, k, self.approach_start, lam, self.cfg.obs_beta, lam_h)
        mpc_hazards = self.hazards if self.cfg.hazard_aware_planner else ()
        u_nom = np.array(self.mpc.solve(pos, heading, v_cap, wp, mpc_hazards, discs, leg))
        if not self.cfg.use_cbf or not self.cfg.apply_filter:
            self.last_info = {"modified": False, "feasible": True}
            return u_nom
        from .clearance import is_fresh
        ell = [(tr.mean, np.eye(2) * tr.sigma**2,
                self.cfg.conf_fresh if self.cfg.conf_fresh and is_fresh(tr, self.cfg.n_fresh) else None) for tr in tracks]
        ell += [(q, np.eye(2) * self.cfg.sigma_floor**2) for q in self.keepout]
        ell += [(f[0], np.eye(2) * f[1]**2, self.cfg.conf_fresh or None) for f in self._fast_active(t)]
        u, info = self.cbf.filter(pos, heading, u_nom, v_cap, self.hazards, ell)
        self.last_info = info
        return u
