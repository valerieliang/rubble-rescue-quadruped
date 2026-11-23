"""Shared interfaces between the simulator and the search/belief code.

Everything here is plain dataclasses + numpy, so either side can change its internals without
touching the other. Frame: world frame, metres, x/y on the ground plane.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Protocol

import numpy as np


class Priority(IntEnum):
    """Triage class assigned by the drone prior; higher is more urgent."""
    STABLE = 0
    URGENT = 1
    CRITICAL = 2


class ResponseState(IntEnum):
    """Hidden survivor state z_i (START-style responsiveness/mobility check). Used from Q4 on."""
    UNRESPONSIVE = 0
    TRAPPED = 1
    MOBILE = 2


# Reward is sum_i w_i * exp(-lambda_i * tau_i) * 1[tau_i <= T]; w and lambda come from the class.
# Weights and decay rates are modelling choices, not calibrated to real triage outcomes.
PRIORITY_WEIGHT = {Priority.STABLE: 1.0, Priority.URGENT: 2.0, Priority.CRITICAL: 4.0}
STANDOFF_RADIUS = 1.5   # m; a survivor counts as rescued once the robot is this close
# Triage urgency also depends on the hidden state (START-style: walking wounded can self-evacuate, unresponsive
# survivors cannot). Used only by the state-weighted metrics of the control-rate Q4-Q5 runs; a modelling choice.
STATE_FACTOR = {ResponseState.UNRESPONSIVE: 1.5, ResponseState.TRAPPED: 1.0, ResponseState.MOBILE: 0.4}
SURVIVOR_KEEPOUT = 1.0  # m; the robot must stay at least this far from a survivor (below STANDOFF_RADIUS)
PRIORITY_DECAY = {Priority.STABLE: 0.002, Priority.URGENT: 0.005, Priority.CRITICAL: 0.01}  # 1/s


@dataclass(frozen=True)
class RewardModel:
    """Reward for reaching a survivor of class c and hidden state z at time tau (control-rate Q4-Q5 runs).

    value = w_c * factor_z * exp(-lambda_c * decay_mult_z * tau) * 1[not (z mobile and tau > evac_time)]

    ``BASE_REWARD`` keeps the original state-weighted score (the state only rescales the reward, so it
    seldom changes the best visiting order). ``CONSEQ_REWARD`` gives the hidden state a real consequence:
    unresponsive survivors deteriorate faster (decay x2) and mobile ones self-evacuate (reward is 0 once
    ``evac_time`` has passed). Both are modelling choices, not calibrated to real triage outcomes.
    """
    factor: dict = field(default_factory=lambda: dict(STATE_FACTOR))
    decay_mult: dict = field(default_factory=lambda: {z: 1.0 for z in ResponseState})
    evac_time: float = float("inf")
    dwell: dict = field(default_factory=lambda: {z: 0.0 for z in ResponseState})   # s of on-site handling after arrival

    def value(self, priority: "Priority", state: "ResponseState", tau: float) -> float:
        if state == ResponseState.MOBILE and tau > self.evac_time:
            return 0.0
        return (PRIORITY_WEIGHT[priority] * self.factor[state]
                * float(np.exp(-PRIORITY_DECAY[priority] * self.decay_mult[state] * tau)))


BASE_REWARD = RewardModel()
CONSEQ_REWARD = RewardModel(decay_mult={ResponseState.UNRESPONSIVE: 2.0, ResponseState.TRAPPED: 1.0,
                                        ResponseState.MOBILE: 1.0}, evac_time=240.0)
CONSEQ_STRONG_REWARD = RewardModel(decay_mult={ResponseState.UNRESPONSIVE: 4.0, ResponseState.TRAPPED: 1.0,
                                               ResponseState.MOBILE: 1.0}, evac_time=120.0)
DWELL = {ResponseState.UNRESPONSIVE: 30.0, ResponseState.TRAPPED: 15.0, ResponseState.MOBILE: 0.0}
TRIAGE_REWARD = RewardModel(decay_mult={ResponseState.UNRESPONSIVE: 2.0, ResponseState.TRAPPED: 1.0,
                                        ResponseState.MOBILE: 1.0}, evac_time=240.0, dwell=DWELL)
REWARDS = {"base": BASE_REWARD, "conseq": CONSEQ_REWARD, "conseq_strong": CONSEQ_STRONG_REWARD, "triage": TRIAGE_REWARD}


@dataclass(frozen=True)
class Survivor:
    id: int
    pos: np.ndarray            # (2,) ground-truth position
    priority: Priority
    state: ResponseState = ResponseState.TRAPPED   # hidden from the robot


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle; kind is 'rubble' or 'hazard'."""
    kind: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class Site:
    """Map format. Generated from a seed so every run is reproducible."""
    seed: int
    bounds: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax
    regions: tuple[Region, ...]
    survivors: tuple[Survivor, ...]
    start: np.ndarray                            # (2,) robot start position


@dataclass(frozen=True)
class Detection:
    """One aerial detection from the drone prior (may be a false positive)."""
    pos: np.ndarray            # (2,) noisy position estimate
    priority: Priority         # (possibly wrong) class estimate
    source_id: int | None = None   # ground-truth survivor id; None for false positives (eval only)
    state_hint: int | None = None  # drone's noisy guess of the hidden ResponseState (e.g. seen moving); None = no cue


@dataclass(frozen=True)
class Observation:
    """What the robot senses at one time step (observation API)."""
    t: float
    robot_pos: np.ndarray      # (2,) estimated robot position
    sensor_radius: float
    detected: tuple[Detection, ...] = ()   # survivors actually sensed within the footprint
    rescued: tuple[np.ndarray, ...] = ()   # positions of survivors reached since the last observation


class Policy(Protocol):
    """Search layer. Called at ~1 Hz; returns the next waypoint (2,) for the waypoint tracker."""

    def act(self, obs: Observation) -> np.ndarray: ...


# Travel-time estimate between two ground points (s). The sim/terrain side owns this.
TravelTime = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class RunLog:
    """Metrics log, one per run, saved as .npz/json by the batch runner."""
    site_seed: int
    method: str
    rescue_times: dict[int, float] = field(default_factory=dict)   # survivor id -> time rescued
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)  # t, x, y
