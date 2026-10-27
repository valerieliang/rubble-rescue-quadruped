"""Shared interfaces between the simulator (Sahil) and search/belief code (Valerie).

DRAFT: to be agreed on Wed Oct 28. Everything here is plain dataclasses + numpy so
either side can change a field without touching the other's internals.
Frame: world frame, metres, x/y on the ground plane.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

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
# Values are placeholders to tune.
PRIORITY_WEIGHT = {Priority.STABLE: 1.0, Priority.URGENT: 2.0, Priority.CRITICAL: 4.0}
PRIORITY_DECAY = {Priority.STABLE: 0.002, Priority.URGENT: 0.005, Priority.CRITICAL: 0.01}  # 1/s


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


@dataclass(frozen=True)
class Observation:
    """What the robot senses at one time step (observation API)."""
    t: float
    robot_pos: np.ndarray      # (2,) estimated robot position
    sensor_radius: float
    detected: tuple[Detection, ...] = ()   # survivors actually sensed within the footprint


@dataclass
class RunLog:
    """Metrics log, one per run, saved as .npz/json by the batch runner."""
    site_seed: int
    method: str
    rescue_times: dict[int, float] = field(default_factory=dict)   # survivor id -> time rescued
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)  # t, x, y
