"""Simulated aerial (drone) survey: the noisy prior over survivor locations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import Detection, Priority, Site


@dataclass(frozen=True)
class PriorConfig:
    sigma_pos: float = 3.0          # std of Gaussian position noise per axis (m)
    p_miss: float = 0.1             # probability a true survivor is not detected
    n_false_pos: float = 1.0        # expected number of false positives (Poisson mean)
    p_class_correct: float = 0.8    # probability the priority class is reported correctly


def generate_detections(site: Site, cfg: PriorConfig, rng: np.random.Generator) -> list[Detection]:
    """Sample the drone's detections for a site."""
    xmin, ymin, xmax, ymax = site.bounds
    dets: list[Detection] = []
    for s in site.survivors:
        if rng.random() < cfg.p_miss:
            continue
        pos = s.pos + rng.normal(0.0, cfg.sigma_pos, size=2)
        dets.append(Detection(pos, _noisy_class(s.priority, cfg.p_class_correct, rng), s.id))
    for _ in range(rng.poisson(cfg.n_false_pos)):
        pos = rng.uniform((xmin, ymin), (xmax, ymax))
        dets.append(Detection(pos, Priority(rng.integers(len(Priority))), None))
    return dets


def _noisy_class(true: Priority, p_correct: float, rng: np.random.Generator) -> Priority:
    if rng.random() < p_correct:
        return true
    others = [p for p in Priority if p != true]
    return others[rng.integers(len(others))]
