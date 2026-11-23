"""Experiment runner: seeded jobs over (site seed, method, prior config), run in parallel to CSV.

A job fixes the site seed and the drone-detection seed, so every method sees the same site and the
same detections (paired comparison).
"""
from __future__ import annotations

import csv
import os
from dataclasses import replace
from functools import lru_cache, partial
from multiprocessing import Pool

import numpy as np

from . import baselines
from .drone_prior import PriorConfig
from .metrics import oracle_score, summarize

if os.environ.get("RESCUE_SIM", "toy") == "go2":
    from .sim import generate_site, run_episode
else:
    from .toy_sim import generate_site, run_episode

METHODS = {"coverage": baselines.coverage_policy,
           "nearest": baselines.nearest_first_policy,
           "priority": baselines.priority_policy}
# Retention remedies (search.PriorSearch ``retention``), each on top of the priority policy.
RETENTION = {"prio_resume": "resume", "prio_region": "region", "prio_keep": "keep",
             "prio_region_keep": "region,keep", "prio_resume_keep": "resume,keep"}
METHODS.update({m: partial(baselines.priority_policy, retention=f) for m, f in RETENTION.items()})
# keep with another region size (sensitivity): keep the track until its k-sigma region is searched
METHODS.update({f"prio_keep_k{k}": partial(baselines.priority_policy, retention="keep", keep_k=float(k))
                for k in (1, 3)})
FIELDS = ["seed", "method", "sigma_pos", "n_false_pos", "p_miss", "p_class_correct",
          "score", "oracle", "norm_score", "time_to_critical", "rescue_rate",
          "T", "n_survivors", "hazard_steps", "survivor_steps", "violation_steps", "violation_rate"]


@lru_cache(maxsize=None)
def _site_and_oracle(seed: int, T: float, n_survivors: int | None = None):
    site = generate_site(seed) if n_survivors is None else generate_site(seed, n_survivors=n_survivors)
    return site, oracle_score(site, T)


def run_job(job: tuple) -> dict:
    seed, method, cfg, T, *rest = job
    n_surv = rest[0] if rest else None
    site, oracle = _site_and_oracle(seed, T, n_surv)
    factory = METHODS[method]
    log = run_episode(site, lambda s, d: factory(s, d, cfg, horizon=T), cfg, seed, method, T=T)
    row = summarize(site, log, T, oracle)
    return {"seed": seed, "method": method, "sigma_pos": cfg.sigma_pos,
            "n_false_pos": cfg.n_false_pos, "p_miss": cfg.p_miss,
            "p_class_correct": cfg.p_class_correct, "T": T,
            "n_survivors": len(site.survivors), **row}


def run_grid(cfgs: list[PriorConfig], methods: list[str], seeds: range, T: float = 600.0,
             workers: int | None = None, n_survivors: int | None = None) -> list[dict]:
    jobs = [(s, m, c, T, n_survivors) for c in cfgs for s in seeds for m in methods]
    with Pool(workers) as pool:
        return pool.map(run_job, jobs, chunksize=8)


def sweep_cfgs(base: PriorConfig, **grid) -> list[PriorConfig]:
    """Cartesian product over the given PriorConfig fields."""
    cfgs = [base]
    for name, values in grid.items():
        cfgs = [replace(c, **{name: v}) for c in cfgs for v in values]
    return cfgs


def save_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return [{k: (v if k == "method" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


def mean_ci(x) -> tuple[float, float]:
    """Mean and 95% half-width (normal approx over sites); NaNs dropped."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def paired_diff(rows: list[dict], a: str, b: str, metric: str) -> dict:
    """a - b per site (same seed + config); returns mean, 95% half-width, and fraction of sites a > b."""
    by = {}
    for r in rows:
        by.setdefault((r["seed"], r["sigma_pos"], r["n_false_pos"], r["p_class_correct"], r.get("T"),
                       r.get("n_survivors")), {})[r["method"]] = r[metric]
    d = np.array([v[a] - v[b] for v in by.values() if a in v and b in v], float)
    d = d[~np.isnan(d)]
    m, h = mean_ci(d)
    return {"mean": m, "ci": h, "win_rate": float((d > 0).mean()) if len(d) else float("nan"), "n": len(d)}
