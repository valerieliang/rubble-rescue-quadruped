"""Experiment runner for Q3-Q4 (control-rate sim) and Q5 (approach harness).

A job is a dict: ``seed``, ``variant`` (a key of ``stack.VARIANTS``), ``label`` (free-text sweep tag)
and any overrides. Override keys are routed by name:

    prior:     sigma_pos, n_false_pos, p_miss, p_class_correct      (PriorConfig)
    sim:       sigma_meas, sensor_radius                            (robot-side localization noise, range)
    site:      n_survivors, frac_mobile                             (behavior.make_site)
    behavior:  p_oom, walk_speed, ...                               (BehaviorConfig)
    reliability                                                     (sets both the true and the assumed q)
    anything else that is a StackConfig field                       (conf_p, hazard_aware_planner, ...)

Every method sees the same site and the same drone detections for a given seed (paired comparison).
"""
from __future__ import annotations

import csv
from dataclasses import replace
from functools import lru_cache
from multiprocessing import Pool

import numpy as np

from .ctrl_behavior import BehaviorConfig, Mode, assign_modes, make_site
from .ctrl_sim import run_episode_ctrl
from .drone_prior import PriorConfig
from .interfaces import REWARDS, ResponseState
from .metrics import oracle_score, rescue_rate, rescue_score, time_to_critical
from .stack import R_RESCUE, VARIANTS, StackConfig

PRIOR_KEYS = set(PriorConfig.__dataclass_fields__)
BEH_KEYS = set(BehaviorConfig.__dataclass_fields__) - {"reliability"}
SITE_KEYS = {"n_survivors", "frac_mobile"}
SIM_KEYS = {"sigma_meas", "sensor_radius"}
STACK_KEYS = set(StackConfig.__dataclass_fields__) - {"reliability"}
META = {"seed", "variant", "label", "T", "plant"}   # plant: "unicycle" (default) or "mujoco" (learned Go2)


def usable_seeds(n: int, start: int = 0) -> list[int]:
    """The first n seeds whose robot start is not inside a hazard (it would count as a violation at t=0)."""
    out, s = [], start
    while len(out) < n:
        site = make_site(s)
        if not any(r.kind == "hazard" and r.xmin <= site.start[0] <= r.xmax and r.ymin <= site.start[1] <= r.ymax
                   for r in site.regions):
            out.append(s)
        s += 1
    return out


@lru_cache(maxsize=None)
def _oracles(seed: int, n_survivors, frac_mobile, T: float, reward: str = "base"):
    site = make_site(seed, n_survivors, frac_mobile)
    return (oracle_score(site, T, standoff=R_RESCUE, state_weighted=REWARDS[reward]),
            oracle_score(site, T, standoff=R_RESCUE))


def _inference_stats(site, log, beh: BehaviorConfig) -> dict:
    modes = assign_modes(site, beh, np.random.default_rng(site.seed + 7919))
    n = ok = mn = mok = en = eok = 0
    t_sum, t_n = 0.0, 0
    for sid, tr in log.state_trace.items():
        s = site.survivors[sid]
        pz = [np.array([p[0], p[1], p[2] + p[3]]) for _, p in tr]
        correct = [int(np.argmax(q)) == int(s.state) for q in pz]
        n += 1
        ok += correct[-1]
        if modes[sid] == Mode.ERRATIC:
            en += 1
            eok += correct[-1]
        else:
            last = tr[-1][1]
            pm = np.array([last[0] + last[1], last[2], last[3]])
            mn += 1
            mok += int(np.argmax(pm)) == int(modes[sid])
        if correct[-1]:
            k = len(correct) - 1
            while k > 0 and correct[k - 1]:
                k -= 1
            t_sum += tr[k][0] - tr[0][0]
            t_n += 1
    return {"n_tracked": n, "n_correct": ok, "mode_n": mn, "mode_correct": mok, "erratic_n": en,
            "erratic_correct": eok, "t_correct_sum": t_sum, "t_correct_n": t_n}


def run_ctrl_job(job: dict) -> dict:
    job = dict(job)
    seed, variant, label, T = job["seed"], job["variant"], job.get("label", ""), job.get("T", 600.0)
    ov = {k: v for k, v in job.items() if k not in META}
    prior = replace(PriorConfig(), **{k: v for k, v in ov.items() if k in PRIOR_KEYS})
    beh_kw = {k: v for k, v in ov.items() if k in BEH_KEYS}
    stack_kw = {k: v for k, v in ov.items() if k in STACK_KEYS}
    if "reliability" in ov:
        beh_kw["reliability"] = ov["reliability"]
        stack_kw["reliability"] = ov["reliability"]
    beh = BehaviorConfig(**beh_kw)
    cfg = replace(VARIANTS[variant], **stack_kw)
    site = make_site(seed, ov.get("n_survivors"), ov.get("frac_mobile"))
    sim_kw = {k: v for k, v in ov.items() if k in SIM_KEYS}
    plant = None
    if job.get("plant") == "mujoco":
        from .mj_plant import Go2MujocoPlant   # local import: needs mujoco and onnxruntime
        plant = Go2MujocoPlant()
    log = run_episode_ctrl(site, prior, cfg, seed, variant, T=T, behavior=beh,
                           dwell=REWARDS[ov.get("reward", "base")].dwell, plant=plant, **sim_kw)
    rw = ov.get("reward", "base")
    o_state, o_class = _oracles(seed, ov.get("n_survivors"), ov.get("frac_mobile"), T, rw)
    sw = rescue_score(site, log, T, state_weighted=REWARDS[rw])
    cw = rescue_score(site, log, T)
    row = {"seed": seed, "variant": variant, "label": label, **ov,
           "plant": job.get("plant", "unicycle"), "fell": int(log.fell_at is not None),
           "n_survivors_site": len(site.survivors),
           "norm_state": sw / o_state if o_state else float("nan"),
           "norm_class": cw / o_class if o_class else float("nan"),
           "time_to_critical": time_to_critical(site, log, T),
           "rescue_rate": rescue_rate(site, log, T),
           "clear_viol_steps": log.clear_viol_steps, "haz_viol_steps": log.haz_viol_steps,
           "any_clear_viol": int(log.clear_viol_steps > 0), "any_haz_viol": int(log.haz_viol_steps > 0),
           "min_clearance": log.min_clearance, "infeasible_steps": log.infeasible_steps,
           "prompts": log.prompts, **_inference_stats(site, log, beh)}
    return row


def run_jobs(jobs: list[dict], workers: int | None = None) -> list[dict]:
    with Pool(workers) as pool:
        return pool.map(run_ctrl_job, jobs, chunksize=2)


def make_jobs(seeds, variants, label: str = "", **grid) -> list[dict]:
    """Jobs for every (seed, variant) at each point of the Cartesian product of ``grid`` values."""
    pts = [{}]
    for k, vals in grid.items():
        pts = [{**p, k: v} for p in pts for v in vals]
    return [{"seed": s, "variant": v, "label": label, **p} for p in pts for s in seeds for v in variants]


def save_rows(rows: list[dict], path: str) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_rows(path: str) -> list[dict]:
    def conv(v):
        if v == "":
            return float("nan")
        try:
            return float(v)
        except ValueError:
            return {"True": True, "False": False}.get(v, v)
    with open(path) as f:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]


def run_approach_job(job: dict) -> dict:
    from .approach import run_approach
    cfg = replace(StackConfig(), legibility=job["lam"], obs_beta=job.get("obs_beta", 0.7),
                  legibility_heading=job.get("lam_head", 0.0))
    r = run_approach(job["seed"], cfg, ResponseState(job["state"]), bool(job.get("blind", False)),
                     n_distractors=job.get("n_distractors", 1))
    return {**job, "t_total": r.t_total, "path_len": r.path_len, "reached": int(r.reached),
            "ident_cost": r.ident["cost"], "ident_heading": r.ident["heading"], "ident_learned": r.ident["learned"],
            "mean_p_cost": r.mean_p["cost"], "mean_p_heading": r.mean_p["heading"],
            "mean_p_learned": r.mean_p["learned"]}


def run_approach_jobs(jobs: list[dict], workers: int | None = None) -> list[dict]:
    with Pool(workers) as pool:
        return pool.map(run_approach_job, jobs, chunksize=4)
