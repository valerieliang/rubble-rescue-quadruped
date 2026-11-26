"""One set of metrics for every safety experiment, so results from different tests can be compared.

Per condition (a list of episode or run rows) ``summarize`` reports:

  viol_rate, viol_lo, viol_hi   share of runs with any entry inside ``D_EVAL`` of a survivor, with a 95% Wilson interval
                                (the interval matters: 0 of 60 is "below 6%", not "zero")
  viol_s                        mean seconds spent inside ``D_EVAL`` per run (exposure; a run grazing the line for 0.1 s
                                and one parked inside it for a minute both count once in viol_rate)
  clear_min, clear_p5           smallest and 5th-percentile closest approach, and ``slack`` = clear_p5 - D_EVAL
                                (how far the worst runs stay from the line: a filter at 0.1 m slack is not the same
                                as one at 1.4 m)
  score, rescue                 mean normalised score and rescue rate
and ``paired_diff`` gives the mean paired score difference with its 95% half-width, the statistic used for every
cost comparison (same sites, same detections, so the site-to-site spread cancels).

Two derived quantities put cost and safety on one axis:

  cost_per_violation_avoided    score lost per percentage point of violation rate removed, between two conditions
  speed_for_zero                the largest speed with no violation in a sweep over speeds
"""
from __future__ import annotations

import numpy as np

D_EVAL = 1.0   # m; a violation is the robot centre within this of a true survivor


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, mid - half)), float(min(1.0, mid + half))


def paired_diff(a, b) -> tuple[float, float]:
    """Mean of b - a over matched pairs and its 95% half-width (normal approximation)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else float("nan"), float("nan")
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


def summarize(rows, dt: float = 0.1, score_key: str = "norm_class") -> dict:
    """Metrics for one condition. Rows are ``run_ctrl_job`` rows (or any dicts with the same keys)."""
    n = len(rows)
    k = int(sum(r["any_clear_viol"] for r in rows))
    lo, hi = wilson(k, n)
    clear = np.array([r["min_clearance"] for r in rows], float)
    out = {"n": n, "viol": k, "viol_rate": k / n if n else float("nan"), "viol_lo": lo, "viol_hi": hi,
           "viol_s": float(np.mean([r["clear_viol_steps"] for r in rows])) * dt,
           "clear_min": float(clear.min()), "clear_p5": float(np.quantile(clear, 0.05)),
           "score": float(np.mean([r[score_key] for r in rows])),
           "rescue": float(np.mean([r["rescue_rate"] for r in rows]))}
    out["slack"] = out["clear_p5"] - D_EVAL
    return out


def fmt(s: dict) -> str:
    return (f"viol {s['viol']:3d}/{s['n']:<3d} ({100 * s['viol_rate']:4.1f}%, 95% {100 * s['viol_lo']:4.1f}-"
            f"{100 * s['viol_hi']:4.1f}) | viol-s {s['viol_s']:5.2f} | closest min {s['clear_min']:4.2f} "
            f"p5 {s['clear_p5']:4.2f} slack {s['slack']:+5.2f} | score {s['score']:.3f} rescue {s['rescue']:.2f}")


def cost_per_violation_avoided(base: dict, other: dict) -> float:
    """Score lost (other vs base) per percentage point of violation rate removed. nan if no violations were removed."""
    removed = 100 * (base["viol_rate"] - other["viol_rate"])
    return float((base["score"] - other["score"]) / removed) if removed > 1e-9 else float("nan")


def speed_for_zero(by_speed: dict[float, int]) -> float:
    """Largest speed (m/s) such that it and every lower speed had zero violations; 0 if even the lowest violates."""
    ok = 0.0
    for v in sorted(by_speed):
        if by_speed[v] > 0:
            break
        ok = v
    return ok
