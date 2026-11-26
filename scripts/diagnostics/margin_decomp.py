"""Where does the closest approach come from?  Head-on decomposition of  closest approach ~ d_clear + 0.5 m.

Usage (repo root):  PYTHONPATH=. python scripts/margin_decomp.py [n_runs] [workers]

The filter's barrier is zero at distance  R = c_p*sigma + d_clear + eps_track + cp_offset  from the survivor estimate,
measured from a control point that sits cp_offset *ahead* of the robot centre. So when the robot drives straight at the
survivor the centre is cp_offset further out still:

    closest approach (centre) = d_clear + eps_track + 2*cp_offset  -  overshoot          (head-on, exact estimate)

where 'overshoot' is how far the robot carries into the barrier edge (zero at the stock cp_offset of 0.25 m and below).
The 0.5 m in the earlier 'd_clear + 0.5' is therefore 2*cp_offset (eps_track was 0 in those runs), not a tracking pad:
cp_offset is counted once in the barrier's margin and once again geometrically in a head-on approach.

This script sweeps d_clear, cp_offset, eps_track and speed on the ideal unicycle (no disturbance, exact estimate,
region collapsed to a point), fits  closest = a + b*d_clear + c*eps + e*cp_offset + f*speed  for cp_offset <= 0.25 and reports the residual
(the overshoot) by cp_offset, alpha and speed. Writes runs/margin_decomp.csv.
"""
import csv
import sys
from multiprocessing import Pool

import numpy as np

from rescue.dist_plant import DisturbedUnicycle
from rescue.safety import CBFFilter, SafetyParams

DT, T, START, W_MAX = 0.1, 15.0, 6.0, 1.0
D_CLEARS = (0.5, 1.0, 1.5)
CPS = (0.0, 0.25, 0.5)
EPS = (0.0, 0.25)
SPEEDS = (0.5, 0.75, 1.0)
ALPHAS = (0.8, 1.6)


def run(job):
    d_clear, cp, eps, v_nom, alpha, seed = job
    rng = np.random.default_rng(seed)
    est = np.zeros(2)
    plant = DisturbedUnicycle(0.0, w_max=W_MAX, seed=seed)
    filt = CBFFilter(SafetyParams(conf_p=0.7, d_clear=d_clear, eps_track=eps, cp_offset=cp, alpha=alpha, w_max=W_MAX))
    ang = rng.uniform(-np.pi, np.pi)
    pos = est + START * np.array([np.cos(ang), np.sin(ang)])
    heading = float(ang + np.pi + rng.choice(np.radians([0, 10, -10])))
    plant.reset(pos, heading)
    S = np.eye(2) * 1e-6
    dmin = np.inf
    for _ in range(int(round(T / DT))):
        d = est - pos
        err = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
        u, _ = filt.filter(pos, heading, np.array([v_nom, float(np.clip(2.0 * err, -W_MAX, W_MAX))]), v_nom, [], [(est, S)])
        pos, heading, _ = plant.step(float(u[0]), float(u[1]), DT)
        dmin = min(dmin, float(np.linalg.norm(pos - est)))
    return dict(d_clear=d_clear, cp=cp, eps=eps, speed=v_nom, alpha=alpha, seed=seed, dmin=dmin)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    jobs = [(d, cp, e, v, a, s) for d in D_CLEARS for cp in CPS for e in EPS for v in SPEEDS for a in ALPHAS for s in range(n)]
    print(len(jobs), "runs ...", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(run, jobs, chunksize=16)
    with open("runs/margin_decomp.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    base = [r for r in rows if r["alpha"] == 0.8 and r["cp"] <= 0.25]      # stock offset and below; 0.5 m is shown separately
    X = np.array([[1.0, r["d_clear"], r["eps"], r["cp"], r["speed"]] for r in base])
    y = np.array([r["dmin"] for r in base])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    print("\nfit (alpha 0.8):  closest = %.2f + %.2f*d_clear + %.2f*eps_track + %.2f*cp_offset + %.2f*speed   (rms residual %.3f m)"
          % (*coef, float(np.sqrt(np.mean(res ** 2)))))
    print("\nclosest approach minus (d_clear + eps + 2*cp_offset): ~0 = the formula holds, negative = the robot ends closer than it")
    print(f"{'cp_offset':>9s} {'alpha':>6s} {'speed':>6s} | {'mean':>6s} {'5th pct':>8s} {'95th pct':>9s}")
    for cp in CPS:
        for a in ALPHAS:
            for v in SPEEDS:
                o = np.array([r["dmin"] - (r["d_clear"] + r["eps"] + 2 * r["cp"]) for r in rows
                              if (r["cp"], r["alpha"], r["speed"]) == (cp, a, v)])
                print(f"{cp:9.2f} {a:6.1f} {v:6.2f} | {o.mean():+6.3f} {np.quantile(o, 0.05):+8.3f} {np.quantile(o, 0.95):+9.3f}")
    print("\nstock settings (d_clear 1.0, cp_offset 0.25, eps 0.0 as in the margin sweep), mean closest approach by speed:")
    for v in SPEEDS:
        r = [x["dmin"] for x in base if (x["d_clear"], x["cp"], x["eps"], x["speed"]) == (1.0, 0.25, 0.0, v)]
        print(f"  speed {v:.2f}: {np.mean(r):.3f} m (d_clear + 2*cp_offset = 1.50)")


if __name__ == "__main__":
    main()
