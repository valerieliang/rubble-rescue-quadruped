"""Does the tracking-error pad (and the confidence ellipse) ever bind? Head-on stress test with injected error.

Usage (repo root):  PYTHONPATH=. python scripts/diagnostics/pad_binding.py [n_runs] [workers]

In every closed-loop rubble run the pad and the ellipse were slack: the robot stayed about 2.4 m from survivors,
so the filter's own margins (d_clear, control-point offset) did all the work. A head-on run is the opposite: a
nominal controller drives at the survivor and the CBF filter is the only thing that stops the robot, so the
barrier is active on every run. To make the *pad* matter, the plant here is the unicycle plus an unmodelled
position disturbance (rescue.dist_plant, OU velocity noise), which the planner and filter do not know about.
The disturbance level is labelled by its measured 1 s tracking error (95% quantile, the statistic the pad is
derived from), comparable to Table 19.

Per (disturbance, pad, speed, case) it reports runs inside 1 m of the true survivor, and the closest approach.
Cases (the filter's region around the estimate; the true survivor is at the estimate plus noise where noted):
  exact     the estimate is exact (region collapses to a point), only tracking error can break the barrier
  ellipse   estimate error std 0.5 m (the stack's floor), confidence 0.7 as in the stack: the true survivor is
            drawn from N(estimate, 0.5^2 I), so the ellipse can bind too
  no_ellipse  as 'ellipse' but confidence 0.01, so the ellipse term is gone and only the pad covers tracking error
Writes runs/pad_binding.csv.
"""
import csv
import sys
from multiprocessing import Pool

import numpy as np

from rescue.dist_plant import DisturbedUnicycle, measured_tracking_error
from rescue.safety import CBFFilter, SafetyParams

D_CLEAR, DT, T, START, W_MAX = 1.0, 0.1, 15.0, 6.0, 1.0
SIGMAS_D = (0.0, 0.2, 0.4, 0.6)
PADS = (0.0, 0.25, 0.5, 1.0)
SPEEDS = (0.5, 1.0)
CASES = {"exact": (1e-3, 0.7, 0.0), "ellipse": (0.5, 0.7, 0.5), "no_ellipse": (0.5, 0.01, 0.5)}   # region std, conf, estimate error std


def run(job):
    sigma_d, pad, v_nom, case, seed = job
    s_reg, conf, s_est = CASES[case]
    rng = np.random.default_rng(seed)
    est = np.zeros(2)
    truth = rng.normal(0.0, s_est, 2) if s_est else est.copy()
    plant = DisturbedUnicycle(sigma_d, w_max=W_MAX, seed=seed + 12345)
    filt = CBFFilter(SafetyParams(conf_p=conf, d_clear=D_CLEAR, eps_track=pad, w_max=W_MAX))
    ang = rng.uniform(-np.pi, np.pi)
    pos = est + START * np.array([np.cos(ang), np.sin(ang)])
    heading = float(ang + np.pi + rng.choice(np.radians([0, 10, 20, 30, 45, -20])))
    plant.reset(pos, heading)
    dmin = np.inf
    S = np.eye(2) * s_reg**2
    for _ in range(int(round(T / DT))):
        d = est - pos
        err = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
        u, _ = filt.filter(pos, heading, np.array([v_nom, float(np.clip(2.0 * err, -W_MAX, W_MAX))]), v_nom, [], [(est, S)])
        pos, heading, _ = plant.step(float(u[0]), float(u[1]), DT)
        dmin = min(dmin, float(np.linalg.norm(pos - truth)))
    return dict(sigma_d=sigma_d, pad=pad, speed=v_nom, case=case, seed=seed, dmin=dmin)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    jobs = [(sd, pad, v, case, s) for sd in SIGMAS_D for pad in PADS for v in SPEEDS for case in CASES for s in range(n)]
    print(len(jobs), "runs ...", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(run, jobs, chunksize=16)
    with open("runs/pad_binding.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    gap = {sd: measured_tracking_error(sd) for sd in SIGMAS_D}
    print("\nmeasured 1 s tracking error (95%) by injected sigma_d:", {k: round(v, 2) for k, v in gap.items()})
    for case in CASES:
        print(f"\ncase {case}: runs inside 1 m / runs (closest approach m); both speeds pooled")
        print(f"{'sigma_d':>8s} {'gap95':>6s} | " + "  ".join(f"pad {p:4.2f}".rjust(16) for p in PADS))
        for sd in SIGMAS_D:
            cells = []
            for pad in PADS:
                r = [x for x in rows if (x["sigma_d"], x["pad"], x["case"]) == (sd, pad, case)]
                cells.append(f"{sum(x['dmin'] < D_CLEAR for x in r):3d}/{len(r)} ({min(x['dmin'] for x in r):4.2f})".rjust(16))
            print(f"{sd:8.2f} {gap[sd]:6.2f} | " + "  ".join(cells))


if __name__ == "__main__":
    main()
