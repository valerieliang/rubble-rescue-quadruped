"""Does the tracking-error pad bind on the real (simulated) Go2?  Head-on test, the MuJoCo counterpart of pad_binding.py.

Usage (repo root, a Python with mujoco and onnxruntime):  PYTHONPATH=. python scripts/mj_pad_binding.py [n_runs] [workers]

pad_binding.py showed the pad binding on a unicycle with *injected* tracking error. Here the error is the Go2 policy's own:
a nominal controller drives the learned Go2 straight at a survivor and the CBF filter is the only thing that stops it, on
flat ground (policy.onnx) and on a rubble patch of a site heightfield (the rough policy, policy_rough.onnx). Measured
1 s tracking errors for these two (Table 19): about 0.2 m flat and 0.36 m on rubble. The same statistic is measured here,
inside the runs: the 95th percentile, over the moving part of every run, of the distance between the robot's pose and the
pose the ideal unicycle would reach from the pose one second earlier under the commands the filter issued.

Cases: 'exact' (the estimate is the survivor's true position, region collapsed) and 'ellipse' (estimate error std 0.5 m, the
stack's floor, confidence 0.7: the true survivor is drawn from N(estimate, 0.5^2 I) so the ellipse can bind too).
Per (plant, case, pad, speed) it reports runs inside 1 m with a Wilson interval, the closest approach, falls, and the
observed gap. Writes runs/mj_pad_binding.csv.
"""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

from rescue.ctrl_behavior import make_site
from rescue.safety import CBFFilter, SafetyParams
from rescue.scorecard import wilson

D_CLEAR, DT, T, START, W_MAX = 1.0, 0.1, 15.0, 6.0, 1.0
PADS = (0.0, 0.25, 0.5)
SPEEDS = (0.5, 1.0)
PLANTS = ("flat", "rubble", "rubble_flatpol")      # rubble_flatpol: rubble with the flat-ground policy (the larger tracking error, Table 19)
CASES = {"exact": (1e-3, 0.7, 0.0), "ellipse": (0.5, 0.7, 0.5)}
ROUGH = os.path.abspath("sim/go2/policy_rough.onnx")
_P: dict = {}


def _plant(name):
    if name not in _P:
        from rescue.mj_plant import Go2MujocoPlant
        _P[name] = (Go2MujocoPlant() if name == "flat" else Go2MujocoPlant(rubble=True, policy=ROUGH) if name == "rubble"
                    else Go2MujocoPlant(rubble=True))
    return _P[name]


def _rubble_site(seed):
    """First seeded site with a rubble region; the survivor goes at its centre, the robot starts START m away inside bounds."""
    s = seed
    while True:
        site = make_site(s)
        regs = [r for r in site.regions if r.kind == "rubble"]
        if regs:
            r = regs[0]
            c = np.array([(r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2])
            xmin, ymin, xmax, ymax = site.bounds
            for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                p = c + START * np.array([np.cos(ang), np.sin(ang)])
                if xmin + 1 < p[0] < xmax - 1 and ymin + 1 < p[1] < ymax - 1:
                    return site, c, p
        s += 1000


def _gap95(poses, cmds):
    """95th percentile 1 s gap between the actual pose and the ideal unicycle driven by the issued commands."""
    n, k = len(poses), int(round(1.0 / DT))
    errs = []
    for i in range(k, n):
        if np.mean([c[0] for c in cmds[i - k:i]]) < 0.2:
            continue
        p, th = poses[i - k][:2].copy(), poses[i - k][2]
        for v, w in cmds[i - k:i]:
            th += w * DT
            p = p + v * DT * np.array([np.cos(th), np.sin(th)])
        errs.append(float(np.linalg.norm(poses[i][:2] - p)))
    return errs


def run(job):
    pname, case, pad, v_nom, seed = job
    s_reg, conf, s_est = CASES[case]
    rng = np.random.default_rng(seed)
    plant = _plant(pname)
    if pname != "flat":
        site, est, pos = _rubble_site(seed)
        plant.set_site(site)
    else:
        est = np.zeros(2)
        ang = rng.uniform(-np.pi, np.pi)
        pos = est + START * np.array([np.cos(ang), np.sin(ang)])
    truth = est + rng.normal(0.0, s_est, 2) if s_est else est.copy()
    heading = float(np.arctan2(*(est - pos)[::-1]) + rng.choice(np.radians([0, 10, 20, -20])))
    plant.reset(pos, heading)
    filt = CBFFilter(SafetyParams(conf_p=conf, d_clear=D_CLEAR, eps_track=pad, w_max=W_MAX))
    S = np.eye(2) * s_reg ** 2
    poses, cmds, dmin, fell = [(pos[0], pos[1], heading)], [], np.inf, 0
    for _ in range(int(round(T / DT))):
        d = est - pos
        err = (np.arctan2(d[1], d[0]) - heading + np.pi) % (2 * np.pi) - np.pi
        u, _ = filt.filter(pos, heading, np.array([v_nom, float(np.clip(2.0 * err, -W_MAX, W_MAX))]), v_nom, [], [(est, S)])
        pos, heading, _ = plant.step(float(u[0]), float(u[1]), DT)
        cmds.append((float(u[0]), float(u[1])))
        poses.append((pos[0], pos[1], heading))
        dmin = min(dmin, float(np.linalg.norm(pos - truth)))
        if plant.fallen:
            fell = 1
            break
    g = _gap95(np.array(poses), cmds)
    return dict(plant=pname, case=case, pad=pad, speed=v_nom, seed=seed, dmin=dmin, fell=fell,
                gap95=float(np.quantile(g, 0.95)) if g else float("nan"))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    jobs = [(p, c, pad, v, s) for p in PLANTS for c in CASES for pad in PADS for v in SPEEDS for s in range(n)]
    jobs.sort(key=lambda j: (j[0], j[4]))                      # one worker tends to keep one plant
    print(len(jobs), "runs ...", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(run, jobs, chunksize=6)
    os.makedirs("runs", exist_ok=True)
    with open("runs/mj_pad_binding.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for p in PLANTS:
        g = [x["gap95"] for x in rows if x["plant"] == p and np.isfinite(x["gap95"])]
        print(f"\n{p}: observed 1 s tracking gap, 95th percentile of per-run values: median {np.median(g):.2f} m, "
              f"95th pct {np.quantile(g, 0.95):.2f} m")
        for case in CASES:
            print(f"  case {case}: runs inside 1 m / runs (95% interval); closest approach min / 5th pct (m); falls")
            for v in SPEEDS:
                cells = []
                for pad in PADS:
                    r = [x for x in rows if (x["plant"], x["case"], x["pad"], x["speed"]) == (p, case, pad, v)]
                    k = sum(x["dmin"] < D_CLEAR for x in r)
                    lo, hi = wilson(k, len(r))
                    d = np.array([x["dmin"] for x in r])
                    cells.append(f"pad {pad:4.2f}: {k:2d}/{len(r)} ({100 * lo:2.0f}-{100 * hi:3.0f}%) {d.min():4.2f}/{np.quantile(d, .05):4.2f} f{sum(x['fell'] for x in r)}")
                print(f"    speed {v:.1f} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
