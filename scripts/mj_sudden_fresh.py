"""Sudden-threat test with a higher confidence on fresh tracks only (experiment 3).

Usage (repo root):  PYTHONPATH=. python scripts/mj_sudden_fresh.py [n_seeds] [workers]

Same setup as mj_sudden_threat.py (robot walking with no survivor known; at 3 s an estimate appears 1.5-6 m ahead,
0.5 m measurement noise), filter on only. While fewer than ``n_fresh`` detections have been fused into the
estimate the filter uses confidence ``conf_fresh`` instead of 0.7; after that it is back to 0.7.
  single: the first noisy estimate is never corrected (n stays 1), so only conf_fresh matters (n_fresh = 1 detection).
  fused (fused=1): 10 Hz re-detection at 90%; "fresh" lasts for the first 1, 5, 10 or 20 detections (about 0.1-2.2 s).
  1 Hz  (fused=2): one detection per second at 90%, what the stack's belief update does; fresh = first 1, 3, 5 or 10.
Writes runs/mj_sudden_fresh.csv and prints violations / runs and the closest approach per setting.
"""
import csv
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "scripts")
import mj_sudden_threat as st   # noqa: E402

CONFS = (None, 0.9, 0.95, 0.99)          # None: baseline (0.7 everywhere)
N_FRESH = {1: (1, 5, 10, 20), 2: (1, 3, 5, 10)}    # fresh = fewer than n+1 detections fused; by fused mode
_PLANT = {}


def work(job):
    name, v, d, lat, seed, fused, conf, n = job
    if name == "mujoco" and "p" not in _PLANT:
        from rescue.mj_plant import Go2MujocoPlant
        _PLANT["p"] = Go2MujocoPlant()
    plant = _PLANT["p"] if name == "mujoco" else None
    r = st.run(plant, v, True, d, lat, seed, bool(fused), conf_fresh=conf, n_fresh=n + 1,
               fuse_every=10 if fused == 2 else 1)
    return dict(plant=name, speed=v, d_appear=d, lateral=lat, seed=seed, fused=fused, conf_fresh=conf or 0.7,
                n_fresh=n if conf else 0, **r)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    jobs = []
    for name in ("unicycle", "mujoco"):
        for v in st.SPEEDS:
            for d in st.D_APPEAR:
                for lat in st.LATERAL:
                    for seed in range(n_seeds):
                        for fused in (0, 1, 2):
                            for conf in CONFS:
                                for n in ((0,) if conf is None else N_FRESH[fused] if fused else (1,)):
                                    jobs.append((name, v, d, lat, seed, fused, conf, n))
    print(len(jobs), "runs ...", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(work, jobs, chunksize=8)
    with open("runs/mj_sudden_fresh.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def summ(sel):
        viol = sum(x["dmin"] < st.D_CLEAR for x in sel)
        return f"{viol:3d}/{len(sel):3d}  closest {min(x['dmin'] for x in sel):4.2f}  infeas {np.mean([x['infeasible'] for x in sel]):5.1f}"

    for fused in (0, 1, 2):
        print(f"\n{['single estimate', 'fused 10 Hz', 'fused 1 Hz'][fused]}: filtered runs within 1 m / runs, all cells (1,440 per setting at 30 seeds)")
        for conf in CONFS:
            for n in ((0,) if conf is None else N_FRESH[fused] if fused else (1,)):
                sel = [x for x in rows if x["fused"] == fused and x["conf_fresh"] == (conf or 0.7) and x["n_fresh"] == (n if conf else 0)]
                tag = "baseline 0.7" if conf is None else f"conf {conf:.2f}, first {n}"
                by_d = "  ".join(f"{d:.1f}:{sum(x['dmin'] < st.D_CLEAR for x in sel if x['d_appear'] == d):2d}" for d in st.D_APPEAR)
                print(f"  {tag:22s} {summ(sel)} | viol by d_appear  {by_d}")


if __name__ == "__main__":
    main()
