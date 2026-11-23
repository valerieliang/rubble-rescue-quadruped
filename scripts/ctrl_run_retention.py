"""Does the 'keep' remedy carry over to the closed-loop stack (10 Hz MPC + CBF, 10 m sensor)?

Usage (repo root):  PYTHONPATH=. python scripts/ctrl_run_retention.py [n_sites] [workers]

Full stack with and without ``StackConfig(retention="keep")`` across drone position noise, same sites and
detections, filter on. Writes runs/ctrl_retention.csv and prints paired differences (keep minus off) and
safety counts (violations must stay at zero).
"""
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds
from rescue.experiments import mean_ci

SIGMAS = [3.0, 6.0, 10.0, 15.0]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    rows = run_jobs(make_jobs(usable_seeds(n), ["full"], "ctrl_retention", retention=["", "keep"],
                              sigma_pos=SIGMAS), workers)
    save_rows(rows, "runs/ctrl_retention.csv")
    print(f"{len(rows)} rows; keep minus off, paired over {n} sites (mean +/- 95% half-width)")
    print(f"{'sigma':>6s} {'score off':>10s} {'score diff':>16s} {'rescue diff':>16s} {'t_crit diff (s)':>18s} "
          f"{'viol off/keep':>14s} {'haz off/keep':>13s}")
    for s in SIGMAS:
        off = {r["seed"]: r for r in rows if r["sigma_pos"] == s and r["retention"] != "keep"}
        on = {r["seed"]: r for r in rows if r["sigma_pos"] == s and r["retention"] == "keep"}
        ks = sorted(set(off) & set(on))
        def d(key):
            x = np.array([on[k][key] - off[k][key] for k in ks], float)
            x = x[~np.isnan(x)]
            m, h = mean_ci(x)
            return m, h
        sc, rr, tc = d("norm_class"), d("rescue_rate"), d("time_to_critical")
        print(f"{s:6.0f} {np.mean([off[k]['norm_class'] for k in ks]):10.3f} {sc[0]:+8.3f}+/-{sc[1]:.3f} "
              f"{rr[0]:+8.3f}+/-{rr[1]:.3f} {tc[0]:+9.1f}+/-{tc[1]:.1f} "
              f"{sum(off[k]['any_clear_viol'] for k in ks):6.0f}/{sum(on[k]['any_clear_viol'] for k in ks):<6.0f} "
              f"{sum(off[k]['any_haz_viol'] for k in ks):6.0f}/{sum(on[k]['any_haz_viol'] for k in ks):<6.0f}")


if __name__ == "__main__":
    main()
