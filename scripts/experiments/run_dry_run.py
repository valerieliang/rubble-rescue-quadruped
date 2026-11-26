"""Dry run: all methods on N sites through experiments.run_grid. Writes runs/dry_run_<sim>.csv.

Usage: python scripts/experiments/run_dry_run.py [n_sites] [--sim {toy,go2}] [--workers W]
"""
import argparse
import os
import time

SWEEP_JOBS_N40 = 6 * 40 * 3 + 6 * 40 * 3 + 36 * 40 * 1 + 160 * 2   # what run_sweeps.py runs by default


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("n_sites", nargs="?", type=int, default=20)
    parser.add_argument("--sim", choices=("toy", "go2"), default="go2")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    # experiments.py picks the simulator at import, and pool workers inherit this.
    os.environ["RESCUE_SIM"] = args.sim
    from rescue import experiments as ex
    from rescue.drone_prior import PriorConfig

    methods = ["coverage", "nearest", "priority"]
    workers = args.workers or os.cpu_count()
    print(f"sim: {ex.run_episode.__module__}, {args.n_sites} sites, T=600 s, {workers} workers")

    start = time.perf_counter()
    rows = ex.run_grid([PriorConfig()], methods, range(args.n_sites), workers=workers)
    elapsed = time.perf_counter() - start

    os.makedirs("runs", exist_ok=True)
    path = f"runs/dry_run_{args.sim}.csv"
    ex.save_csv(rows, path)

    print(f"\n{'method':10s} {'norm score':>16s} {'time to critical (s)':>22s} {'rescue rate':>16s}")
    for m in methods:
        cells = [ex.mean_ci([r[k] for r in rows if r["method"] == m])
                 for k in ("norm_score", "time_to_critical", "rescue_rate")]
        print(f"{m:10s} {cells[0][0]:9.3f} ± {cells[0][1]:.3f} {cells[1][0]:14.1f} ± {cells[1][1]:5.1f} "
              f"{cells[2][0]:9.3f} ± {cells[2][1]:.3f}")

    print("\npriority - nearest, paired over sites (mean ± 95% CI)")
    for k in ("norm_score", "time_to_critical", "rescue_rate"):
        d = ex.paired_diff(rows, "priority", "nearest", k)
        print(f"  {k:17s} {d['mean']:+.3f} ± {d['ci']:.3f}   priority higher on {d['win_rate']:.0%}, n={d['n']}")

    per_job = elapsed / len(rows)
    print(f"\n{len(rows)} jobs in {elapsed:.1f} s ({per_job:.2f} s/job across {workers} workers)")
    print(f"wrote {path}")
    print(f"default run_sweeps.py is {SWEEP_JOBS_N40} jobs; at this rate about "
          f"{SWEEP_JOBS_N40 * per_job / 60:.1f} min")


if __name__ == "__main__":
    main()
