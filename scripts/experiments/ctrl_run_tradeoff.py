"""A benchmark where safety margins can cost something, and where the unfiltered baseline can get worse.

Usage (repo root):  PYTHONPATH=. python scripts/experiments/ctrl_run_tradeoff.py [n_sites] [workers]

Two earlier results were uninformative by construction:
  * score did not move as the margins shrank (0.667-0.681 on rubble). The robot confirms a survivor from up to
    ``rescue_radius`` = 3.5 m, and every standoff the margins asked for (about 1.8-2.7 m) is already inside that, so the
    margins never decided whether or when a survivor was reached.
  * the no-filter baseline violated in ~60% of episodes at every noise level, because without the filter the robot's
    path does not depend on the belief, so nothing the noise does can change it.

Part A (trade-off): vary the margins and the confirm range together. The standoff is max(r_stand, barrier + 0.1) with
r_stand = 1 m here, so it follows the margins; a survivor can only be rescued if the standoff is within the confirm range.
At a short confirm range (2.0 m: close enough to talk to or see a person's state) wide margins make rescue impossible and
tight margins do not, which is the trade-off the 3.5 m range hid.

Part B (noise): sigma_meas 0.25-2.0 m on a dense site (8 survivors), full stack versus two baselines: 'mpc_only' (the MPC
sees the keep-out discs but the CBF-QP is off: a soft constraint) and 'no_cbf' (neither). These can respond to noise
because the discs are sized by the belief's confidence ellipse.

Both parts report the metrics in rescue.scorecard. Writes runs/ctrl_tradeoff_a.csv and runs/ctrl_tradeoff_b.csv.
"""
import sys

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds
from rescue.scorecard import cost_per_violation_avoided, fmt, paired_diff, summarize

# (name, conf_p, d_clear, eps_track): barrier radius at sigma 0.5 m = c_p*0.5 + d_clear + eps + 0.25
MARGINS = [("wide", 0.9, 1.0, 0.25), ("stock", 0.7, 1.0, 0.25), ("tight", 0.3, 1.0, 0.0), ("min", 0.01, 0.5, 0.0)]
RESCUE_RADII = (3.5, 2.5, 2.0)
SIGMAS = (0.25, 0.5, 1.0, 2.0)
DENSE = 8      # the oracle brute-forces n! visiting orders, so 8 (40,320) is the practical maximum


def part_a(seeds, workers):
    jobs = []
    for rr in RESCUE_RADII:
        for name, c, d, e in MARGINS:
            jobs += make_jobs(seeds, ["full"], label=f"{name}_rr{rr}", conf_p=[c], d_clear=[d], eps_track=[e],
                              r_stand=[1.0], rescue_radius=[rr], sigma_meas=[0.5])
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/ctrl_tradeoff_a.csv")
    print("\nPART A: margins x confirm range (unicycle, full stack, r_stand 1 m)")
    for rr in RESCUE_RADII:
        print(f"\nconfirm range {rr} m")
        base = None
        for name, c, d, e in MARGINS:
            r = [x for x in rows if x["label"] == f"{name}_rr{rr}"]
            s = summarize(r)
            base = base or (s, {x["seed"]: x["norm_class"] for x in r})
            dm, hw = paired_diff([base[1][x["seed"]] for x in r], [x["norm_class"] for x in r])
            print(f"  {name:6s} (c {c:4.2f} d {d:.2f} eps {e:.2f}) {fmt(s)} | vs wide {dm:+.3f} +/- {hw:.3f}")
    return rows


def part_b(seeds, workers):
    jobs = []
    for sg in SIGMAS:
        for var, kw in (("full", {}), ("mpc_only", {"apply_filter": [False]}), ("no_cbf", {})):
            jobs += make_jobs(seeds, ["no_cbf" if var == "no_cbf" else "full"], label=f"{var}_s{sg}",
                              sigma_meas=[sg], n_survivors=[DENSE], **kw)
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/ctrl_tradeoff_b.csv")
    print(f"\nPART B: noise sweep on a dense site ({DENSE} survivors)")
    for sg in SIGMAS:
        print(f"\nsigma_meas {sg} m")
        ref = None
        for var in ("full", "mpc_only", "no_cbf"):
            r = [x for x in rows if x["label"] == f"{var}_s{sg}"]
            s = summarize(r)
            ref = ref or s
            extra = f" | score cost of the full filter per point of violation rate it removes vs this: {cost_per_violation_avoided(s, ref):+.4f}" if var != "full" else ""
            print(f"  {var:9s} {fmt(s)}{extra}")
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    which = sys.argv[3] if len(sys.argv) > 3 else "ab"      # "a" or "b" to run one part
    seeds = usable_seeds(n)
    if "a" in which:
        part_a(seeds, workers)
    if "b" in which:
        part_b(seeds, workers)


if __name__ == "__main__":
    main()
