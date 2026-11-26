"""Q3 on physical rubble: does the safety result hold when the Go2 walks over a heightfield?

Usage (repo root):  PYTHONPATH=. python scripts/mujoco/mj_run_q3_rubble.py [n_sites] [workers]

Same sites and drone detections, full stack (MPC + CBF-QP, pad 0.25 m, confidence 0.7) and baseline 3 (no
filter), on four plants (label in brackets):
  unicycle                          the reduced-order plant used for Q1-Q5
  mujoco, flat policy               flat ground, the policy trained on flat ground (as in the earlier checks)
  mujoco_rubble, flat policy        rubble heightfield, flat-trained policy
  mujoco_rubble, rough policy       rubble heightfield, policy trained on rough terrain (sim/go2/policy_rough.onnx)
Writes runs/mj_q3_rubble.csv and prints a summary. A violation on a rubble plant that the unicycle did not have
means the 0.25 m pad is too small there.
"""
import os
import sys

import numpy as np

from rescue.ctrl_experiments import make_jobs, run_jobs, save_rows, usable_seeds

ROUGH = os.path.abspath("sim/go2/policy_rough.onnx")
PLANTS = [("unicycle", "unicycle", None), ("mujoco_flat", "mujoco", None),
          ("rubble_flatpolicy", "mujoco_rubble", None), ("rubble_roughpolicy", "mujoco_rubble", ROUGH)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    os.makedirs("runs", exist_ok=True)
    seeds = usable_seeds(n)
    jobs = []
    for label, plant, policy in PLANTS:
        extra = {"policy": [policy]} if policy else {}
        jobs += make_jobs(seeds, ["full", "no_cbf"], label=label, plant=[plant], **extra)
    print(f"{len(jobs)} episodes ...")
    rows = run_jobs(jobs, workers)
    save_rows(rows, "runs/mj_q3_rubble.csv")

    print(f"\n{'plant':20s} {'variant':7s} {'n':>3s} {'score':>6s} {'rescued':>8s} {'sites w/ clear viol':>20s} "
          f"{'sites w/ hazard viol':>21s} {'min clear (m)':>14s} {'falls':>5s}")
    for label, _, _ in PLANTS:
        for variant in ("full", "no_cbf"):
            r = [x for x in rows if x["label"] == label and x["variant"] == variant]
            if not r:
                continue
            m = lambda k: float(np.mean([x[k] for x in r]))
            print(f"{label:20s} {variant:7s} {len(r):3d} {m('norm_class'):6.3f} {m('rescue_rate'):8.2f} "
                  f"{m('any_clear_viol'):20.2f} {m('any_haz_viol'):21.2f} {min(x['min_clearance'] for x in r):14.2f} "
                  f"{sum(x['fell'] for x in r):5d}")


if __name__ == "__main__":
    main()
