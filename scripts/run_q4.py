"""Q4 offline evaluation of behaviour/state inference. Writes runs/q4_*.csv and runs/figures/q4_inference.png.

Tracks come from the stand-in generator ``behavior.synthetic_track`` (survivor state ~ uniform, mode ~
``mode_given_state``, constant-velocity motion plus observation noise), so the numbers measure the
inference given that model, not performance on the simulator's survivor motion. Baseline 4 is no
behaviour model: it predicts the prior's most likely mode / state from the first sample on.

Reports, per observation noise sigma_obs and observation length:
  * mode accuracy and state accuracy of the Bayes filter vs. baseline 4, plus the state-accuracy
    ceiling (what perfect mode knowledge would give, since modes do not identify states uniquely);
  * speed: the first time the mode posterior reaches 0.9, how often that happens within the window
    and how often it is correct;
  * robustness: how often lateral (out-of-model) motion is flagged by the surprise statistic vs.
    in-model motion.

Usage: PYTHONPATH=. python scripts/run_q4.py [n_tracks]
"""
import csv
import os
import sys

import numpy as np

from rescue.behavior import Behavior, BehaviorModel, ModeBelief, synthetic_track
from rescue.interfaces import ResponseState

SIGMAS = [0.1, 0.25, 0.5, 1.0, 2.0]
N_OBS = 10                 # samples per track, 1 s apart
CONF = 0.9                 # posterior needed to commit to a mode
SURPRISE_FLAG = 9.21       # chi-square(2 dof) 99th percentile
MODEL = BehaviorModel()


def sample_tracks(n: int, sigma_obs: float, rng: np.random.Generator, lateral: bool = False):
    states = rng.integers(3, size=n)
    modes = np.array([rng.choice(3, p=MODEL.mode_given_state[s]) for s in states])
    tracks = [synthetic_track(Behavior(m), rng, n=N_OBS, sigma_obs=sigma_obs, speed=MODEL.speed,
                              lateral=lateral, speed_std=MODEL.s_move)
              for m in modes]
    return states, modes, tracks


def run_filter(track, sigma_obs):
    """Posteriors after each observation k = 1..N_OBS: arrays (N_OBS, 3) mode and state, and surprise."""
    mb = ModeBelief(MODEL, sigma_obs)
    mode_p, state_p, surp = [], [], []
    for t, z, r in zip(*track):
        mb.observe(t, z, r)
        mode_p.append(mb.mode_posterior)
        state_p.append(mb.state_posterior)
        surp.append(mb.surprise)
    return np.array(mode_p), np.array(state_p), np.array(surp)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rng = np.random.default_rng(0)
    os.makedirs("runs/figures", exist_ok=True)
    base_mode = int(np.argmax(MODEL.prior_mode))
    base_state = int(np.argmax(MODEL.prior_state))
    # State accuracy with perfect mode knowledge: choose argmax_z p(z) p(m|z) per mode, then average.
    joint = MODEL.prior_state[:, None] * MODEL.mode_given_state
    ceiling = float(joint.max(axis=0).sum())

    acc_rows, dec_rows, rob_rows = [], [], []
    for sg in SIGMAS:
        states, modes, tracks = sample_tracks(n, sg, rng)
        res = [run_filter(t, sg) for t in tracks]
        mode_p = np.array([r[0] for r in res])      # (n, N_OBS, 3)
        state_p = np.array([r[1] for r in res])
        for k in range(2, N_OBS + 1):
            acc_rows.append({"sigma_obs": sg, "n_obs": k,
                             "mode_acc": float(np.mean(mode_p[:, k - 1].argmax(1) == modes)),
                             "mode_acc_baseline": float(np.mean(modes == base_mode)),
                             "state_acc": float(np.mean(state_p[:, k - 1].argmax(1) == states)),
                             "state_acc_baseline": float(np.mean(states == base_state)),
                             "state_acc_ceiling": ceiling})
        # Speed: first k >= 2 where the mode posterior reaches CONF.
        hit = mode_p.max(2) >= CONF
        hit[:, 0] = False
        first = np.where(hit.any(1), hit.argmax(1), -1)
        decided = first >= 0
        correct = np.array([mode_p[i, first[i]].argmax() == modes[i] for i in range(n) if decided[i]])
        dec_rows.append({"sigma_obs": sg, "frac_decided": float(decided.mean()),
                         "acc_when_decided": float(correct.mean()) if decided.any() else float("nan"),
                         "mean_seconds_to_decision": float(first[decided].mean()) if decided.any() else float("nan")})
        # Robustness: lateral motion is outside the three modes.
        _, _, lat = sample_tracks(n, sg, rng, lateral=True)
        s_lat = np.array([run_filter(t, sg)[2][-1] for t in lat])
        s_in = np.array([r[2][-1] for r in res])
        rob_rows.append({"sigma_obs": sg, "flag_rate_lateral": float(np.mean(s_lat > SURPRISE_FLAG)),
                         "flag_rate_in_model": float(np.mean(s_in > SURPRISE_FLAG))})
    for name, rows in (("q4_accuracy", acc_rows), ("q4_speed", dec_rows), ("q4_robustness", rob_rows)):
        with open(f"runs/{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"prior mode baseline {np.mean(modes == base_mode):.2f}; state ceiling {ceiling:.2f}")
    print("sigma_obs  mode_acc@3s  @6s  @10s | state_acc@10s | decided  acc  sec | lateral flagged / in-model flagged")
    for sg, d, rb in zip(SIGMAS, dec_rows, rob_rows):
        a = {r["n_obs"]: r for r in acc_rows if r["sigma_obs"] == sg}
        print(f"{sg:8.2f}   {a[3]['mode_acc']:.2f}  {a[6]['mode_acc']:.2f}  {a[10]['mode_acc']:.2f} |"
              f" {a[10]['state_acc']:.2f} | {d['frac_decided']:.2f} {d['acc_when_decided']:.2f}"
              f" {d['mean_seconds_to_decision']:.1f} | {rb['flag_rate_lateral']:.2f} / {rb['flag_rate_in_model']:.2f}")
    plot(acc_rows, dec_rows)


def plot(acc_rows, dec_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": grid,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    ks = sorted({r["n_obs"] for r in acc_rows})
    for sg, color in zip((0.25, 0.5, 1.0, 2.0), ("#2a78d6", "#eb6834", "#2f9e6b", "#8a5fd0")):
        rows = {r["n_obs"]: r for r in acc_rows if r["sigma_obs"] == sg}
        a.plot([k - 1 for k in ks], [rows[k]["mode_acc"] for k in ks], color=color, lw=1.5, marker="o", ms=3,
               label=f"sigma_obs = {sg} m")
    base = next(r["mode_acc_baseline"] for r in acc_rows)
    a.axhline(base, color="#8a8984", ls="--", lw=1.2, label="no behaviour model (baseline 4)")
    a.set_xlabel("Seconds of observed motion")
    a.set_ylabel("Mode accuracy")
    a.legend(frameon=False, fontsize=7.5)
    b.plot([r["sigma_obs"] for r in dec_rows], [r["mean_seconds_to_decision"] for r in dec_rows],
           color="#2a78d6", marker="o", ms=4, lw=1.5)
    b.set_xscale("log")
    b.set_xlabel("Observation noise sigma_obs (m)")
    b.set_ylabel(f"Mean seconds to P(mode) >= {CONF}")
    fig.savefig("runs/figures/q4_inference.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
