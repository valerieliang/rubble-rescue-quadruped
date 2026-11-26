"""Figure: clearance violations and closest approach vs. the required clearance d_clear, on rubble.

Usage (repo root):  PYTHONPATH=. python scripts/plot_rubble_margins.py
Reads runs/mj_q3_rubble_margins.csv and runs/mj_q3_rubble_margins_low.csv (scripts/mj_run_q3_rubble_margins.py);
writes figures/mujoco/rubble_margins.png.

Main series: confidence 0.01, no tracking pad, d_clear from 1.0 m down to 0 (everything that can be shrunk is gone,
so d_clear is the only margin left). Gray points at d_clear 1.0: the other confidence / pad settings of the sweep.
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, GRAY, INK, MUTED = "#2a6fb0", "#9aa0a6", "#202124", "#5f6368"


def load(path):
    return [{k: float(v) if k in ("conf_p", "eps_track", "d_clear", "min_clearance", "any_clear_viol") else v
             for k, v in r.items()} for r in csv.DictReader(open(path))]


rows = load("runs/mj_q3_rubble_margins.csv") + load("runs/mj_q3_rubble_margins_low.csv")
main = {}
for r in rows:
    if r["conf_p"] == 0.01 and r["eps_track"] == 0.0:
        main.setdefault(r["d_clear"], []).append(r)
ds = sorted(main)
viol = [np.mean([r["any_clear_viol"] for r in main[d]]) for d in ds]
dmin = [min(r["min_clearance"] for r in main[d]) for d in ds]
dmed = [float(np.median([r["min_clearance"] for r in main[d]])) for d in ds]
others = {}
for r in rows:
    if r["d_clear"] == 1.0 and not (r["conf_p"] == 0.01 and r["eps_track"] == 0.0):
        others.setdefault((r["conf_p"], r["eps_track"]), []).append(r["min_clearance"])
other_min = [min(v) for v in others.values()]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6))

a.plot(ds, viol, "o-", color=BLUE, lw=2, ms=6, mfc=BLUE, mec="white", mew=1.5)
for d, v in zip(ds, viol):
    off = (18, 2) if d == 0.4 else (0, 9)
    a.annotate(f"{v:.0%}", (d, v), textcoords="offset points", xytext=off, ha="center", fontsize=9, color=INK)
a.set_xlabel("required clearance d_clear (m)")
a.set_ylabel("episodes with a clearance violation")
a.set_ylim(-0.03, 0.45)
a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
a.set_title("Violations appear below d_clear = 0.5 m", loc="left", fontsize=10.5, color=INK)

xs = np.linspace(0, 1.05, 50)
b.plot(xs, xs + 0.5, "--", color=GRAY, lw=1.2)
b.text(1.06, 1.1, "d_clear + 0.5 m", color=MUTED, fontsize=9, ha="right")
b.axhline(1.0, color=INK, lw=1.0)
b.text(0.0, 1.03, "1 m: violation below this line", color=INK, fontsize=9, va="bottom")
b.plot([1.0] * len(other_min), other_min, "o", color=GRAY, ms=5, mfc="white", mew=1.2)
b.annotate("other confidence / pad\nsettings at d_clear 1.0", (1.0, np.mean(other_min)), textcoords="offset points",
           xytext=(-10, 14), ha="right", fontsize=8.5, color=MUTED)
b.plot(ds, dmed, "o-", color=BLUE, lw=1.4, ms=4, alpha=0.55)
b.plot(ds, dmin, "o-", color=BLUE, lw=2, ms=6, mfc=BLUE, mec="white", mew=1.5)
b.text(ds[0] + 0.02, dmin[0] - 0.12, "closest of 100 episodes", color=BLUE, fontsize=9, va="top")
b.text(ds[1] - 0.03, dmed[1] + 0.12, "median episode", color=BLUE, alpha=0.8, fontsize=9, ha="right")
b.set_xlabel("required clearance d_clear (m)")
b.set_ylabel("closest approach to a survivor (m)")
b.set_ylim(0, 2.7)
b.set_title("Closest approach tracks d_clear + about 0.5 m", loc="left", fontsize=10.5, color=INK)

for ax in (a, b):
    ax.set_xlim(-0.05, 1.08)
    ax.grid(axis="y", color="#e3e5e8", lw=0.8)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/mujoco/rubble_margins.png", dpi=170)
print("closest approach (min over 100):", dict(zip(ds, np.round(dmin, 2))))
