"""Pipeline diagram for the report and talk: drone detections to robot command (vector art).

Usage: python scripts/render_pipeline.py [--out figures]   (writes pipeline.svg, pipeline.pdf, pipeline.png)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
C_IN, C_PLAN, C_SAFE, C_OUT = "#e8eef5", "#eef3e8", "#fbe9e7", "#efede6"
EDGE = {C_IN: "#1f6fb4", C_PLAN: "#2e7d32", C_SAFE: "#c62828", C_OUT: "#52514e"}


def box(ax, x, y, w, h, title, lines, fc, rate=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc,
                                ec=EDGE[fc], lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.55, "\n".join(lines), ha="center", va="top", fontsize=8.3, color=MUTED,
            linespacing=1.35)
    if rate:
        ax.text(x + w - 0.1, y + 0.1, rate, ha="right", va="bottom", fontsize=8, color=EDGE[fc],
                fontweight="bold")


def arrow(ax, p, q, label=None, rad=0.0, dy=0.14, ls="-", color=INK, lpos=0.5, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3, shrinkA=0, shrinkB=0))
    if label:
        ax.text(p[0] + (q[0] - p[0]) * lpos, p[1] + (q[1] - p[1]) * lpos + dy, label, ha=ha, va="bottom",
                fontsize=7.8, color=color, style="italic")


def render(out):
    fig, ax = plt.subplots(figsize=(13, 5.6), facecolor=SURFACE)
    ax.set(xlim=(0, 13), ylim=(0, 5.6))
    ax.axis("off")
    W, H, Y = 2.1, 1.6, 2.9
    xs = [0.15, 2.75, 5.35, 7.95, 10.55]
    box(ax, xs[0], Y, W, H, "Drone detections",
        ["noisy position, class,", "false positives, misses", "(aerial prior, once)"], C_IN)
    box(ax, xs[1], Y, W, H, "Belief and search",
        ["tracks with covariance;", "priority-weighted search", "picks the next survivor"], C_PLAN, "1 Hz")
    box(ax, xs[2], Y, W, H, "State inference",
        ["motion tracker + prompts", "give P(state) per survivor;", "rescales search weights"], C_PLAN, "1 Hz")
    box(ax, xs[3], Y, W, H, "Approach MPC",
        ["unicycle plan to a standoff", "pose, terrain-aware;", "+ legibility term"], C_PLAN, "10 Hz")
    box(ax, xs[4], Y, W, H, "CBF safety filter",
        ["QP minimally edits the", "command: 1 m clearance,", "hazard keep-out"], C_SAFE, "10 Hz")

    ym = Y + H / 2
    for a, b, lab in [(0, 1, "detections"), (1, 2, "target track"), (2, 3, "state, goal"), (3, 4, "u_nom")]:
        arrow(ax, (xs[a] + W + 0.02, ym), (xs[b] - 0.02, ym))
    # state inference feeds the search weights (loop back above the row)
    arrow(ax, (xs[2] + 0.5, Y + H + 0.03), (xs[1] + W - 0.5, Y + H + 0.03), "state-scaled weights",
          rad=0.5, dy=0.38, color=EDGE[C_PLAN], lpos=0.5)

    # robot (bottom right)
    box(ax, 10.55, 0.35, W, 1.35, "Robot (Go2)", ["velocity command (v, w)", "unicycle plant"], C_OUT)
    arrow(ax, (xs[4] + W / 2, Y - 0.02), (xs[4] + W / 2, 1.72), "safe command")
    # onboard sensing feeds belief and inference
    ax.add_patch(FancyBboxPatch((3.2, 0.35), 5.9, 1.35, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=C_IN, ec=EDGE[C_IN], lw=1.4, zorder=2))
    ax.text(6.15, 1.5, "Onboard sensing", ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(6.15, 1.1, "person detector (10 m), noisy positions;\nprompt responses and observed motion",
            ha="center", va="top", fontsize=8.3, color=MUTED, linespacing=1.35)
    arrow(ax, (10.52, 1.0), (9.13, 1.0), "robot pose", dy=0.1)
    arrow(ax, (4.2, 1.72), (xs[1] + 0.9, Y - 0.02), None, color=EDGE[C_IN])
    arrow(ax, (6.65, 1.72), (xs[2] + 1.1, Y - 0.02), None, color=EDGE[C_IN])
    ax.text(0.2, 1.0, "Safety is scored against true\nsurvivor positions and terrain,\nnever against the stack's belief.",
            fontsize=8, color=MUTED, va="center", style="italic")
    ax.text(0.15, 5.6, "Rubble-rescue stack: from aerial prior to a filtered velocity command", fontsize=11.5,
            fontweight="bold", color=INK, va="top")
    os.makedirs(out, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(os.path.join(out, f"pipeline.{ext}"), facecolor=SURFACE, dpi=200, bbox_inches="tight",
                    pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="figures")
    render(ap.parse_args().out)
