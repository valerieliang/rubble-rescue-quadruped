"""Physical rubble for the MuJoCo Go2: a heightfield built from a Site's regions.

The rescue stack treats rubble and hazard regions as speed caps. Here the same rectangles also become
uneven ground: random per-cell heights (quantized to ``step`` metres) inside each region and flat ground
elsewhere, so the legged robot has to walk over them. The amplitudes sit inside the range the rough-terrain
policy was trained on (``HfRandomUniformTerrainCfg`` noise_range 0.02-0.10 m, 0.02 m steps).

  rubble: heights uniform in [0, RUBBLE_AMP]    hazard: heights uniform in [0, HAZARD_AMP]

A disc around the robot start is kept flat so the robot starts on level ground.
"""
from __future__ import annotations

import numpy as np

RES = 0.1            # m, heightfield cell size
RUBBLE_AMP = 0.06    # m, peak height of rubble cells
HAZARD_AMP = 0.10    # m, peak height of hazard cells (the training maximum)
STEP = 0.02          # m, heights are multiples of this (as in training)
START_FLAT = 1.5     # m, radius around the start kept flat
MARGIN = 10.0        # m, flat ground added beyond the site bounds (the plant is not clipped to the site)


def make_heights(site, res: float = RES, rubble_amp: float = RUBBLE_AMP, hazard_amp: float = HAZARD_AMP,
                 step: float = STEP, seed: int | None = None) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Heights (rows along y, columns along x, metres, >= 0) covering ``site.bounds`` plus a flat margin, and the
    bounds used."""
    xmin, ymin, xmax, ymax = site.bounds
    xmin, ymin, xmax, ymax = xmin - MARGIN, ymin - MARGIN, xmax + MARGIN, ymax + MARGIN
    nx, ny = int(round((xmax - xmin) / res)) + 1, int(round((ymax - ymin) / res)) + 1
    rng = np.random.default_rng(site.seed + 424242 if seed is None else seed)
    xs = xmin + res * np.arange(nx)
    ys = ymin + res * np.arange(ny)
    X, Y = np.meshgrid(xs, ys)
    h = np.zeros((ny, nx))
    for kind, amp in (("rubble", rubble_amp), ("hazard", hazard_amp)):      # hazards drawn last, so they win
        mask = np.zeros_like(h, bool)
        for r in site.regions:
            if r.kind == kind:
                mask |= (X >= r.xmin) & (X <= r.xmax) & (Y >= r.ymin) & (Y <= r.ymax)
        levels = int(round(amp / step))
        h[mask] = step * rng.integers(0, levels + 1, int(mask.sum()))
    sx, sy = site.start
    h[np.hypot(X - sx, Y - sy) < START_FLAT] = 0.0
    return h, (float(xs[0]), float(ys[0]), float(xs[-1]), float(ys[-1]))
