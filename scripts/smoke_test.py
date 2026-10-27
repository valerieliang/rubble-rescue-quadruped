"""Run the three methods over a few seeded sites and print mean metrics.

Usage: python scripts/smoke_test.py [n_sites]
"""
import sys
from functools import partial

import numpy as np

from rescue import baselines
from rescue.drone_prior import PriorConfig
from rescue.metrics import oracle_score, summarize
from rescue.toy_sim import generate_site, run_episode

T = 600.0
cfg = PriorConfig()
methods = {"coverage (no prior)": baselines.coverage_policy,
           "prior + nearest-first": baselines.nearest_first_policy,
           "prior + priority": baselines.priority_policy}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rows = {m: [] for m in methods}
for seed in range(n):
    site = generate_site(seed)
    oracle = oracle_score(site, T)
    for m, factory in methods.items():
        mk = lambda s, d, f=factory: f(s, d, cfg, horizon=T)
        log = run_episode(site, mk, cfg, seed, m, T=T)
        rows[m].append(summarize(site, log, T, oracle))
print(f"{n} sites, T={T:.0f}s")
print(f"{'method':24s} {'norm score':>10s} {'t_critical':>10s} {'rescue rate':>11s}")
for m, r in rows.items():
    print(f"{m:24s} {np.nanmean([x['norm_score'] for x in r]):10.3f} "
          f"{np.nanmean([x['time_to_critical'] for x in r]):10.1f} "
          f"{np.mean([x['rescue_rate'] for x in r]):11.3f}")
