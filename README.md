# Rubble Rescue Quadruped

Drone-informed, priority-aware search and rescue with a quadruped robot. Course project for Human-Centered Robotics (Valerie Liang, Sahil Patel).

After a collapse, a simulated quadrotor surveys the site and hands a Unitree Go2 a list of likely survivor locations, each with position uncertainty and a triage priority (critical, urgent, stable). The quadruped must decide whom to reach first, how to approach safely, and (later) how to infer each survivor's state when they cannot answer.

## Approach

Three layers approximate the underlying POMDP:

1. **Search layer (~1 Hz):** greedy, priority-weighted, urgency-discounted search over a belief seeded by the aerial prior. Score per track is `r * w * exp(-lambda * t_arrive) / (tau + 1)^k`; `k = 1` (reward per second of travel) was best in a 60-site check (k = 0.5, 0.25, 0 were worse on score and time to critical), so it is the default (`dist_exp`).
2. **Approach planner (~10 Hz):** MPC with a unicycle model and a standoff pose as goal, plus a legibility term.
3. **Safety filter (every control step):** CBF-QP enforcing clearance from survivors and hazardous terrain.

Objective: maximize `E[ sum_i w_i * exp(-lambda_i * tau_i) * 1[tau_i <= T] ]` subject to a safety constraint, where `tau_i` is the time survivor `i` is reached.

## Research questions

| | Question |
|---|---|
| Q1 | How much does the aerial prior help vs. prior-free coverage, and when does it stop helping? |
| Q2 | Does priority-weighted search reach critical survivors sooner than nearest-first? |
| Q3 | Can a CBF filter using a high-probability region of the belief eliminate violations, and what does it cost as localization uncertainty grows? |
| Q4 | How fast and accurately can survivor state (unresponsive / trapped / mobile) be inferred? |
| Q5 | Does a legibility term let an observer identify the goal earlier, and at what cost? |

Baselines: (1) coverage with no prior, (2) prior with nearest-first, (3) no safety filter, (4) no state inference, (5) communication oracle.

Metrics: priority-weighted rescue score (oracle-normalized), time to critical survivors, rescue rate within `T`, state-inference accuracy, safety violations.

## Status

Implemented (Q1-Q2):

- Drone prior: position noise, misses, false positives, class errors
- Belief with Bayes update
- Greedy search layer (priority-weighted, urgency-discounted) and baselines 1-2
- Metrics: weighted rescue score, oracle normalization, time to critical, rescue rate
- Simulators: point-robot reference (`toy_sim.py`) and reduced-order Go2 (`sim.py`), with a parallel batch runner
- Q1 sweep (prior noise, false-positive rate), Q2 comparison with a class-accuracy sweep, and plots

Not yet implemented: MPC approach planner, CBF safety filter, state inference, legibility (Q3-Q5).

## Results (Go2 sim, T = 600 s, default prior: sigma 3 m, 1 false positive, 10% miss, 80% class accuracy)

Oracle-normalized score, mean over 40 sites per config (`runs/q1_sigma.csv`, `runs/q1_fp.csv`):

| sigma (m) | 1 | 3 | 6 | 10 | 15 | 25 |
|---|---|---|---|---|---|---|
| no-prior coverage | 0.228 | 0.228 | 0.228 | 0.228 | 0.228 | 0.228 |
| prior + priority | 0.810 | 0.729 | 0.560 | 0.386 | 0.261 | 0.097 |

The oracle is a terrain-aware upper bound (shortest paths over rubble and hazards, standoff radius at each survivor, no robot dynamics), so normalized scores never exceed 1 (observed maximum 0.993).

- **Q1:** the prior triples the score at sigma = 3 m (0.73 vs 0.23) and the gain shrinks steadily with noise until sigma is about 15 m, where it is roughly equal to coverage; at 25 m it is worse than no prior. False positives hurt much less: even 20 expected false positives leave the score at 0.46, about twice coverage.
- **Q2:** priority-weighted vs nearest-first, paired over 160 sites: score +0.013 +/- 0.014, time to critical -9.6 s +/- 15.2 s, rescue rate +0.005 +/- 0.009. No significant difference at the default config. The score gap grows with drone class accuracy (-0.003 at 40%, +0.026 at 100%) but stays within the CI at every level.

Regenerate with `scripts/run_sweeps.py` and `scripts/plot_results.py`. The Go2 and toy simulators give near-identical numbers; hazards are slow but traversable and robot pose is true pose.

## Layout

```
rescue/
  interfaces.py    # Site, Survivor, Detection, Observation, Policy, RunLog, priority classes
  drone_prior.py   # PriorConfig, generate_detections
  belief.py        # Track, Belief (Bayes update from robot observations)
  search.py        # PriorSearch (greedy), Coverage (lawnmower)
  baselines.py     # policy factories: coverage, nearest-first, priority
  metrics.py       # rescue score, oracle, time to critical, rescue rate
  terrain.py       # terrain speed + travel-time estimate
  toy_sim.py       # point-robot reference simulator + seeded site generator
  sim.py           # reduced-order Go2 simulator (used for reported results)
  experiments.py   # parallel seeded job runner, CSV, paired statistics
scripts/smoke_test.py   # compare methods over N sites
scripts/run_sweeps.py   # Q1/Q2 sweeps -> runs/*.csv
scripts/plot_results.py # figures from runs/*.csv
scripts/render_site.py  # site and rollout figures
tests/
docs/INTERFACES.md
```

## Setup

```
pip install -r requirements.txt
python -m pytest
PYTHONPATH=. python scripts/smoke_test.py 20 --sim go2
RESCUE_SIM=go2 PYTHONPATH=. python scripts/run_sweeps.py 40   # -> runs/*.csv
PYTHONPATH=. python scripts/plot_results.py                    # -> runs/figures/
```

Requires Python 3.10+, numpy, scipy, matplotlib.
