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

Midterm (due Nov 11): Q1-Q2.

- [x] Shared interface drafts (`rescue/interfaces.py`, see `docs/INTERFACES.md`)
- [x] Drone prior: position noise, misses, false positives, class errors
- [x] Belief with Bayes update
- [x] Greedy search layer (priority-weighted, urgency-discounted) and baselines 1-2
- [x] Metrics: weighted rescue score, oracle normalization, time to critical, rescue rate
- [x] Toy simulator (stand-in) so the above run end to end
- [x] Reduced-order Go2 simulator (`rescue/sim.py`: unicycle, waypoint tracker, disc sensor) and batch runner (`rescue/experiments.py`)
- [x] Q1 sweep (prior noise, false-positive rate), Q2 comparison (+ class-accuracy sweep), plots (`scripts/run_sweeps.py`, `scripts/plot_results.py`)
- [ ] Results freeze (Nov 6) and midterm write-up
- [ ] MPC, CBF filter, state inference, legibility (after midterm)

## Layout

```
rescue/
  interfaces.py    # Site, Survivor, Detection, Observation, Policy, RunLog, priority classes
  drone_prior.py   # PriorConfig, generate_detections
  belief.py        # Track, Belief (Bayes update from robot observations)
  search.py        # PriorSearch (greedy), Coverage (lawnmower)
  baselines.py     # policy factories: coverage, nearest-first, priority
  metrics.py       # rescue score, oracle, time to critical, rescue rate
  terrain.py       # terrain speed + travel-time estimate (placeholder)
  toy_sim.py       # point-robot stand-in simulator + seeded site generator
  sim.py           # reduced-order Go2 simulator (used for reported results)
  experiments.py   # parallel seeded job runner, CSV, paired statistics
scripts/smoke_test.py   # compare methods over N sites
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
