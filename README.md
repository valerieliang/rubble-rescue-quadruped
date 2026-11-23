# Rubble Rescue Quadruped

Drone-informed, priority-aware search and rescue with a quadruped robot. Course project for Human-Centered Robotics (Valerie Liang, Sahil Patel).

After a collapse, a simulated quadrotor surveys the site and hands a Unitree Go2 a list of likely survivor locations, each with position uncertainty and a triage priority (critical, urgent, stable). The quadruped must decide whom to reach first, how to approach safely, and how to infer each survivor's state when they cannot answer.

## Approach

Three layers approximate the underlying POMDP:

1. **Search layer (~1 Hz):** greedy, priority-weighted, urgency-discounted search over a belief seeded by the aerial prior. Score per track is `r * w * exp(-lambda * t_arrive) / (tau + 1)^k`; `k = 1` (reward per second of travel) was best in a 60-site check (k = 0.5, 0.25, 0 were worse on score and time to critical), so it is the default (`dist_exp`).
2. **Approach planner (~10 Hz):** MPC with a unicycle model and a standoff pose as goal, plus a legibility term.
3. **Safety filter (every control step):** CBF-QP enforcing clearance from survivors and hazardous terrain.

Objective: maximize `E[ sum_i w_i * exp(-lambda_i * tau_i) * 1[tau_i <= T] ]` subject to a safety constraint, where `tau_i` is the time survivor `i` is reached.

## Research questions

|    | Question                                                                                                                                      |
| -- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1 | How much does the aerial prior help vs. prior-free coverage, and when does it stop helping?                                                   |
| Q2 | Does priority-weighted search reach critical survivors sooner than nearest-first?                                                             |
| Q3 | Can a CBF filter using a high-probability region of the belief eliminate violations, and what does it cost as localization uncertainty grows? |
| Q4 | How fast and accurately can survivor state (unresponsive / trapped / mobile) be inferred?                                                     |
| Q5 | Does a legibility term let an observer identify the goal earlier, and at what cost?                                                           |

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

Q3-Q5 standalone components (unit-tested and evaluated on their own: a local unicycle integrator, synthetic tracks; the closed-loop Q3-Q5 below uses them through `StackConfig(controller="components")`):

- Clearance regions: confidence-ellipse keep-out region per belief track, with barrier `h(x)` and gradient (`rescue/clearance.py`, `PriorSearch.clearance_regions()`)
- Approach planner: sampling-based MPC (MPPI) at 10 Hz on the unicycle model with a standoff pose as goal, terrain speed caps, soft barrier costs and a one-detour router around hazards and other survivors' regions (`rescue/mpc.py`)
- Safety filter: CBF-QP over hazard rectangles and clearance discs, closed-form because the constraints are linear in forward speed only (`rescue/cbf.py`)
- Legibility: observer-model term in the MPC cost, weighted by the target's inferred mobility (`rescue/mpc.py`, `rescue/observer.py`); Q5 rollouts in `scripts/run_q5.py`
- Behaviour inference: Bayes filter over still / approaching / retreating and the induced response-state belief, with a surprise statistic for out-of-model motion (`rescue/behavior.py`); offline evaluation and baseline 4 in `scripts/run_q4.py`, on a stand-in track generator
- Safety-violation metric: trajectory steps inside a hazard region or within 1 m of a survivor (`metrics.safety_violations`, in the run CSVs); Q3 baseline 3 (no filter) in `scripts/run_q3.py`

Closed-loop Q3-Q5 (control-rate simulator, `ctrl_sim.py`; the Q1-Q2 `sim.py` is unchanged):

- `ctrl_sim.py`: 10 Hz MPC + CBF on the Go2 plant, moving survivors (still / approaching / retreating / erratic), prompts and responses, violation logging
- `stack.py`: the three layers assembled, with variants for baselines 3 (no filter), 4 (no state inference) and 5 (communication oracle)
- `ctrl_mpc.py`, `safety.py`: receding-horizon MPC with a terrain path planner, and a CBF-QP with ellipse and hazard barriers padded by a measured tracking error
- `ctrl_behavior.py`: survivor simulation and online state inference; `legibility.py`, `approach.py`: legibility term, simulated observers and the Q5 approach scenarios
- `ctrl_experiments.py` and `scripts/ctrl_run_q3.py`, `ctrl_run_q4.py`, `ctrl_run_q5.py`, `ctrl_run_consistency.py`, `ctrl_plot.py`, `ctrl_stats.py` (paired differences with confidence intervals)
- Q4 reward models (`interfaces.RewardModel`): `base` (state rescales the reward) and `triage` (state-dependent handling time, faster decay for unresponsive survivors, self-evacuation of mobile ones), used consistently by the search layer, the simulator and the oracle; an optional drone state cue (`PriorConfig.p_state_cue`, off by default) and the `oracle_global` upper-bound variant
- Q5: a heading-legibility term in the planner (`StackConfig.legibility_heading`) and a learned third observer (`legibility.LearnedObserver`)
- Q3: safety-region confidence sweep; optional 10 Hz measurement fusion (`StackConfig.avg_meas`, off: it did not help)

## Results

Go2 sim, T = 600 s, 40 sites per config (160 for Q2). Default prior: sigma 3 m, 1 false positive, 10% miss, 80% class accuracy. Each sweep varies one setting and holds the rest at default. Figures are committed in [`figures/metrics/`](figures/metrics/). The raw per-site results (`runs/*.csv`) are git-ignored; regenerate them, and fresh figures in `runs/figures/`, with the commands under Setup.

**Metrics.** *Score* is the priority-weighted rescue score divided by the oracle's, so 1.0 is perfect. The oracle is a terrain-aware upper bound (shortest paths over rubble and hazards, standoff radius at each survivor, no robot dynamics), so scores never exceed 1 (observed maximum 0.993). *Rescue rate* is the fraction of survivors reached within T. *Time to critical* is, per site, the mean over its critical survivors of the time each is reached, counting T (600 s) for any not reached; sites with no critical survivor are excluded.

### Q1: how much does the aerial prior help?

The coverage baseline uses no prior, so it is the same in every column. It scores 0.228, reaches 42% of survivors and averages about 430 s to reach its critical survivors.

**Table 1.** Value of the aerial prior vs. position noise.

| sigma                                  | 1    | 3    | 6    | 10   | 15   | 25   |
| -------------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Score, prior + priority                | 0.81 | 0.73 | 0.56 | 0.39 | 0.26 | 0.10 |
| Score, coverage                        | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| Rescue rate, prior + priority          | 0.94 | 0.85 | 0.69 | 0.56 | 0.45 | 0.22 |
| Time to critical (s), prior + priority | 157  | 175  | 212  | 318  | 432  | 533  |

*Caption: sigma is the standard deviation (m) of the Gaussian noise added to each axis of the drone's survivor position estimates; larger sigma means a less accurate prior. Score is oracle-normalized (1.0 is perfect). Each column is the mean over 40 sites, all other prior settings at default. Coverage uses no prior, so it does not depend on sigma.*

![Figure 1](figures/metrics/q1_sigma.png)

**Figure 1.** Score, time to critical and rescue rate vs. position noise sigma (log axis). Lines are means over 40 sites; bands are 95% confidence intervals. The grey dashed line and band are the no-prior coverage baseline. Priority-weighted and nearest-first overlap almost exactly, since the prior, not the ordering, drives the gap to coverage.

**Table 2.** Value of the aerial prior vs. false-positive rate.

| false positives                        | 0    | 1    | 3    | 6    | 10   | 20   |
| -------------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Score, prior + priority                | 0.79 | 0.73 | 0.68 | 0.62 | 0.53 | 0.46 |
| Score, coverage                        | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 | 0.23 |
| Rescue rate, prior + priority          | 0.90 | 0.85 | 0.84 | 0.78 | 0.71 | 0.65 |
| Time to critical (s), prior + priority | 143  | 175  | 215  | 208  | 266  | 344  |

*Caption: false positives is the mean of a Poisson count of spurious detections added per site, so the robot may be sent to survivors that do not exist. Mean over 40 sites per column, all other prior settings at default (sigma 3 m).*

![Figure 2](figures/metrics/q1_fp.png)

**Figure 2.** Same three metrics vs. expected false positives per site, with the same lines, bands and baseline as Figure 1.

![Figure 3](figures/metrics/q1_heatmap.png)

**Figure 3.** Prior + priority score minus no-prior coverage score over the full grid of position noise and false positives (40 sites per cell). Blue means the prior helps and red means it is worse than no prior; white is break-even. Break-even is at sigma of roughly 15-25 m and shifts toward larger sigma as false positives increase; at low noise, extra false positives cost the prior up to about 0.35 of score.

**Answer.** The prior helps a lot when it is accurate: at sigma = 3 m the score is 0.73 against 0.23 for coverage, about 3x. The gain shrinks steadily as noise grows. At sigma = 15 m the prior only matches coverage (0.26 vs 0.23, and time to critical is no better), and at 25 m it is worse than having no prior (0.10). False positives hurt much less. Even 20 spurious detections per site leave the score at 0.46, twice coverage. Position noise is what limits the prior, not clutter.

### Q2: does priority-weighted search reach critical survivors sooner than nearest-first?

Both methods use the same prior; they differ only in whether the triage class sets the target's weight. Differences are priority minus nearest, paired by site, shown as mean +/- 95% CI half-width.

**Table 3.** Priority-weighted vs. nearest-first search at the default config.

| Metric               | Nearest-first | Priority | Difference                                         |
| -------------------- | ------------- | -------- | -------------------------------------------------- |
| Score                | 0.720         | 0.733    | +0.013 +/- 0.014                                   |
| Rescue rate          | 0.854         | 0.859    | +0.005 +/- 0.009                                   |
| Time to critical (s) | 198           | 188      | -9.6 +/- 15.2 (123 sites have a critical survivor) |

*Caption: means over 160 sites. Difference is priority minus nearest-first, paired by site, with a 95% confidence half-width. For time to critical a negative difference means priority is faster, and only the 123 sites that contain a critical survivor count. No difference is significant here, because every interval includes zero.*

**Table 4.** Priority-weighted vs. nearest-first search as the drone's triage-class accuracy varies.

| Class accuracy       | 40%              | 60%              | 80% (default)    | 90%              | 100%             |
| -------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| Score, nearest-first | 0.709            | 0.729            | 0.720            | 0.722            | 0.713            |
| Score, priority      | 0.707            | 0.733            | 0.733            | 0.740            | 0.740            |
| Score difference     | -0.003 +/- 0.016 | +0.004 +/- 0.015 | +0.013 +/- 0.014 | +0.018 +/- 0.014 | +0.026 +/- 0.015 |

*Caption: class accuracy is the probability that the drone labels a survivor with its correct triage class (critical, urgent, stable); otherwise the label is wrong. 160 paired sites per column, all other prior settings at default. Differences are priority minus nearest-first, mean +/- 95% half-width, and are not corrected for the five comparisons. At 100% accuracy, time to critical is also 17.6 +/- 14.1 s faster with priority weighting.*

![Figure 4](figures/metrics/q2_class.png)

**Figure 4.** Paired priority-minus-nearest-first difference vs. drone class accuracy for score (higher is better), time to critical (lower is better) and rescue rate. Points are means over 160 sites and bars are 95% confidence intervals; an interval that excludes the zero line is a detectable difference.

**Answer.** Only partly, and only when the classes are reliable. Priority weighting does nothing at 40-80% class accuracy. Only the 90% and 100% points exclude zero, and those intervals are not corrected for the five comparisons. Even at 100% the gain is about 3 points of normalized score, so the effect is small. Most of the benefit of the pipeline comes from the prior (Q1), not from priority ordering.

The Go2 and toy simulators give near-identical numbers. Hazards are slow but traversable, and robot pose is true pose.

### Case studies: why survivors are lost

Aggregate scores do not show why a survivor is lost. Each panel of Figure 5 is one seeded Go2-sim site under the default prior (black path is prior + priority), chosen to isolate one way the prior can be wrong. The panels illustrate mechanisms and are not a sample; Table 5 gives how often each occurs.

![Figure 5](figures/path_planning/cases_grid.png)

**Figure 5.** Six rollouts. Dashed circles are 2 sigma around each drone detection, colored by reported class; markers are true survivors; black rings were rescued (labelled with the time in s), red rings were not. In E and F the labels are priority/nearest-first times ("-" = not rescued), and E also draws the nearest-first path. A-B: drone misses. C: position error. D: false positives. E-F: priority vs. nearest-first.

- **A, B: the drone misses a survivor (sites 0, 1).** The policy cannot target an unreported survivor, so it is rescued only if the 5 m onboard sensor happens to see it. In A the route to the critical survivor passes within 0.6 m of the missed one, which is rescued at 136 s. In B the three reported survivors are done by 160 s, no viable track remains, and the policy falls back to the lawnmower sweep. The sweep restarts at the waypoint nearest the start, not the robot, so it first crosses the site (about 95 m) and by 600 s has covered two rows and part of a third of eleven. The survivor lies on the sixth, and the robot passed 11.6 m from it at 114 s.
- **C: a reported position is too far off (site 139).** The drone reports the urgent survivor 4.6 m from its true position. The robot approaches the reported point and each tick without a sighting lowers the track's existence probability `r`, which falls below 0.1 at 88 s while the robot is still 6.3 m away (closest approach 5.5 m, sensor radius 5 m). The track is dropped and the survivor is never rescued, by either ordering; at sigma = 2 m the same site is 3/3.
- **D: false positives cost time, not rescues (site 48).** Two spurious detections send the planner toward nonexistent survivors early in the run. All four real survivors are still rescued, but the critical one at 314 s instead of 95 s, and the score falls from 0.91 to 0.54.
- **E, F: priority vs. nearest-first (sites 152, 155).** In E, priority rescues the critical survivor first (52 s) and all four by 235 s (score 0.94). Nearest-first clears two nearby stable survivors first, its track for the critical one falls below 0.1 while the robot is about 7 m away, and it is never rescued (0.50). In F the drone labels one critical survivor stable, so priority reaches it at 172 s instead of 36 s and scores 0.48 against 0.79; with true labels priority scores 0.80. E and F are tails: over the 160 paired sites priority scores higher on 38, lower on 32 and within 0.01 on 90, with a per-site standard deviation of 0.09 around the +0.013 mean, so wins like E and losses like F largely cancel.

**Table 5.** Why survivors are not rescued (prior + priority, 160 sites, 864 survivors, pooled).

| sigma (m)    | 1   | 3   | 6   | 10  | 15  |
| ------------ | --- | --- | --- | --- | --- |
| Unrescued    | 54  | 117 | 244 | 385 | 496 |
| Not reported | 81% | 38% | 18% | 12% | 10% |
| Dropped      | 11% | 54% | 75% | 81% | 73% |
| Other        | 7%  | 8%  | 7%  | 7%  | 17% |

*Caption: Not reported = missed by the drone. Dropped = reported, but its track fell below r = 0.1 at some point. Other = reported and never dropped (still active at T, or deleted). Reproduce with `--budget` (see Setup).*

**Takeaway.** The prior controls where the robot goes; the onboard sensor controls what it finds, and a survivor is confirmed only when the robot comes within 5 m. At sigma = 1 m misses (A, B) are 81% of the failures, which is why the rescue rate stops at 0.94 (Table 1). At the default sigma = 3 m dropped tracks (C) are the larger share (54% vs. 38%), and beyond it they dominate, one mechanism behind the loss of the prior's benefit in Q1. Three cheap remedies follow, none implemented or measured yet: (i) resume the sweep from the waypoint nearest the current position, skipping rows already covered; (ii) when no track remains, search the largest unobserved region instead of a fixed pattern; (iii) keep a track until its 2-sigma region has been searched.

## Closed-loop results: Q3-Q5

Control-rate Go2 sim, T = 600 s (300 s in the Q4 triage regime). **The default safety-region confidence is 0.7 (`StackConfig.conf_p`); every closed-loop result below uses it unless a table says otherwise.** **Q3 and Q4 use 100 sites per configuration (200 for the Q4 triage regime and 300 for the violation check); Q5 uses 100 scenarios.** Every comparison between methods is paired by site (same site, same drone detections) and reported as mean +/- 95% confidence half-width; a `*` marks an interval that excludes zero, uncorrected for multiple comparisons. `scripts/ctrl_stats.py` and `scripts/ctrl_plot.py` regenerate these tables and the figures from `runs/*.csv`. Figures are committed in [`figures/closed_loop/`](figures/closed_loop/).

Differences from the Q1-Q2 simulator: sensor radius 10 m (an onboard person detector), rescue radius 3.5 m (the robot confirms from a standoff outside the 1 m clearance), and survivors can move. Q3-Q4 scores are therefore not comparable with the Q1-Q2 tables in absolute terms; the consistency check below compares the trends. The survivor behavior, the reward models and the prompt-response model are modelling choices, not calibrated to people.

### Q3: safety under uncertainty

A clearance violation is the robot centre within 1 m of any survivor (including ones already reached). Localization uncertainty is the robot's onboard measurement noise sigma_meas; the safety region is the confidence ellipse of the belief, padded by the clearance and a measured tracking error. Score is oracle-normalized (class weights).

**Table 6.** Safety filter vs. no filter (default confidence 0.7, 100 sites).

| sigma_meas (m) | Episodes with violation, full | Episodes with violation, no filter | Score, full | Score diff. (full - no filter) | Rescue-rate diff. | Time-to-critical diff. (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 0.00 | 0.61 | 0.75 | -0.012 +/- 0.015 | -0.014 +/- 0.019 | +5.8 +/- 15.8 |
| 0.5 | 0.00 | 0.58 | 0.74 | -0.007 +/- 0.010 | -0.011 +/- 0.012 | -5.3 +/- 10.9 |
| 1.0 | 0.00 | 0.44 | 0.60 | +0.004 +/- 0.015 | +0.000 +/- 0.037 | -0.2 +/- 16.7 |
| 1.5 | 0.00 | 0.44 | 0.56 | +0.000 +/- 0.021 | +0.013 +/- 0.042 | +10.9 +/- 24.3 |
| 2.0 | 0.00 | 0.39 | 0.52 | -0.015 +/- 0.025 | -0.006 +/- 0.049 | +11.7 +/- 26.3 |

![Q3: violations, score and rescue rate vs. localization noise](figures/closed_loop/q3_uncertainty.png)

**Figure 6.** Violations, score and rescue rate vs. onboard localization noise, full system vs. no filter (lines are means over 100 sites, bands 95% CIs).

**What the filter costs, and how to reduce it.** The cost is the conservative region around survivors the robot has not yet localized. At confidence 0.9 (the previous default) it was 1 to 5 points of score and 25 to 45 s on time to critical, detectable at four of five noise levels; at the new default of 0.7 no difference is detectable at any noise level (Table 6). Lowering the confidence of the region removes nearly all of it with no loss of safety:

**Table 7.** Score difference vs. no filter by confidence level of the safety region (100 sites).

| sigma_meas (m) | conf 0.5 | conf 0.7 | conf 0.8 | conf 0.9 |
| --- | --- | --- | --- | --- |
| 0.5 | -0.001 +/- 0.012 | -0.007 +/- 0.010 | -0.018 +/- 0.016* | -0.055 +/- 0.020* |
| 1.0 | +0.006 +/- 0.013 | +0.004 +/- 0.015 | +0.002 +/- 0.024 | -0.009 +/- 0.016 |
| 2.0 | -0.023 +/- 0.021* | -0.015 +/- 0.025 | -0.040 +/- 0.029* | -0.031 +/- 0.031* |

Episodes with a violation were 0 in every cell. A larger check at sigma_meas 1.0 and 2.0 and confidence 0.5, 0.7 and 0.9 (300 sites each, 1800 episodes) again had **0 clearance violations and 0 hazard entries**, with minimum robot-survivor distance never below 2.1 m against the 1 m requirement, so the 95% upper bound on the violation rate is about 1% per setting. The reason the lower confidence is free is that the fixed margins (1 m clearance, tracking-error pad, control-point offset) already keep the robot well clear: the region's confidence level adds little safety here but costs time. The lower levels rest on those margins; with a smaller margin the confidence level would matter again. Raising the confidence to 0.95 and 0.99 lowers the score to 0.61 and 0.47 (sigma_meas 0.5). The closed-loop default is now 0.7 (`StackConfig.conf_p`) and the standalone `SafetyParams` default stays at 0.9, and Q3-Q5 were all rerun at it. The 0.9 rows in the sweeps are the earlier default.

![Q3: confidence level](figures/closed_loop/q3_confidence.png)

**Figure 7.** Violations and score difference vs. no filter by region confidence level, for three noise levels.

**Tried, and did not help.** Fusing all 10 Hz detections in a second (`StackConfig(avg_meas=True)`: mean with std sigma_meas/sqrt(n), so the region shrinks faster) lowered the filtered score (0.48 vs 0.59 at sigma_meas 1.0, 0.33 vs 0.51 at 2.0) and is left off.

**Hazard stress test.** With a hazard-blind planner the unfiltered stack entered a hazard in 57% of episodes and had a clearance violation in 66%; the filtered one had neither, at about 1 point of score (0.71 vs 0.72). With the hazard-aware planner neither variant entered a hazard, so the hazard barrier matters only when the planner fails.

**Answer.** Yes. The filter eliminated every clearance violation in every condition (over 4,000 filtered episodes in total), against 40-61% of unfiltered episodes with one. At the default confidence of 0.7 the cost is within noise of zero at every noise level (point estimates between -1.5 and +0.4 points; the 0.9 setting cost 1 to 5), with violations still at zero. The loss from localization uncertainty itself (0.75 to 0.52 as sigma_meas grows) is larger than the filter's. The filter needs active uncertainty reduction: when the safety region is as large as the sensing range, the robot circles the region until it sees the survivor.

### Q4: survivor state

Inference quality at the default config: the full system infers the state with **86% accuracy** (communication oracle 97%, no-inference baseline 35%, the accuracy of the prior alone) and settles on the right state on average **3.2 s** after the first sighting. Accuracy falls as survivors move out of model: 0.87 with no erratic survivors, 0.74 with half of the mobile survivors erratic, 0.56 with all of them, and only 37-45% on the erratic survivors themselves (see [`q4_ablations.png`](figures/closed_loop/q4_ablations.png) for the other ablations: prior noise, false positives, survivor count, fraction mobile, prompt reliability). Two of those panels need care. The prompt-reliability panel is flat because the default-scene score is insensitive to inference quality, not because reliability has no effect: state accuracy rises from 73% (q = 0.4) to 95% (q = 1.0) while the score stays at 0.742. The erratic-fraction panel rises (0.706 to 0.733) because erratic survivors do not flee the robot the way retreating ones do, so it is confounded with how hard survivors are to reach; read the accuracy numbers above for robustness.

**Does inferring the state help the rescue score?** It depends on whether state changes the best decision. We test two configurations, reported side by side.

*Default scene (reward `base`).* The state only rescales each survivor's reward (unresponsive 1.5, trapped 1.0, mobile 0.4), 3-8 survivors, T = 600 s. Here inference does nothing: state-weighted score (paired difference vs. no inference, 100 sites) is **-0.003 +/- 0.006** for the full system and **-0.003 +/- 0.007** for the communication oracle; the no-inference score is 0.744. The interval is tight, so this is a clean null, not an underpowered one. It is also explained: an exhaustive search over visiting orders with perfect state knowledge improves the oracle-relative score by only 0.8% (`base`), 1.1% (state-dependent decay, `conseq`) or 3.7% (stronger decay and earlier self-evacuation, `conseq_strong`) over the best order that ignores the state (25-60 sites). With 3-8 survivors the nearest-first order is close to optimal whatever their state, so the state cannot matter much, and no reward-weight tuning in this scene changes that.

*Triage regime (reward `triage`).* State must change what acting costs, not only what it is worth. Unresponsive survivors need 30 s of on-site assessment, trapped ones 15 s and mobile ones none (a prompt is enough); unresponsive survivors also deteriorate twice as fast, and mobile survivors self-evacuate after 240 s (reward 0 after that). The robot cannot reach everyone: 8 survivors, T = 300 s. Both the search layer and the oracle use this reward (the search maximizes expected reward per unit of travel plus expected handling time under its belief over the state; the oracle's best order is brute-forced with the same handling times). In this regime knowing the state is worth up to 7.9% of the score (the same exhaustive-order calculation, 25 sites; with 8 survivors and T = 600 s it is 1.9%, and 1.2% in the default scene), and inference captures part of it:

**Table 8.** State-weighted score minus the no-inference baseline, triage regime (8 survivors, T = 300 s, 200 sites).

| Drone state cue | No-inference score | Full (inference) | Communication oracle (in range) | Global oracle (all states known at t = 0, upper bound) |
| --- | --- | --- | --- | --- |
| none | 0.590 | +0.007 +/- 0.009 | +0.009 +/- 0.009* | +0.028 +/- 0.022* |
| 0.7 accuracy | 0.583 | +0.009 +/- 0.022 | +0.009 +/- 0.023 | +0.013 +/- 0.023 |

![Q4: triage regime](figures/closed_loop/q4_triage.png)

**Figure 8.** Left: paired gain over no inference in the triage regime, with and without a noisy drone state cue. Right: gain vs. number of survivors (no cue, Table 9). Bars are 95% CIs.

**Table 9.** The gain grows with how overloaded the robot is (triage regime, T = 300 s, no drone cue, 200 sites).

| Survivors | No-inference score | Full (inference) | Communication oracle (in range) | Global oracle (upper bound) |
| --- | --- | --- | --- | --- |
| 4 | 0.678 | -0.000 +/- 0.001 | -0.003 +/- 0.005 | +0.012 +/- 0.022 |
| 6 | 0.638 | -0.004 +/- 0.007 | -0.001 +/- 0.008 | +0.015 +/- 0.017 |
| 8 | 0.590 | +0.007 +/- 0.009 | +0.009 +/- 0.009* | +0.028 +/- 0.022* |

With 4 survivors the robot can reach everyone and state information is worth nothing; any gain appears only as the site becomes capacity-limited. Only the oracles' 8-survivor gains exclude zero; the full system's gain (+0.007 +/- 0.009) does not.

**Replication on fresh sites.** To check whether the 8-survivor gain was real, we reran the same configuration (no drone cue) on 500 new sites (seeds 1000 and up, disjoint from the original 200; `scripts/ctrl_run_q4_triage_large.py`, `runs/q4_triage_large.csv`):

**Table 9b.** State-weighted score minus the no-inference baseline, triage regime, 8 survivors, T = 300 s, no drone cue.

| Sites | n | No-inference score | Full (inference) | Communication oracle (in range) | Global oracle (upper bound) |
| --- | --- | --- | --- | --- | --- |
| Original | 200 | 0.590 | +0.007 +/- 0.009 | +0.009 +/- 0.009* | +0.028 +/- 0.022* |
| Fresh | 500 | 0.580 | +0.003 +/- 0.005 | +0.001 +/- 0.005 | +0.016 +/- 0.014* |
| Pooled | 700 | 0.583 | +0.004 +/- 0.004 | +0.003 +/- 0.005 | +0.019 +/- 0.012* |

The full system's gain stays within noise on the fresh sites (+0.003 +/- 0.005; it improved on 65 sites and worsened on 67) and is smaller than on the original ones. The in-range oracle's original gain did not replicate (+0.001 +/- 0.005 on the fresh sites), so it was most likely chance. Only the global-knowledge bound is detectably positive in every row, at about 1.5-3 points. The full system again had no clearance violations or hazard entries in any of the 700 episodes.

**Answer.** Inference works (86% accurate, about 3 s to settle), but what it is worth depends on whether state changes the decision. In the default scene it is worth nothing measurable (-0.003 +/- 0.006), and a perfect-knowledge ceiling of 1-4% shows why. When the robot is capacity-limited and state sets the handling time, the full system's point estimate is slightly positive but not distinguishable from zero: +0.007 +/- 0.009 at 200 sites, +0.003 +/- 0.005 on 500 fresh sites, and +0.004 +/- 0.004 pooled over 700 (Table 9b). The global-knowledge bound is detectably positive (+0.019 +/- 0.012 pooled), so a gain exists in principle, and the full system captures at most a small part of it. So we cannot claim a measured benefit from inference; the data are consistent with a gain of up to about half a point and with none. State information is only valuable before the robot commits to a survivor, and the onboard sensor delivers it too late to capture much of the gain. A noisy drone cue (70% accurate) gives no further detectable gain. The triage effect rests on modelling choices (handling times, decay, self-evacuation) that are not calibrated to people.

### Q5: legibility

On open ground with one distractor on the same side, mobile survivor, 100 scenarios. Three simulated observers score each approach: a cost-model observer (same family as the planner's cost term, so partly by construction), a heading-extrapolation observer, and a **learned observer**: a conditional-logit goal classifier trained on synthetic noisy-rational walkers (features: motion direction toward each goal and progress toward it), which shares neither the planner's model nor a hand-set likelihood. The planner has two legibility terms: the cost-model term (weight lambda) and a heading term (weight lambda_head) that rewards motion pointing at the true goal rather than the distractors.

**Table 10.** Mean P(true goal) over the approach and approach time, mobile survivor.

| | Approach time (s) | Cost-model observer | Heading observer | Learned observer |
| --- | --- | --- | --- | --- |
| No legibility | 19.3 | 0.77 | 0.78 | 0.89 |
| Cost term, lambda = 300 | 25.7 (+33%) | 0.91 | 0.81 | 0.94 |
| Heading term, lambda_head = 3 | 20.3 (+5%) | 0.84 | 0.83 | 0.95 |

**Table 11.** Paired differences vs. no legibility (mobile survivor; time in s, others in P(goal)).

| lambda (cost) | lambda_head | Approach time | Cost-model observer | Heading observer | Learned observer |
| --- | --- | --- | --- | --- | --- |
| 300 | 0 | +6.41 +/- 0.48* | +0.14 +/- 0.01* | +0.03 +/- 0.00* | +0.05 +/- 0.01* |
| 0 | 1 | +0.08 +/- 0.01* | +0.02 +/- 0.00* | +0.02 +/- 0.00* | +0.02 +/- 0.00* |
| 0 | 2 | +0.49 +/- 0.03* | +0.05 +/- 0.00* | +0.04 +/- 0.00* | +0.04 +/- 0.01* |
| 0 | 3 | +0.97 +/- 0.06* | +0.07 +/- 0.01* | +0.05 +/- 0.00* | +0.06 +/- 0.01* |
| 0 | 5 | +2.01 +/- 0.15* | +0.09 +/- 0.01* | +0.06 +/- 0.00* | +0.07 +/- 0.01* |
| 300 | 3 | +7.68 +/- 0.50* | +0.14 +/- 0.01* | +0.03 +/- 0.00* | +0.05 +/- 0.01* |
| 300 | 5 | +8.58 +/- 0.51* | +0.15 +/- 0.01* | +0.03 +/- 0.00* | +0.04 +/- 0.01* |

![Q5: legibility weight vs. approach time and observer belief](figures/closed_loop/q5_legibility.png)

**Figure 9.** Legibility weight vs. approach time and each observer's mean P(true goal), by true survivor state and for a state-blind planner (dashed).

**Answer.** Yes, and the result no longer rests on one observer model: both terms raise P(true goal) for all three observers, including the independent learned one (cost term +0.05 +/- 0.01, heading term +0.06 +/- 0.01). They differ in cost. The cost term at lambda = 300 buys its gain at about a third more approach time (+6.4 s); the heading term at lambda_head = 3 buys a slightly larger gain for the learned observer (+0.06) and a larger one for the heading observer (+0.05 vs. +0.03), for about 1 s (+5%). Raising lambda_head to 5 adds a little more (learned +0.07, heading +0.06) for 2 s (+10%). Combining the two terms adds time without adding to the independent observers' gain (lambda = 300 with lambda_head = 5: +8.6 s, learned +0.04). At the earlier 0.9 confidence default, lambda_head = 5 made the robot hover and lose legibility; that failure did not reproduce at 0.7, so we do not claim a narrow useful range, only diminishing returns above about 3. The cost-model observer's gain is still partly by construction. Weighting by inferred state removes the cost for unresponsive survivors (19.3 s, no change) and lowers it for trapped ones, whereas a state-blind planner pays the full cost for everyone.

### Consistency with Q1-Q2

The two simulators differ on purpose, so we compare trends. Running the closed-loop sim over the Q1 prior sweeps (safety filter off, 100 sites, class-weighted score) and comparing relative score (score / score at the best setting) with the frozen Q1 tables (`scripts/ctrl_run_consistency.py`):

**Table 12.** Relative score (score / best setting): closed-loop sim vs. frozen Q1.

| Position noise sigma (m) | 1 | 3 | 6 | 10 | 15 | 25 |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen Q1 | 1.00 | 0.90 | 0.69 | 0.48 | 0.32 | 0.12 |
| Closed-loop, 5 m sensor (as in Q1) | 1.00 | 0.85 | 0.65 | 0.46 | 0.27 | 0.14 |
| Closed-loop, 10 m sensor (default) | 1.00 | 0.94 | 0.88 | 0.76 | 0.68 | 0.49 |

| False positives per site | 0 | 1 | 3 | 6 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen Q1 | 1.00 | 0.92 | 0.86 | 0.78 | 0.67 | 0.58 |
| Closed-loop, 5 m sensor | 1.00 | 0.98 | 0.95 | 0.86 | 0.75 | 0.63 |
| Closed-loop, 10 m sensor (default) | 1.00 | 0.93 | 0.87 | 0.83 | 0.69 | 0.65 |

Both sims degrade monotonically (Spearman -1.00) in both sweeps. With the same 5 m sensing range the closed-loop sim reproduces the Q1 noise curve to within 0.05 in relative score; the default 10 m sensor flattens it (relative score 0.49 vs. 0.12 at sigma 25), because a wider sensor finds survivors that a poor prior misplaces. The false-positive curves agree within 0.07 (10 m) and 0.09 (5 m). The closed-loop sim is therefore consistent with the frozen Q1 trends, and the differences are explained by the sensing range.

**Which implementation drives the closed loop.** The standalone components (`mpc.py` MPPI planner, `cbf.py` filter, `clearance.py`, `observer.py`, `behavior.py`) and the closed-loop modules (`ctrl_mpc.py`, `safety.py`, `stack.py`, `ctrl_sim.py`, `ctrl_behavior.py`) were compared by running both controllers in the same control-rate simulation (`StackConfig(controller="closed_loop" | "components")`), 30 sites, same search layer, sensors and survivors (run before the Q3-Q5 changes above, so at the earlier safety-region confidence of 0.9, and not rerun at 0.7; the comparison is of the controllers, not of the confidence level):

| sigma_meas (m) | controller | filter | score | rescue rate | episodes with clearance violation |
| --- | --- | --- | --- | --- | --- |
| 0.5 | closed-loop (`ctrl_mpc` + `safety`) | on | 0.72 | 0.88 | 0.00 |
| 0.5 | components (`mpc` + `cbf`) | on | 0.58 | 0.78 | 0.00 |
| 0.5 | closed-loop | off | 0.78 | 0.94 | 0.50 |
| 0.5 | components | off | 0.66 | 0.90 | 0.60 |
| 1.5 | closed-loop | on | 0.57 | 0.75 | 0.00 |
| 1.5 | components | on | 0.50 | 0.74 | 0.00 |
| 1.5 | closed-loop | off | 0.62 | 0.85 | 0.40 |
| 1.5 | components | off | 0.63 | 0.87 | 0.67 |

Both filters eliminate clearance violations. The closed-loop controller is the default for all reported Q3-Q5 numbers because it was developed and tuned for this simulator; **the components controller was not retuned**, so this table compares a tuned controller with an untuned one and the speed gap is not evidence that the closed-loop design is intrinsically better. The standalone components are kept: the Bayes mode filter, terrain-aware observer and clearance regions are tested on their own (`tests/test_*`), and their controller can be switched in with `controller="components"`. It was not retuned for this simulator (its standoff is 1.2 m and it was written against a 1.5 m rescue radius).

## Limitations

Survivors are a crude stand-in for people: three states, constant-speed motion, a fixed response model, and handling times, decay rates and self-evacuation that are modelling choices. The Q4 triage effect is small (+0.004 +/- 0.004 pooled over 700 sites, not detectable at 200 or at 500 fresh sites) and could exist only when the robot is capacity-limited and state sets the handling time; it should not be read as a measured benefit for real triage. Locomotion is a reduced-order unicycle with a measured tracking-error pad, not a full-body Go2 with a learned policy, and the drone detector is parametric noise, not image-derived. The oracle for Q3-Q4 uses survivors' initial positions. The lower-confidence safety result (zero violations at 0.5-0.7) depends on the 1 m clearance and the fixed margins and was checked at three confidence levels and two noise levels, not at every setting. The Q5 observers are simulated people (the learned one is trained on synthetic walkers), not human subjects.

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
  clearance.py, mpc.py, cbf.py, behavior.py, observer.py   # standalone Q3-Q5 components
  ctrl_sim.py, stack.py, ctrl_mpc.py, safety.py, ctrl_behavior.py, legibility.py,
  approach.py, ctrl_experiments.py                          # closed-loop Q3-Q5
scripts/smoke_test.py   # compare methods over N sites
scripts/run_sweeps.py   # Q1/Q2 sweeps -> runs/*.csv
scripts/plot_results.py # figures from runs/*.csv
scripts/render_site.py  # site and rollout figures
scripts/render_cases.py # six-panel case-study figure, --stats / --budget
scripts/run_q3.py, run_q4.py, run_q5.py   # standalone component evaluations
scripts/ctrl_run_q3.py, ctrl_run_q4.py, ctrl_run_q5.py   # closed-loop Q3-Q5 runs
scripts/ctrl_run_q4_triage_large.py   # Q4 triage replication on 500 fresh sites
scripts/render_rollout.py, render_pipeline.py   # filtered vs. unfiltered rollout figure, pipeline diagram
scripts/ctrl_run_consistency.py   # closed-loop sim vs. the frozen Q1 trends
scripts/ctrl_stats.py, ctrl_plot.py   # paired-difference tables and figures
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
PYTHONPATH=. python scripts/render_cases.py --stats --budget   # -> figures/path_planning/cases_grid.png
# closed-loop Q3-Q5, as reported (last argument: workers; keep ~8 on Windows). Each Q4 triage study is its own sweep.
PYTHONPATH=. python scripts/ctrl_run_q3.py 100 8                # Q3 -> runs/q3_*.csv (incl. confidence sweep)
PYTHONPATH=. python scripts/ctrl_run_q4.py 100 8 default        # Q4 default scene -> runs/q4_*.csv
PYTHONPATH=. python scripts/ctrl_run_q4.py 200 8 triage         # Q4 triage regime -> runs/q4_triage*.csv
PYTHONPATH=. python scripts/ctrl_run_q4_triage_large.py 500 8   # triage replication, 500 fresh sites -> runs/q4_triage_large.csv
PYTHONPATH=. python scripts/ctrl_run_q5.py 100 8                # Q5 (three observers, heading-term sweep) -> runs/q5.csv
PYTHONPATH=. python scripts/ctrl_run_consistency.py 100 8       # vs. frozen Q1; add a 4th argument 5 for a 5 m sensor
PYTHONPATH=. python scripts/ctrl_stats.py                       # paired differences with 95% CIs (markdown)
PYTHONPATH=. python scripts/ctrl_plot.py                        # -> runs/figures/ and markdown tables
```

Requires Python 3.10+, numpy, scipy, matplotlib.
