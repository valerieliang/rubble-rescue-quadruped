# Interfaces

Source of truth: [rescue/interfaces.py](../rescue/interfaces.py). Frame: world, metres, ground-plane x/y.

## Contract between the simulator and the search layer

| Interface | Shape | Notes |
|---|---|---|
| Map | `Site(seed, bounds, regions, survivors, start)` | Rectangles of kind `rubble` / `hazard`; `seed` reproduces the whole site. |
| Survivor list | `Survivor(id, pos, priority, state)` | `priority` in {STABLE, URGENT, CRITICAL}; `state` (unresponsive/trapped/mobile) is hidden from the robot and unused until state inference. |
| Observation (sim -> policy, 1 Hz) | `Observation(t, robot_pos, sensor_radius, detected, rescued)` | `detected`: survivors sensed inside the footprint (noisy position, true class). `rescued`: positions reached since the last call. |
| Policy (policy -> sim) | `act(obs) -> waypoint (2,)` | The simulator's waypoint tracker drives the robot to it. |
| Travel time | `travel_time(a, b) -> seconds` | Used by the search to score targets (`rescue/terrain.py`). |
| Metrics log | `RunLog(site_seed, method, rescue_times{id: t}, trajectory[(t,x,y)])` | Metrics are computed from this plus the `Site`. |

## Modelling choices (implemented in `rescue/sim.py`)

1. **Sensor:** disc of radius 5 m, detection probability 0.9, position noise 0.5 m, true class.
2. **Robot pose:** the policy receives the true pose.
3. **Rescue:** reaching the 1.5 m standoff radius, with no dwell time.
4. **Hazards:** traversable at 0.25x speed (rubble 0.5x); there are no barrier constraints.
5. **Travel time:** the straight-line estimate integrates 1/speed along the segment.
6. **Clock:** the simulator owns time; dynamics step at 0.1 s and the policy runs at 1 Hz.
7. **Run length:** T = 600 s.
8. **Logging:** one `RunLog` per run; the batch runner writes one CSV row of metrics per (site, method, prior config).
9. **Locomotion:** reduced-order Go2 (unicycle with rate-limited forward speed and turn rate), the velocity interface the real Go2 accepts.

## Contract between the search layer and the approach planner / safety filter (Q3)

| Interface | Shape | Notes |
|---|---|---|
| Clearance regions | `PriorSearch.clearance_regions(confidence=0.95, keepout=SURVIVOR_KEEPOUT) -> list[ClearanceRegion]` | One per belief track with existence probability above the search layer's threshold. Recomputed each search step (~1 Hz); the planner and filter read the latest list. |
| Region | `ClearanceRegion(center, cov, confidence, keepout, priority, existence)` | `contains(p)` tests the confidence ellipse; `circle()` gives the enclosing keep-out disc `(centre, R)`, `R = largest semi-axis + keepout`. |
| Barrier | `h(p) = |p - c|^2 - R^2`, `barrier_grad(p) = 2 (p - c)` | `h >= 0` is safe. Hazard rectangles are the other barrier family and come from `Site.regions`. |
| Violation metric | `metrics.safety_violations(site, log)` | Counts 1 Hz trajectory samples inside a hazard or within `SURVIVOR_KEEPOUT` (1 m) of a true survivor position. |

A track's region is far larger than the rescue standoff radius (1.5 m) at realistic drone noise (about 9 m at sigma 3 m and 95%). A filter that forbids the whole region would forbid the rescue, so the planner has to treat the region as a keep-out for survivors it is not approaching (or shrink it with the confirmed track, whose sigma falls to about 0.5 m once the robot senses it).

## Approach controller and safety filter (`rescue/mpc.py`, `rescue/cbf.py`)

`ApproachController.command(pos, heading, v, target, speed_fn, hazards, regions, other_goals, state_posterior) -> (v_cmd, omega_cmd)` runs every control step (10 Hz): MPC toward the standoff pose of `target`, then the CBF filter if one was given.

| Input | Source |
|---|---|
| `target` | the search layer's waypoint (a track mean) |
| `speed_fn(points) -> speeds` | vectorized terrain speed cap; `None` for open ground |
| `hazards` | `Region`s of kind `hazard` from `Site.regions` |
| `regions` | `PriorSearch.clearance_regions()`; the region centred on `target` is relaxed to a 1 m physical keep-out |
| `other_goals`, `state_posterior` | other live track means (legibility alternatives) and `ModeBelief.state_posterior` of the target; only used when `legibility_base > 0` |

The filter returns the nearest safe command: forward speed is clipped by `a v >= -alpha h` for each barrier, the turn rate is untouched (barriers depend on position only), and at a boundary it turns along the tangent toward the goal. The guarantee is for the commanded velocity and the estimated pose; it does not model the Go2's acceleration limit beyond a 0.3 m braking margin or localization error.

The controller is wired into a simulator by `rescue/ctrl_sim.py` (see the last section): `stack.RescueStack` builds one controller per episode with `StackConfig(controller="components")`, calls `command` every 0.1 s step, and passes `speed_fn` as the vectorized form of `terrain.speed_at`. `sim.run_episode` (Q1-Q2) is unchanged.


## Control-rate simulator (`rescue/ctrl_sim.py`, closed-loop Q3-Q5)

Same plant as `sim.py`; the Q1-Q2 simulator is unchanged. `StackConfig.controller` selects the controller: `"closed_loop"` (`ctrl_mpc.py` + `safety.py`, default) or `"components"` (`mpc.ApproachController` + `cbf.SafetyFilter`).

| Item | Value |
|---|---|
| Control loop | MPC + CBF-QP at 10 Hz output `(v, omega)`, the Go2 velocity interface; search runs at 1 Hz |
| Sensor | disc of radius 10 m, detection probability 0.9, noise `sigma_meas` (default 0.5 m) |
| Rescue | robot within 3.5 m (`stack.R_RESCUE`) of the true survivor position |
| Violation | robot centre within 1 m (`SURVIVOR_KEEPOUT`) of any survivor (rescued ones included), or inside a hazard rectangle |
| Survivors | hidden state z; only MOBILE move (approaching / retreating / erratic, 0.3 m/s); rescued survivors stop |
| Prompts | up to 3 per survivor within 6 m; response in {answer, wave, nothing} with reliability q |
| Reward (Q4-Q5) | class weight x `STATE_FACTOR[z]` (unresponsive 1.5, trapped 1.0, mobile 0.4), oracle-normalized (`metrics.rescue_score(..., state_weighted=True)`) |
| Log | `CtrlLog`: rescue times, violation counts, minimum clearance, per-survivor state posterior trace |
