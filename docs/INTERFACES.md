# Interface proposal (for Wed Oct 28 with Sahil)

Source of truth: [rescue/interfaces.py](../rescue/interfaces.py). Frame: world, metres, ground-plane x/y.

## What Valerie's side needs from the simulator

| Interface | Shape | Notes |
|---|---|---|
| Map | `Site(seed, bounds, regions, survivors, start)` | Rectangles of kind `rubble` / `hazard`; `seed` reproduces the whole site. |
| Survivor list | `Survivor(id, pos, priority, state)` | `priority` in {STABLE, URGENT, CRITICAL}; `state` (unresponsive/trapped/mobile) is hidden, used from Q4. |
| Observation (sim -> policy, ~1 Hz) | `Observation(t, robot_pos, sensor_radius, detected, rescued)` | `detected`: survivors sensed inside the footprint (noisy pos, true class). `rescued`: positions reached since last call. |
| Policy (policy -> sim) | `act(obs) -> waypoint (2,)` | Sim's waypoint tracker drives the robot to it. |
| Travel time | `travel_time(a, b) -> seconds` | Used by the search to score targets. `rescue/terrain.py` is a placeholder. |
| Metrics log | `RunLog(site_seed, method, rescue_times{id: t}, trajectory[(t,x,y)])` | Metrics are computed from this plus the `Site`. |

## Questions to settle

1. **Sensor model**: disc of radius `sensor_radius` with detection probability ~0.9, position noise ~0.5 m? Or something else (e.g. cone)?
2. **Robot pose**: does the policy get true or estimated pose in `robot_pos`? (Q3 needs estimated.)
3. **Rescue condition**: reaching a standoff radius (currently 1.5 m) = rescued. Does the sim also need a confirmation dwell time?
4. **Hazards**: are hazard regions traversable (slow) or forbidden? The toy sim treats them as slow; CBF barriers come later.
5. **Terrain cost**: can the sim expose a `travel_time(a, b)` that reflects rubble, or do we keep the straight-line + region-speed estimate?
6. **Tick rate**: policy at 1 Hz; waypoint tracker rate and who owns time (sim clock)?
7. **Run length** `T`: 600 s?
8. **Logging**: file format for runs (npz per run? one JSON per run?) so the batch runner and metrics agree.
9. **Reduced-order vs Isaac Sim** for the midterm (proposal says Isaac; timeline says reduced-order Go2). Suggest: reduced-order for midterm, state it in the report.
