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
