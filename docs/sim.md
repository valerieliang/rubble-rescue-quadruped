# Simulation notes

How the three simulators fit together, how the locomotion policies were trained, and how to reproduce it. Results tables are in [`README.md`](../README.md); this file is the working record. This repository: <https://github.com/valerieliang/rubble-rescue-quadruped>. The robot and training code (the "doggy in sim") is Unitree's [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab), which we ran in WSL2 and did not modify (see [Reproducing](#reproducing-from-scratch)).

## Reproducing from scratch

Two separate environments: one to train policies (needs an NVIDIA GPU, Linux or WSL2) and one to run the rescue experiments (CPU only). The trained policies are already committed in `sim/go2/`, so training only needs repeating if you want to regenerate them.

**1. Train a policy** (WSL2 Ubuntu, NVIDIA GPU; we used a laptop RTX 5070 Ti, 12 GB):

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git ~/unitree_rl_mjlab
cd ~/unitree_rl_mjlab
git checkout 1425b15f73bd4095f0df53709d7c389c3eb9e790     # "Fix the warnings during rough-terrain training", 2026-04-13; no local changes
conda create -n unitree_rl_mjlab python=3.11 -y && conda activate unitree_rl_mjlab
pip install -e .
pip install "mujoco==3.5.0" "warp-lang==1.12.0" scipy      # the versions pip picks by default break mjlab 1.2.0
python scripts/train.py Unitree-Go2-Flat  --env.scene.num-envs=4096 --agent.logger tensorboard --agent.max-iterations 3000
python scripts/train.py Unitree-Go2-Rough --env.scene.num-envs=2048 --agent.logger tensorboard --agent.max-iterations 3000
```

Versions used: Python 3.11.18, mjlab 1.2.0, mujoco 3.5.0, mujoco-warp 3.5.0, warp-lang 1.12.0, torch 2.11.0+cu128, rsl-rl-lib 5.0.1, onnx 1.23.2, numpy 2.4.6. Training is not bit-for-bit reproducible (GPU PPO), so a re-run gives a similar, not identical, policy; the committed `policy.onnx` files are the ones every reported number uses.

Each run writes `logs/rsl_rl/go2_velocity/<timestamp>/` with `model_*.pt`, the training configs (`params/env.yaml`, `params/agent.yaml`) and an exported `policy.onnx`. Copy the ONNX file and the configs into this repo:

```bash
RUN=~/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/<timestamp>
cp $RUN/policy.onnx /mnt/c/<path to repo>/sim/go2/policy.onnx            # flat
cp $RUN/policy.onnx /mnt/c/<path to repo>/sim/go2/policy_rough.onnx      # rough
```

The robot model (`sim/go2/model/`: `go2.xml`, meshes, constants) comes from the same repo (`src/assets/robots/unitree_go2/`).

**2. Run the experiments** (WSL or any Python 3.10+ with MuJoCo; the Windows Python we used for the rest has no `mujoco`):

```bash
git clone https://github.com/valerieliang/rubble-rescue-quadruped.git && cd rubble-rescue-quadruped
python -m venv ~/venv-rescue && source ~/venv-rescue/bin/activate
pip install -r requirements.txt "mujoco==3.5.0" onnxruntime        # numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, onnxruntime 1.32.0
export PYTHONPATH=.
python scripts/mujoco/mj_calibrate.py                                       # flat-ground tracking and the 0.25 m pad
python scripts/mujoco/mj_rubble_calibrate.py --n 100                        # flat policy on rubble (the baseline below)
python scripts/mujoco/mj_rubble_calibrate.py --n 100 --policy sim/go2/policy_rough.onnx --label rough
python scripts/mujoco/mj_run_q3.py 100 8                                    # Q3 on unicycle and flat Go2
python scripts/mujoco/mj_run_q3_rubble.py 100 8                             # Q3 on rubble (last argument: workers)
```

Everything is seeded: a site seed fixes the survivors, rubble, hazards and drone detections, and the rubble heightfield is a function of the site seed (`mj_terrain.make_heights`), so the same command reproduces the same terrain and the same paired comparisons.

## The simulators

| Simulator | File | Robot | Ground | Used for |
| --- | --- | --- | --- | --- |
| Search simulator | `rescue/sim.py` (and `toy_sim.py`, a point-robot reference) | Reduced-order Go2 unicycle (turn 1.5 rad/s, accel 1 m/s^2), 5 m sensor, 1.5 m rescue radius | Rubble and hazards are speed caps (0.5x, 0.25x) | Q1, Q2, retention |
| Control-rate simulator | `rescue/ctrl_sim.py` | Same unicycle at 10 Hz, 10 m sensor, 3.5 m rescue radius, moving survivors, prompts | Speed caps | Q3-Q5 |
| MuJoCo Go2 | `rescue/mj_plant.py` behind `ctrl_sim.py` | Full Go2 model walking with a learned velocity policy | Flat plane, or a rubble heightfield (`rubble=True`) | Q3 checks on a legged robot |

The unicycle is the default plant everywhere. The MuJoCo plant is optional: `run_episode_ctrl(..., plant=Go2MujocoPlant())`, or `plant="mujoco"` / `"mujoco_rubble"` in an experiment job. The stack's 10 Hz `(v, omega)` command is clipped to the policy's training range (forward -0.5 to 1.0 m/s, yaw +/-1.0 rad/s), turned into the policy's velocity command, and the pose comes back from the simulated robot.

All simulators draw sites from the same seeded generator, so a seed gives the same survivors, rubble and hazards, and the same drone detections, on every plant. That is what makes the paired comparisons between plants valid.

## Locomotion policies

Both are PPO velocity-tracking policies from [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab) (mjlab, MuJoCo Warp), exported to ONNX. Both output 12 joint-position targets. The flat policy's actor takes 47 inputs (base angular velocity, projected gravity, velocity command, gait phase, joint positions, joint velocities, last action). The rough policy's actor takes the same 47 plus a **height scan** (234 inputs): a 1.6 m x 1.0 m grid of 17 x 11 downward rays, aligned with the base's yaw, giving base height minus ground height (5 m for a miss), scaled by 1/5. `mj_plant.py` supplies the scan from the plane or the heightfield, and picks the layout from the policy's input size.

| | Flat policy | Rough policy |
| --- | --- | --- |
| Task | `Unitree-Go2-Flat` | `Unitree-Go2-Rough` |
| Terrain | Plane | Curriculum over stairs (to 10 cm), slopes, waves, random rough (2-10 cm noise, 2 cm steps), flat |
| Iterations | 3000 | 3000 |
| Environments | 4096 | 2048 (4096 crashed, see below) |
| Wall time | 1 h 23 min | about 1 h 55 min |
| File | `sim/go2/policy.onnx` (+ `model_2999.pt`) | `sim/go2/policy_rough.onnx` (to be added) |

Hardware and versions are listed under [Reproducing](#reproducing-from-scratch).

### Training commands (WSL)

```bash
cd ~/unitree_rl_mjlab
# flat (as reported)
python scripts/train.py Unitree-Go2-Flat  --env.scene.num-envs=4096 --agent.logger tensorboard --agent.max-iterations 3000
# rough (4096 envs failed on this GPU, see notes)
python scripts/train.py Unitree-Go2-Rough --env.scene.num-envs=2048 --agent.logger tensorboard --agent.max-iterations 3000
```

Output lands in `logs/rsl_rl/go2_velocity/<timestamp>/` (checkpoints every 100 iterations, `policy.onnx` at the end); see Reproducing for copying it into `sim/go2/`.

### Notes from the rough run

- **4096 environments failed** at the first reset with `Warp CUDA error 600: device not ready` (graph creation). The rough task is heavier than flat (`ccd_iterations` 500, contact `maxmatch` 500, a ray-cast sensor). 2048 environments used about 7 GB and ran at about 2.5 s per iteration.
- **Early curve:** the terrain curriculum level started near 0 (the robot could not yet walk the rough tiles), reached 0.44 by iteration 650, 1.0 by 1250 and about 1.3 from iteration 2000 on, where it flattened. Mean reward went 9.7 (it. 74), 23.8 (270), 30.8 (646), 36.4 (1234), 41.5 (1986), 41.9 (2806). Velocity-tracking error (`error_vel_xy`) fell from 0.68 to about 0.53-0.55. Illegal-contact terminations ended at about 0.13-0.25.
- The policy converged by about iteration 2000; the last 1000 iterations changed little.

## The MuJoCo plant

`rescue/mj_plant.py` rebuilds the training robot in plain MuJoCo: position actuators with the training gains and effort limits, joint armature, feet-only collision against the ground, the training solver settings, policy at 50 Hz on 200 Hz physics (decimation 4). Joint order, default pose, gains, action scale and the observation layout are read from, and checked against, the ONNX metadata, so a wrong policy file fails loudly. The robot's assets are in `sim/go2/model/`.

### Rubble terrain

`rescue/mj_terrain.py` turns a site's regions into a heightfield (`Go2MujocoPlant(rubble=True)`, then `plant.set_site(site)`, which `run_episode_ctrl` calls for you):

- 0.1 m cells, one heightfield over the whole 100 m x 100 m site, built in about 0.25 s.
- Rubble cells: random heights in {0, 2, 4, 6} cm. Hazard cells: {0, ..., 10} cm (the training maximum). Elsewhere flat. Heights are multiples of 2 cm as in training.
- A 1.5 m disc around the robot start is flat, so the robot starts on level ground, and 10 m of flat ground surrounds the site (the plant is not clipped to the site, so a robot that walks off the edge must not fall off the map).
- The rescue stack still caps speed in these regions (0.5x, 0.25x). The cap is how the planner estimates travel time; the terrain is now additionally physical.

### Tracking on rubble, flat-trained vs rough-trained policy

`scripts/mujoco/mj_rubble_calibrate.py --n 200`, random starts and headings on 40 m x 40 m patches of each ground type, error after 1 s of a random constant command against the ideal unicycle (the same measurement as the 0.25 m pad):

| Ground | Policy | Falls | Median | 95% | 99% | Distance ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Flat | flat-trained | 0 / 200 | 0.086 m | 0.204 m | 0.269 m | 0.89 |
| Flat | rough-trained | 0 / 200 | 0.085 m | 0.211 m | 0.247 m | 0.92 |
| Rubble | flat-trained | 6 / 200 | 0.241 m | 0.589 m | 0.937 m | 0.64 |
| Rubble | rough-trained | 4 / 200 | 0.152 m | 0.355 m | 0.507 m | 0.79 |
| Hazard | flat-trained | 34 / 200 | 0.426 m | 0.847 m | 0.952 m | 0.36 |
| Hazard | rough-trained | 9 / 200 | 0.284 m | 0.801 m | 0.920 m | 0.53 |

The rough policy cuts falls and tracking error on rubble and costs nothing on flat ground. Its rubble 95% error (0.36 m) is still above the stack's 0.25 m pad (`StackConfig.eps_track`, the flat 95% number), and hazard ground (10 cm random cells every 10 cm, the training maximum) is still hard. The pad was left at 0.25 m because the Q3 rerun below shows no violations with it.

### Q3 on rubble

`scripts/mujoco/mj_run_q3_rubble.py 100 16` (`runs/mj_q3_rubble.csv`): 100 sites, sigma_meas 0.5 m, confidence 0.7, pad 0.25 m, same sites and drone detections on every plant. An episode ends if the robot falls.

| Plant | Variant | Score | Rescue rate | Episodes with clearance violation | Hazard entries | Min. distance | Falls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Unicycle | full | 0.738 | 0.90 | 0.00 | 0.00 | 2.39 | - |
| Unicycle | no filter | 0.749 | 0.92 | 0.59 | 0.00 | 0.04 | - |
| Go2, flat ground | full | 0.674 | 0.91 | 0.00 | 0.00 | 2.41 | 0 |
| Go2, flat ground | no filter | 0.685 | 0.91 | 0.59 | 0.00 | 0.03 | 0 |
| Go2, rubble, flat policy | full | 0.558 | 0.71 | 0.00 | 0.00 | 2.35 | 42 |
| Go2, rubble, flat policy | no filter | 0.552 | 0.70 | 0.49 | 0.00 | 0.03 | 46 |
| Go2, rubble, rough policy | full | 0.670 | 0.89 | 0.00 | 0.00 | 2.41 | 0 |
| Go2, rubble, rough policy | no filter | 0.682 | 0.90 | 0.61 | 0.00 | 0.01 | 0 |

Paired by site, full stack vs the unicycle: flat ground -0.064 +/- 0.015, rubble with the flat policy -0.179 +/- 0.036 (rescue rate -0.195 +/- 0.058), rubble with the rough policy -0.068 +/- 0.015 (rescue rate -0.011 +/- 0.019). The filter's own cost on the rough-policy rubble runs is -0.012 +/- 0.013, within noise.

- **Safety holds on rubble.** With the filter, 0 clearance violations and 0 hazard entries in 100 episodes with the rough policy (closest approach 2.41 m); without it 61% of episodes violated.
- **The rough-trained policy is what makes rubble workable.** It did not fall in any of 200 episodes and its score sits where flat ground does (the gap to the unicycle is the same, about 0.065, as on flat ground). The flat-trained policy fell in 42 of 100 filtered episodes.
- **An earlier 40-site run had a bug and is superseded.** The heightfield ended exactly at the site bounds, so a robot that walked past the edge fell off the map (the old flat plane was infinite); that produced 6 of 40 falls with the rough policy. The heightfield now has 10 m of flat margin (`mj_terrain.MARGIN`). The flat policy's 42 falls are mostly terrain: the rough policy, on the same paths, had none.
- **One noise level, one pad.** sigma_meas 0.5 m and a 0.25 m pad; the rubble tracking error (0.36 m at 95%) is above the pad, but the fixed margins keep the robot about 2.4 m from survivors.

## Scripts

| Script | What it does |
| --- | --- |
| `scripts/mujoco/mj_calibrate.py` | Stand, walk and turn checks and the tracking-error pad on flat ground (`--policy` for another policy) |
| `scripts/mujoco/mj_rubble_calibrate.py` | Falls and tracking error on flat, rubble and hazard patches; run per policy |
| `scripts/mujoco/mj_run_q3.py` | Q3 on unicycle and flat-ground Go2 (the existing results) |
| `scripts/mujoco/mj_run_q3_rubble.py` | Q3 on four plants: unicycle, flat Go2, rubble with the flat policy, rubble with the rough policy |
| `scripts/mujoco/mj_run_conf.py`, `mj_cbf_stress.py`, `mj_sudden_threat.py` | Smaller safety regions, head-on and sudden-threat tests (flat ground) |
| `scripts/figures/render_mj.py` | Figures in `figures/mujoco/` |

The Windows Python environment has no `mujoco`; the MuJoCo scripts were run from WSL (`~/venv-rescue`, which has `mujoco==3.5.0` and `onnxruntime`):

```bash
cd /mnt/c/Users/TheAn/Documents/GitHub/rubble-rescue-quadruped
source ~/venv-rescue/bin/activate
PYTHONPATH=. python scripts/mujoco/mj_rubble_calibrate.py --n 200 --policy sim/go2/policy_rough.onnx --label rough
```

## Status and next steps

- [x] Rubble heightfield plant and recalibration script
- [x] Rough policy trained (3000 iterations) and exported to `sim/go2/policy_rough.onnx`
- [x] Rubble calibration with both policies; `eps_track` left at 0.25 m
- [x] Q3 on rubble (`mj_run_q3_rubble.py`), README, writeup and slides updated
- [x] 100-site rubble check (`runs/mj_q3_rubble.csv`) and the rubble render (`scripts/figures/render_mj_rubble.py`, `figures/mujoco/go2_rubble_site_074.png`)
- [ ] Several noise levels and a larger pad (0.35-0.40 m) on rubble
- [ ] Hazard ground is still hard (9 of 200 falls and 0.80 m tracking error in the calibration); it only matters if the planner ever sends the robot there
