"""Go2 in MuJoCo, driven by a learned locomotion policy: the plant behind the control-rate simulator.

The policy is ``Unitree-Go2-Flat`` from unitree_rl_mjlab (velocity tracking, PPO, trained in mjlab /
MuJoCo Warp) and was exported to ``sim/go2/policy.onnx``. This module rebuilds the training robot in plain
MuJoCo and runs the policy on it, so the rescue stack's ``(v, omega)`` command is tracked by a legged
robot instead of the rate-limited unicycle in ``sim.step_unicycle``.

What matches training (from the run's ``env.yaml`` and the ONNX metadata):
  * policy at 50 Hz, physics at 200 Hz (decimation 4); action = 12 joint position targets,
    ``target = default_pose + action_scale * action``;
  * joint order, default pose, PD gains and action scale are read from the ONNX metadata;
  * actor observation (47): base angular velocity (gyro), gravity in the body frame, command
    (vx, vy, wz), gait phase (sin, cos; zero when the command is ~0), joint positions relative to the
    default pose, joint velocities, last action; the layout is asserted against the metadata;
  * position actuators with the training gains and effort limits, joint armature, feet-only collision
    against a flat plane, the training solver settings.

What does not: the terrain is a flat plane (rubble stays a speed cap in the rescue stack), contact
parameters of the plane are MuJoCo defaults (training randomizes foot friction), and the policy was
trained on commands vx in [-0.5, 1.0] m/s, vy in [-0.5, 0.5] and yaw rate in [-1.0, 1.0] rad/s, so
``step`` clips the unicycle command to that range.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ASSET_DIR = Path(__file__).resolve().parents[1] / "sim" / "go2"

POLICY_DT = 0.02       # s, policy / environment step
PHYS_DT = 0.005        # s, physics step (decimation 4)
DECIMATION = 4
PERIOD_STEPS = 30      # gait phase period 0.6 s / 0.02 s
INIT_HEIGHT = 0.32     # m, base height at reset (training init state)
OBS_NAMES = ("base_ang_vel", "projected_gravity", "command", "phase", "joint_pos", "joint_vel", "actions")
# Training observation noise (uniform, +-): applied when ``obs_noise`` is on.
NOISE = {"ang_vel": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 1.5}
# Fallbacks if the ONNX carries no metadata: kind -> (kp, kd, effort limit N m, armature).
_FALLBACK = {"hip": (20.0, 1.0, 23.5, 0.01), "thigh": (20.0, 1.0, 23.5, 0.01), "calf": (40.0, 2.0, 45.0, 0.02)}
_FOOT = re.compile(r"^[FR][LR]_foot_collision$")


def _csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",")]


def _read_metadata(sess: ort.InferenceSession) -> dict:
    meta = sess.get_modelmeta().custom_metadata_map
    if "joint_names" not in meta:
        return {}
    out = {"joint_names": meta["joint_names"].split(",")}
    for k in ("joint_stiffness", "joint_damping", "default_joint_pos"):
        if k in meta:
            out[k] = _csv_floats(meta[k])
    if "action_scale" in meta:
        out["action_scale"] = _csv_floats(meta["action_scale"]) if "," in meta["action_scale"] else float(meta["action_scale"])
    if "observation_names" in meta:
        out["observation_names"] = meta["observation_names"].split(",")
    return out


def build_model(xml: Path, joints: list[str], kp: list[float], kd: list[float]) -> mujoco.MjModel:
    """The training robot: ``go2.xml`` + position actuators + feet-only collision + a plane."""
    spec = mujoco.MjSpec.from_file(str(xml))
    o = spec.option
    o.timestep = PHYS_DT
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.impratio = 1.0
    o.iterations = 10
    o.tolerance = 1e-8
    o.ls_iterations = 20
    o.ls_tolerance = 0.01
    o.ccd_iterations = 50

    for g in spec.geoms:                       # collision geoms touch the plane only; feet get friction
        name = g.name or ""
        if not name.endswith("_collision"):
            continue
        g.contype, g.conaffinity = 1, 0
        if _FOOT.match(name):
            g.condim, g.priority = 3, 1
            g.friction = [0.6, 0.005, 0.0001]
            g.solimp = [0.9, 0.95, 0.023, 0.5, 2.0]
        else:
            g.condim = 1
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            contype=1, conaffinity=1)

    for name, p, d in zip(joints, kp, kd):     # actuators in the policy's joint order
        kind = name.split("_")[1]
        _, _, effort, armature = _FALLBACK[kind]
        j = spec.joint(name)
        j.armature, j.frictionloss, j.damping = armature, 0.0, 0.0
        a = spec.add_actuator(name=name.replace("_joint", ""), target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm = [p] + [0.0] * 9
        a.biasprm = [0.0, -p, -d] + [0.0] * 7
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-effort, effort]
    return spec.compile()


class Go2MujocoPlant:
    """``step(v_cmd, omega_cmd, dt) -> (xy, yaw, forward speed)``, the same contract as ``step_unicycle``
    but with the robot's true (simulated) pose and speed instead of an integrated command."""

    def __init__(self, asset_dir: Path | str = ASSET_DIR, policy: Path | str | None = None,
                 v_range: tuple[float, float] = (-0.5, 1.0), w_max: float = 1.0,
                 obs_noise: bool = False, seed: int = 0, strict: bool = True):
        asset_dir = Path(asset_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1    # one thread: experiments run one plant per worker
        self.sess = ort.InferenceSession(str(policy or asset_dir / "policy.onnx"), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        if inp.shape[-1] != 47:
            raise ValueError(f"policy expects {inp.shape[-1]} inputs, this plant builds 47 (see module docstring)")
        self._in = inp.name

        meta = _read_metadata(self.sess)
        if meta:
            if strict and meta.get("observation_names") and tuple(meta["observation_names"]) != OBS_NAMES:
                raise ValueError(f"observation layout differs from the plant's: {meta['observation_names']} vs {OBS_NAMES}")
            self.joints = meta["joint_names"]
            self.default = np.array(meta["default_joint_pos"])
            kp, kd = meta["joint_stiffness"], meta["joint_damping"]
            self.scale = meta.get("action_scale", 0.25)
        else:
            print("[mj_plant] WARNING: no ONNX metadata; assuming FL,FR,RL,RR x hip,thigh,calf and training defaults")
            self.joints = [f"{leg}_{k}_joint" for leg in ("FL", "FR", "RL", "RR") for k in ("hip", "thigh", "calf")]
            hip = {"FL": -0.1, "FR": 0.1, "RL": -0.1, "RR": 0.1}
            self.default = np.array([hip[j[:2]] if "hip" in j else (0.9 if "thigh" in j else -1.8) for j in self.joints])
            kp = [_FALLBACK[j.split("_")[1]][0] for j in self.joints]
            kd = [_FALLBACK[j.split("_")[1]][1] for j in self.joints]
            self.scale = 0.25

        self.scale = np.asarray(self.scale, float)
        self.model = build_model(asset_dir / "model" / "xmls" / "go2.xml", self.joints, kp, kd)
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in self.joints])
        self.vadr = np.array([self.model.joint(j).dofadr[0] for j in self.joints])
        self.v_range, self.w_max = v_range, w_max
        self.obs_noise = obs_noise
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self, xy=(0.0, 0.0), yaw: float = 0.0, settle: float = 1.0) -> None:
        """Stand the robot at ``xy`` facing ``yaw`` and let it settle for ``settle`` s of zero command."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = (xy[0], xy[1], INIT_HEIGHT)
        d.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        d.qpos[self.qadr] = self.default
        d.ctrl[:] = self.default
        mujoco.mj_forward(m, d)
        self.last_action = np.zeros(12)
        self.n = 0
        self.fallen = False
        for _ in range(int(round(settle / POLICY_DT))):
            self._policy_step(np.zeros(3))
        d.qpos[0:2] = xy                      # undo the small drift while settling
        mujoco.mj_forward(m, d)
        self.n = 0

    def _rot(self) -> np.ndarray:
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return R.reshape(3, 3)

    def pose(self) -> tuple[np.ndarray, float, float]:
        """(xy, yaw, forward speed) of the base."""
        d = self.data
        w, x, y, z = d.qpos[3:7]
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        v_fwd = float(self._rot()[:, 0] @ d.qvel[0:3])
        return d.qpos[0:2].copy(), yaw, v_fwd

    # ------------------------------------------------------------------ policy

    def _obs(self, cmd: np.ndarray) -> np.ndarray:
        d = self.data
        ang = d.sensor("imu_ang_vel").data.copy()
        grav = -self._rot()[2]                          # gravity (0, 0, -1) in the body frame
        q = d.qpos[self.qadr] - self.default
        qd = d.qvel[self.vadr].copy()
        if self.obs_noise:
            u = self.rng.uniform
            ang = ang + u(-NOISE["ang_vel"], NOISE["ang_vel"], 3)
            grav = grav + u(-NOISE["gravity"], NOISE["gravity"], 3)
            q = q + u(-NOISE["joint_pos"], NOISE["joint_pos"], 12)
            qd = qd + u(-NOISE["joint_vel"], NOISE["joint_vel"], 12)
        if np.linalg.norm(cmd) < 0.1:
            phase = np.zeros(2)
        else:
            a = 2 * np.pi * (self.n % PERIOD_STEPS) / PERIOD_STEPS
            phase = np.array([np.sin(a), np.cos(a)])
        return np.concatenate([ang, grav, cmd, phase, q, qd, self.last_action]).astype(np.float32)

    def _policy_step(self, cmd: np.ndarray) -> None:
        act = self.sess.run(None, {self._in: self._obs(cmd)[None]})[0][0]
        self.last_action = np.asarray(act, float)
        self.data.ctrl[:] = self.default + self.scale * self.last_action
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        self.n += 1
        if self.data.qpos[2] < 0.12 or self._rot()[2][2] < 0.3:    # base low, or tilted more than ~72 deg
            self.fallen = True

    def step(self, v_cmd: float, omega_cmd: float, dt: float = 0.1) -> tuple[np.ndarray, float, float]:
        """Hold the command for ``dt`` seconds (5 policy steps at dt = 0.1) and return the new pose."""
        cmd = np.array([np.clip(v_cmd, *self.v_range), 0.0, np.clip(omega_cmd, -self.w_max, self.w_max)])
        for _ in range(max(1, int(round(dt / POLICY_DT)))):
            self._policy_step(cmd)
        return self.pose()
