"""A9 — The frozen Python <-> C# policy interface (DESIGN.md §3).

Single source of truth for actuator order, default pose, action mapping, observation layout and the gait
phase clock. `export_contract()` writes contract.json, which the Unity ObservationBuilder consumes; any change
here must be mirrored in C# and is caught by parity gate G2.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "assets" / "scene_matt.xml"
CONTRACT_JSON = ROOT.parent / "parity" / "contract.json"

CONTRACT_VERSION = 3
DECIMATION = 4
ACTION_SCALE = 0.25  # rad per unit action
JOINT_VEL_SCALE = 0.05
GAIT_HZ_BASE = 0.8  # stride frequency (Hz) = GAIT_HZ_BASE + GAIT_HZ_PER_MPS * speed  (human-like: 0.9 Hz @ 0.5 m/s,
GAIT_HZ_PER_MPS = 0.2  # 1.4 Hz @ 3 m/s); speed = |(vx, vy)| + GAIT_HZ_YAW_WEIGHT·|wz|. Advanced only while moving.
# v3 (2026-09-28): yaw weight 0.5 -> 1.2. At a 2.5 rad/s pivot the v2 clock ran 1.05 Hz, i.e. ~60 deg of turn per step —
# at MATT's +-40 deg hip-rotation limit — and capped the 360 Turntable at ~3.2 s; 1.4 Hz lets it step faster.
GAIT_HZ_YAW_WEIGHT = 1.2
PHASE_CMD_THRESHOLD = 0.1  # |(vx, vy, wz)| below this => phase frozen at 0
# Lane-keeping steering (outside the policy; Python evaluator and Unity PolicyRunner use the same law):
#   heading_target = atan(-LANE_GAIN * lane_offset_y * dir);  wz = clip(HEADING_GAIN * wrap(target - yaw), ±STEER_WZ_LIMIT)
# Lane direction is world +x; dir = -1 when running backwards (vx command < 0: yawing the other way moves the athlete
# back towards the centre line), else +1. Keeps wz within the trained ±0.5 rad/s. (Rung 1 G1: heading-only hold at gain 0.5 let a
# 5° gait heading bias and shove displacements accumulate to 1-3 m over 30 m — see rl_optimization_log.md.)
HEADING_GAIN = 2.0
LANE_GAIN = 0.3
STEER_WZ_LIMIT = 0.5

OBS_LAYOUT = [  # (name, size) — order is the contract
    ("base_lin_vel_heading", 3),
    ("base_ang_vel_local", 3),
    ("projected_gravity", 3),
    ("base_height", 1),
    ("command", 3),
    ("gait_phase_sincos", 2),
    ("joint_pos_rel", 23),
    ("joint_vel_scaled", 23),
    ("last_action", 23),
]
OBS_DIM = sum(s for _, s in OBS_LAYOUT)
# Per-term uniform observation-noise amplitudes used in training (matt_env actor terms). An athlete's "sensor noise"
# trait scales these (0 = clean, 1 = training level).
OBS_NOISE = {"base_lin_vel_heading": 0.1, "base_ang_vel_local": 0.2, "projected_gravity": 0.05, "base_height": 0.02,
             "joint_pos_rel": 0.01, "joint_vel_scaled": 0.05}
# Athlete traits (DESIGN §1: per-lane stat differences -> odds). Same ranges as the training domain randomisation.
TRAIT_RANGES = {"strength": (0.85, 1.15), "latency_substeps": (0, 4), "obs_noise": (0.0, 1.0)}
NUM_ACTIONS = 23


@dataclass(frozen=True)
class Athlete:
    """Index bindings of one athlete inside a compiled model (resolved by name, so works for any scene)."""

    actuator_names: tuple[str, ...]
    actuator_ids: np.ndarray
    joint_qposadr: np.ndarray
    joint_dofadr: np.ndarray
    root_qposadr: int
    root_dofadr: int
    default_pos: np.ndarray  # rad, per actuator
    range_lo: np.ndarray
    range_hi: np.ndarray

    @staticmethod
    def bind(m: mujoco.MjModel, prefix: str = "") -> "Athlete":
        names, aids, qadr, dadr, lo, hi = [], [], [], [], [], []
        for i in range(m.nu):
            name = m.actuator(i).name
            if prefix and not name.startswith(prefix):
                continue
            j = m.actuator_trnid[i, 0]
            names.append(name[len(prefix):])
            aids.append(i)
            qadr.append(m.jnt_qposadr[j])
            dadr.append(m.jnt_dofadr[j])
            lo.append(m.jnt_range[j, 0])
            hi.append(m.jnt_range[j, 1])
        if len(names) != NUM_ACTIONS:
            raise ValueError(f"expected {NUM_ACTIONS} actuators, found {len(names)}")
        root = m.joint(prefix + "root").id
        key_names = [m.key(k).name for k in range(m.nkey)]
        if "default" in key_names:
            default = m.key_qpos[m.key("default").id][np.array(qadr)]
        else:  # e.g. an mjlab-composed model: take the default pose from the scene MJCF, by actuator name
            ref = mujoco.MjModel.from_xml_path(str(SCENE_XML))
            kq = ref.key_qpos[ref.key("default").id]
            default = np.array([kq[ref.jnt_qposadr[ref.actuator_trnid[ref.actuator(n).id, 0]]] for n in names])
        return Athlete(tuple(names), np.array(aids), np.array(qadr), np.array(dadr), int(m.jnt_qposadr[root]),
                       int(m.jnt_dofadr[root]), np.array(default), np.array(lo), np.array(hi))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def steer_yaw_rate(quat: np.ndarray, lane_offset_y: float, vx_command: float = 1.0) -> float:
    """Lane-keeping yaw-rate command from the pelvis quaternion, the lateral offset from the lane centre line and the
    sign of the forward-speed command (facing +x, running forwards or backwards)."""
    direction = -1.0 if vx_command < 0 else 1.0
    target = math.atan(-LANE_GAIN * lane_offset_y * direction)
    err = (target - yaw_of(quat) + math.pi) % (2 * math.pi) - math.pi
    return float(np.clip(HEADING_GAIN * err, -STEER_WZ_LIMIT, STEER_WZ_LIMIT))


def advance_phase(phase: float, command: np.ndarray) -> float:
    """Phase clock update, called once per control tick BEFORE building the observation."""
    if float(np.linalg.norm(command)) < PHASE_CMD_THRESHOLD:
        return 0.0
    return (phase + gait_hz(command) * DECIMATION * 0.005) % 1.0


def gait_hz(command: np.ndarray) -> float:
    speed = math.hypot(float(command[0]), float(command[1])) + GAIT_HZ_YAW_WEIGHT * abs(float(command[2]))
    return GAIT_HZ_BASE + GAIT_HZ_PER_MPS * speed


def build_obs(ath: Athlete, qpos: np.ndarray, qvel: np.ndarray, command: np.ndarray, phase: float,
              last_action: np.ndarray) -> np.ndarray:
    """84-dim observation, computed in float64 then cast to float32 (the ONNX input dtype)."""
    r = ath.root_qposadr
    dv = ath.root_dofadr
    quat = qpos[r + 3 : r + 7]
    R = quat_to_mat(quat)
    lin_world = qvel[dv : dv + 3]  # free joint: linear velocity in world frame
    ang_local = qvel[dv + 3 : dv + 6]  # free joint: angular velocity in body frame
    yaw = yaw_of(quat)
    c, s = math.cos(yaw), math.sin(yaw)
    lin_heading = np.array([c * lin_world[0] + s * lin_world[1], -s * lin_world[0] + c * lin_world[1], lin_world[2]])
    grav = R.T @ np.array([0.0, 0.0, -1.0])
    obs = np.concatenate([
        lin_heading,
        ang_local,
        grav,
        [qpos[r + 2]],
        command,
        [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
        qpos[ath.joint_qposadr] - ath.default_pos,
        qvel[ath.joint_dofadr] * JOINT_VEL_SCALE,
        last_action,
    ])
    assert obs.shape == (OBS_DIM,)
    return obs.astype(np.float32)


def action_to_ctrl(ath: Athlete, action: np.ndarray) -> np.ndarray:
    """Reference implementation of the mapping baked into the ONNX graph (C# never does this)."""
    return np.clip(ath.default_pos + ACTION_SCALE * action, ath.range_lo, ath.range_hi)


def export_contract(fingerprint_sha256: str | None = None) -> dict:
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    ath = Athlete.bind(m)
    offsets, o = {}, 0
    for name, size in OBS_LAYOUT:
        offsets[name] = {"offset": o, "size": size}
        o += size
    contract = {
        "contract_version": CONTRACT_VERSION,
        "mujoco_version": mujoco.__version__,
        "timestep": float(m.opt.timestep),
        "decimation": DECIMATION,
        "control_hz": 1.0 / (m.opt.timestep * DECIMATION),
        "action_scale": ACTION_SCALE,
        "joint_vel_scale": JOINT_VEL_SCALE,
        "gait_hz_base": GAIT_HZ_BASE,
        "gait_hz_per_mps": GAIT_HZ_PER_MPS,
        "gait_hz_yaw_weight": GAIT_HZ_YAW_WEIGHT,
        "phase_cmd_threshold": PHASE_CMD_THRESHOLD,
        "obs_noise": [{"term": k, "offset": offsets[k]["offset"], "size": offsets[k]["size"], "amplitude": v}
                      for k, v in OBS_NOISE.items()],
        "trait_ranges": {"strength": list(TRAIT_RANGES["strength"]), "latency_substeps": list(TRAIT_RANGES["latency_substeps"]),
                         "obs_noise": list(TRAIT_RANGES["obs_noise"])},
        "steering": {"heading_gain": HEADING_GAIN, "lane_gain": LANE_GAIN, "wz_limit": STEER_WZ_LIMIT,
                     "law": "wz = clip(heading_gain * wrap(atan(-lane_gain * lane_offset_y * dir) - yaw), +-wz_limit); dir = -1 if vx_cmd < 0 else 1"},
        "obs_dim": OBS_DIM,
        "num_actions": NUM_ACTIONS,
        "obs_layout": offsets,
        "root_joint": "root",
        "actuators": [
            {"name": n, "joint": n, "default_rad": float(d), "range_lo_rad": float(lo), "range_hi_rad": float(hi)}
            for n, d, lo, hi in zip(ath.actuator_names, ath.default_pos, ath.range_lo, ath.range_hi)
        ],
        "default_qpos_scene": [float(x) for x in m.key("default").qpos],
        # default state keyed by joint name (Unity resolves addresses itself; never rely on qpos ordering)
        "default_joint_qpos": [
            {"joint": m.joint(j).name,
             "qpos": [float(x) for x in m.key("default").qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + (7 if m.jnt_type[j] == 0 else 4 if m.jnt_type[j] == 1 else 1)]]}
            for j in range(m.njnt)
        ],
        "tick_order": ["read_state", "advance_phase", "build_obs", "infer", "write_ctrl", "apply_disturbance",
                       f"mj_step x{DECIMATION}"],
        "frames": {"root_lin_vel": "world (free joint qvel[0:3]) rotated by -yaw into heading frame",
                   "root_ang_vel": "pelvis local (free joint qvel[3:6])",
                   "projected_gravity": "R_pelvis^T @ (0,0,-1)"},
        "fingerprint_sha256": fingerprint_sha256,
    }
    CONTRACT_JSON.parent.mkdir(exist_ok=True)
    CONTRACT_JSON.write_text(json.dumps(contract, indent=1))
    return contract
