"""MDP terms for the PoOlympics athlete tasks (mjlab 1.6).

Observations reproduce training/poolympic/contract.py::build_obs EXACTLY (same terms, frames, order and actuator
order), so the policy's input in training is the input Unity's C# ObservationBuilder produces (gate G2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

from .. import contract as C

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# --------------------------------------------------------------------------------------------------------------
# Index cache: contract actuator order -> entity joint indices; free-joint qpos/qvel addresses.
# --------------------------------------------------------------------------------------------------------------
class _Idx:
    def __init__(self, env: "ManagerBasedRlEnv", entity_name: str = "robot"):
        ent = env.scene[entity_name]
        names = list(CONTRACT_ACTUATORS)
        ids, found = ent.find_joints(names, preserve_order=True)
        assert list(found) == names, f"joint order mismatch: {found}"
        dev = env.device
        self.joint_ids = torch.as_tensor(ids, device=dev, dtype=torch.long)
        self.q_free = ent.indexing.free_joint_q_adr.to(dev)
        self.v_free = ent.indexing.free_joint_v_adr.to(dev)
        self.default = torch.as_tensor(CONTRACT_DEFAULTS, device=dev, dtype=torch.float32)
        self.entity = ent


def _idx(env) -> _Idx:
    cache = getattr(env, "_poolympic_idx", None)
    if cache is None:
        cache = _Idx(env)
        env._poolympic_idx = cache
    return cache


def _load_contract_actuators():
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    ath = C.Athlete.bind(m)
    return ath.actuator_names, [float(x) for x in ath.default_pos], [float(x) for x in ath.range_lo], [float(x) for x in ath.range_hi]


CONTRACT_ACTUATORS, CONTRACT_DEFAULTS, CONTRACT_RANGE_LO, CONTRACT_RANGE_HI = _load_contract_actuators()


def _root(env):
    ix = _idx(env)
    qpos = env.sim.data.qpos
    qvel = env.sim.data.qvel
    return ix, qpos[:, ix.q_free], qvel[:, ix.v_free]


def _quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R(q)^T v for wxyz quaternions."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


# ---- observation terms (contract order) ----------------------------------------------------------------------
def obs_base_lin_vel_heading(env) -> torch.Tensor:
    _, qp, qv = _root(env)
    q = qp[:, 3:7]
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    vx, vy, vz = qv[:, 0], qv[:, 1], qv[:, 2]
    return torch.stack([c * vx + s * vy, -s * vx + c * vy, vz], dim=-1)


def obs_base_ang_vel_local(env) -> torch.Tensor:
    _, _, qv = _root(env)
    return qv[:, 3:6]


def obs_projected_gravity(env) -> torch.Tensor:
    _, qp, _ = _root(env)
    g = torch.zeros(qp.shape[0], 3, device=qp.device, dtype=qp.dtype)
    g[:, 2] = -1.0
    return _quat_rotate_inv(qp[:, 3:7], g)


def obs_base_height(env) -> torch.Tensor:
    _, qp, _ = _root(env)
    return qp[:, 2:3]


def obs_command(env, command_name: str = "athlete") -> torch.Tensor:
    return env.command_manager.get_term(command_name).command


def obs_gait_phase(env, command_name: str = "athlete") -> torch.Tensor:
    ph = env.command_manager.get_term(command_name).phase
    return torch.stack([torch.sin(2 * math.pi * ph), torch.cos(2 * math.pi * ph)], dim=-1)


def obs_joint_pos_rel(env) -> torch.Tensor:
    ix = _idx(env)
    return ix.entity.data.joint_pos[:, ix.joint_ids] - ix.default


def obs_joint_vel_scaled(env) -> torch.Tensor:
    ix = _idx(env)
    return ix.entity.data.joint_vel[:, ix.joint_ids] * C.JOINT_VEL_SCALE


def obs_last_action(env) -> torch.Tensor:
    return env.action_manager.action


# ---- command with contract phase clock ----------------------------------------------------------------------
class AthleteCommand(UniformVelocityCommand):
    """Uniform (vx, vy, wz) command + the contract's gait phase clock (advanced once per control tick)."""

    def __init__(self, cfg: "AthleteCommandCfg", env):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        out = super().reset(env_ids)
        if env_ids is None:
            self.phase.zero_()
        else:
            self.phase[env_ids] = 0.0
        return out

    def compute(self, dt, env_ids=None):
        super().compute(dt, env_ids)
        if env_ids is None:  # per-step path: contract.advance_phase with the current command
            moving = torch.linalg.norm(self.command, dim=-1) >= C.PHASE_CMD_THRESHOLD
            cmd = self.command
            speed = torch.linalg.norm(cmd[:, :2], dim=-1) + C.GAIT_HZ_YAW_WEIGHT * cmd[:, 2].abs()
            hz = C.GAIT_HZ_BASE + C.GAIT_HZ_PER_MPS * speed
            adv = torch.remainder(self.phase + hz * C.DECIMATION * 0.005, 1.0)
            self.phase = torch.where(moving, adv, torch.zeros_like(self.phase))


@dataclass(kw_only=True)
class AthleteCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return AthleteCommand(self, env)


# ---- rewards -----------------------------------------------------------------------------------------------
def base_height_tracking(env, target: float, std: float) -> torch.Tensor:
    _, qp, _ = _root(env)
    return torch.exp(-((qp[:, 2] - target) ** 2) / std**2)


def stay_near_origin(env, std: float) -> torch.Tensor:
    """Keep the pelvis over its spawn point (Iron Pedestal: feet must stay inside a 1 m x 1 m box)."""
    _, qp, _ = _root(env)
    d = qp[:, :2] - env.scene.env_origins[:, :2]
    return torch.exp(-(d**2).sum(-1) / std**2)


def joint_vel_limit_excess(env, limit: float) -> torch.Tensor:
    ix = _idx(env)
    qd = ix.entity.data.joint_vel[:, ix.joint_ids].abs()
    return torch.relu(qd - limit).sum(-1)


def torso_tilt_rad(env, body_name: str = "torso") -> torch.Tensor:
    ent = env.scene["robot"]
    bid = ent.find_bodies(body_name)[0][0]
    q = ent.data.body_link_quat_w[:, bid]
    z = torch.zeros(q.shape[0], 3, device=q.device, dtype=q.dtype)
    z[:, 2] = 1.0
    # world z of the body's local z axis
    w, x, y, zz = q.unbind(-1)
    bz = 1.0 - 2.0 * (x * x + y * y)
    return torch.acos(torch.clamp(bz, -1.0, 1.0))


def torso_upright(env, std: float) -> torch.Tensor:
    return torch.exp(-(torso_tilt_rad(env) ** 2) / std**2)


# ---- terminations --------------------------------------------------------------------------------------------
def torso_tilt_exceeds(env, limit_deg: float) -> torch.Tensor:
    return torso_tilt_rad(env) > math.radians(limit_deg)


def pelvis_below(env, minimum_height: float) -> torch.Tensor:
    _, qp, _ = _root(env)
    return qp[:, 2] < minimum_height


def nonfoot_ground_contact(env, sensor_name: str) -> torch.Tensor:
    return (env.scene[sensor_name].data.found > 0).any(-1)


# ---- events ------------------------------------------------------------------------------------------------
def drop_cube_on_athlete(env, env_ids, cube_names: tuple[str, ...], height_above_shoulder: float,
                         xy_jitter: float = 0.15, shoulder_above_pelvis: float = 0.55,
                         down_speed_range: tuple[float, float] = (0.0, 0.0)):
    """Teleport a pooled cube above the athlete (native MuJoCo free-joint state; mirrors MjCubePool.DropOnAthlete)."""
    if env_ids is None or len(env_ids) == 0:
        return
    ix = _idx(env)
    n = len(env_ids)
    dev = env.device
    root = env.sim.data.qpos[env_ids][:, ix.q_free]
    which = torch.randint(0, len(cube_names), (n,), device=dev)
    for k, name in enumerate(cube_names):
        sel = env_ids[which == k]
        if len(sel) == 0:
            continue
        r = root[which == k]
        m = len(sel)
        pose = torch.zeros(m, 7, device=dev)
        pose[:, 0:2] = r[:, 0:2] + (torch.rand(m, 2, device=dev) * 2 - 1) * xy_jitter
        pose[:, 2] = r[:, 2] + shoulder_above_pelvis + height_above_shoulder
        pose[:, 3] = 1.0
        vel = torch.zeros(m, 6, device=dev)
        lo, hi = down_speed_range
        vel[:, 2] = -(lo + torch.rand(m, device=dev) * (hi - lo))
        cube = env.scene[name]
        cube.write_root_link_pose_to_sim(pose, env_ids=sel)
        cube.write_root_link_velocity_to_sim(vel, env_ids=sel)


def robot_cfg(name: str = "robot") -> SceneEntityCfg:
    return SceneEntityCfg(name)


# ---- Rung 1+ (locomotion) ------------------------------------------------------------------------------------
FOOT_SENSOR = "feet_ground"


def _foot_contact(env) -> torch.Tensor:
    """[N, 2] bool, order (foot_l, foot_r) — subtree contact (foot + toe) with the ground."""
    return env.scene[FOOT_SENSOR].data.found > 0


def phase_contact(env, command_name: str = "athlete") -> torch.Tensor:
    """Gait phase ↔ stance: left foot planted for phase ∈ [0, 0.5), right for [0.5, 1); both planted when standing."""
    term = env.command_manager.get_term(command_name)
    contact = _foot_contact(env).float()
    moving = torch.linalg.norm(term.command, dim=-1) >= C.PHASE_CMD_THRESHOLD
    left_stance = (term.phase < 0.5).float()
    want = torch.stack([left_stance, 1.0 - left_stance], dim=-1)
    want = torch.where(moving[:, None], want, torch.ones_like(want))
    return (contact == want).float().mean(-1)


def foot_slip(env) -> torch.Tensor:
    ent = env.scene["robot"]
    ids = getattr(env, "_poolympic_foot_ids", None)
    if ids is None:
        ids = torch.as_tensor(ent.find_bodies(("foot_l", "foot_r"), preserve_order=True)[0], device=env.device)
        env._poolympic_foot_ids = ids
    v = ent.data.body_link_lin_vel_w[:, ids, :2]
    return ((v**2).sum(-1) * _foot_contact(env).float()).sum(-1)


def heading_yaw(qpos_quat: torch.Tensor) -> torch.Tensor:
    """mjlab heading: yaw of the pelvis x-axis (EntityData.heading_w). Unity's heading controller uses the same."""
    w, x, y, z = qpos_quat.unbind(-1)
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    return torch.atan2(fy, fx)


# ------------------------------------------------------------------ Iron Pedestal fine-tune (Event 1)
FOOT_GEOMS = ("foot_l_geom0", "toe_l_geom0", "foot_r_geom0", "toe_r_geom0")


def _foot_geom_xy(env) -> torch.Tensor:
    """World xy of the 4 foot/toe box centres relative to the env origin (= pedestal centre). (num_envs, 4, 2)"""
    ent = env.scene["robot"]
    ids = getattr(env, "_poolympic_foot_geom_ids", None)
    if ids is None:
        local = ent.find_geoms(FOOT_GEOMS, preserve_order=True)[0]
        ids = torch.as_tensor([ent.indexing.geom_ids[i] for i in local], device=env.device)
        env._poolympic_foot_geom_ids = ids
    xy = ent.data.data.geom_xpos[:, ids, :2]
    return xy - env.scene.env_origins[:, None, :2]


def feet_off_pedestal(env, half: float) -> torch.Tensor:
    """Termination: a foot/toe box centre left the square pedestal (|x| or |y| > half) — on the real 1 m x 1 m block
    more than half the foot would hang over the edge (Unity AthleteJudge: 'STEPPED OFF')."""
    return (_foot_geom_xy(env).abs() > half).any(-1).any(-1)


def feet_centred(env, std: float) -> torch.Tensor:
    """Reward: feet close to the pedestal centre (mean squared foot-centre distance)."""
    return torch.exp(-(_foot_geom_xy(env) ** 2).sum(-1).mean(-1) / std**2)


# ------------------------------------------------------------------ Iron Pedestal v2: heat-shaped gusts + adaptive level
def push_gust(env, env_ids, asset_cfg=None) -> None:
    """Heat gust: instantaneous horizontal Δv of the CURRENT curriculum magnitude in a uniformly random direction
    (IronPedestalHeat / iron_pedestal.py apply exactly this)."""
    ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=env.device)
    level = getattr(env, "_poolympic_gust", 0.3)
    asset = env.scene["robot"]
    vel = asset.data.root_link_vel_w[ids].clone()
    a = torch.rand(len(ids), device=env.device) * 2 * math.pi
    vel[:, 0] += level * torch.cos(a)
    vel[:, 1] += level * torch.sin(a)
    asset.write_root_link_velocity_to_sim(vel, env_ids=ids)


def gust_curriculum(env, env_ids, start: float, step: float, max_level: float, up: float, down: float) -> dict:
    """Adaptive gust magnitude: among the envs being reset, the fraction that lasted the full episode decides —
    > up: harder (+step), < down: easier (−step). Keeps training at the edge of what the athlete can survive."""
    if not hasattr(env, "_poolympic_gust"):
        env._poolympic_gust = start
    if env_ids is not None and len(env_ids) > 0:
        full = (env.episode_length_buf[env_ids] >= env.max_episode_length - 1).float().mean().item()
        if full > up:
            env._poolympic_gust = min(max_level, env._poolympic_gust + step)
        elif full < down:
            env._poolympic_gust = max(start, env._poolympic_gust - step)
    return {"gust_mps": torch.tensor(env._poolympic_gust)}
