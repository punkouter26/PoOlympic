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

from .. import biomech
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


    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        frac = getattr(self.cfg, "sprint_fraction", 0.0)
        if frac:  # r2_v8: extra mass on fast running with a mild turn (the G1 margin misses: 3.2-3.8 m/s, |wz| 0.2-0.5)
            n = len(env_ids)
            pick = torch.rand(n, device=self.device) < frac      # masked writes: no GPU->CPU sync
            lo, hi = self.cfg.sprint_vx
            cur = self.vel_command_b[env_ids]
            sprint = torch.stack([torch.empty(n, device=self.device).uniform_(lo, hi), torch.zeros(n, device=self.device),
                                  torch.empty(n, device=self.device).uniform_(-self.cfg.sprint_wz, self.cfg.sprint_wz)], -1)
            self.vel_command_b[env_ids] = torch.where(pick[:, None], sprint, cur)
            if hasattr(self, "is_standing_env"):
                self.is_standing_env[env_ids] = self.is_standing_env[env_ids] & ~pick
        cfrac = getattr(self.cfg, "crab_fraction", 0.0)
        if cfrac:  # g2_v7: extra mass on side-stepping (the G1 crab segments: small vx, |vy| up to the envelope edge, no turn)
            n = len(env_ids)
            pick = torch.rand(n, device=self.device) < cfrac
            side = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
            crab = torch.stack([torch.empty(n, device=self.device).uniform_(*self.cfg.crab_vx),
                                side * torch.empty(n, device=self.device).uniform_(*self.cfg.crab_vy),
                                torch.zeros(n, device=self.device)], -1)
            self.vel_command_b[env_ids] = torch.where(pick[:, None], crab, self.vel_command_b[env_ids])
            if hasattr(self, "is_standing_env"):
                self.is_standing_env[env_ids] = self.is_standing_env[env_ids] & ~pick
        for frac, vx, vy_abs, wz_max in getattr(self.cfg, "bands", ()):
            # g2_v7: extra mass on named parts of the envelope (fraction, vx range, |vy| range either side, |wz| max);
            # drawn in order, a later band overrides an earlier one
            n = len(env_ids)
            pick = torch.rand(n, device=self.device) < frac
            u = lambda lo, hi: torch.empty(n, device=self.device).uniform_(lo, hi)
            side = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
            band = torch.stack([u(*vx), side * u(*vy_abs), u(-wz_max, wz_max)], -1)
            self.vel_command_b[env_ids] = torch.where(pick[:, None], band, self.vel_command_b[env_ids])
            if hasattr(self, "is_standing_env"):
                self.is_standing_env[env_ids] = self.is_standing_env[env_ids] & ~pick
        a_max = getattr(self.cfg, "max_lateral_accel", None)
        if a_max:  # |wz| <= a_max / |v|: no physically impossible sprint-and-spin commands (r2_v6)
            v = torch.linalg.norm(self.vel_command_b[env_ids, :2], dim=-1).clamp(min=1e-3)
            lim = a_max / v
            self.vel_command_b[env_ids, 2] = torch.maximum(torch.minimum(self.vel_command_b[env_ids, 2], lim), -lim)


@dataclass(kw_only=True)
class AthleteCommandCfg(UniformVelocityCommandCfg):
    max_lateral_accel: float | None = None  # m/s^2; None = independent uniform sampling
    sprint_fraction: float = 0.0            # share of resamples drawn from the sprint band below
    sprint_vx: tuple[float, float] = (2.5, 4.0)
    sprint_wz: float = 0.6
    crab_fraction: float = 0.0              # share of resamples drawn from the side-step band below (after the sprint draw)
    crab_vx: tuple[float, float] = (-0.5, 0.5)
    crab_vy: tuple[float, float] = (0.4, 1.1)   # |vy|, either side
    bands: tuple = ()                       # ((fraction, (vx lo, hi), (|vy| lo, hi), |wz| max), ...) after the draws above

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


def _body_id(env, name: str) -> int:
    """Entity body index, looked up once per env (find_bodies is a regex search over all bodies)."""
    cache = getattr(env, "_poolympic_body_ids", None)
    if cache is None:
        cache = env._poolympic_body_ids = {}
    if name not in cache:
        cache[name] = env.scene["robot"].find_bodies(name)[0][0]
    return cache[name]


def torso_tilt_rad(env, body_name: str = "torso") -> torch.Tensor:
    ent = env.scene["robot"]
    bid = _body_id(env, body_name)
    q = ent.data.body_link_quat_w[:, bid]
    z = torch.zeros(q.shape[0], 3, device=q.device, dtype=q.dtype)
    z[:, 2] = 1.0
    # world z of the body's local z axis
    w, x, y, zz = q.unbind(-1)
    bz = 1.0 - 2.0 * (x * x + y * y)
    return torch.acos(torch.clamp(bz, -1.0, 1.0))


def torso_upright(env, std: float) -> torch.Tensor:
    return torch.exp(-(torso_tilt_rad(env) ** 2) / std**2)


def reset_lying(env, env_ids, prone_fraction: float = 0.5, z: tuple[float, float] = (0.2, 0.25),
                roll: float = 0.3, pitch_jitter: float = 0.2, asset_cfg=None) -> None:
    """Get-up reset: each env lying on its back (pitch -90°) or, with probability prone_fraction, on its front (+90°),
    pitch ± pitch_jitter, roll ± roll, any yaw, pelvis at z metres; zero velocity (mjlab reset_root_state_uniform can
    only sample one interval per angle, not two poses)."""
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
    asset = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=env.device)
    prone = torch.rand(n, device=env.device) < prone_fraction
    pitch = torch.where(prone, torch.full((n,), math.pi / 2, device=env.device),
                        torch.full((n,), -math.pi / 2, device=env.device)) + u(-pitch_jitter, pitch_jitter)
    q = quat_from_euler_xyz(u(-roll, roll), pitch, u(-math.pi, math.pi))
    root = asset.data.default_root_state[env_ids].clone()
    pos = root[:, 0:3] + env.scene.env_origins[env_ids]
    pos[:, 2] = u(*z)
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat_mul(root[:, 3:7], q)], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)


def getup_assist(env, env_ids, asset_cfg, max_fraction: float = 0.6, decay_steps: int = 36000,
                 body_weight_n: float | None = 80.0 * 9.81) -> None:
    """Get-up curriculum (reset event): a constant upward world force on the torso for the whole episode,
    U(0, current max) × body weight, the max fading linearly from max_fraction to 0 over decay_steps env steps
    (1500 its × 24). From flat on the floor no small motion changes height or uprightness — getup_v1/v2 lay still
    (action std 0.5 -> 0.04) — the lift makes the first rising motions pay. Afterwards: unassisted, as in Unity.
    body_weight_n: the default is MATT's 80 kg (PoOlympic-Matt-Getup3/4/5, the only tasks that rely on it); None = each
    env's own model mass (robot_mass: any body, mass DR included) — for new variants."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    cur = max(0.0, max_fraction * (1.0 - float(env.common_step_counter) / decay_steps))
    if body_weight_n is None:
        body_weight_n = robot_mass(env)[env_ids] * 9.81
    fz = torch.rand(n, device=env.device) * cur * body_weight_n
    nb = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 1
    force = torch.zeros(n, nb, 3, device=env.device)
    force[:, :, 2] = fz[:, None]
    asset.write_external_wrench_to_sim(force, torch.zeros_like(force), env_ids=env_ids, body_ids=asset_cfg.body_ids)
    env.extras.setdefault("log", {})["Metrics/getup_assist_max"] = cur


def height_progress(env, target: float) -> torch.Tensor:
    """clip(pelvis z / target, 0, 1): a get-up reward with a gradient all the way from lying to standing (the exp
    height kernel is ~0 below ~0.7 m)."""
    _, qp, _ = _root(env)
    return torch.clamp(qp[:, 2] / target, 0.0, 1.0)


def upright_linear(env) -> torch.Tensor:
    """(1 + cos(torso tilt)) / 2: 0 upside down, 0.5 lying flat, 1 upright."""
    return 0.5 * (1.0 + torch.cos(torso_tilt_rad(env)))


def rise(env, target: float) -> torch.Tensor:
    """height_progress × upright_linear²: pays only for a pelvis that is high AND a torso that is vertical together
    (getup_v4 raised the pelvis on all fours with the torso flat)."""
    return height_progress(env, target) * upright_linear(env) ** 2


HAND_SENSOR = "hands_ground"


def torso_tilt_target(env, target_deg: float, std_deg: float) -> torch.Tensor:
    """exp(-(tilt − target)² / std²) on the torso's tilt from vertical (crawling: ~75-90°, trunk near horizontal)."""
    return torch.exp(-torch.square(torso_tilt_rad(env) - math.radians(target_deg)) / math.radians(std_deg) ** 2)


def forward_progress(env, command_name: str = "athlete", min_cmd: float = 0.1) -> torch.Tensor:
    """clip(forward speed in the heading frame / commanded vx, -0.5, 1) while vx is commanded (else 0): pays for any
    forward motion at once — the exp speed kernels pay ~0 until the athlete already moves near the command (crawl_v1
    held a perfect static all-fours pose: posture terms ~3.5 per step, speed ~0)."""
    cmd = env.command_manager.get_command(command_name)[:, 0]
    _, qp, qv = _root(env)
    w, x, y, z = qp[:, 3], qp[:, 4], qp[:, 5], qp[:, 6]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    v = torch.cos(yaw) * qv[:, 0] + torch.sin(yaw) * qv[:, 1]
    active = cmd > min_cmd
    return torch.where(active, torch.clamp(v / torch.clamp(cmd, min=min_cmd), -0.5, 1.0), torch.zeros_like(cmd))


def crawl_heading(qp: torch.Tensor) -> torch.Tensor:
    """(N, 2) unit horizontal heading that stays valid on all fours: the horizontal projection of (pelvis x axis +
    pelvis z axis). Standing it is the facing direction (x axis horizontal, z vertical); crawling it is the head
    direction (x axis points down, z axis along the spine) — the contract's yaw (x axis only) flips towards the feet
    as soon as the hips are higher than the shoulders."""
    w, x, y, z = qp[:, 3], qp[:, 4], qp[:, 5], qp[:, 6]
    xa = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)], -1)       # R[:, 0] horizontal part
    za = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x)], -1)           # R[:, 2] horizontal part
    h = xa + za
    return h / torch.clamp(torch.linalg.norm(h, dim=-1, keepdim=True), min=1e-6)


def _crawl_vel(env):
    _, qp, qv = _root(env)
    h = crawl_heading(qp)
    fwd = h[:, 0] * qv[:, 0] + h[:, 1] * qv[:, 1]
    lat = -h[:, 1] * qv[:, 0] + h[:, 0] * qv[:, 1]
    return fwd, lat


def crawl_track_lin(env, std: float, command_name: str = "athlete") -> torch.Tensor:
    """Speed tracking in the crawl heading frame (forward = towards the head), exp kernel."""
    cmd = env.command_manager.get_command(command_name)
    fwd, lat = _crawl_vel(env)
    return torch.exp(-(torch.square(fwd - cmd[:, 0]) + torch.square(lat - cmd[:, 1])) / std**2)


def crawl_progress(env, command_name: str = "athlete", min_cmd: float = 0.1) -> torch.Tensor:
    """forward_progress in the crawl heading frame."""
    cmd = env.command_manager.get_command(command_name)[:, 0]
    fwd, _ = _crawl_vel(env)
    return torch.where(cmd > min_cmd, torch.clamp(fwd / torch.clamp(cmd, min=min_cmd), -0.5, 1.0), torch.zeros_like(cmd))


def crawl_progress_gated(env, target_z: float, command_name: str = "athlete") -> torch.Tensor:
    """crawl_progress × clip(pelvis z / target_z, 0, 1)²: forward motion only pays up on all fours (zcrawl_v1
    belly-slid forward: progress 2.45 / 3 with the pelvis below the crawl band and the hands off the ground)."""
    _, qp, _ = _root(env)
    return crawl_progress(env, command_name) * torch.square(torch.clamp(qp[:, 2] / target_z, 0.0, 1.0))


def yaw_rate_world(env, std: float, command_name: str = "athlete") -> torch.Tensor:
    """Turn-rate tracking about the world vertical (the pelvis z axis is horizontal on all fours)."""
    cmd = env.command_manager.get_command(command_name)[:, 2]
    _, qp, qv = _root(env)
    # free-joint angular velocity is in the body frame: world z component = R[2, :] · ω_body
    w, x, y, z = qp[:, 3], qp[:, 4], qp[:, 5], qp[:, 6]
    r2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    wz = (r2 * qv[:, 3:6]).sum(-1)
    return torch.exp(-torch.square(cmd - wz) / std**2)


def yaw_rate_world_l1(env, command_name: str = "athlete") -> torch.Tensor:
    """|commanded − actual turn rate about the world vertical| (use with a negative weight)."""
    cmd = env.command_manager.get_command(command_name)[:, 2]
    _, qp, qv = _root(env)
    w, x, y, z = qp[:, 3], qp[:, 4], qp[:, 5], qp[:, 6]
    r2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return torch.abs(cmd - (r2 * qv[:, 3:6]).sum(-1))


def hands_on_ground(env) -> torch.Tensor:
    """Fraction of the two hands (far end of the forearm capsules: forearm_l / forearm_r bodies) touching the ground —
    all fours means the hands carry weight."""
    return (env.scene[HAND_SENSOR].data.found > 0).float().mean(-1)


def standing_tall(env, min_height: float, max_tilt_deg: float) -> torch.Tensor:
    """1 while the pelvis is above min_height and the torso within max_tilt_deg of vertical (up and done)."""
    _, qp, _ = _root(env)
    return ((qp[:, 2] > min_height) & (torso_tilt_rad(env) < math.radians(max_tilt_deg))).float()


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
    if len(cube_names) == 1:                                   # recipe v5: no per-cube masking (no GPU->CPU sync)
        pose = torch.zeros(n, 7, device=dev)
        pose[:, 0:2] = root[:, 0:2] + (torch.rand(n, 2, device=dev) * 2 - 1) * xy_jitter
        pose[:, 2] = root[:, 2] + shoulder_above_pelvis + height_above_shoulder
        pose[:, 3] = 1.0
        vel = torch.zeros(n, 6, device=dev)
        lo, hi = down_speed_range
        vel[:, 2] = -(lo + torch.rand(n, device=dev) * (hi - lo))
        cube = env.scene[cube_names[0]]
        cube.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        cube.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
        return
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


def _yaw_rate_error(env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return command[:, 2] - env.scene["robot"].data.root_link_ang_vel_b[:, 2]


def track_yaw_rate(env, std: float, command_name: str = "athlete") -> torch.Tensor:
    """exp(-e²/std²) on the yaw-rate error only. mjlab's track_angular_velocity adds the roll/pitch rates to the error:
    the zombie's rocking shuffle keeps that term near zero even when it turns at the commanded rate."""
    return torch.exp(-torch.square(_yaw_rate_error(env, command_name)) / std**2)


def yaw_rate_l1(env, command_name: str = "athlete") -> torch.Tensor:
    """|yaw-rate error| (use with a negative weight): a gradient that does not vanish at large errors, where the kernels
    pay nothing (z2_v4: training yaw error ~4x MATT's, track_ang 0.15 of 2.0)."""
    return torch.abs(_yaw_rate_error(env, command_name))


def _filtered_yaw_error(env, command_name: str, tau: float) -> torch.Tensor:
    """Commanded minus actual yaw rate, the actual one low-pass filtered (EMA, time constant tau ≈ one stride): the
    pelvis' natural per-step rotation averages out, a sustained turn error does not. One EMA state per env, reset to the
    current rate on the first step of an episode; advanced once per env step (reward terms run once per step — both
    filtered terms share the state, only the first call of a step advances it)."""
    rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    st = getattr(env, "_poolympic_yaw_ema", None)
    step = int(env.common_step_counter)
    if st is None or st[0].shape != rate.shape:
        st = [rate.clone(), -1]
        env._poolympic_yaw_ema = st
    if st[1] != step:
        alpha = min(1.0, env.step_dt / tau)
        fresh = env.episode_length_buf <= 1
        st[0] = torch.where(fresh, rate, st[0] + alpha * (rate - st[0]))
        st[1] = step
    return env.command_manager.get_command(command_name)[:, 2] - st[0]


def track_yaw_rate_filtered(env, std: float, tau: float = 0.5, command_name: str = "athlete") -> torch.Tensor:
    return torch.exp(-torch.square(_filtered_yaw_error(env, command_name, tau)) / std**2)


def yaw_rate_filtered_l1(env, tau: float = 0.5, command_name: str = "athlete") -> torch.Tensor:
    return torch.abs(_filtered_yaw_error(env, command_name, tau))


def yaw_wobble_l2(env, tau: float = 0.5, command_name: str = "athlete") -> torch.Tensor:
    """(yaw rate − its stride-filtered value)²: the per-stride pelvis rotation only (use with a negative weight).
    Sustained turning and the steering error are left to the filtered terms."""
    _filtered_yaw_error(env, command_name, tau)                 # advances the shared EMA state for this step
    rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return torch.square(rate - env._poolympic_yaw_ema[0])


def flight_phase(env, command_name: str = "athlete", speed_threshold: float = 2.2) -> torch.Tensor:
    """1 while both feet are off the ground and the commanded planar speed is above speed_threshold (running with an
    aerial phase — Event 13, Steeplechase Jog); 0 otherwise. phase_contact keeps the footfalls alternating."""
    term = env.command_manager.get_term(command_name)
    airborne = ~_foot_contact(env).any(dim=-1)
    fast = torch.linalg.norm(term.command[:, :2], dim=-1) > speed_threshold
    return (airborne & fast).float()


def flight_landing(env, command_name: str = "athlete", speed_threshold: float = 2.2, min_s: float = 0.04,
                   cap_s: float = 0.2) -> torch.Tensor:
    """Paid once per flight, at touchdown: (flight duration − min_s) clipped to [0, cap_s − min_s], divided by the step
    length (the reward manager multiplies by dt, so each flight earns its seconds above min_s × weight). r2f_v1's
    per-step airborne reward bought airborne time with many ~50 ms hops (4.8 flights/s at 3.5 m/s); this pays for
    long flights, not for many."""
    term = env.command_manager.get_term(command_name)
    airborne = ~_foot_contact(env).any(dim=-1)
    t = getattr(env, "_poolympic_flight_t", None)
    if t is None or t.shape != airborne.shape:
        t = torch.zeros(airborne.shape, device=airborne.device)
    fresh = env.episode_length_buf <= 1
    landed = (~airborne) & (t > 0) & ~fresh
    fast = torch.linalg.norm(term.command[:, :2], dim=-1) > speed_threshold
    pay = torch.where(landed & fast, torch.clamp(t - min_s, 0.0, cap_s - min_s), torch.zeros_like(t)) / env.step_dt
    env._poolympic_flight_t = torch.where(airborne & ~fresh, t + env.step_dt, torch.zeros_like(t))
    return pay


def foot_slip(env) -> torch.Tensor:
    ent = env.scene["robot"]
    ids = getattr(env, "_poolympic_foot_ids", None)
    if ids is None:
        ids = torch.as_tensor(ent.find_bodies(("foot_l", "foot_r"), preserve_order=True)[0], device=env.device)
        env._poolympic_foot_ids = ids
    v = ent.data.body_link_lin_vel_w[:, ids, :2]
    return ((v**2).sum(-1) * _foot_contact(env).float()).sum(-1)


# ---- bio-realism (recipe v5, reward-side only: the MJCF, contract and ONNX graph are untouched) ----------------
def _act_torque_vel(env) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, 23] actuator torque and joint speed, both in contract order (actuators are declared in joint-tree order ==
    contract order; asserted once)."""
    ix = _idx(env)
    ids = getattr(env, "_poolympic_act_ids", None)
    if ids is None:
        a, names = ix.entity.find_actuators(list(CONTRACT_ACTUATORS), preserve_order=True)
        assert list(names) == list(CONTRACT_ACTUATORS), f"actuator order mismatch: {names}"
        ids = torch.as_tensor(a, device=env.device, dtype=torch.long)
        env._poolympic_act_ids = ids
    return ix.entity.data.actuator_force[:, ids], ix.entity.data.joint_vel[:, ix.joint_ids]


def mechanical_power(env) -> torch.Tensor:
    """Σ|τ·q̇| (W) — positive + negative joint work, a metabolic-cost proxy (use with a negative weight). Minimising
    energy is what makes gaits look human (Fu et al. 2021, "Minimizing energy consumption leads to the emergence of
    gaits"); τ² alone lets the policy spin light joints fast and stiffen heavy ones."""
    tau, qd = _act_torque_vel(env)
    return (tau * qd).abs().sum(-1)


# Hill-type torque-velocity envelope: the torque a muscle group can still produce falls linearly from the actuator cap
# at rest to 0 at w_max (joint-group shortening speed, rad/s; sprint data: knee ~20+, hip ~12-15, ankle ~15-20).
HILL_W_MAX = biomech.HILL_W_MAX      # one definition, shared with the physics-side limit (bio_force_limits)


def _hill_tables(env):
    t = getattr(env, "_poolympic_hill", None)
    if t is None:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
        cap, wmax = [], []
        for n in CONTRACT_ACTUATORS:
            a = m.actuator(n).id
            cap.append(float(max(abs(m.actuator_forcerange[a][0]), abs(m.actuator_forcerange[a][1]))))
            group = next(g for g in HILL_W_MAX if n.startswith(g))
            wmax.append(HILL_W_MAX[group])
        t = (torch.as_tensor(cap, device=env.device), torch.as_tensor(wmax, device=env.device))
        env._poolympic_hill = t
    return t


def torque_speed_excess(env) -> torch.Tensor:
    """Σ relu(|τ| − cap·(1 − |q̇|/w_max)) / cap while the joint moves WITH the torque (concentric work; eccentric
    braking may exceed it, as in muscle). The physics keeps the flat cap (Unity parity); this teaches the policy not to
    ask for peak torque at peak speed, which real legs cannot do (use with a negative weight)."""
    tau, qd = _act_torque_vel(env)
    cap, wmax = _hill_tables(env)
    concentric = (tau * qd) > 0
    allowed = cap * torch.clamp(1.0 - qd.abs() / wmax, min=0.0)
    return (torch.relu(tau.abs() - allowed) * concentric / cap).sum(-1)


def foot_impact(env, body_weight_n: float, limit_bw: float = 3.0) -> torch.Tensor:
    """Σ over feet of relu(|F_ground| − limit_bw · body weight) / body weight: soft landings. Human ground reaction
    peaks at ~1.2 BW walking and ~2.5-3 BW running (bio_probe: rung2.onnx per foot p95 1.9 / max 2.6 BW at 3.0 m/s,
    r2f_v3 max 3.4 BW) — a guard against stamping, not a shaping term (use with a negative weight; needs the feet_ground sensor with fields ('found', 'force'))."""
    f = env.scene[FOOT_SENSOR].data.force                        # [N, 2, 3] net force per foot subtree
    return (torch.relu(torch.linalg.norm(f, dim=-1) - limit_bw * body_weight_n) / body_weight_n).sum(-1)


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


def feet_under_pelvis(env, std: float, max_height: float = 0.08, height_std: float = 0.10) -> torch.Tensor:
    """Get-up: exp(-d^2 / std^2) on the horizontal distance d between the pelvis and the centre of the four foot / toe
    boxes, times a factor that fades when the feet leave the ground (mean box height above max_height) so that legs
    held in the air over the pelvis earn nothing. 1 when standing or squatting, ~0 in a long sit: the gradient that
    pulls the feet back under the body (getup_rev_v2 straightened its legs and sat, at any assist)."""
    ent = env.scene["robot"]
    xy = _foot_geom_xy(env).mean(1)
    _, qp, _ = _root(env)
    d2 = ((xy - (qp[:, :2] - env.scene.env_origins[:, :2])) ** 2).sum(-1)
    z = ent.data.data.geom_xpos[:, env._poolympic_foot_geom_ids, 2].mean(-1)
    return torch.exp(-d2 / std**2) * torch.exp(-(torch.relu(z - max_height) / height_std) ** 2)


# ------------------------------------------------------------------ Iron Pedestal v2: heat-shaped gusts + adaptive level
def push_gust(env, env_ids, asset_cfg=None, min_fraction: float = 1.0, record: bool = False) -> None:
    """Heat gust: instantaneous horizontal Δv of the CURRENT curriculum magnitude in a uniformly random direction
    (IronPedestalHeat / iron_pedestal.py apply exactly this). min_fraction < 1: magnitude U(min_fraction, 1) x the
    level (the shove ramp: Rung 0-2 shoves vary in size). record: keep the Δv and its step for priv_shove."""
    ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=env.device)
    level = getattr(env, "_poolympic_gust", 0.3)
    asset = env.scene["robot"]
    vel = asset.data.root_link_vel_w[ids].clone()
    a = torch.rand(len(ids), device=env.device) * 2 * math.pi
    if min_fraction < 1.0:
        level = level * (min_fraction + (1.0 - min_fraction) * torch.rand(len(ids), device=env.device))
    dv = torch.stack([level * torch.cos(a), level * torch.sin(a)], dim=-1)
    vel[:, 0] += dv[:, 0]
    vel[:, 1] += dv[:, 1]
    asset.write_root_link_velocity_to_sim(vel, env_ids=ids)
    if record:
        st = _shove_state(env)
        st[0][ids], st[1][ids] = dv, int(env.common_step_counter)


def gust_curriculum(env, env_ids, start: float, step: float, max_level: float, up: float, down: float,
                    steps_per_it: int = 0, min_count: int = 64) -> dict:
    """Adaptive gust magnitude: among the envs being reset, the fraction that lasted the full episode decides —
    > up: harder (+step), < down: easier (−step). Keeps training at the edge of what the athlete can survive.
    steps_per_it = 0 (Pedestal2): one decision per reset batch. steps_per_it = 24 (the shove ramp, DESIGN §5 "intensity
    ramps when success > 80 %"): the ended episodes are pooled and ONE decision is taken per PPO iteration, once
    min_count episodes have ended (per-reset decisions arrive in waves and oscillate: ped_v2, getup_rev)."""
    if not hasattr(env, "_poolympic_gust"):
        env._poolympic_gust = start
        env._poolympic_gust_pool = [0.0, 0.0, -1]            # full episodes, ended episodes, last iteration

    def decide(full: float) -> None:
        if full > up:
            env._poolympic_gust = min(max_level, env._poolympic_gust + step)
        elif full < down:
            env._poolympic_gust = max(start, env._poolympic_gust - step)

    pool = env._poolympic_gust_pool
    if env_ids is not None and len(env_ids) > 0:
        full = (env.episode_length_buf[env_ids] >= env.max_episode_length - 1).float()
        if steps_per_it:
            pool[0] += full.sum().item()
            pool[1] += len(env_ids)
        else:
            decide(full.mean().item())
    if steps_per_it:
        it = int(env.common_step_counter) // steps_per_it
        if it != pool[2] and pool[1] >= min_count:
            decide(pool[0] / pool[1])
            pool[0] = pool[1] = 0.0
            pool[2] = it
    return {"gust_mps": torch.tensor(env._poolympic_gust)}


# ------------------------------------------------------------------ Phase Z: zombie style (movement personality)
def _heading_frame_xy(env, v_xy: torch.Tensor) -> torch.Tensor:
    """World xy vectors (..., 2) rotated into the pelvis heading frame (x forward, y left)."""
    _, qp, _ = _root(env)
    yaw = heading_yaw(qp[:, 3:7])
    c, s = torch.cos(yaw), torch.sin(yaw)
    while c.dim() < v_xy.dim() - 1:
        c, s = c[:, None], s[:, None]
    return torch.stack([c * v_xy[..., 0] + s * v_xy[..., 1], -s * v_xy[..., 0] + c * v_xy[..., 1]], dim=-1)


def torso_pitch_tracking(env, target_deg: float, std_deg: float, body_name: str = "torso") -> torch.Tensor:
    """Reward the torso leaning forward by target_deg (zombie hunch) instead of standing upright: signed pitch of the
    torso z axis towards the heading direction."""
    ent = env.scene["robot"]
    bid = _body_id(env, body_name)
    w, x, y, z = ent.data.body_link_quat_w[:, bid].unbind(-1)
    axis = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x)], dim=-1)          # torso z axis, world xy part
    up = 1.0 - 2.0 * (x * x + y * y)
    fwd = _heading_frame_xy(env, axis)[:, 0]
    pitch = torch.atan2(fwd, up)
    return torch.exp(-((pitch - math.radians(target_deg)) ** 2) / math.radians(std_deg) ** 2)


def feet_width(env, target: float, std: float) -> torch.Tensor:
    """Reward a lateral foot spacing of `target` metres (heading frame) — the zombie's wide stance."""
    xy = _foot_geom_xy(env)                                                        # (N, 4, 2): foot_l, toe_l, foot_r, toe_r
    lat = _heading_frame_xy(env, xy)[..., 1]
    width = lat[:, 0] - lat[:, 2]
    return torch.exp(-((width - target) ** 2) / std**2)


def feet_low(env, max_height: float, std: float) -> torch.Tensor:
    """Reward keeping both foot boxes below max_height (geom centre, m above the ground) — the zombie shuffle: feet
    skim the ground instead of lifting high."""
    ent = env.scene["robot"]
    ids = getattr(env, "_poolympic_foot_geom_ids", None)
    if ids is None:
        _foot_geom_xy(env)
        ids = env._poolympic_foot_geom_ids
    h = ent.data.data.geom_xpos[:, ids[[0, 2]], 2]
    excess = torch.relu(h - max_height)
    return torch.exp(-(excess**2).sum(-1) / std**2)


# ====================================================================================================================
# 2026-10-11 audit additions — every term below is OPT-IN: no task registered before that date refers to them.
# ====================================================================================================================
def _per_world(field: torch.Tensor, n: int) -> torch.Tensor:
    """A model field as [num_envs, ...] whether or not domain randomisation expanded it per world."""
    return field if field.shape[0] == n else field.expand(n, *field.shape[1:])


def robot_mass(env) -> torch.Tensor:
    """[N] total mass of each env's athlete (kg), read from the model: any body, mass DR included."""
    ent = env.scene["robot"]
    return _per_world(env.sim.model.body_mass, env.num_envs)[:, ent.indexing.body_ids].sum(-1)


# ---- rewards -------------------------------------------------------------------------------------------------------
SELF_SENSOR = "self_contact"


def self_contact_force(env, body_weight_n: float, sensor_name: str = SELF_SENSOR, deadband_bw: float = 0.0) -> torch.Tensor:
    """Σ over the self-contact slots of relu(|F| − deadband) / body weight (use with a small negative weight; AGENTS.md
    "Self-collision": self-contact never ends an episode — if the policy leans on it, charge the force). Needs the sensor
    of matt_env.self_contact_sensor_cfg (athlete subtree against itself, fields found + force)."""
    f = torch.linalg.norm(env.scene[sensor_name].data.force, dim=-1)              # [N, slots]
    return torch.relu(f / body_weight_n - deadband_bw).sum(-1)


def com_over_support(env, std: float) -> torch.Tensor:
    """exp(-d² / std²), d = horizontal distance between the whole-body centre of mass and the centre of the four foot /
    toe boxes (DESIGN §5 Rung 0 "CoM over support"). Standing: 1; leaning until the CoM leaves the feet: -> 0."""
    ent = env.scene["robot"]
    gid = getattr(env, "_poolympic_root_gid", None)
    if gid is None:
        gid = env._poolympic_root_gid = int(ent.indexing.body_ids[_body_id(env, "pelvis")])
    com = env.sim.data.subtree_com[:, gid, :2] - env.scene.env_origins[:, :2]
    d2 = ((com - _foot_geom_xy(env).mean(1)) ** 2).sum(-1)
    return torch.exp(-d2 / std**2)


# ---- domain randomisation --------------------------------------------------------------------------------------------
def _solref_event():
    from mjlab.envs.mdp.dr import _core
    from mjlab.managers.event_manager import requires_model_fields

    @requires_model_fields("geom_solref")
    def dr_geom_solref(env, env_ids, ranges, asset_cfg, operation: str = "scale", shared_random: bool = True) -> None:
        """Contact softness / restitution (DESIGN §4 "restitution via solref"): randomise geom_solref = (time constant,
        damping ratio). ranges {0: (lo, hi), 1: (lo, hi)}; a damping ratio below 1 makes the contact bounce. The time
        constant must stay >= 2 x the timestep (0.01 s here; the default is 0.02 s)."""
        _core._randomize_model_field(env, env_ids, "geom_solref", entity_type="geom", ranges=ranges, operation=operation,
                                     asset_cfg=asset_cfg, shared_random=shared_random, default_axes=[0, 1],
                                     valid_axes=[0, 1])
    return dr_geom_solref


dr_geom_solref = _solref_event()


# ---- biomechanics in the physics (poolympic/biomech.py; contract v5 proposal) -----------------------------------------
def _bio_event():
    from types import SimpleNamespace

    from mjlab.managers.event_manager import requires_model_fields

    def state(env):
        st = getattr(env, "_poolympic_bio", None)
        if st is None:
            ix = _idx(env)
            _act_torque_vel(env)                                           # resolves env._poolympic_act_ids (contract order)
            gi, mean, wmax = biomech.tables(CONTRACT_ACTUATORS)
            ctrl = ix.entity.indexing.ctrl_ids[env._poolympic_act_ids]
            base = env.sim.model.actuator_forcerange[:, ctrl].clone()      # per env: after the strength DR (startup)
            f32 = lambda a: torch.as_tensor(a, device=env.device, dtype=torch.float32)  # noqa: E731
            st = env._poolympic_bio = SimpleNamespace(
                ctrl=ctrl, base=base, cap=base.abs().amax(-1), gi=torch.as_tensor(gi, device=env.device), mean=f32(mean),
                wmax=f32(wmax), level=torch.zeros(env.num_envs, len(biomech.JOINT_GROUPS), device=env.device))
        return st

    @requires_model_fields("actuator_forcerange")
    def bio_force_limits(env, env_ids, hill: bool = True, fatigue: bool = True, init_level_max: float = 0.0) -> None:
        """Register TWICE: mode="step" (once per control tick, after its substeps: update the fatigue levels from the
        tick's actuator force, then write each actuator's force range for the next tick — biomech.force_scale) and
        mode="reset" (level U(0, init_level_max) per group, force range = its fatigue scale; 0 = fresh, as every CPU /
        Unity episode starts — a training episode is 20 s, too short to reach the tired states by effort alone).
        CPU mirror: biomech.BioLimits."""
        st = state(env)
        fr = env.sim.model.actuator_forcerange
        if env_ids is not None:                                            # reset
            st.level[env_ids] = torch.rand(len(env_ids), st.level.shape[1], device=env.device) * init_level_max
            scale, _ = biomech.force_scale(st.level[env_ids][:, st.gi], 0.0, st.wmax, False, fatigue)
            fr[env_ids[:, None], st.ctrl] = st.base[env_ids] * scale[..., None]
            return
        tau, qd = _act_torque_vel(env)
        st.level = biomech.fatigue_step(st.level, (tau.abs() / st.cap) @ st.mean, env.step_dt)
        lo, hi = biomech.force_scale(st.level[:, st.gi], qd, st.wmax, hill, fatigue)
        fr[:, st.ctrl, 0] = st.base[..., 0] * lo
        fr[:, st.ctrl, 1] = st.base[..., 1] * hi
        env.extras.setdefault("log", {})["Metrics/fatigue_max_group"] = st.level.mean(0).max()
    return bio_force_limits


bio_force_limits = _bio_event()


def obs_fatigue(env) -> torch.Tensor:
    """Contract v5 block: the fatigue level of each joint group (biomech.JOINT_GROUPS order), 0 fresh .. 1 exhausted.
    Zeros until bio_force_limits has run (it owns the state: the force-range baseline must be read after startup DR)."""
    st = getattr(env, "_poolympic_bio", None)
    if st is None:
        return torch.zeros(env.num_envs, len(biomech.JOINT_GROUPS), device=env.device)
    return st.level


# ---- privileged critic observations (asymmetric critic; never in the actor group, never exported) ----------------------
PRIV_DIM = 8          # foot forces 4 + friction 1 + mass 1 + shove 2 (tasks/symmetry.py mirrors this block)


def priv_foot_force(env, body_weight_n: float) -> torch.Tensor:
    """[N, 4] ground reaction per foot subtree in body weights: (left vertical, left horizontal, right vertical, right
    horizontal). Needs the feet_ground sensor (Rung 1+)."""
    f = env.scene[FOOT_SENSOR].data.force / body_weight_n                    # [N, 2, 3] world frame (netforce)
    return torch.stack([f[:, 0, 2].abs(), torch.linalg.norm(f[:, 0, :2], dim=-1),
                        f[:, 1, 2].abs(), torch.linalg.norm(f[:, 1, :2], dim=-1)], dim=-1)


def priv_friction_scale(env) -> torch.Tensor:
    """[N, 1] foot friction of this env / nominal (dr_foot_friction: one shared scale for the four foot geoms)."""
    _foot_geom_xy(env)
    g = int(env._poolympic_foot_geom_ids[0])
    cur = _per_world(env.sim.model.geom_friction, env.num_envs)[:, g, 0]
    return (cur / env.sim.get_default_field("geom_friction")[g, 0])[:, None]


def priv_mass_scale(env) -> torch.Tensor:
    """[N, 1] athlete mass of this env / nominal (dr_mass)."""
    ids = env.scene["robot"].indexing.body_ids
    return (robot_mass(env) / env.sim.get_default_field("body_mass")[ids].sum())[:, None]


def _shove_state(env):
    st = getattr(env, "_poolympic_shove", None)
    if st is None:
        st = env._poolympic_shove = [torch.zeros(env.num_envs, 2, device=env.device),
                                     torch.full((env.num_envs,), -10**6, device=env.device, dtype=torch.long)]
    return st


def priv_shove(env, hold_steps: int = 25) -> torch.Tensor:
    """[N, 2] the last shove's Δv (m/s) in the heading frame, for hold_steps control ticks after it (0.5 s), else 0.
    Shoves must come from push_gust(record=True)."""
    dv, at = _shove_state(env)
    age = int(env.common_step_counter) - at
    # not a previous episode's shove: one from this episode is younger than the episode (events fire after the step
    # counter moved on); one from the last episode's terminal step is exactly as old as the new episode
    live = ((age < hold_steps) & (age < env.episode_length_buf)).float()[:, None]
    return _heading_frame_xy(env, dv) * live
