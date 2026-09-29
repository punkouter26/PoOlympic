"""Contract v4 stance skills — mjlab MDP terms for Rung S (docs/CONTRACT_V4_STANCE_PROPOSAL.md).

The measured quantities mirror poolympic/skills.py (numpy, used by the G1 drills) — keep the two in sync:
  torso aim  chest x-axis yaw relative to the pelvis heading, forward pitch = asin(-x_z)
  hand       forearm tip (far end of the forearm capsule) in the pelvis heading frame
  knee rise  shin body origin z − its standing z;  foot rise: foot body z − its standing z
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .. import contract as C
from .mdp import AthleteCommand, AthleteCommandCfg, _foot_contact, _root, torso_upright

SKILL_MODES = ("locomotion", "squat", "flamingo", "march", "torso", "reach")
MODE = {n: i for i, n in enumerate(SKILL_MODES)}


def _skill_geometry() -> dict:
    """CPU, once: standing pelvis / knee / foot heights, forearm tip offsets, shoulders in the heading frame."""
    import mujoco

    from .. import skills as K
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
    mujoco.mj_forward(m, d)
    sb = K.SkillBodies.bind(m)
    feet = [float(d.xpos[m.body(n).id][2]) for n in ("foot_l", "foot_r")]
    return {"pelvis_z0": sb.pelvis_z0, "knee_z0": sb.knee_z0, "foot_z0": tuple(feet), "tip_local": sb.tip_local,
            "shoulder": (K.shoulder_in_heading(d, m, -1), K.shoulder_in_heading(d, m, 1))}


SKILL_GEOM = _skill_geometry()
SKILL_RANGES = C.skill_block()["ranges"]
REACH_AZ = (math.radians(-30), math.radians(90))      # target direction from the shoulder: forward .. arm's side
REACH_EL = (math.radians(-40), math.radians(60))
REACH_FRAC = (0.45, 0.95)                             # of the reach radius


def _qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R(q) v for wxyz quaternions (batched)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


class _SkillIdx:
    def __init__(self, env):
        ent = env.scene["robot"]
        names = ("pelvis", "chest", "shin_l", "shin_r", "forearm_l", "forearm_r", "foot_l", "foot_r")
        ids = ent.find_bodies(names, preserve_order=True)[0]
        self.b = dict(zip(names, ids))
        dev = env.device
        self.tip = torch.as_tensor(np.stack(SKILL_GEOM["tip_local"]), device=dev, dtype=torch.float32)
        self.knee_z0 = torch.as_tensor(SKILL_GEOM["knee_z0"], device=dev, dtype=torch.float32)
        self.foot_z0 = torch.as_tensor(SKILL_GEOM["foot_z0"], device=dev, dtype=torch.float32)
        self.ent = ent


def _sidx(env) -> _SkillIdx:
    c = getattr(env, "_poolympic_sidx", None)
    if c is None:
        c = _SkillIdx(env)
        env._poolympic_sidx = c
    return c


def _pelvis_yaw(env) -> torch.Tensor:
    _, qp, _ = _root(env)
    w, x, y, z = qp[:, 3:7].unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def measure_torso_aim(env) -> torch.Tensor:
    """[N, 2] chest (yaw rel. pelvis heading, forward pitch) — skills.torso_aim."""
    s = _sidx(env)
    q = s.ent.data.body_link_quat_w[:, s.b["chest"]]
    ex = torch.zeros(q.shape[0], 3, device=q.device, dtype=q.dtype)
    ex[:, 0] = 1.0
    x = _qrot(q, ex)
    yaw = torch.remainder(torch.atan2(x[:, 1], x[:, 0]) - _pelvis_yaw(env) + math.pi, 2 * math.pi) - math.pi
    return torch.stack([yaw, torch.asin(torch.clamp(-x[:, 2], -1.0, 1.0))], dim=-1)


def measure_hand(env, arm: torch.Tensor) -> torch.Tensor:
    """[N, 3] forearm tip of `arm` (-1 left / +1 right, per env) in the pelvis heading frame — skills.hand_in_heading."""
    s = _sidx(env)
    right = arm > 0
    fb = torch.where(right, torch.full(arm.shape, s.b["forearm_r"], device=arm.device, dtype=torch.long),
                     torch.full(arm.shape, s.b["forearm_l"], device=arm.device, dtype=torch.long))
    rows = torch.arange(arm.shape[0], device=arm.device)
    pos = s.ent.data.body_link_pos_w[rows, fb]
    quat = s.ent.data.body_link_quat_w[rows, fb]
    tip = pos + _qrot(quat, s.tip[right.long()])
    rel = tip - s.ent.data.body_link_pos_w[:, s.b["pelvis"]]
    yaw = _pelvis_yaw(env)
    c, sn = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * rel[:, 0] + sn * rel[:, 1], -sn * rel[:, 0] + c * rel[:, 1], rel[:, 2]], dim=-1)


def measure_knee_rise(env) -> torch.Tensor:
    s = _sidx(env)
    return s.ent.data.body_link_pos_w[:, [s.b["shin_l"], s.b["shin_r"]], 2] - s.knee_z0


def measure_foot_rise(env) -> torch.Tensor:
    s = _sidx(env)
    return s.ent.data.body_link_pos_w[:, [s.b["foot_l"], s.b["foot_r"]], 2] - s.foot_z0


# ---- command ---------------------------------------------------------------------------------------------------
class AthleteSkillCommand(AthleteCommand):
    """AthleteCommand + the v4 skill block. Every resample draws a mode per env (cfg.mode_probs over SKILL_MODES):
    locomotion = the Rung 2 velocity command with a zero skill block; every stance skill zeroes the velocity command
    (stands still) and samples its values in the approved ranges. A march cadence > 0 drives the gait clock
    (contract.advance_phase with cadence)."""

    def __init__(self, cfg: "AthleteSkillCommandCfg", env):
        super().__init__(cfg, env)
        self.skill = torch.zeros(self.num_envs, C.SKILL_DIM, device=self.device)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._shoulder = torch.as_tensor(np.stack(SKILL_GEOM["shoulder"]), device=self.device, dtype=torch.float32)

    def compute(self, dt, env_ids=None):
        prev = self.phase.clone()
        super().compute(dt, env_ids)
        if env_ids is None:
            cad = self.skill[:, 3]
            adv = torch.remainder(prev + cad * C.DECIMATION * 0.005, 1.0)
            self.phase = torch.where(cad > 0.0, adv, self.phase)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        mode = torch.multinomial(torch.as_tensor(self.cfg.mode_probs, device=dev, dtype=torch.float32), n, replacement=True)
        self.mode[env_ids] = mode
        sk = torch.zeros(n, C.SKILL_DIM, device=dev)

        def U(lo, hi, k):
            return torch.empty(k, device=dev).uniform_(lo, hi)

        r = SKILL_RANGES
        m = mode == MODE["squat"]
        sk[m, 0] = U(*r["pelvis_height"], int(m.sum()))
        m = mode == MODE["flamingo"]
        left = torch.rand(n, device=dev) < 0.5
        sk[m & left, 1] = 1.0
        sk[m & ~left, 2] = 1.0
        m = mode == MODE["march"]
        sk[m, 3] = U(*r["march_hz"], int(m.sum()))
        sk[m, 4] = U(*r["knee_lift"], int(m.sum()))
        m = mode == MODE["torso"]
        sk[m, 5] = U(*r["torso_yaw"], int(m.sum()))
        sk[m, 6] = U(*r["torso_pitch"], int(m.sum()))
        m = mode == MODE["reach"]
        k = int(m.sum())
        if k:
            arm = torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
            az, el = U(*REACH_AZ, k), U(*REACH_EL, k)
            rad = U(*REACH_FRAC, k) * r["hand_reach"]
            dirn = torch.stack([torch.cos(el) * torch.cos(az), -arm * torch.cos(el) * torch.sin(az), torch.sin(el)], -1)
            sk[m, 7:10] = self._shoulder[(arm > 0).long()] + dirn * rad[:, None]
            sk[m, 10] = arm
        self.skill[env_ids] = sk
        still = env_ids[mode != MODE["locomotion"]]
        if len(still):
            self.is_standing_env[still] = True
            self.vel_command_b[still] = 0.0


@dataclass(kw_only=True)
class AthleteSkillCommandCfg(AthleteCommandCfg):
    # locomotion, squat, flamingo, march, torso, reach (approved: 20 % locomotion, the rest shared by the skills)
    mode_probs: tuple[float, ...] = (0.20, 0.16, 0.16, 0.16, 0.16, 0.16)

    def build(self, env):
        return AthleteSkillCommand(self, env)


def obs_skill(env, command_name: str = "athlete") -> torch.Tensor:
    return env.command_manager.get_term(command_name).skill


def _term(env, command_name: str):
    return env.command_manager.get_term(command_name)


# ---- rewards: each skill term pays only in its own mode ----------------------------------------------------------
def skill_pelvis_height(env, std: float, command_name: str = "athlete") -> torch.Tensor:
    """Every mode: pelvis at the standing height + the squat target (0 outside squat) — replaces the fixed height
    reward, so squatting is never penalised."""
    _, qp, _ = _root(env)
    target = SKILL_GEOM["pelvis_z0"] + _term(env, command_name).skill[:, 0]
    return torch.exp(-((qp[:, 2] - target) ** 2) / std**2)


def skill_squat(env, std: float, command_name: str = "athlete") -> torch.Tensor:
    return skill_pelvis_height(env, std, command_name) * (_term(env, command_name).mode == MODE["squat"])


def skill_flamingo(env, clearance: float = 0.10, command_name: str = "athlete") -> torch.Tensor:
    """Lifted foot off the ground and at least `clearance` up; the stance foot planted."""
    term = _term(env, command_name)
    lift = term.skill[:, 1:3]
    contact = _foot_contact(env).float()
    rise = measure_foot_rise(env)
    lifted_ok = ((1 - contact) * torch.clamp(rise / clearance, 0.0, 1.0) * lift).sum(-1)
    stance_ok = (contact * (1 - lift)).sum(-1)
    return lifted_ok * stance_ok * (term.mode == MODE["flamingo"])


def march_profile(phase: torch.Tensor, lift: torch.Tensor) -> torch.Tensor:
    """[N, 2] desired knee rise (left swings for phase in [0.5, 1), right for [0, 0.5); half-sine up to `lift`)."""
    return torch.stack([lift * torch.clamp(torch.sin(2 * math.pi * (phase - 0.5)), min=0.0),
                        lift * torch.clamp(torch.sin(2 * math.pi * phase), min=0.0)], dim=-1)


def skill_march(env, std: float, command_name: str = "athlete") -> torch.Tensor:
    term = _term(env, command_name)
    err = ((measure_knee_rise(env) - march_profile(term.phase, term.skill[:, 4])) ** 2).sum(-1)
    return torch.exp(-err / std**2) * (term.mode == MODE["march"])


def skill_torso(env, std: float, command_name: str = "athlete") -> torch.Tensor:
    term = _term(env, command_name)
    err = ((measure_torso_aim(env) - term.skill[:, 5:7]) ** 2).sum(-1)
    return torch.exp(-err / std**2) * (term.mode == MODE["torso"])


def skill_reach(env, std: float, command_name: str = "athlete") -> torch.Tensor:
    term = _term(env, command_name)
    d2 = ((measure_hand(env, term.skill[:, 10]) - term.skill[:, 7:10]) ** 2).sum(-1)
    return torch.exp(-d2 / std**2) * (term.mode == MODE["reach"])


def phase_contact_v4(env, command_name: str = "athlete") -> torch.Tensor:
    """mdp.phase_contact with the march clock (marching in place alternates stance feet) and flamingo (only the stance
    foot down)."""
    term = _term(env, command_name)
    contact = _foot_contact(env).float()
    moving = (torch.linalg.norm(term.command, dim=-1) >= C.PHASE_CMD_THRESHOLD) | (term.skill[:, 3] > 0)
    left_stance = (term.phase < 0.5).float()
    want = torch.stack([left_stance, 1.0 - left_stance], dim=-1)
    want = torch.where(moving[:, None], want, torch.ones_like(want))
    want = torch.where((term.mode == MODE["flamingo"])[:, None], 1.0 - term.skill[:, 1:3], want)
    return (contact == want).float().mean(-1)


def _posture_locomotion_cls():
    from mjlab.tasks.velocity import mdp as vel_mdp

    class posture_locomotion(vel_mdp.variable_posture):  # noqa: N801 (mjlab class-based reward term)
        """Rung 2 variable posture, paid only in locomotion mode (stance skills move joints far from the default)."""

        def __call__(self, env, command_name: str = "athlete", **params) -> torch.Tensor:
            return super().__call__(env, command_name=command_name, **params) * (_term(env, command_name).mode == MODE["locomotion"])

    return posture_locomotion


posture_locomotion = _posture_locomotion_cls()


def upright_unless_aiming(env, std: float, command_name: str = "athlete") -> torch.Tensor:
    """torso_upright except while aiming the torso or squatting (both lean the trunk on purpose)."""
    m = _term(env, command_name).mode
    return torso_upright(env, std) * ((m != MODE["torso"]) & (m != MODE["squat"]))


def pelvis_below_skill(env, minimum_height: float, squat_margin: float = 0.10, command_name: str = "athlete") -> torch.Tensor:
    """Fall line lowered by the squat target (+ a margin while squatting): a deep squat is not a fall."""
    _, qp, _ = _root(env)
    term = _term(env, command_name)
    line = minimum_height + term.skill[:, 0] - (term.mode == MODE["squat"]).float() * squat_margin
    return qp[:, 2] < line
