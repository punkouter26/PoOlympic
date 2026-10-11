"""Left/right mirror symmetry for PPO data augmentation (rsl_rl Symmetry extension; Mittal et al., ICRA 2024).

Mirror = reflection across the athlete's sagittal plane (heading frame y -> -y). With MATT's joint conventions
(build_mjcf.py: every hinge's positive direction is anatomical, right-side axes are mirror images of the left) the limb
joints map onto their partner with NO sign change; the midline abdomen lateral-bend and twist flip sign.
Observation (contract.OBS_LAYOUT, 84):
  lin vel (heading)  (x, y, z) -> (x, -y, z)        ang vel (pelvis)  (x, y, z) -> (-x, y, -z)
  gravity (pelvis)   (x, y, z) -> (x, -y, z)        height -> height
  command (vx, vy, wz) -> (vx, -vy, -wz)            gait phase -> phase + 0.5 while moving ((sin, cos) -> (-sin, -cos));
                                                    unchanged while standing (clock frozen at 0)
  joint pos / joint vel / last action -> partner joint (x sign)
Contract v4 skill block (95-dim obs): pelvis height, march -> same; lift foot (l, r) -> (r, l); torso aim (yaw, pitch)
-> (-yaw, pitch); hand target (x, y, z) -> (x, -y, z); arm -> -arm. A march cadence runs the clock, so it shifts too.
Verified against physically mirrored MuJoCo states in tests/test_symmetry.py.
"""

from __future__ import annotations

import json

import torch

from .. import contract as C

_NAMES = [a["name"] for a in json.loads(C.CONTRACT_JSON.read_text())["actuators"]]
_FLIP = {"abdomen_lat", "abdomen_twist"}


def _partner(n: str) -> str:
    return n[:-2] + "_r" if n.endswith("_l") else n[:-2] + "_l" if n.endswith("_r") else n


PERM = [_NAMES.index(_partner(n)) for n in _NAMES]
SIGN = [-1.0 if n in _FLIP else 1.0 for n in _NAMES]


def _obs_maps() -> tuple[list[int], list[float]]:
    lay = dict(C.OBS_LAYOUT)
    idx, sgn = [], []
    off = 0
    for name, size in C.OBS_LAYOUT:
        if name in ("base_lin_vel_heading", "projected_gravity"):
            idx += [off, off + 1, off + 2]; sgn += [1, -1, 1]
        elif name == "base_ang_vel_local":
            idx += [off, off + 1, off + 2]; sgn += [-1, 1, -1]
        elif name == "base_height":
            idx += [off]; sgn += [1]
        elif name == "command":
            idx += [off, off + 1, off + 2]; sgn += [1, -1, -1]
        elif name == "gait_phase_sincos":
            idx += [off, off + 1]; sgn += [-1, -1]
        else:  # joint blocks
            assert size == len(_NAMES), name
            idx += [off + p for p in PERM]; sgn += SIGN
        off += size
    assert off == C.OBS_DIM and len(lay) == len(C.OBS_LAYOUT)
    return idx, sgn


OBS_IDX, OBS_SIGN = _obs_maps()
# contract v4 skill block (C.SKILL_LAYOUT order, after the 84 v3 obs)
_S = C.OBS_DIM
SKILL_IDX = [_S + 0, _S + 2, _S + 1, _S + 3, _S + 4, _S + 5, _S + 6, _S + 7, _S + 8, _S + 9, _S + 10]
SKILL_SIGN = [1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1]
assert len(SKILL_IDX) == C.SKILL_DIM
MARCH_HZ = _S + 3
# 2026-10-11 (opt-in blocks). Contract v5 fatigue block: one level per joint group, both sides share it -> unchanged.
# Privileged critic block (mdp.PRIV_DIM: foot forces (L vertical, L horizontal, R vertical, R horizontal), friction,
# mass, shove (x, y) in the heading frame): the feet swap, the shove's y flips. Layouts by input size:
#   84 v3 · 95 v4 (skill) · 90 v5 (fatigue) · 92 v3 + privileged · 98 v5 + privileged
_PRIV_PERM, _PRIV_SIGN = [2, 3, 0, 1, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1, 1, -1]


def _tails() -> dict:
    from .mdp import PRIV_DIM
    assert len(_PRIV_PERM) == PRIV_DIM
    fat = (list(range(_S, _S + C.FATIGUE_DIM)), [1] * C.FATIGUE_DIM)
    priv = lambda o: ([o + p for p in _PRIV_PERM], _PRIV_SIGN)  # noqa: E731
    return {C.OBS_DIM: ([], []), C.OBS_DIM_V4: (SKILL_IDX, SKILL_SIGN), C.OBS_DIM_V5: fat, _S + PRIV_DIM: priv(_S),
            C.OBS_DIM_V5 + PRIV_DIM: (fat[0] + priv(C.OBS_DIM_V5)[0], fat[1] + _PRIV_SIGN)}


TAILS = _tails()


_OFF = {}
_o = 0
for _n, _sz in C.OBS_LAYOUT:
    _OFF[_n] = (_o, _o + _sz)
    _o += _sz
CMD_SLICE, PHASE_SLICE = slice(*_OFF["command"]), slice(*_OFF["gait_phase_sincos"])


def mirror_obs(x: torch.Tensor) -> torch.Tensor:
    v4 = x.shape[-1] == C.OBS_DIM_V4
    tail_idx, tail_sgn = TAILS[x.shape[-1]]
    idx = torch.as_tensor(OBS_IDX + tail_idx, device=x.device)
    sgn = torch.as_tensor(OBS_SIGN + tail_sgn, device=x.device, dtype=x.dtype)
    y = x[..., idx] * sgn
    # The gait clock is frozen at phase 0 while standing (|cmd| < threshold): a mirrored standing athlete is still at
    # phase 0, so only a running clock shifts by half a stride. (r2_v3 bug: unconditional flip made every mirrored
    # standing sample carry phase 0.5 — an impossible state.) v4: a march cadence > 0 runs the clock too.
    moving = x[..., CMD_SLICE].norm(dim=-1, keepdim=True) >= C.PHASE_CMD_THRESHOLD
    if v4:
        moving = moving | (x[..., MARCH_HZ:MARCH_HZ + 1] > 0)
    y[..., PHASE_SLICE] = torch.where(moving, y[..., PHASE_SLICE], x[..., PHASE_SLICE])
    return y


def mirror_actions(a: torch.Tensor) -> torch.Tensor:
    idx = torch.as_tensor(PERM, device=a.device)
    sgn = torch.as_tensor(SIGN, device=a.device, dtype=a.dtype)
    return a[..., idx] * sgn


def augment(env=None, obs=None, actions=None):
    """rsl_rl data_augmentation_func: returns (original ++ mirrored) along the batch dimension."""
    out_obs = out_act = None
    if obs is not None:
        out_obs = obs.clone()
        mirrored = obs.clone()
        for k in mirrored.keys():
            mirrored[k] = mirror_obs(obs[k])
        out_obs = torch.cat([obs, mirrored], dim=0)
    if actions is not None:
        out_act = torch.cat([actions, mirror_actions(actions)], dim=0)
    return out_obs, out_act


def _runner_base():
    from mjlab.rl import MjlabOnPolicyRunner
    return MjlabOnPolicyRunner


class SymmetricRunner(_runner_base()):
    """MjlabOnPolicyRunner with rsl_rl mirror data augmentation (every PPO mini-batch + its left/right mirror)."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        train_cfg["algorithm"]["symmetry_cfg"] = {
            "data_augmentation_func": augment,
            "use_data_augmentation": True,
            # r2_v3b: augmentation alone let the mirror loss rise (0.36 -> 0.70) — mirrored samples reuse the original
            # action's old log-prob, so for an asymmetric starting policy their PPO ratios are clipped away. The explicit
            # mirror loss penalises pi(mirror(o)) != mirror(pi(o)) directly.
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.5,
        }
        super().__init__(env, train_cfg, log_dir, device)
