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


_OFF = {}
_o = 0
for _n, _sz in C.OBS_LAYOUT:
    _OFF[_n] = (_o, _o + _sz)
    _o += _sz
CMD_SLICE, PHASE_SLICE = slice(*_OFF["command"]), slice(*_OFF["gait_phase_sincos"])


def mirror_obs(x: torch.Tensor) -> torch.Tensor:
    idx = torch.as_tensor(OBS_IDX, device=x.device)
    sgn = torch.as_tensor(OBS_SIGN, device=x.device, dtype=x.dtype)
    y = x[..., idx] * sgn
    # The gait clock is frozen at phase 0 while standing (|cmd| < threshold): a mirrored standing athlete is still at
    # phase 0, so only a running clock shifts by half a stride. (r2_v3 bug: unconditional flip made every mirrored
    # standing sample carry phase 0.5 — an impossible state.)
    moving = x[..., CMD_SLICE].norm(dim=-1, keepdim=True) >= C.PHASE_CMD_THRESHOLD
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
