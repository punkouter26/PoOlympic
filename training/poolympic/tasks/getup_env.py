"""R3 get-up (Event 27, The Resurrection Dash) — reverse curriculum (recipe v5, 2026-09-30).

getup_v1-v5 always started flat on the floor and never found the feet-under transition (they sit up or reach all fours,
rl_optimization_log.md). Here episodes start closer to standing and move back towards lying as the athlete succeeds:

  stage 0 squat  (feet flat, hips / knees deeply bent)      stage 2 long sit (pelvis on the floor, legs forward)
  stage 1 kneel  (tall kneeling, shins on the floor)          stage 3 lying    (supine / prone, the event start)

Each reset draws a stage: the frontier stage 50 %, the next one 10 % (preview), the easier ones share the rest. Success =
"standing tall" (pelvis > 0.85 m, torso < 20°) at the end of the episode. The frontier's success rate is smoothed ONCE
per PPO iteration (ped_v2 lesson: per-reset updates arrive in waves and oscillate) and the frontier advances at > 60 %.
Rewards = getup_v5 (rise = height × uprightness², standing tall) + bio terms; fading torso assist over 600 iterations.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .. import contract as C
from . import mdp
from .matt_env import CUBE_NAMES, V5_ENVS, add_bio_rewards, matt_getup5_env_cfg

STAGES = ("squat", "kneel", "sit", "lying")
# joint angles in degrees (both sides), on top of the default pose; root pitch (+ = trunk forward), pelvis on the ground
POSES = {
    "squat": ({"abdomen_flex": 35, "hip_flex": 105, "knee": 125, "ankle_dorsi": 25}, math.radians(20)),
    "kneel": ({"hip_flex": 0, "knee": 100, "ankle_dorsi": -45}, 0.0),
    "sit": ({"hip_flex": 90, "knee": 15, "ankle_dorsi": 0, "abdomen_flex": 10}, 0.0),
}
UP_Z, UP_TILT_DEG = 0.85, 20.0
ADVANCE_AT = 0.6
MIN_ITS_PER_STAGE = 40


def _pose_table() -> dict:
    """Per stage: joint targets in contract order + the root height that puts the lowest geom 2 mm above the ground
    (CPU FK of the active body's scene, once)."""
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    ath = C.Athlete.bind(m)
    names = list(mdp.CONTRACT_ACTUATORS)
    out = {}
    root_q = ath.root_qposadr
    robot_geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name not in ("world",)
                   and not m.body(m.geom_bodyid[g]).name.startswith("cube")]
    for stage, (angles, pitch) in POSES.items():
        q = np.array(mdp.CONTRACT_DEFAULTS, dtype=float)
        for i, n in enumerate(names):
            base = n[:-2] if n.endswith(("_l", "_r")) else n
            if base in angles:
                q[i] = math.radians(angles[base])
        q = np.clip(q, mdp.CONTRACT_RANGE_LO, mdp.CONTRACT_RANGE_HI)
        mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
        for i, a in enumerate(ath.actuator_ids):
            d.qpos[m.jnt_qposadr[m.actuator_trnid[a][0]]] = q[i]
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([0.0, pitch, 0.0]), "XYZ")
        d.qpos[root_q:root_q + 3] = [0.0, 0.0, 1.0]
        d.qpos[root_q + 3:root_q + 7] = quat
        mujoco.mj_kinematics(m, d)
        low = np.inf
        for g in robot_geoms:
            if m.geom_bodyid[g] == 0:
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            z, s = d.geom_xpos[g][2], m.geom_size[g]
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                bottom = z - s[0]
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                bottom = z - abs(R[2, 2]) * s[1] - s[0]
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                bottom = z - abs(R[2, 0]) * s[0] - abs(R[2, 1]) * s[1] - abs(R[2, 2]) * s[2]
            else:
                continue
            low = min(low, bottom)
        out[stage] = (q, 1.0 - low + 0.002, quat)
    return out


def reset_pose_mix(env, env_ids, joint_noise: float = 0.05, prone_fraction: float = 0.5) -> None:
    """Reset each env into a stage drawn from the curriculum's current mix. Split per stage (boolean indexing syncs,
    but only at episode ends): lying = mdp.reset_lying with the default joint pose, the other stages write their joint
    pose and a root placed on the ground at a random yaw."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    dev = env.device
    st = _state(env)
    stage = torch.multinomial(st["probs"], len(env_ids), replacement=True)
    st["start"][env_ids] = stage
    asset = env.scene["robot"]
    ix = mdp._idx(env)
    for k, name in enumerate(STAGES):
        ids = env_ids[stage == k]
        n = len(ids)
        if n == 0:
            continue
        jp = asset.data.default_joint_pos[ids].clone()
        if name != "lying":
            q, z, quat = st["table"][name]
            jp[:, ix.joint_ids] = q
        jp = jp + (torch.rand_like(jp) * 2 - 1) * joint_noise
        asset.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
        if name == "lying":
            mdp.reset_lying(env, ids, prone_fraction=prone_fraction)
            continue
        half = (torch.rand(n, device=dev) * 2 - 1) * math.pi / 2          # yaw / 2
        qy = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], -1)
        w2, x2, y2, z2 = (float(v) for v in quat)
        w1, z1 = qy[:, 0], qy[:, 3]                                         # yaw ∘ pitch (x1 = y1 = 0)
        qq = torch.stack([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2], -1)
        pos = env.scene.env_origins[ids].clone()
        pos[:, 2] = z
        asset.write_root_link_pose_to_sim(torch.cat([pos, qq], -1), env_ids=ids)
        asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)


def _state(env) -> dict:
    st = getattr(env, "_poolympic_getup_rev", None)
    if st is None:
        st = {"level": 0, "ema": 0.0, "since": 0, "last_it": -1, "succ": 0.0, "count": 0.0,
              "start": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
              "probs": _mix(0, env.device), "table": {k: (torch.as_tensor(v[0], device=env.device, dtype=torch.float32),
                                                          float(v[1]), v[2]) for k, v in _pose_table().items()}}
        env._poolympic_getup_rev = st
    return st


def _mix(level: int, device) -> torch.Tensor:
    p = torch.zeros(len(STAGES), device=device)
    p[level] = 0.5 if level > 0 else 0.9
    if level + 1 < len(STAGES):
        p[level + 1] = 0.1
    if level > 0:
        p[:level] = 0.4 / level
    return p / p.sum()


def getup_rev_curriculum(env, env_ids, steps_per_it: int = 24) -> dict:
    """Called with the envs about to reset (terminal state): count frontier-stage episodes that ended standing tall;
    once per iteration fold the batch into an EMA and advance the frontier when it passes ADVANCE_AT."""
    st = _state(env)
    if env_ids is not None and len(env_ids) > 0:
        _, qp, _ = mdp._root(env)
        up = (qp[env_ids, 2] > UP_Z) & (mdp.torso_tilt_rad(env)[env_ids] < math.radians(UP_TILT_DEG))
        front = st["start"][env_ids] == st["level"]
        st["succ"] += float((up & front).sum())            # one sync per reset batch (resets already synced)
        st["count"] += float(front.sum())
    it = int(env.common_step_counter) // steps_per_it
    if it != st["last_it"]:
        st["last_it"] = it
        if st["count"] >= 64:
            rate = st["succ"] / st["count"]
            st["ema"] = 0.8 * st["ema"] + 0.2 * rate
            st["succ"] = st["count"] = 0.0
        st["since"] += 1
        if st["ema"] > ADVANCE_AT and st["since"] >= MIN_ITS_PER_STAGE and st["level"] + 1 < len(STAGES):
            st["level"] += 1
            st["ema"], st["since"] = 0.0, 0
            st["probs"] = _mix(st["level"], env.device)
    return {"getup_stage": torch.tensor(float(st["level"])), "getup_frontier_success": torch.tensor(st["ema"])}


def matt_getup_rev_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = matt_getup5_env_cfg(play=play)
    for n in CUBE_NAMES:                                    # no cube drops in get-up: no pool at all
        cfg.scene.entities.pop(n, None)
    ev = cfg.events
    ev.pop("reset_joints", None)
    ev["reset_base"] = EventTermCfg(func=reset_pose_mix, mode="reset")
    if "getup_assist" in ev:                                # fade the torso assist over 600 its (8192 envs: 2x samples)
        ev["getup_assist"].params.update({"max_fraction": 0.4, "decay_steps": 600 * 24,
                                          "body_weight_n": C.BODY.total_mass * 9.81})
    cfg.curriculum = {"getup_rev": CurriculumTermCfg(func=getup_rev_curriculum, params={})}
    add_bio_rewards(cfg)
    cfg.scene.num_envs = V5_ENVS
    return cfg
