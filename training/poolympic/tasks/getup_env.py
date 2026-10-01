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


# ---- getup_rev_v2 (2026-10-01): pose ladder with interpolated stages + a success-driven assist ---------------------
# getup_rev_v1 stood up from the squat within 50 its, then sat at 0 % from the kneel for 1400: the next stage was a
# different posture with no way up the policy already knew, and the one global assist had faded by it 600 without a
# single success. Here
#   - the ladder is squat -> tuck -> supine (the event start is on the back) with LADDER_SUB interpolated start poses
#     between the key poses (joint targets and trunk pitch blended, root placed on the ground): every stage starts a
#     small step away from one the athlete can already get up from. squat -> tuck keeps the squat's joints and rocks the
#     body back onto its back (knees to the chest); tuck -> supine stretches out on the floor. Run in reverse that is a
#     rock-up onto the feet. (First tried squat -> long sit -> supine: its squat/sit blends hang the pelvis 0.4 m in the
#     air on the heels and drop the athlete onto its back, harder than the stages after them.)
#   - the upward torso assist belongs to the frontier stage and follows its success rate instead of the clock: no assist
#     for the first ASSIST_GRACE its of a stage, then it rises while fewer than ADVANCE_AT of the frontier episodes end
#     standing and falls while more do. The frontier advances only after ASSIST_ZERO_ITS its without assist. Easier
#     stages are never assisted.
LADDER_KEYS = ("squat", "tuck", "supine")
LADDER_POSES = {"squat": POSES["squat"], "tuck": (POSES["squat"][0], -math.pi / 2),   # the squat's joints, on the back
                "supine": ({}, -math.pi / 2)}                                          # default joint pose, flat on the back
LADDER_SUB = 3
LADDER_EPISODE_S = 6.0          # a stage's verdict arrives one episode later: 12.5 its instead of 21
LADDER_MIN_ITS = 15
ASSIST_GRACE = 15               # its of a new stage before the assist may rise (one episode + the EMA warm-up)
ASSIST_GAIN = 0.05              # per it: +0.03 body weights at 0 % success, -0.02 at 100 %
ASSIST_MAX = 0.6
ASSIST_ZERO_ITS = 5


def ladder_poses() -> list[tuple[np.ndarray, float, np.ndarray]]:
    """(joint targets in contract order, root height, root quaternion) per ladder stage: stage 0 = the first key pose,
    then LADDER_SUB blends towards each next key pose (the last of them = that key pose)."""
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    ath = C.Athlete.bind(m)
    names = list(mdp.CONTRACT_ACTUATORS)
    root_q = ath.root_qposadr

    def key(stage: str) -> tuple[np.ndarray, float]:
        angles, pitch = LADDER_POSES[stage]
        q = np.array(mdp.CONTRACT_DEFAULTS, dtype=float)
        for i, n in enumerate(names):
            base = n[:-2] if n.endswith(("_l", "_r")) else n
            if base in angles:
                q[i] = math.radians(angles[base])
        return np.clip(q, mdp.CONTRACT_RANGE_LO, mdp.CONTRACT_RANGE_HI), pitch

    def place(q: np.ndarray, pitch: float) -> tuple[np.ndarray, float, np.ndarray]:
        mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
        for i, a in enumerate(ath.actuator_ids):
            d.qpos[m.jnt_qposadr[m.actuator_trnid[a][0]]] = q[i]
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([0.0, pitch, 0.0]), "XYZ")
        d.qpos[root_q:root_q + 3] = [0.0, 0.0, 1.0]
        d.qpos[root_q + 3:root_q + 7] = quat
        mujoco.mj_kinematics(m, d)
        low = np.inf
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b == 0 or m.body(b).name.startswith("cube"):
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            z, s, t = d.geom_xpos[g][2], m.geom_size[g], m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                low = min(low, z - s[0])
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                low = min(low, z - abs(R[2, 2]) * s[1] - s[0])
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                low = min(low, z - abs(R[2, 0]) * s[0] - abs(R[2, 1]) * s[1] - abs(R[2, 2]) * s[2])
        return q, 1.0 - low + 0.002, quat

    keys = [key(k) for k in LADDER_KEYS]
    out = [place(*keys[0])]
    for (qa, pa), (qb, pb) in zip(keys, keys[1:]):
        for j in range(1, LADDER_SUB + 1):
            a = j / LADDER_SUB
            out.append(place((1 - a) * qa + a * qb, (1 - a) * pa + a * pb))
    return out


def _ladder_state(env, start_level: int = 0) -> dict:
    """start_level: the frontier a run starts at (a warm start that already passed the stages below it)."""
    st = getattr(env, "_poolympic_getup_ladder", None)
    if st is None:
        table = [(torch.as_tensor(q, device=env.device, dtype=torch.float32), float(z), quat) for q, z, quat in ladder_poses()]
        st = {"level": start_level, "ema": 0.0, "since": 0, "last_it": -1, "succ": 0.0, "count": 0.0, "assist": 0.0, "zero_its": 0,
              "start": torch.zeros(env.num_envs, dtype=torch.long, device=env.device), "table": table,
              "probs": _ladder_mix(start_level, len(table), env.device)}
        env._poolympic_getup_ladder = st
    return st


def _ladder_mix(level: int, n: int, device) -> torch.Tensor:
    """Frontier 50 % (90 % at stage 0), the next stage 10 % (preview), the easier ones share the rest."""
    p = torch.zeros(n, device=device)
    p[level] = 0.5 if level > 0 else 0.9
    if level + 1 < n:
        p[level + 1] = 0.1
    if level > 0:
        p[:level] = 0.4 / level
    return p / p.sum()


def reset_ladder(env, env_ids, joint_noise: float = 0.05, start_level: int = 0) -> None:
    """Reset each env into a ladder stage drawn from the curriculum's current mix. The last stage is the event start
    (mdp.reset_lying, supine: tools/getup_probe.py); the others write their joint pose and a root placed on the ground
    at a random yaw."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    dev = env.device
    st = _ladder_state(env, start_level)
    stage = torch.multinomial(st["probs"], len(env_ids), replacement=True)
    st["start"][env_ids] = stage
    asset = env.scene["robot"]
    ix = mdp._idx(env)
    last = len(st["table"]) - 1
    for k, (q, z, quat) in enumerate(st["table"]):
        ids = env_ids[stage == k]
        n = len(ids)
        if n == 0:
            continue
        jp = asset.data.default_joint_pos[ids].clone()
        if k != last:
            jp[:, ix.joint_ids] = q
        jp = jp + (torch.rand_like(jp) * 2 - 1) * joint_noise
        asset.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
        if k == last:
            mdp.reset_lying(env, ids, prone_fraction=0.0)
            continue
        half = (torch.rand(n, device=dev) * 2 - 1) * math.pi / 2          # yaw / 2
        w1, z1 = torch.cos(half), torch.sin(half)
        w2, x2, y2, z2 = (float(v) for v in quat)
        qq = torch.stack([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2], -1)   # yaw * pitch
        pos = env.scene.env_origins[ids].clone()
        pos[:, 2] = z
        asset.write_root_link_pose_to_sim(torch.cat([pos, qq], -1), env_ids=ids)
        asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)


def ladder_assist(env, env_ids, asset_cfg, body_weight_n: float) -> None:
    """Reset event, after reset_ladder: a constant upward world force on the torso for the episode, U(0, assist) x body
    weight, for episodes that start at the frontier stage or beyond; none for the easier stages."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    st = _ladder_state(env)
    n = len(env_ids)
    on = (st["start"][env_ids] >= st["level"]).float()
    fz = torch.rand(n, device=env.device) * st["assist"] * body_weight_n * on
    nb = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else 1
    force = torch.zeros(n, nb, 3, device=env.device)
    force[:, :, 2] = fz[:, None]
    asset.write_external_wrench_to_sim(force, torch.zeros_like(force), env_ids=env_ids, body_ids=asset_cfg.body_ids)
    env.extras.setdefault("log", {})["Metrics/getup_assist_max"] = st["assist"]


def getup_ladder_curriculum(env, env_ids, steps_per_it: int = 24, assist_max: float = ASSIST_MAX, start_level: int = 0) -> dict:
    """getup_rev_curriculum on the ladder, plus the assist controller (once per iteration, with the success EMA)."""
    st = _ladder_state(env, start_level)
    if env_ids is not None and len(env_ids) > 0:
        _, qp, _ = mdp._root(env)
        up = (qp[env_ids, 2] > UP_Z) & (mdp.torso_tilt_rad(env)[env_ids] < math.radians(UP_TILT_DEG))
        front = st["start"][env_ids] == st["level"]
        st["succ"] += float((up & front).sum())
        st["count"] += float(front.sum())
    it = int(env.common_step_counter) // steps_per_it
    if it != st["last_it"]:
        st["last_it"] = it
        if st["count"] >= 64:
            st["ema"] = 0.8 * st["ema"] + 0.2 * st["succ"] / st["count"]
            st["succ"] = st["count"] = 0.0
        st["since"] += 1
        if st["since"] > ASSIST_GRACE:
            st["assist"] = min(assist_max, max(0.0, st["assist"] + ASSIST_GAIN * (ADVANCE_AT - st["ema"])))
        st["zero_its"] = st["zero_its"] + 1 if st["assist"] <= 0.0 else 0
        if (st["ema"] > ADVANCE_AT and st["since"] >= LADDER_MIN_ITS and st["zero_its"] >= ASSIST_ZERO_ITS
                and st["level"] + 1 < len(st["table"])):
            st["level"] += 1
            st["ema"], st["since"] = 0.0, 0
            st["probs"] = _ladder_mix(st["level"], len(st["table"]), env.device)
    return {"getup_stage": torch.tensor(float(st["level"])), "getup_frontier_success": torch.tensor(st["ema"])}


def matt_getup_ladder_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = matt_getup5_env_cfg(play=play)
    for n in CUBE_NAMES:                                    # no cube drops in get-up: no pool at all
        cfg.scene.entities.pop(n, None)
    if not play:
        cfg.episode_length_s = LADDER_EPISODE_S
    ev = cfg.events
    ev.pop("reset_joints", None)
    ev.pop("getup_assist", None)
    ev["reset_base"] = EventTermCfg(func=reset_ladder, mode="reset")
    if not play:                                            # after reset_base: the assist reads the stage it drew
        ev["ladder_assist"] = EventTermCfg(func=ladder_assist, mode="reset", params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)), "body_weight_n": C.BODY.total_mass * 9.81})
    cfg.curriculum = {"getup_rev": CurriculumTermCfg(func=getup_ladder_curriculum, params={})}
    add_bio_rewards(cfg)
    cfg.scene.num_envs = V5_ENVS
    return cfg


def matt_getup_ladder_b_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """getup_rev_v2b: v2 passed the squat and the first rock-back stage, then sat at 0 % on stage 2 (seated, leaning back)
    for 130 its with the assist at its 0.6 cap. Rolled out on CPU it straightens its legs and sits upright, propped on its
    arms, at any assist: an upright trunk pays, nothing pays for feet under the body. Added: feet_under_pelvis (w 3);
    assist cap 0.6 -> 0.8."""
    from mjlab.managers.reward_manager import RewardTermCfg
    cfg = matt_getup_ladder_env_cfg(play=play)
    cfg.rewards["feet_under"] = RewardTermCfg(func=mdp.feet_under_pelvis, weight=3.0, params={"std": 0.3})
    cfg.curriculum["getup_rev"].params["assist_max"] = 0.8
    return cfg


def matt_getup_ladder_c_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """getup_rev_v2c: v2b still straightened its legs on stage 2 (0 % for 60 its, assist at 0.8: it hangs from the assist
    with straight legs). Zero action = the standing pose, so the knees open within 0.3 s, and with std 0.3 feet_under is
    ~0 in the long sit (feet 0.85 m ahead: e^-8): no gradient left to pull them back. std 0.3 -> 0.5 (long sit 0.06,
    tuck 0.6, squat 0.96), weight 3 -> 4."""
    cfg = matt_getup_ladder_b_env_cfg(play=play)
    cfg.rewards["feet_under"].weight = 4.0
    cfg.rewards["feet_under"].params["std"] = 0.5
    return cfg


_TUCK_JOINTS = ("hip_flex", "knee", "ankle_dorsi")


def tuck_when_low(env, std: float, target: float) -> torch.Tensor:
    """(1 - pelvis height / target) x exp(-mean squared leg-joint error to the squat pose / std^2): while the pelvis is
    low, legs folded as in the squat (feet ready under the body); fades to 0 as the athlete rises, so standing up costs
    nothing. Joint-space on purpose (rs_v5 lesson): the task-space terms gave the knees no usable gradient."""
    st = _ladder_state(env)
    ix = mdp._idx(env)
    legs = st.get("tuck_ix")
    if legs is None:
        legs = st["tuck_ix"] = torch.as_tensor(
            [i for i, n in enumerate(mdp.CONTRACT_ACTUATORS) if n[:-2] in _TUCK_JOINTS], device=env.device)
    q = ix.entity.data.joint_pos[:, ix.joint_ids][:, legs]
    err = ((q - st["table"][0][0][legs]) ** 2).mean(-1)
    return (1.0 - mdp.height_progress(env, target)) * torch.exp(-err / std**2)


def upright_when_high(env, std: float, target: float) -> torch.Tensor:
    """Rung 0's uprightness kernel, paid in proportion to the pelvis height: an upright trunk on the floor earns nothing."""
    return mdp.torso_upright(env, std) * mdp.height_progress(env, target)


def matt_getup_ladder_d_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """getup_rev_v2d: v2c sat at 0 % on stage 2 for 250 its. CPU rollout: in the first 0.2 s it throws its knees open
    (knee action -1.6, target 0 deg) and uses the kick to sit bolt upright (tilt 1-5 deg, pelvis 0.16 m) with straight
    legs, at any assist; the upright kernel pays that in full and feet_under moved nothing. Changes: the upright kernel
    pays in proportion to height; tuck_when_low (w 3) pays for keeping the legs folded while the pelvis is low; the run
    starts with stage 2 as its frontier (the warm start passes stages 0 and 1)."""
    from mjlab.managers.reward_manager import RewardTermCfg
    from .matt_env import DEFAULT_ROOT_Z
    cfg = matt_getup_ladder_c_env_cfg(play=play)
    cfg.rewards["upright"] = RewardTermCfg(func=upright_when_high, weight=1.0,
                                           params={"std": math.radians(20), "target": DEFAULT_ROOT_Z})
    cfg.rewards["tuck_low"] = RewardTermCfg(func=tuck_when_low, weight=3.0, params={"std": 0.6, "target": DEFAULT_ROOT_Z})
    cfg.events["reset_base"].params["start_level"] = 2
    return cfg


def legs_folded_when_low(env, target: float, start_level: int = 0) -> torch.Tensor:
    """(1 - pelvis height / target) x mean over hips and knees of clamp(1 - |q - q_squat| / |q_squat - q_default|, -0.5, 1):
    1 with the legs folded as in the squat, 0 at the standing pose, negative past it. Linear, so there is a gradient at
    every knee angle: v2d's kernel version (tuck_when_low) was flat once the knees had opened, and every sampled action
    opens them (zero action = the standing pose; holding 125 deg of knee flexion takes an action of +7.3)."""
    st = _ladder_state(env, start_level)
    ix = mdp._idx(env)
    legs = st.get("fold_ix")
    if legs is None:
        legs = st["fold_ix"] = torch.as_tensor(
            [i for i, n in enumerate(mdp.CONTRACT_ACTUATORS) if n[:-2] in ("hip_flex", "knee")], device=env.device)
        st["fold_span"] = (st["table"][0][0][legs] - ix.default[legs]).abs()
    q = ix.entity.data.joint_pos[:, ix.joint_ids][:, legs]
    frac = torch.clamp(1.0 - (q - st["table"][0][0][legs]).abs() / st["fold_span"], -0.5, 1.0).mean(-1)
    return (1.0 - mdp.height_progress(env, target)) * frac


def matt_getup_ladder_e_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """getup_rev_v2e: v2d changed nothing on stage 2 in 115 its (tuck_low 0.036 = the first frames only). The kernel
    reward is flat once the knees are open, and every sampled action opens them. tuck_when_low -> legs_folded_when_low
    (linear, w 3); the frontier really starts at stage 2 (v2d's start_level was read after the state existed); the init
    gets an action std floor of 0.5."""
    from mjlab.managers.reward_manager import RewardTermCfg
    from .matt_env import DEFAULT_ROOT_Z
    cfg = matt_getup_ladder_d_env_cfg(play=play)
    cfg.rewards.pop("tuck_low")
    cfg.rewards["legs_folded"] = RewardTermCfg(func=legs_folded_when_low, weight=3.0,
                                               params={"target": DEFAULT_ROOT_Z, "start_level": 2})
    cfg.curriculum["getup_rev"].params["start_level"] = 2
    return cfg
