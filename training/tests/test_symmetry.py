"""The PPO mirror map equals physics: obs(mirrored MuJoCo state) == mirror(obs(state))."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from poolympic import contract as C
from poolympic.tasks import symmetry as S


def _mirror_state(m, ath, qpos, qvel):
    """Reflect the athlete across the world XZ plane (y -> -y) and swap left/right joints."""
    q, v = qpos.copy(), qvel.copy()
    r, dv = ath.root_qposadr, ath.root_dofadr
    q[r + 1] = -qpos[r + 1]
    w, x, y, z = qpos[r + 3: r + 7]
    q[r + 3: r + 7] = [w, -x, y, -z]                          # reflected orientation (proper rotation)
    v[dv: dv + 3] = qvel[dv: dv + 3] * [1, -1, 1]             # world linear velocity
    v[dv + 3: dv + 6] = qvel[dv + 3: dv + 6] * [-1, 1, -1]    # body angular velocity (pseudovector)
    for i, j in enumerate(S.PERM):
        q[ath.joint_qposadr[i]] = S.SIGN[i] * (qpos[ath.joint_qposadr[j]] - ath.default_pos[j]) + ath.default_pos[i]
        v[ath.joint_dofadr[i]] = S.SIGN[i] * qvel[ath.joint_dofadr[j]]
    return q, v


def test_mirror_is_physical():
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    ath = C.Athlete.bind(m)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
        qpos = d.qpos.copy()
        qpos[ath.joint_qposadr] += rng.normal(0, 0.3, len(ath.joint_qposadr))
        yaw = rng.uniform(-np.pi, np.pi)
        tilt = rng.normal(0, 0.2, 3)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, [tilt[0], tilt[1], yaw], "xyz")
        qpos[ath.root_qposadr + 3: ath.root_qposadr + 7] = quat
        qpos[ath.root_qposadr: ath.root_qposadr + 2] = rng.normal(0, 3, 2)
        qvel = rng.normal(0, 1.0, m.nv)
        standing = rng.uniform() < 0.3
        cmd = np.zeros(3) if standing else rng.uniform(-2, 2, 3)
        if not standing and np.linalg.norm(cmd) < C.PHASE_CMD_THRESHOLD:
            cmd[0] = 1.0
        phase = 0.0 if standing else rng.uniform()     # the clock is frozen at 0 while standing
        last = rng.normal(0, 1, C.NUM_ACTIONS)
        o = C.build_obs(ath, qpos, qvel, cmd, phase, last)
        qm, vm = _mirror_state(m, ath, qpos, qvel)
        om = C.build_obs(ath, qm, vm, cmd * [1, -1, -1], phase if standing else (phase + 0.5) % 1.0,
                         S.mirror_actions(torch.as_tensor(last)).numpy())
        pred = S.mirror_obs(torch.as_tensor(o)).numpy()
        worst = max(worst, float(np.abs(pred - om).max()))
    assert worst < 1e-5, worst


def test_mirror_is_an_involution():
    x = torch.randn(7, C.OBS_DIM)
    a = torch.randn(7, C.NUM_ACTIONS)
    assert torch.allclose(S.mirror_obs(S.mirror_obs(x)), x)
    assert torch.allclose(S.mirror_actions(S.mirror_actions(a)), a)


def _partner_body(name: str) -> str:
    return name[:-2] + "_r" if name.endswith("_l") else name[:-2] + "_l" if name.endswith("_r") else name


def test_mirrored_joint_map_mirrors_the_body():
    """Forward kinematics: every body of the mirrored state sits at the reflection (y -> -y) of its partner body in the
    original state — i.e. the joint permutation + signs really are the physical mirror (validates the axis conventions)."""
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d, dm = mujoco.MjData(m), mujoco.MjData(m)
    ath = C.Athlete.bind(m)
    rng = np.random.default_rng(1)
    bodies = [m.body(b).name for b in range(1, m.nbody) if not m.body(b).name.startswith("cube")]
    worst = 0.0
    for _ in range(30):
        mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
        d.qpos[ath.joint_qposadr] += rng.normal(0, 0.4, len(ath.joint_qposadr))
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, rng.normal(0, 0.3, 3), "xyz")
        d.qpos[ath.root_qposadr + 3: ath.root_qposadr + 7] = quat
        dm.qpos[:], _ = _mirror_state(m, ath, d.qpos, d.qvel)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_kinematics(m, dm)
        for b in bodies:
            p = d.xpos[m.body(b).id] * [1, -1, 1]
            pm = dm.xpos[m.body(_partner_body(b)).id]
            worst = max(worst, float(np.abs(p - pm).max()))
    assert worst < 2e-3, worst   # MATT's skeleton is L/R-symmetric to < 1 mm


def _mirror_skill(skill: np.ndarray) -> np.ndarray:
    s = C.SkillCommand.from_array(skill)
    return C.SkillCommand(s.pelvis_height, {"l": "r", "r": "l"}.get(s.lift_foot, ""), s.march_hz, s.knee_lift,
                          -s.torso_yaw, s.torso_pitch, (s.hand[0], -s.hand[1], s.hand[2]), -s.arm).to_array()


def test_mirror_v4_skill_block():
    """95-dim obs: mirror(obs(state, skill)) == obs(mirrored state, mirrored skill); a march cadence runs the clock."""
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    ath = C.Athlete.bind(m)
    rng = np.random.default_rng(3)
    worst = 0.0
    for k in range(30):
        mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
        qpos = d.qpos.copy()
        qpos[ath.joint_qposadr] += rng.normal(0, 0.3, len(ath.joint_qposadr))
        qvel = rng.normal(0, 1.0, m.nv)
        march = k % 2 == 0
        skill = C.SkillCommand(pelvis_height=-rng.uniform(0, 0.4), lift_foot=["", "l", "r"][k % 3],
                               march_hz=1.2 if march else 0.0, knee_lift=0.2 if march else 0.0,
                               torso_yaw=rng.uniform(-1, 1), torso_pitch=rng.uniform(-0.3, 0.6),
                               hand=tuple(rng.normal(0, 0.4, 3)), arm=[-1, 0, 1][k % 3]).to_array()
        phase = rng.uniform() if march else 0.0          # standing, no march: clock frozen at 0
        last = rng.normal(0, 1, C.NUM_ACTIONS)
        o = C.build_obs(ath, qpos, qvel, np.zeros(3), phase, last, skill)
        qm, vm = _mirror_state(m, ath, qpos, qvel)
        om = C.build_obs(ath, qm, vm, np.zeros(3), (phase + 0.5) % 1.0 if march else phase,
                         S.mirror_actions(torch.as_tensor(last)).numpy(), _mirror_skill(skill))
        worst = max(worst, float(np.abs(S.mirror_obs(torch.as_tensor(o)).numpy() - om).max()))
    assert worst < 1e-5, worst
    x = torch.randn(5, C.OBS_DIM_V4)
    assert torch.allclose(S.mirror_obs(S.mirror_obs(x)), x)


def test_skill_measures_mirror_physically():
    """The quantities the skill rewards measure mirror the way the skill block does: torso aim (yaw, pitch) ->
    (-yaw, pitch), hand of arm a (x, y, z) -> hand of arm -a (x, -y, z), knee rise / foot heights swap sides."""
    from poolympic import skills as K
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d, dm = mujoco.MjData(m), mujoco.MjData(m)
    ath = C.Athlete.bind(m)
    sb = K.SkillBodies.bind(m)
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(20):
        mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
        d.qpos[ath.joint_qposadr] += rng.normal(0, 0.3, len(ath.joint_qposadr))
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, [0.0, 0.0, rng.uniform(-np.pi, np.pi)], "xyz")
        d.qpos[ath.root_qposadr + 3: ath.root_qposadr + 7] = quat
        dm.qpos[:], _ = _mirror_state(m, ath, d.qpos, d.qvel)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_kinematics(m, dm)
        y, p = K.torso_aim(d, sb)
        ym, pm = K.torso_aim(dm, sb)
        worst = max(worst, abs(ym + y), abs(pm - p))
        for arm in (-1, 1):
            worst = max(worst, float(np.abs(K.hand_in_heading(dm, sb, -arm) - K.hand_in_heading(d, sb, arm) * [1, -1, 1]).max()))
        kl, kr = K.knee_rise(d, sb)
        kml, kmr = K.knee_rise(dm, sb)
        worst = max(worst, abs(kml - kr), abs(kmr - kl))
    assert worst < 3e-3, worst   # skeleton L/R symmetric to < 1 mm; the forearm tip lever adds a little
