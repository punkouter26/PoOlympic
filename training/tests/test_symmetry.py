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
        cmd = rng.uniform(-2, 2, 3)
        phase = rng.uniform()
        last = rng.normal(0, 1, C.NUM_ACTIONS)
        o = C.build_obs(ath, qpos, qvel, cmd, phase, last)
        qm, vm = _mirror_state(m, ath, qpos, qvel)
        om = C.build_obs(ath, qm, vm, cmd * [1, -1, -1], (phase + 0.5) % 1.0,
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
