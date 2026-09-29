"""A9 — Observation/action contract unit tests (frames, layout, phase clock)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from poolympic import contract as C


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(C.SCENE_XML))


@pytest.fixture(scope="module")
def ath(model):
    return C.Athlete.bind(model)


def _obs(ath, model, yaw=0.0, lin_world=(0, 0, 0), ang_local=(0, 0, 0)):
    qpos = model.key("default").qpos.copy()
    qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    qvel = np.zeros(model.nv)
    qvel[0:3] = lin_world
    qvel[3:6] = ang_local
    return C.build_obs(ath, qpos, qvel, np.zeros(3), 0.0, np.zeros(C.NUM_ACTIONS)), qpos


def sl(name):
    o = 0
    for n, s in C.OBS_LAYOUT:
        if n == name:
            return slice(o, o + s)
        o += s
    raise KeyError(name)


def test_layout(ath):
    assert C.OBS_DIM == 84 and len(ath.actuator_names) == 23


def test_standing_default(ath, model):
    obs, qpos = _obs(ath, model)
    assert np.allclose(obs[sl("projected_gravity")], [0, 0, -1], atol=1e-6)
    assert obs[sl("base_height")][0] == pytest.approx(qpos[2], abs=1e-6)
    assert np.allclose(obs[sl("joint_pos_rel")], 0, atol=1e-6)
    assert np.allclose(obs[sl("gait_phase_sincos")], [0, 1])


def test_heading_frame_linear_velocity(ath, model):
    # facing +y (yaw 90°) and moving +y in world => forward velocity in heading frame
    obs, _ = _obs(ath, model, yaw=math.pi / 2, lin_world=(0, 1.5, 0))
    assert np.allclose(obs[sl("base_lin_vel_heading")], [1.5, 0, 0], atol=1e-6)
    # gravity unaffected by yaw
    assert np.allclose(obs[sl("projected_gravity")], [0, 0, -1], atol=1e-6)


def test_projected_gravity_pitch(ath, model):
    qpos = model.key("default").qpos.copy()
    pitch = math.radians(30)  # nose-down pitch about +y
    qpos[3:7] = [math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]
    obs = C.build_obs(ath, qpos, np.zeros(model.nv), np.zeros(3), 0.0, np.zeros(23))
    assert np.allclose(obs[sl("projected_gravity")], [math.sin(pitch), 0, -math.cos(pitch)], atol=1e-6)


def test_phase_clock():
    assert C.advance_phase(0.3, np.zeros(3)) == 0.0
    p = C.advance_phase(0.0, np.array([1.0, 0, 0]))
    assert p == pytest.approx((C.GAIT_HZ_BASE + C.GAIT_HZ_PER_MPS * 1.0) * 0.02)
    assert C.gait_hz(np.array([3.0, 0, 0])) == pytest.approx(1.4)
    assert 0.0 <= C.advance_phase(0.99, np.array([1.0, 0, 0])) < 1.0


def test_action_mapping_clips_to_range(ath):
    big = np.full(23, 100.0)
    assert np.allclose(C.action_to_ctrl(ath, big), ath.range_hi)
    assert np.allclose(C.action_to_ctrl(ath, -big), ath.range_lo)
    assert np.allclose(C.action_to_ctrl(ath, np.zeros(23)), ath.default_pos)


# ---------------------------------------------------------------- contract v4 (stance-skill block)

def test_skill_block_layout():
    assert C.SKILL_DIM == 11 and C.OBS_DIM_V4 == 95
    s = C.SkillCommand(pelvis_height=-0.3, lift_foot="r", march_hz=1.5, knee_lift=0.2, torso_yaw=0.4, torso_pitch=-0.1,
                       hand=(0.5, -0.2, 0.3), arm=1)
    a = s.to_array()
    assert a.tolist() == [-0.3, 0, 1, 1.5, 0.2, 0.4, -0.1, 0.5, -0.2, 0.3, 1]
    assert C.SkillCommand.from_array(a) == s
    assert not C.SkillCommand().to_array().any()


def test_obs_v4_appends_skill_block(ath, model):
    qpos = model.key("default").qpos.copy()
    qvel = np.zeros(model.nv)
    v3 = C.build_obs(ath, qpos, qvel, np.zeros(3), 0.25, np.zeros(23))
    zero = C.build_obs(ath, qpos, qvel, np.zeros(3), 0.25, np.zeros(23), C.SkillCommand().to_array())
    assert zero.shape == (95,) and np.array_equal(zero[:84], v3) and not zero[84:].any()
    skill = C.SkillCommand(pelvis_height=-0.2, lift_foot="l", hand=(0.4, 0.3, 0.2), arm=-1).to_array()
    obs = C.build_obs(ath, qpos, qvel, np.zeros(3), 0.25, np.zeros(23), skill)
    assert np.array_equal(obs[:84], v3) and np.allclose(obs[84:], skill.astype(np.float32))


def test_march_cadence_drives_clock():
    assert C.advance_phase(0.0, np.zeros(3), cadence=1.5) == pytest.approx(1.5 * C.DECIMATION * 0.005)
    assert C.advance_phase(0.3, np.zeros(3), cadence=0.0) == 0.0          # v3: frozen at zero command
    assert C.advance_phase(0.1, np.array([1.0, 0, 0]), cadence=0.0) == C.advance_phase(0.1, np.array([1.0, 0, 0]))
    g = (C.GAIT_HZ_BASE, C.GAIT_HZ_PER_MPS, C.GAIT_HZ_YAW_WEIGHT)
    assert C.advance_phase_clock(0.2, np.zeros(3), g, 1.2) == C.advance_phase(0.2, np.zeros(3), 1.2)


def test_contract_json_skill_block():
    import json
    c = json.loads(C.CONTRACT_JSON.read_text())
    sb = c["skill_block"]
    assert c["obs_dim"] == 84 and sb["offset"] == 84 and sb["obs_dim"] == 95 and sb["version"] == C.SKILL_VERSION
    assert [k for k in sb["layout"]] == [n for n, _ in C.SKILL_LAYOUT]
    assert sb["layout"]["hand_target"] == {"offset": 91, "size": 4}
