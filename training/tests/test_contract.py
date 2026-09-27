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
    assert p == pytest.approx(C.GAIT_HZ * 0.02)
    assert 0.0 <= C.advance_phase(0.99, np.array([1.0, 0, 0])) < 1.0


def test_action_mapping_clips_to_range(ath):
    big = np.full(23, 100.0)
    assert np.allclose(C.action_to_ctrl(ath, big), ath.range_hi)
    assert np.allclose(C.action_to_ctrl(ath, -big), ath.range_lo)
    assert np.allclose(C.action_to_ctrl(ath, np.zeros(23)), ath.default_pos)
