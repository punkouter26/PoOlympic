"""Rung S (contract v4 stance skills): measurement module, reach-target sampling and the G1 drill harness."""

import math

import mujoco
import numpy as np
import pytest

from poolympic import contract as C
from poolympic import skills as K

V4_BRAIN = C.ROOT.parent / "parity" / "brains" / "random_brain_v4.onnx"


def _default():
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
    mujoco.mj_forward(m, d)
    return m, d, K.SkillBodies.bind(m)


def test_measures_at_the_default_stance():
    m, d, sb = _default()
    assert K.torso_aim(d, sb) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert K.pelvis_height_rel(d, sb) == pytest.approx(0.0) and K.knee_rise(d, sb) == pytest.approx((0.0, 0.0))
    left, right = K.hand_in_heading(d, sb, -1), K.hand_in_heading(d, sb, 1)
    assert left[1] > 0.2 and right[1] < -0.2 and left == pytest.approx(right * [1, -1, 1], abs=2e-3)   # arms by the sides


def test_reach_targets_are_reachable_and_on_the_arm_side():
    m, d, sb = _default()
    reach = K.body_ranges()["hand_reach"]
    rng = np.random.default_rng(0)
    for arm in (-1, 1):
        sh = K.shoulder_in_heading(d, m, arm)
        for _ in range(200):
            t = K.sample_reach_target(rng, sh, arm, reach)
            r = np.linalg.norm(t - sh)
            assert K.REACH_FRAC[0] * reach - 1e-9 <= r <= K.REACH_FRAC[1] * reach + 1e-9
            u = (t - sh) / r
            assert u[0] >= math.cos(math.radians(90)) * math.cos(math.radians(60)) - 1e-9 or u[1] * -arm >= 0   # not behind
            assert -arm * u[1] >= -math.sin(math.radians(30)) - 1e-9                                       # not across


@pytest.mark.skipif(not V4_BRAIN.exists(), reason="random_brain_v4.onnx not exported")
def test_drill_harness_runs_a_v4_brain():
    from poolympic import evaluate_stance as ES
    sim = ES.SkillSim(V4_BRAIN)
    res = ES.drill_squat(sim, np.random.default_rng(0))
    assert set(res) >= {"pass", "fall", "worst_err_m"}
    assert res["pass"] is False             # the random test brain cannot squat
    assert sim.last_action.shape == (C.NUM_ACTIONS,)
