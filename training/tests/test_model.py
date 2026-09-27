"""A7 — MJCF validation for MATT (DESIGN.md §2). Run: uv run pytest tests/test_model.py -q"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SKEL = json.loads((ASSETS / "derived" / "skeleton_matt.json").read_text())
BONE_POS = {j["name"]: np.array(j["pos"]) for j in SKEL["joints"]}

BODY_BONE = {
    "pelvis": "Hips", "torso": "Spine", "chest": "Spine2", "head": "Neck",
    **{f"{b}_{s}": f"{S}{bone}" for s, S in (("l", "Left"), ("r", "Right"))
       for b, bone in (("upper_arm", "Arm"), ("forearm", "ForeArm"), ("thigh", "UpLeg"), ("shin", "Leg"),
                       ("foot", "Foot"), ("toe", "ToeBase"))},
}
MASS_TABLE = {"pelvis": 8.94, "torso": 13.06, "chest": 12.77, "head": 5.55, "upper_arm": 2.17, "forearm": 1.79,
              "thigh": 11.33, "shin": 3.46, "foot": 0.85, "toe": 0.25}
GAINS = {"hip": (300, 30, 280), "knee": (300, 30, 280), "ankle": (200, 20, 220), "abdomen": (300, 30, 200),
         "shoulder": (60, 6, 80), "elbow": (50, 5, 70)}


@pytest.fixture(scope="module")
def robot():
    return mujoco.MjModel.from_xml_path(str(ASSETS / "matt.xml"))


@pytest.fixture(scope="module")
def scene():
    return mujoco.MjModel.from_xml_path(str(ASSETS / "scene_matt.xml"))


def robot_mass(m, body):
    return float(m.body_mass[m.body(body).id])


def test_dimensions(robot):
    assert (robot.nq, robot.nv, robot.nu) == (32, 31, 23)
    assert robot.opt.timestep == 0.005
    assert robot.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert robot.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert robot.opt.iterations == 20
    assert robot.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL


def test_compiles_in_mujoco_warp(scene):
    import mujoco_warp as mjw

    mjw.put_model(scene)


def test_total_and_segment_mass(robot):
    total = sum(robot_mass(robot, b) for b in BODY_BONE)
    assert abs(total - 80.0) < 0.05
    scale = 80.0 / sum(v * (2 if k not in ("pelvis", "torso", "chest", "head") else 1) for k, v in MASS_TABLE.items())
    for body in BODY_BONE:
        key = body[:-2] if body[-2:] in ("_l", "_r") else body
        assert robot_mass(robot, body) == pytest.approx(MASS_TABLE[key] * scale, abs=1e-4), body


def test_qpos0_is_bind_pose(robot):
    d = mujoco.MjData(robot)
    mujoco.mj_kinematics(robot, d)
    for body, bone in BODY_BONE.items():
        err = np.linalg.norm(d.xpos[robot.body(body).id] - BONE_POS[bone])
        assert err < 1e-3, f"{body} vs {bone}: {err * 1000:.3f} mm"
        assert np.allclose(d.xquat[robot.body(body).id], [1, 0, 0, 0], atol=1e-9), body  # world-aligned frames


def test_left_right_symmetry(robot):
    mirror_axis = np.array([-1, 1, -1])
    for i in range(robot.njnt):
        name = robot.joint(i).name
        if not name.endswith("_l"):
            continue
        r = robot.joint(name[:-2] + "_r").id
        assert np.allclose(robot.jnt_range[i], robot.jnt_range[r]), name
        assert np.allclose(robot.jnt_axis[i] * mirror_axis, robot.jnt_axis[r]), name
        assert robot.dof_armature[robot.jnt_dofadr[i]] == robot.dof_armature[robot.jnt_dofadr[r]]
    for b in BODY_BONE:
        if b.endswith("_l"):
            rb = b[:-2] + "_r"
            assert robot_mass(robot, b) == robot_mass(robot, rb)
            assert np.allclose(robot.body_inertia[robot.body(b).id], robot.body_inertia[robot.body(rb).id])


def test_actuators_match_design(robot):
    for i in range(robot.nu):
        name = robot.actuator(i).name
        group = next(g for g in GAINS if name.startswith(g))
        kp, kv, cap = GAINS[group]
        assert robot.actuator_trnid[i, 0] == robot.joint(name).id
        assert robot.actuator_gainprm[i, 0] == kp
        assert robot.actuator_biasprm[i, 1] == -kp and robot.actuator_biasprm[i, 2] == -kv
        assert robot.actuator_forcelimited[i] and np.allclose(robot.actuator_forcerange[i], [-cap, cap])
        assert not robot.actuator_ctrllimited[i]  # clipping happens inside the ONNX graph


def _site_after(robot, joint, deg, body, local=np.zeros(3), base=None):
    d = mujoco.MjData(robot)
    if base is not None:
        d.qpos[:] = base
    d.qpos[robot.jnt_qposadr[robot.joint(joint).id]] += math.radians(deg)
    mujoco.mj_kinematics(robot, d)
    b = robot.body(body).id
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ local


@pytest.mark.parametrize("side,sy", [("l", 1), ("r", -1)])
def test_joint_sign_semantics(robot, side, sy):
    """Positive angle = anatomical positive direction, mirrored correctly on both sides."""
    q0 = robot.qpos0.copy()
    down = np.array([0, 0, -0.4])
    fwd_toe = np.array([0.1, 0, 0])
    p = lambda *a, **k: _site_after(robot, *a, **k)  # noqa: E731
    # hip flexion swings the knee forward; knee flexion moves the ankle back
    assert p(f"hip_flex_{side}", 30, f"thigh_{side}", down)[0] > p(f"hip_flex_{side}", 0, f"thigh_{side}", down)[0] + 0.1
    assert p(f"knee_{side}", 30, f"shin_{side}", down)[0] < p(f"knee_{side}", 0, f"shin_{side}", down)[0] - 0.1
    # hip abduction moves the foot away from the midline
    assert sy * p(f"hip_abd_{side}", 20, f"thigh_{side}", down)[1] > sy * p(f"hip_abd_{side}", 0, f"thigh_{side}", down)[1] + 0.05
    # ankle dorsiflexion lifts the toes
    assert p(f"ankle_dorsi_{side}", 20, f"foot_{side}", fwd_toe)[2] > p(f"ankle_dorsi_{side}", 0, f"foot_{side}", fwd_toe)[2] + 0.02
    # ankle inversion turns the sole to face the midline (lateral edge goes down)
    lateral_edge = np.array([0, sy * 0.05, 0])
    assert p(f"ankle_inv_{side}", 20, f"foot_{side}", lateral_edge)[2] < p(f"ankle_inv_{side}", 0, f"foot_{side}", lateral_edge)[2] - 0.01
    # shoulder elevation raises the hand from the T-pose; negative brings it to the side
    hand = np.array([0, sy * 0.3, 0])
    assert p(f"shoulder_elev_{side}", 30, f"forearm_{side}", hand)[2] > q0[2] + 0.1 + p(f"shoulder_elev_{side}", 0, f"forearm_{side}", hand)[2] - q0[2]
    # with arms hanging (elev -80), shoulder flexion swings the hand forward; elbow flexion too
    hang = q0.copy()
    hang[robot.jnt_qposadr[robot.joint(f"shoulder_elev_{side}").id]] = math.radians(-80)
    assert p(f"shoulder_flex_{side}", 40, f"forearm_{side}", hand, base=hang)[0] > p(f"shoulder_flex_{side}", 0, f"forearm_{side}", hand, base=hang)[0] + 0.1
    assert p(f"elbow_{side}", 60, f"forearm_{side}", hand, base=hang)[0] > p(f"elbow_{side}", 0, f"forearm_{side}", hand, base=hang)[0] + 0.1


def _hold_default(scene, seconds):
    d = mujoco.MjData(scene)
    mujoco.mj_resetDataKeyframe(scene, d, scene.key("default").id)
    for i in range(scene.nu):
        d.ctrl[i] = d.qpos[scene.jnt_qposadr[scene.actuator_trnid[i, 0]]]
    ground = scene.geom("ground").id
    max_pen, heights = 0.0, []
    for _ in range(int(seconds / scene.opt.timestep)):
        mujoco.mj_step(scene, d)
        assert np.all(np.isfinite(d.qpos)), "simulation exploded"
        for c in d.contact[: d.ncon]:
            if ground in (c.geom1, c.geom2):
                max_pen = max(max_pen, -c.dist)
        heights.append(d.qpos[2])
    return d, max_pen, np.array(heights)


def test_passive_pd_hold_settles(scene):
    d, max_pen, h = _hold_default(scene, 1.0)
    assert max_pen < 0.005, f"penetration {max_pen * 1000:.2f} mm"
    last = h[int(0.8 / scene.opt.timestep):]
    assert np.ptp(last) < 0.005, "root height still moving after 0.8 s"
    assert h[-1] > 0.85


def test_contacts_only_feet_on_ground_at_rest(scene):
    d, _, _ = _hold_default(scene, 0.5)
    ground = scene.geom("ground").id
    for c in d.contact[: d.ncon]:
        other = c.geom2 if c.geom1 == ground else c.geom1 if c.geom2 == ground else None
        if other is None:
            continue
        name = scene.geom(other).name
        assert name.startswith(("foot", "toe", "cube")), f"unexpected ground contact: {name}"


def test_actuators_declared_in_joint_order(robot):
    """ctrl order must follow joint id order (mjlab XmlActuator pairing; see build_mjcf.compose_model)."""
    joint_ids = [int(robot.actuator_trnid[i, 0]) for i in range(robot.nu)]
    assert joint_ids == sorted(joint_ids)


# ---------------------------------------------------------------- C6: 8-lane meet scene
MEET = ROOT / "assets" / "scene_meet8.xml"


@pytest.fixture(scope="module")
def meet():
    return mujoco.MjModel.from_xml_path(str(MEET))


def _lane_of(m, g) -> int | None:
    name = m.body(m.body_rootid[m.geom_bodyid[g]]).name
    return int(name[1:name.index("_")]) if name.startswith("L") else None


def test_meet_lanes_never_collide(meet):
    """Lane isolation by construction: no geom of lane a can ever touch a geom of lane b != a; every athlete geom
    still touches the ground and the cubes."""
    lanes = [_lane_of(meet, g) for g in range(meet.ngeom)]
    ct, ca = meet.geom_contype, meet.geom_conaffinity
    shared = [g for g in range(meet.ngeom) if lanes[g] is None]
    for a in range(meet.ngeom):
        for b in range(a + 1, meet.ngeom):
            can = bool((ct[a] & ca[b]) | (ct[b] & ca[a]))
            if lanes[a] is not None and lanes[b] is not None and lanes[a] != lanes[b]:
                assert not can, (meet.geom(a).name, meet.geom(b).name)
    for a in range(meet.ngeom):
        if lanes[a] is not None:
            assert all((ct[a] & ca[s]) | (ct[s] & ca[a]) for s in shared), meet.geom(a).name


def test_meet_lane_is_the_training_athlete(meet, scene):
    """Each lane's athlete matches the solo scene body-for-body (masses, inertias, joints, actuators); only names,
    collision bits and the pelvis origin differ."""
    from poolympic.meet import load_layout

    for lane in load_layout():
        for i in range(1, scene.nbody):
            name = scene.body(i).name
            if name.startswith("cube"):
                continue
            j = meet.body(lane.prefix + name).id
            assert meet.body_mass[j] == scene.body_mass[i]
            np.testing.assert_array_equal(meet.body_inertia[j], scene.body_inertia[i])
            if name != "pelvis":
                np.testing.assert_array_equal(meet.body_pos[j], scene.body_pos[i])
        np.testing.assert_allclose(meet.body_pos[meet.body(lane.prefix + "pelvis").id],
                                   scene.body_pos[scene.body("pelvis").id] + lane.origin, atol=1e-6)
        for a in range(scene.nu):
            b = meet.actuator(lane.prefix + scene.actuator(a).name).id
            np.testing.assert_array_equal(meet.actuator_gainprm[b], scene.actuator_gainprm[a])
            np.testing.assert_array_equal(meet.actuator_forcerange[b], scene.actuator_forcerange[a])
        k0 = meet.key("default").qpos
        ks = scene.key("default").qpos
        ra = meet.jnt_qposadr[meet.joint(lane.prefix + "root").id]
        np.testing.assert_allclose(k0[ra:ra + 32] - np.r_[lane.origin, np.zeros(29)], ks[:32], atol=1e-6)
