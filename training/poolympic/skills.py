"""Contract v4 stance skills — what each skill command MEANS, measured from MuJoCo state (numpy, CPU).

Shared by the G1 drills (evaluate.py) and mirrored in torch by tasks/mdp.py (skill rewards), so training rewards and
the pass bars measure the same quantities. Frames: MuJoCo world, x forward at qpos = 0, z up (DESIGN.md §2).

  pelvis_height  pelvis z − the default standing pelvis z                                   (event 3 Deep Squat)
  lifted foot    min z of the foot + toe geoms of the lifted leg; ground contact of that leg (event 6 Flamingo)
  knee rise      shin body origin (= the knee) z − its standing z                            (event 7 Cadence March)
  torso aim      chest x-axis yaw relative to the pelvis heading; pitch = forward lean of the chest (event 2)
  hand           forearm tip (far end of the forearm capsule; wrists are welded) in the heading frame of the pelvis
                 (x forward, y left, z up, origin = pelvis)                                  (event 4 Javelin Reach)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import contract as C

SKILL_MODES = ("locomotion", "squat", "flamingo", "march", "torso", "reach")


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class SkillBodies:
    """Ids + local tip offsets for one athlete (prefix) in a compiled model."""
    pelvis: int
    chest: int
    shin: tuple[int, int]            # (left, right)
    forearm: tuple[int, int]
    tip_local: tuple[np.ndarray, np.ndarray]
    foot_geoms: tuple[tuple[int, ...], tuple[int, ...]]
    ground: int
    knee_z0: tuple[float, float]     # standing knee heights (default keyframe)
    pelvis_z0: float

    @staticmethod
    def bind(m: mujoco.MjModel, prefix: str = "", keyframe: str | None = "default", ground: str = "ground") -> "SkillBodies":
        """keyframe None: no standing reference (heights relative to 0) — e.g. the mjlab training model."""
        b = lambda n: m.body(prefix + n).id  # noqa: E731
        fore = (b("forearm_l"), b("forearm_r"))
        tips = tuple(forearm_tip_local(m, fb) for fb in fore)
        feet = tuple(tuple(m.geom(prefix + f"{p}_{s}_geom0").id for p in ("foot", "toe")) for s in ("l", "r"))
        shin = (b("shin_l"), b("shin_r"))
        knee0, pel0 = (0.0, 0.0), 0.0
        if keyframe is not None:
            d = mujoco.MjData(m)
            mujoco.mj_resetDataKeyframe(m, d, m.key(keyframe).id)
            mujoco.mj_forward(m, d)
            knee0, pel0 = (float(d.xpos[shin[0]][2]), float(d.xpos[shin[1]][2])), float(d.xpos[b("pelvis")][2])
        return SkillBodies(b("pelvis"), b("chest"), shin, fore, tips, feet, m.geom(ground).id, knee0, pel0)


def forearm_tip_local(m: mujoco.MjModel, body: int) -> np.ndarray:
    """Far end of the body's capsule (+ radius) from the body origin (the elbow), in the body frame."""
    g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == body)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    axis = R.reshape(3, 3)[:, 2]
    r, half = m.geom_size[g][0], m.geom_size[g][1]
    ends = [m.geom_pos[g] + s * axis * half for s in (1, -1)]
    far = max(ends, key=np.linalg.norm)
    return far + r * far / np.linalg.norm(far)


def pelvis_yaw(d: mujoco.MjData, sb: SkillBodies) -> float:
    R = d.xmat[sb.pelvis].reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def torso_aim(d: mujoco.MjData, sb: SkillBodies) -> tuple[float, float]:
    """(yaw, pitch) of the chest: yaw of its x-axis relative to the pelvis heading, pitch = forward lean (> 0)."""
    x = d.xmat[sb.chest].reshape(3, 3)[:, 0]
    return wrap(math.atan2(x[1], x[0]) - pelvis_yaw(d, sb)), math.asin(float(np.clip(-x[2], -1, 1)))


def hand_in_heading(d: mujoco.MjData, sb: SkillBodies, arm: int) -> np.ndarray:
    """Forearm tip of arm (-1 left, +1 right) in the pelvis heading frame."""
    k = 0 if arm < 0 else 1
    tip = d.xpos[sb.forearm[k]] + d.xmat[sb.forearm[k]].reshape(3, 3) @ sb.tip_local[k]
    rel = tip - d.xpos[sb.pelvis]
    c, s = math.cos(pelvis_yaw(d, sb)), math.sin(pelvis_yaw(d, sb))
    return np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1], rel[2]])


def pelvis_height_rel(d: mujoco.MjData, sb: SkillBodies) -> float:
    return float(d.xpos[sb.pelvis][2]) - sb.pelvis_z0


def knee_rise(d: mujoco.MjData, sb: SkillBodies) -> tuple[float, float]:
    return float(d.xpos[sb.shin[0]][2]) - sb.knee_z0[0], float(d.xpos[sb.shin[1]][2]) - sb.knee_z0[1]


def foot_on_ground(d: mujoco.MjData, sb: SkillBodies) -> tuple[bool, bool]:
    on = [False, False]
    for c in d.contact[: d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        other = g2 if g1 == sb.ground else g1 if g2 == sb.ground else -1
        for k in (0, 1):
            if other in sb.foot_geoms[k]:
                on[k] = True
    return on[0], on[1]


def shoulder_in_heading(d: mujoco.MjData, m: mujoco.MjModel, arm: int, prefix: str = "") -> np.ndarray:
    """Shoulder (upper-arm body origin) in the pelvis heading frame — the reach sphere's centre."""
    ua = m.body(prefix + ("upper_arm_l" if arm < 0 else "upper_arm_r")).id
    pel = m.body(prefix + "pelvis").id
    R = d.xmat[pel].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    rel = d.xpos[ua] - d.xpos[pel]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1], rel[2]])


REACH_AZ = (math.radians(-30), math.radians(90))      # = tasks/skill_mdp.py: forward .. the arm's own side
REACH_EL = (math.radians(-40), math.radians(60))
REACH_FRAC = (0.45, 0.95)                             # of the reach radius


def sample_reach_target(rng: np.random.Generator, shoulder: np.ndarray, arm: int, reach: float) -> np.ndarray:
    """A reachable target from the shoulder (heading frame), sampled like the training command (skill_mdp)."""
    az, el = rng.uniform(*REACH_AZ), rng.uniform(*REACH_EL)
    u = np.array([math.cos(el) * math.cos(az), -arm * math.cos(el) * math.sin(az), math.sin(el)])
    return shoulder + u * reach * rng.uniform(*REACH_FRAC)


def body_ranges(contract: dict | None = None) -> dict:
    """The skill ranges of the process body (contract.json skill_block ranges)."""
    return (contract or {}).get("skill_block", {}).get("ranges") or C.skill_block()["ranges"]
