"""A5/A6 — Derive MATT's MuJoCo physics body from the extracted skeleton + skin (DESIGN.md §2).

Inputs : assets/derived/skeleton_matt.json, assets/derived/skin_matt.npz  (from extract_skeleton.py)
Outputs: assets/matt.xml         robot only (for mjlab)
         assets/scene_matt.xml   flattened: options + ground + robot + cube pool + keyframe (CPU eval, Unity import)
         assets/scene_meet8.xml  8 lane-isolated athletes (names prefixed L<k>_) + 16-cube pool (G6)
         assets/scene_<event>8.xml  event scenes: crowd contact between lanes (crowd_bits), venue layouts from venues.json
         assets/scene_pedestal.xml Event 1 Iron Pedestal: solo scene on a 1 m x 1 m x 0.5 m block (top at z = 0)
         assets/meet8_layout.json lane table (prefix, origin, cube slots) shared by Python and Unity
         assets/derived/body_report.json

Conventions (DESIGN §2):
  * Every body frame is world-aligned at qpos = 0, placed at its bone pivot => qpos = 0 is MATT's bind (T) pose.
  * Hinge axes are signed so that a positive angle is the anatomical positive direction
    (flexion / abduction / dorsiflexion / inversion / elevation). Left/right axes are mirror images.
  * Flat MJCF: no <default> classes, explicit <inertial>, integer-degree joint ranges, floats with <= 7 sig. digits
    (lossless round-trip through the Unity plug-in's float32 components).
"""

from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import bodies  # noqa: E402

ASSETS = ROOT / "assets"
DERIVED = ASSETS / "derived"
# Phase Z: the body is chosen with $POOLYMPIC_BODY (default matt). Every size-dependent number below is MATT's value times
# a scale that is exactly 1.0 for MATT, so his artefacts stay byte-identical.
BODY = bodies.current()
L = BODY.length_scale

TOTAL_MASS = BODY.total_mass
TIMESTEP = 0.005
SOLVER_ITERATIONS = 20
N_CUBES_TRAINING = 4
CUBE_HALF = 0.1
CUBE_MASS = 2.0
CUBE_FRICTION = 0.8
GROUND_FRICTION = (1.0, 0.005, 0.0001)
BODY_FRICTION = (1.0, 0.005, 0.0001)
ALL_BITS = 0xFFFF  # ground + cubes collide with every lane
LANE = 0  # training scene is always lane 0; the meet scene gives lane k its own bits
N_LANES = 8
N_CUBES_MEET = 16  # 2 per lane for scripted disturbances; the HUD pool cycles through all of them
LANE_WIDTH = 1.22  # m (World Athletics lane width)
# Event 1 Iron Pedestal: 1 m x 1 m block (= the Rung 0 foot box). Its top is z = 0 and the ground drops to -PEDESTAL_H,
# so the athlete's default pose, pelvis-height observation and fall rule are exactly those it was trained with.
PEDESTAL_HALF = 0.5
PEDESTAL_H = 0.5


def lane_prefix(lane: int) -> str:
    return f"L{lane}_"


def lane_origin(lane: int) -> np.ndarray:
    """World origin of lane k: lanes side by side along y (lane 0 leftmost, +y), running along +x."""
    return np.array([0.0, (0.5 * (N_LANES - 1) - lane) * LANE_WIDTH, 0.0])


def f(x: float) -> str:
    s = f"{x:.7g}"
    return "0" if s in ("-0", "0") else s


def vec(v) -> str:
    return " ".join(f(x) for x in v)


# Body pairs that never collide (MuJoCo already skips parent/child). The inner thighs touch at rest on every body.
# A self-colliding body adds the pairs that touch in its zero / default pose (filled in by build_zombie_excludes).
EXCLUDES: list[tuple[str, str]] = [("thigh_l", "thigh_r")]


def lane_bits(lane: int, leg: bool) -> tuple[int, int]:
    """(contype, conaffinity). Upper body: collides with ground/cubes only. Legs: also leg<->leg within lane."""
    if leg:
        b = 1 << (lane + 8)
        return b, b
    return 1 << lane, 0


# Crowd contact (event scenes): lane k's athlete also owns bit CROWD_SHIFT + k and has every OTHER lane's crowd bit in
# its conaffinity, so every body part of an athlete collides with every body part of every other athlete, while its own
# geoms still never see each other (no new self-collision). Bits 16-23 are outside ALL_BITS: ground / props / cubes are
# unaffected. The G6 testbed (scene_meet8) keeps full lane isolation (every lane == its solo run).
CROWD_SHIFT = 16
CROWD_ALL = 0xFF << CROWD_SHIFT


def crowd_bits(ct: int, ca: int, lane: int) -> tuple[int, int]:
    own = 1 << (CROWD_SHIFT + lane)
    return ct | own, ca | (CROWD_ALL & ~own)


# --------------------------------------------------------------------------------------------------
# Joint + actuator spec (DESIGN §2). axis given for the LEFT side; right side mirrors x/z, keeps y.
# --------------------------------------------------------------------------------------------------
@dataclass
class JointSpec:
    name: str
    axis: tuple[float, float, float]
    range_deg: tuple[int, int]
    group: str | None  # actuator group, None = passive
    default_deg: float = 0.0
    mirror: bool = True  # mirror axis for right side (x and z components flip sign)


MATT_GAINS = {  # kp (Nm/rad), kv (Nm s/rad), force cap (Nm), armature
    "hip": (300.0, 30.0, 280.0, 0.02),
    "knee": (300.0, 30.0, 280.0, 0.02),
    "ankle": (200.0, 20.0, 220.0, 0.01),
    "abdomen": (300.0, 30.0, 200.0, 0.02),
    "shoulder": (60.0, 6.0, 80.0, 0.01),
    "elbow": (50.0, 5.0, 70.0, 0.01),
}
# size scaling: stiffness and torque cap ∝ mass × length × strength, damping additionally × time scale, armature ∝ m l²
_TS, _IS, _TT = BODY.torque_scale, BODY.inertia_scale, BODY.time_scale
GAINS = MATT_GAINS if BODY.name == "matt" else {
    k: (kp * _TS, kv * _TS * _TT, cap * _TS, arm * _IS) for k, (kp, kv, cap, arm) in MATT_GAINS.items()}
_PS = BODY.mass_scale * L        # passive toe spring: size-scaled, not weakened
TOE_PASSIVE = dict(stiffness=20.0, damping=1.0, armature=0.005) if BODY.name == "matt" else dict(
    stiffness=20.0 * _PS, damping=1.0 * _PS * _TT, armature=0.005 * _IS)

ABDOMEN = [
    JointSpec("abdomen_flex", (0, 1, 0), (-30, 60), "abdomen"),
    JointSpec("abdomen_lat", (1, 0, 0), (-35, 35), "abdomen"),
    JointSpec("abdomen_twist", (0, 0, 1), (-45, 45), "abdomen"),
]
HIP = [
    JointSpec("hip_flex", (0, -1, 0), (-30, 120), "hip", 10.0),
    JointSpec("hip_abd", (1, 0, 0), (-30, 45), "hip"),
    JointSpec("hip_rot", (0, 0, 1), (-40, 40), "hip"),
]
KNEE = [JointSpec("knee", (0, 1, 0), (0, 150), "knee", 20.0)]
ANKLE = [
    JointSpec("ankle_dorsi", (0, -1, 0), (-50, 25), "ankle", 10.0),
    JointSpec("ankle_inv", (-1, 0, 0), (-20, 30), "ankle"),
]
TOE = [JointSpec("toe", (0, -1, 0), (-30, 60), None)]
SHOULDER = [
    JointSpec("shoulder_elev", (1, 0, 0), (-95, 80), "shoulder", -80.0),
    JointSpec("shoulder_flex", (0, 0, -1), (-60, 170), "shoulder"),
    JointSpec("shoulder_twist", (0, 1, 0), (-80, 80), "shoulder"),
]
ELBOW = [JointSpec("elbow", (0, 0, -1), (0, 145), "elbow", 15.0)]

# de Leva (1996) adult male, scaled to 80 kg (DESIGN §2; torso = Spine + Spine1 segments).
MASS = {
    "pelvis": 8.94, "torso": 13.06, "chest": 12.77, "head": 5.55,
    "upper_arm": 2.17, "forearm": 1.79, "thigh": 11.33, "shin": 3.46, "foot": 0.85, "toe": 0.25,
}


@dataclass
class BodySpec:
    name: str
    parent: str | None
    pivot_bone: str
    bones: list[str]  # skin bones whose vertices belong to this body
    mass_key: str
    joints: list[JointSpec] = field(default_factory=list)
    leg: bool = False
    side: str = ""  # "l", "r" or ""


def body_specs() -> list[BodySpec]:
    specs = [
        BodySpec("pelvis", None, "Hips", ["Hips"], "pelvis"),
        BodySpec("torso", "pelvis", "Spine", ["Spine", "Spine1"], "torso", ABDOMEN),
        BodySpec("chest", "torso", "Spine2", ["Spine2", "LeftShoulder", "RightShoulder"], "chest"),
        BodySpec("head", "chest", "Neck", ["Neck", "Head"], "head"),
    ]
    for side, S in (("l", "Left"), ("r", "Right")):
        fingers = [f"{S}Hand{n}{i}" for n in ("Thumb", "Index", "Middle", "Ring", "Pinky") for i in (1, 2, 3)]
        specs += [
            BodySpec(f"upper_arm_{side}", "chest", f"{S}Arm", [f"{S}Arm"], "upper_arm", SHOULDER, side=side),
            BodySpec(f"forearm_{side}", f"upper_arm_{side}", f"{S}ForeArm", [f"{S}ForeArm", f"{S}Hand", *fingers],
                     "forearm", ELBOW, side=side),
            BodySpec(f"thigh_{side}", "pelvis", f"{S}UpLeg", [f"{S}UpLeg"], "thigh", HIP, leg=True, side=side),
            BodySpec(f"shin_{side}", f"thigh_{side}", f"{S}Leg", [f"{S}Leg"], "shin", KNEE, leg=True, side=side),
            BodySpec(f"foot_{side}", f"shin_{side}", f"{S}Foot", [f"{S}Foot"], "foot", ANKLE, leg=True, side=side),
            BodySpec(f"toe_{side}", f"foot_{side}", f"{S}ToeBase", [f"{S}ToeBase"], "toe", TOE, leg=True, side=side),
        ]
    return specs


# --------------------------------------------------------------------------------------------------
# Geometry fitting (all in world frame at bind pose; fitted on the LEFT/centre, mirrored to the right)
# --------------------------------------------------------------------------------------------------
MIRROR = np.array([1.0, -1.0, 1.0])


class Skin:
    def __init__(self):
        sk = json.loads(BODY.skeleton_json.read_text())
        self.joint_pos = {j["name"]: np.array(j["pos"]) for j in sk["joints"]}
        z = np.load(BODY.skin_npz)
        names = list(z["joint_names"])
        mesh_names = list(z["mesh_names"])
        keep = np.array([("hair" not in mesh_names[m]) for m in z["mesh"]])  # hair is visual only
        self.verts = z["verts"][keep]
        self.dom = np.array(names)[z["dominant"][keep]]
        self.sole_z = float(self.verts[np.isin(self.dom, ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]), 2].min())

    def of(self, bones: list[str]) -> np.ndarray:
        return self.verts[np.isin(self.dom, bones)]

    def hand_tip(self, side: str) -> np.ndarray:
        """End of the forearm capsule: MATT's middle finger (LeftHandMiddle2); rigs without finger bones (zombie) use
        the hand vertex furthest along the forearm direction (98th percentile)."""
        S = "Left" if side == "l" else "Right"
        if f"{S}HandMiddle2" in self.joint_pos:
            return self.joint_pos[f"{S}HandMiddle2"]
        p0, p1 = self.joint_pos[f"{S}ForeArm"], self.joint_pos[f"{S}Hand"]
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        t = float(np.percentile((self.of([f"{S}Hand"]) - p1) @ u, 98))
        return p1 + u * t


def fit_limb_capsule(p0, p1, verts, pct=65.0, shrink=0.5, p1_extra=0.0):
    u = p1 - p0
    length = np.linalg.norm(u)
    uh = u / length
    t = (verts - p0) @ uh
    inside = (t > 0.15 * length) & (t < 0.85 * length)
    d = np.linalg.norm((verts - p0) - np.outer(t, uh), axis=1)[inside]
    r = float(np.percentile(d, pct))
    a = p0 + uh * (shrink * r)
    b = p1 + uh * (p1_extra - shrink * r)
    return {"type": "capsule", "fromto": np.concatenate([a, b]), "size": [r]}


def fit_trunk_capsule(verts, z_lo=None, z_hi=None):
    """Horizontal (y-axis) capsule through a trunk slab: radius = half depth (x), length from width (y)."""
    v = verts
    if z_lo is not None:
        v = v[(v[:, 2] >= z_lo) & (v[:, 2] <= z_hi)]
    x5, x95 = np.percentile(v[:, 0], [5, 95])
    y5, y95 = np.percentile(np.abs(v[:, 1]), [5, 95])
    r = 0.5 * (x95 - x5)
    cx = 0.5 * (x95 + x5)
    cz = float(np.median(v[:, 2]))
    half = max(0.0, y95 - r)
    return {"type": "capsule", "fromto": np.array([cx, -half, cz, cx, half, cz]), "size": [r]}


def fit_sphere(verts, pct=85.0):
    c = verts.mean(0)
    c[1] = 0.0
    r = float(np.percentile(np.linalg.norm(verts - c, axis=1), pct))
    return {"type": "sphere", "pos": c, "size": [r]}


def fit_box(verts, x_lo, x_hi, sole_z, height):
    y1, y99 = np.percentile(verts[:, 1], [2, 98])
    lo = np.array([x_lo, y1, sole_z])
    hi = np.array([x_hi, y99, sole_z + height])
    return {"type": "box", "pos": 0.5 * (lo + hi), "size": list(0.5 * (hi - lo))}


def fit_geoms(skin: Skin) -> dict[str, list[dict]]:
    J = skin.joint_pos
    g: dict[str, list[dict]] = {}
    # GRANDMA's automatic (bone heat) weights give the hip / buttock skin to the thighs, so her Hips-only skin is
    # narrower than deep (zero-length capsule): her pelvis is fitted to all skin in the hip slab
    pelvis_v = skin.verts if BODY.family == "grandma" else skin.of(["Hips"])
    g["pelvis"] = [fit_trunk_capsule(pelvis_v, J["Hips"][2] - 0.08 * L, J["Spine"][2] + 0.02 * L)]
    g["torso"] = [fit_trunk_capsule(skin.of(["Spine", "Spine1"]))]
    g["chest"] = [fit_trunk_capsule(skin.of(["Spine2"]), J["Spine2"][2], J["LeftArm"][2] - 0.03 * L)]
    neck = fit_limb_capsule(J["Neck"], J["Head"], skin.of(["Neck"]), pct=60, shrink=0.0)
    head = fit_sphere(skin.of(["Head"]))
    g["head"] = [neck, head]
    # Left limbs (mirrored later)
    g["upper_arm_l"] = [fit_limb_capsule(J["LeftArm"], J["LeftForeArm"], skin.of(["LeftArm"]))]
    g["forearm_l"] = [fit_limb_capsule(J["LeftForeArm"], skin.hand_tip("l"), skin.of(["LeftForeArm"]), pct=70)]
    g["thigh_l"] = [fit_limb_capsule(J["LeftUpLeg"], J["LeftLeg"], skin.of(["LeftUpLeg"]), pct=60)]
    g["shin_l"] = [fit_limb_capsule(J["LeftLeg"], J["LeftFoot"], skin.of(["LeftLeg"]), pct=60)]
    foot_v = skin.of(["LeftFoot", "LeftToeBase"])
    toe_x = J["LeftToeBase"][0]
    heel_x = float(np.percentile(foot_v[:, 0], 0.5))
    if BODY.family != "matt":
        # AccuRig weights the heel skin to the calf: take the heel from all skin near the sole under this foot
        sole = skin.verts[(skin.verts[:, 2] < skin.sole_z + 0.03 * L)
                          & (np.abs(skin.verts[:, 1] - J["LeftFoot"][1]) < 0.07 * L)]
        heel_x = float(np.percentile(sole[:, 0], 0.5))
    g["foot_l"] = [fit_box(foot_v, heel_x, toe_x, skin.sole_z, 0.06 * L)]
    g["toe_l"] = [fit_box(foot_v[foot_v[:, 0] > toe_x - 0.01 * L], toe_x, float(np.percentile(foot_v[:, 0], 99.5)),
                          skin.sole_z, 0.035 * L)]
    for b in ("upper_arm", "forearm", "thigh", "shin", "foot", "toe"):
        g[f"{b}_r"] = [mirror_geom(x) for x in g[f"{b}_l"]]
    return g


def mirror_geom(geom: dict) -> dict:
    m = dict(geom)
    if "fromto" in m:
        ft = np.asarray(m["fromto"])
        m["fromto"] = np.concatenate([ft[:3] * MIRROR, ft[3:] * MIRROR])
    if "pos" in m:
        m["pos"] = np.asarray(m["pos"]) * MIRROR
    return m


# --------------------------------------------------------------------------------------------------
# XML emission
# --------------------------------------------------------------------------------------------------
def joint_axis(js: JointSpec, side: str) -> np.ndarray:
    a = np.array(js.axis, dtype=float)
    if side == "r" and js.mirror:
        a = a * np.array([-1.0, 1.0, -1.0])
    return a


def build_robot(skin: Skin, geoms, inertials=None, lane: int = LANE, crowd: bool = False) -> tuple[ET.Element, list, list]:
    """Returns (<body name="pelvis"> element, actuator specs, joint specs in qpos order)."""
    specs = body_specs()
    by_name = {s.name: s for s in specs}
    pivot = {s.name: skin.joint_pos[s.pivot_bone] for s in specs}
    elems: dict[str, ET.Element] = {}
    actuators, joints_in_order = [], []

    for s in specs:
        parent_pos = pivot[s.parent] if s.parent else np.zeros(3)
        attrs = {"name": s.name, "pos": vec(pivot[s.name] - parent_pos)}
        el = ET.Element("body", attrs) if s.parent is None else ET.SubElement(elems[s.parent], "body", attrs)
        elems[s.name] = el
        if inertials:
            ine = inertials[s.name]
            ET.SubElement(el, "inertial", {"pos": vec(ine["pos"]), "quat": vec(ine["quat"]),
                                           "mass": f(ine["mass"]), "diaginertia": vec(ine["diag"])})
        if s.parent is None:
            ET.SubElement(el, "freejoint", {"name": "root"})
        for js in s.joints:
            jname = f"{js.name}_{s.side}" if s.side else js.name
            ja = {"name": jname, "type": "hinge", "pos": "0 0 0", "axis": vec(joint_axis(js, s.side)),
                  "range": f"{js.range_deg[0]} {js.range_deg[1]}"}
            if js.group is None:
                ja.update({k: f(v) for k, v in TOE_PASSIVE.items()})
            else:
                ja["armature"] = f(GAINS[js.group][3])
                actuators.append((jname, js.group))
            ET.SubElement(el, "joint", ja)
            joints_in_order.append((jname, js))
        ct, ca = lane_bits(lane, s.leg or BODY.self_collision)   # self-colliding body: every geom on the leg bit
        if crowd:
            ct, ca = crowd_bits(ct, ca, lane)
        for gi, g in enumerate(geoms[s.name]):
            ga = {"name": f"{s.name}_geom{gi}", "type": g["type"], "size": vec(g["size"]),
                  "contype": str(ct), "conaffinity": str(ca), "condim": "3", "friction": vec(BODY_FRICTION)}
            if not inertials:
                ga["density"] = "1000"
            if "fromto" in g:
                # Emit capsules as pos/quat/(radius, half-length), NOT fromto: the Unity plug-in rebuilds a fromto
                # frame in float32 (FromToRotation), which drifts ~3e-6 rad for near-vertical limbs (G0). Computing
                # the quaternion once here in float64 means both sides parse identical text.
                ft = np.asarray(g["fromto"], float)
                a, b = ft[:3] - pivot[s.name], ft[3:] - pivot[s.name]
                d = b - a
                half = 0.5 * float(np.linalg.norm(d))
                q = np.zeros(4)
                mujoco.mju_quatZ2Vec(q, d / np.linalg.norm(d))
                if q[0] < 0:
                    q = -q
                ga["pos"] = vec(0.5 * (a + b))
                ga["quat"] = vec(q)
                ga["size"] = vec([g["size"][0], half])
            else:
                ga["pos"] = vec(np.asarray(g["pos"]) - pivot[s.name])
            ET.SubElement(el, "geom", ga)
    return elems["pelvis"], actuators, joints_in_order


def actuator_block(parent: ET.Element, actuators):
    act = ET.SubElement(parent, "actuator")
    for jname, group in actuators:
        kp, kv, cap, _arm = GAINS[group]
        lo, hi = -cap, cap
        if BODY.torque_caps:        # joint- and direction-specific caps (bodies.BIO_TORQUE_CAPS), never above the group cap
            base = jname[:-2] if jname.endswith(("_l", "_r")) else jname
            against, towards = BODY.torque_caps[base]
            lo, hi = -min(against, cap), min(towards, cap)
        ET.SubElement(act, "position", {"name": jname, "joint": jname, "kp": f(kp), "kv": f(kv),
                                        "forcelimited": "true", "forcerange": f"{f(lo)} {f(hi)}"})


def option_block(root: ET.Element):
    ET.SubElement(root, "compiler", {"angle": "degree", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": f(TIMESTEP), "gravity": "0 0 -9.81", "integrator": "implicitfast",
                                   "solver": "Newton", "iterations": str(SOLVER_ITERATIONS), "cone": "pyramidal",
                                   "jacobian": "auto"})
    # Explicit: the Unity plug-in (3.11) defaults MultiCCD to "disable" while MuJoCo defaults it on — pin it (G0).
    ET.SubElement(root.find("option"), "flag", {"multiccd": "disable"})


def indent(el: ET.Element) -> str:
    ET.indent(el, space="  ")
    return ET.tostring(el, encoding="unicode")


def compose_model(skin, geoms, inertials, with_scene: bool, n_cubes: int, default_qpos=None,
                  pedestal_h: float = 0.0) -> str:
    """pedestal_h > 0: Iron Pedestal scene — ground plane at z = -pedestal_h, a static 1 m x 1 m box with its top at z = 0
    under the athlete (same surface as the ground), parked cubes lowered onto the ground."""
    root = ET.Element("mujoco", {"model": ("pedestal_scene" if pedestal_h else f"{BODY.name}_scene") if with_scene else BODY.name})
    option_block(root)
    wb = ET.SubElement(root, "worldbody")
    if with_scene:
        ET.SubElement(wb, "light", {"name": "sun", "pos": "0 0 5", "dir": "0 0 -1", "directional": "true"})
        ground = {"name": "ground", "type": "plane", "size": "0 0 0.05", "contype": str(ALL_BITS),
                  "conaffinity": str(ALL_BITS), "condim": "3", "friction": vec(GROUND_FRICTION)}
        if pedestal_h:
            ground["pos"] = vec([0, 0, -pedestal_h])
        ET.SubElement(wb, "geom", ground)
        if pedestal_h:
            ET.SubElement(wb, "geom", {"name": "pedestal", "type": "box", "pos": vec([0, 0, -pedestal_h / 2]),
                                       "size": vec([PEDESTAL_HALF, PEDESTAL_HALF, pedestal_h / 2]),
                                       "contype": str(ALL_BITS), "conaffinity": str(ALL_BITS), "condim": "3",
                                       "friction": vec(GROUND_FRICTION)})
    pelvis, actuators, joints = build_robot(skin, geoms, inertials)
    # Declare actuators in joint-tree (depth-first) order == MuJoCo joint id order. mjlab's XmlActuator pairs joint
    # targets (joint order) with ctrl slots (declaration order); any other order silently scrambles the action wiring.
    tree_order = [j.get("name") for j in pelvis.iter("joint")]
    actuators = sorted(actuators, key=lambda a: tree_order.index(a[0]))
    wb.append(pelvis)
    if with_scene:
        inertia = CUBE_MASS * (2 * CUBE_HALF) ** 2 / 6.0
        for i in range(n_cubes):
            cb = ET.SubElement(wb, "body", {"name": f"cube{i}", "pos": vec(cube_park_pos(i) - [0, 0, pedestal_h])})
            ET.SubElement(cb, "inertial", {"pos": "0 0 0", "mass": f(CUBE_MASS), "diaginertia": vec([inertia] * 3)})
            ET.SubElement(cb, "freejoint", {"name": f"cube{i}_free"})
            ET.SubElement(cb, "geom", {"name": f"cube{i}_geom", "type": "box", "size": vec([CUBE_HALF] * 3),
                                       "contype": str(ALL_BITS), "conaffinity": str(ALL_BITS), "condim": "3",
                                       "friction": f"{f(CUBE_FRICTION)} 0.005 0.0001"})
    # Inner thighs overlap at rest (hip pivots 0.172 m apart, thigh radius ~0.089 m): exclude that pair only;
    # shin/foot/toe still collide across legs (leg-crossing detection).
    contact = ET.SubElement(root, "contact")
    for b1, b2 in EXCLUDES:
        ET.SubElement(contact, "exclude", {"body1": b1, "body2": b2})
    actuator_block(root, actuators)
    if with_scene and default_qpos is not None:
        kf = ET.SubElement(root, "keyframe")
        ET.SubElement(kf, "key", {"name": "default", "qpos": vec(default_qpos)})
    return indent(root)


def static_box(parent: ET.Element, name: str, pos, size) -> None:
    """A static event prop (collides with every lane)."""
    ET.SubElement(parent, "geom", {"name": name, "type": "box", "pos": vec(pos), "size": vec(size),
                                   "contype": str(ALL_BITS), "conaffinity": str(ALL_BITS), "condim": "3",
                                   "friction": vec(GROUND_FRICTION)})


def shaker_body(parent: ET.Element, prefix: str, pos, sh: dict) -> np.ndarray:
    """Spring-mounted platform (body/geom `<prefix>shaker`, slide joints `<prefix>shaker_x/_y`, top at z = 0 when pos is
    the athlete spot). sh["half"]: half side (square) or [half x, half y]. Returns its 2 keyframe qpos."""
    hx, hy = (sh["half"], sh["half"]) if np.isscalar(sh["half"]) else sh["half"]
    body = ET.SubElement(parent, "body", {"name": prefix + "shaker", "pos": vec(np.asarray(pos, float) + [0, 0, -sh["h"] / 2])})
    half = [hx, hy, sh["h"] / 2]
    ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": f(sh["mass"]), "diaginertia": vec(
        [sh["mass"] * (half[1] ** 2 + half[2] ** 2) / 3, sh["mass"] * (half[0] ** 2 + half[2] ** 2) / 3,
         sh["mass"] * (half[0] ** 2 + half[1] ** 2) / 3])})
    for ax, axis in (("x", "1 0 0"), ("y", "0 1 0")):
        ET.SubElement(body, "joint", {"name": f"{prefix}shaker_{ax}", "type": "slide", "axis": axis,
                                      "stiffness": f(sh["stiffness"]), "damping": f(sh["damping"]),
                                      "limited": "true", "range": vec([-sh["range"], sh["range"]])})
    ET.SubElement(body, "geom", {"name": prefix + "shaker", "type": "box", "size": vec(half),
                                 "contype": str(ALL_BITS), "conaffinity": str(ALL_BITS), "condim": "3",
                                 "friction": vec(GROUND_FRICTION)})
    return np.zeros(2)


def stage_shared(wb: ET.Element, pedestal_h: float, shaker: dict | None, beam: dict | None) -> list[np.ndarray]:
    """Stage pieces all athletes share (crowd events): the Iron Pedestal beam (geom `pedestal`, {pos, half: [x, y]}, top
    at z = 0) and the Gust Gauntlet shaker floor (shaker["shared"], centred at shaker["pos"]). Returns keyframe qpos."""
    if beam:
        static_box(wb, "pedestal", np.asarray(beam["pos"], float) + [0, 0, -pedestal_h / 2],
                   [beam["half"][0], beam["half"][1], pedestal_h / 2])
    if shaker and shaker.get("shared"):
        return [shaker_body(wb, "", shaker["pos"], shaker)]
    return []


def stage_lane(wb: ET.Element, p: str, o: np.ndarray, pedestal_h: float, shaker: dict | None, beam: dict | None) -> list[np.ndarray]:
    """Per-lane stage pieces: its own 1 m pedestal / shaker platform (unless the event shares one)."""
    if pedestal_h and not beam:
        static_box(wb, p + "pedestal", o + [0, 0, -pedestal_h / 2], [PEDESTAL_HALF, PEDESTAL_HALF, pedestal_h / 2])
    if shaker and not shaker.get("shared"):
        return [shaker_body(wb, p, o, shaker)]
    return []


def compose_meet(skin, geoms, inertials, default_qpos: np.ndarray, n_lanes: int = N_LANES,
                 n_cubes: int = N_CUBES_MEET, origins: list[np.ndarray] | None = None, pedestal_h: float = 0.0,
                 model: str | None = None, park_offset=(0.0, 0.0, 0.0), props: list[dict] | None = None,
                 shaker: dict | None = None, beam: dict | None = None, crowd: bool = False) -> tuple[str, dict]:
    """Multi-athlete scene (G6 / events): lane k = the training athlete with every name prefixed `L<k>_`, its own
    collision bits and its pelvis shifted to origins[k] (default lane_origin(k)). Ground + cube pool as in the solo
    scene. crowd: athletes of different lanes collide (crowd_bits; every event scene) — otherwise full lane isolation
    (G6). pedestal_h > 0: every lane stands on its own 1 m x 1 m pedestal (`L<k>_pedestal`, top at z = 0) — or all on
    one shared beam {pos, half} (geom `pedestal`) — and the ground drops to -pedestal_h (Iron Pedestal heat). props:
    static event boxes {name, pos, size (half)} that collide with every lane (rails, poles). shaker {half, h, mass,
    stiffness, damping, range}: every lane stands on its own spring-mounted platform (body `L<k>_shaker`, slide joints
    `L<k>_shaker_x/_y`, top at z = 0; ground at -h) — or, shaker["shared"], all on one floor (`shaker`, centre
    shaker["pos"]) — events shake it with velocity kicks. Returns (xml, layout) — the lane table Unity and the evaluators
    share."""
    root = ET.Element("mujoco", {"model": model or f"meet{n_lanes}"})
    option_block(root)
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"name": "sun", "pos": "0 0 5", "dir": "0 0 -1", "directional": "true"})
    ground = {"name": "ground", "type": "plane", "size": "0 0 0.05", "contype": str(ALL_BITS),
              "conaffinity": str(ALL_BITS), "condim": "3", "friction": vec(GROUND_FRICTION)}
    if pedestal_h or shaker:
        ground["pos"] = vec([0, 0, -(pedestal_h or shaker["h"])])
    ET.SubElement(wb, "geom", ground)
    for pr in props or []:
        static_box(wb, pr["name"], pr["pos"], pr["size"])
    contact = ET.Element("contact")
    all_actuators, lanes = [], []
    key_qpos = stage_shared(wb, pedestal_h, shaker, beam)
    for k in range(n_lanes):
        p, o = lane_prefix(k), (np.asarray(origins[k], float) if origins is not None else lane_origin(k))
        key_qpos += stage_lane(wb, p, o, pedestal_h, shaker, beam)
        pelvis, actuators, _ = build_robot(skin, geoms, inertials, lane=k, crowd=crowd)
        tree_order = [j.get("name") for j in pelvis.iter("joint")]
        actuators = sorted(actuators, key=lambda a: tree_order.index(a[0]))
        for el in pelvis.iter():
            if "name" in el.attrib:
                el.set("name", p + el.get("name"))
        pelvis.set("pos", vec(np.array([float(x) for x in pelvis.get("pos").split()]) + o))
        wb.append(pelvis)
        for b1, b2 in EXCLUDES:
            ET.SubElement(contact, "exclude", {"body1": p + b1, "body2": p + b2})
        all_actuators += [(p + j, g) for j, g in actuators]
        q = default_qpos.copy()
        q[0:3] += o
        key_qpos.append(q)
        lanes.append({"lane": k, "prefix": p, "origin": o.tolist(), "cubes": [2 * k, 2 * k + 1]})
    inertia = CUBE_MASS * (2 * CUBE_HALF) ** 2 / 6.0
    for i in range(n_cubes):
        park = cube_park_pos(i) - [0, 0, pedestal_h or (shaker['h'] if shaker else 0.0)] + np.asarray(park_offset, float)
        cb = ET.SubElement(wb, "body", {"name": f"cube{i}", "pos": vec(park)})
        ET.SubElement(cb, "inertial", {"pos": "0 0 0", "mass": f(CUBE_MASS), "diaginertia": vec([inertia] * 3)})
        ET.SubElement(cb, "freejoint", {"name": f"cube{i}_free"})
        ET.SubElement(cb, "geom", {"name": f"cube{i}_geom", "type": "box", "size": vec([CUBE_HALF] * 3),
                                   "contype": str(ALL_BITS), "conaffinity": str(ALL_BITS), "condim": "3",
                                   "friction": f"{f(CUBE_FRICTION)} 0.005 0.0001"})
        key_qpos.append(np.concatenate([park, [1, 0, 0, 0]]))
    root.append(contact)
    actuator_block(root, all_actuators)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", {"name": "default", "qpos": vec(np.concatenate(key_qpos))})
    layout = {"n_lanes": n_lanes, "lane_width": LANE_WIDTH, "n_cubes": n_cubes, "lanes": lanes, "pedestal_h": pedestal_h,
              "crowd": crowd,
              "note": "lane k: names prefixed L<k>_, pelvis shifted by origin; solo-scene cube i -> meet cube cubes[i]"}
    stage_layout(layout, shaker, beam)
    if props:
        layout["props"] = [{"name": pr["name"], "pos": np.asarray(pr["pos"], float).tolist(), "size": list(pr["size"])}
                           for pr in props]
    return indent(root), layout


def stage_layout(layout: dict, shaker: dict | None, beam: dict | None) -> None:
    """Layout entries of the stage (JSON-safe): shaker platform spec, shared beam."""
    js = lambda v: np.asarray(v, float).tolist() if not np.isscalar(v) else v
    if shaker:
        layout["shaker"] = {k: js(v) for k, v in shaker.items()}
    if beam:
        layout["beam"] = {k: js(v) for k, v in beam.items()}


SHAKER = {"half": 0.8, "h": 0.1, "mass": 150.0, "stiffness": 36300.0, "damping": 1160.0, "range": 0.25}
NL = chr(10)


VENUES_JSON = ROOT.parent / "SourceArt" / "Stadium" / "venues.json"


def venue_lane_origins(event: int, reference_lane: int, extra_yaw_deg: float = 0.0) -> list[np.ndarray]:
    """Competitor spots of a stadium event (venues.json, written by SourceArt/Stadium/build_venues.py) in the athlete
    frame: the reference lane's spot is the origin and its facing is +x (EventScenes.PlaceStadium does the same turn in
    Unity). extra_yaw_deg turns the athlete relative to the event direction (90 = facing the event's left, crab events).
    Heights are relative to the reference spot (the surface the athlete stands on)."""
    return [venue_to_athlete(event, reference_lane, lane["pos"], extra_yaw_deg)
            for lane in json.loads(VENUES_JSON.read_text())["events"][f"{event:02d}"]["lanes"]]


def venue_to_athlete(event: int, reference_lane: int, pos, extra_yaw_deg: float = 0.0) -> np.ndarray:
    """A stadium point (MuJoCo axes, venues.json frame) in the frame of venue_lane_origins."""
    ref = json.loads(VENUES_JSON.read_text())["events"][f"{event:02d}"]["lanes"][reference_lane]
    p0, yaw = np.asarray(ref["pos"], float), math.radians(ref["yaw_deg"] + extra_yaw_deg)
    c, s = math.cos(-yaw), math.sin(-yaw)
    d = np.asarray(pos, float) - p0
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


def venue_box(event: int, key: str) -> dict:
    """A shared stage box of a crowd venue (venues.json events[event][key] = {pos: top centre, size_m}) in the athlete
    frame of reference lane 3: {pos (top centre), half [x, y]} — the E01 beam lies across the athletes' facing."""
    ev = json.loads(VENUES_JSON.read_text())["events"][f"{event:02d}"]
    box, yaw = ev[key], math.radians(ev["lanes"][3]["yaw_deg"])
    sx, sy = box["size_m"][0] / 2, box["size_m"][1] / 2
    turned = abs(math.cos(yaw)) < 0.5
    return {"pos": venue_to_athlete(event, 3, box["pos"]), "half": [sy, sx] if turned else [sx, sy]}


# 5 Gust Gauntlet crowd stage: ONE spring-mounted floor under all 8 (venue "floor"); mass, stiffness and damping are 8 x
# the per-lane platform's, so with the 8 athletes aboard it still rings at 2.0 Hz, damping ratio 0.2
SHAKER_FLOOR = {"h": 0.1, "mass": 8 * 150.0, "stiffness": 8 * 36300.0, "damping": 8 * 1160.0, "range": 0.25, "shared": True}


def shaker_floor() -> dict:
    fl = venue_box(5, "floor")
    return {**SHAKER_FLOOR, "half": fl["half"], "pos": fl["pos"]}


def crab_rails() -> list[dict]:
    """10 Crab Shuffle rails (build_venues.py: a 4 cm steel bar centred 0.3 m up on every lane line, over the 20 m course)
    as MuJoCo boxes in the crab frame (reference lane 3, athletes facing the event's left)."""
    ev = json.loads(VENUES_JSON.read_text())["events"]["10"]
    ys = [l["pos"][1] for l in ev["lanes"]]
    x0, length, lw = ev["lanes"][0]["pos"][0], float(ev["length_m"]), abs(ys[0] - ys[1])
    rails = []
    for j in range(len(ys) + 1):
        y = ys[0] + lw / 2 - j * lw            # lane lines from the lane-1 side
        c = venue_to_athlete(10, 3, [x0 + length / 2, y, 0.3], 90.0)
        rails.append({"name": f"rail{j}", "pos": c, "size": [0.02, length / 2, 0.02]})
    return rails


def slalom_poles() -> list[dict]:
    """11 Slalom Sprint poles (build_venues.py: 4 cm x 1.2 m steel-orange poles on every lane's centre line; venues.json
    "poles" = per lane the pole positions) as MuJoCo boxes in the athlete frame of reference lane 3."""
    ev = json.loads(VENUES_JSON.read_text())["events"]["11"]
    return [{"name": f"pole{k}_{g}", "pos": venue_to_athlete(11, 3, [x, y, 0.6]), "size": [0.02, 0.02, 0.6]}
            for k, lane in enumerate(ev["poles"]) for g, (x, y) in enumerate(lane)]


TRENCH_POST_OUT = 1.2     # m — the ceiling overhangs the lane block by TRENCH_OVERHANG, its posts stand this far outside it
TRENCH_OVERHANG = 0.8
TRENCH_UNDERSIDE = 0.72   # m — Event 23 ceiling underside: MATT's crawl brain squeezes through (15-19 s / 16 m), the zombie
                          # is untouched (tools/trench_probe.py sweep: 0.62-0.70 m stalls MATTs, 0.78 m is no obstacle)


def trench_props() -> list[dict]:
    """Event 23 Trench Crawl (build_venues.py): a 12 m ceiling (4 cm slab, underside TRENCH_UNDERSIDE) over the whole
    8-lane block (+ TRENCH_OVERHANG each side) from 2 m past the start, carried by 5 cm posts every 4 m along both sides,
    TRENCH_POST_OUT outside the block — as MuJoCo boxes in the athlete frame of reference lane 3. (Posts on every lane
    line, 0.61 m from each lane centre, hooked 7 of 8 MATT crawlers; an edge flush with the outer lane line / posts 0.4 m
    out stopped the lane-1 crawler, which drifts outwards: tools/trench_probe.py + CPU heats.)"""
    ev = json.loads(VENUES_JSON.read_text())["events"]["23"]
    ys = [l["pos"][1] for l in ev["lanes"]]
    lo, lw = ev["lanes"][0]["pos"][0], abs(ys[0] - ys[1])
    cy, width = (ys[0] + ys[-1]) / 2, lw * len(ys)
    props = [{"name": "trench_ceiling", "pos": venue_to_athlete(23, 3, [lo + 8.0, cy, TRENCH_UNDERSIDE + 0.02]),
              "size": [6.0, width / 2 + TRENCH_OVERHANG, 0.02]}]
    for j, y in enumerate((cy - width / 2 - TRENCH_POST_OUT, cy + width / 2 + TRENCH_POST_OUT)):
        for k in range(4):
            props.append({"name": f"trench_post{j}_{k}",
                          "pos": venue_to_athlete(23, 3, [lo + 2.0 + 4.0 * k, y, TRENCH_UNDERSIDE / 2]),
                          "size": [0.025, 0.025, TRENCH_UNDERSIDE / 2]})
    return props


def cube_entity_xml() -> str:
    """One pooled cube as a standalone model (mjlab entity for training) — identical to the scene's cube bodies."""
    root = ET.Element("mujoco", {"model": "cube"})
    ET.SubElement(root, "compiler", {"angle": "degree", "autolimits": "true"})
    wb = ET.SubElement(root, "worldbody")
    cb = ET.SubElement(wb, "body", {"name": "cube", "pos": vec(cube_park_pos(0))})
    inertia = CUBE_MASS * (2 * CUBE_HALF) ** 2 / 6.0
    ET.SubElement(cb, "inertial", {"pos": "0 0 0", "mass": f(CUBE_MASS), "diaginertia": vec([inertia] * 3)})
    ET.SubElement(cb, "freejoint", {"name": "cube_free"})
    ET.SubElement(cb, "geom", {"name": "cube_geom", "type": "box", "size": vec([CUBE_HALF] * 3),
                               "contype": str(ALL_BITS), "conaffinity": str(ALL_BITS), "condim": "3",
                               "friction": f"{f(CUBE_FRICTION)} 0.005 0.0001"})
    return indent(root)


def cube_park_pos(i: int) -> np.ndarray:
    return np.array([50.0 + 2.0 * i, 0.0, CUBE_HALF])


# --------------------------------------------------------------------------------------------------
def compute_inertials(skin, geoms) -> dict:
    """Uniform-density inertia from geoms, rescaled to the de Leva target mass (principal frame kept)."""
    xml = compose_model(skin, geoms, None, with_scene=False, n_cubes=0)
    m = mujoco.MjModel.from_xml_string(xml)
    specs = body_specs()
    geom_mass = sum(float(m.body_mass[m.body(s.name).id]) for s in specs)
    targets = {s.name: MASS[s.mass_key] for s in specs}
    scale_total = TOTAL_MASS / sum(targets.values())
    out = {}
    for s in specs:
        b = m.body(s.name).id
        mass = targets[s.name] * scale_total
        k = mass / float(m.body_mass[b])
        out[s.name] = {"mass": mass, "pos": m.body_ipos[b].copy(), "quat": m.body_iquat[b].copy(),
                       "diag": m.body_inertia[b] * k}
    # enforce exact L/R symmetry of the inertial description
    for s in specs:
        if s.side == "r":
            l = out[s.name.replace("_r", "_l")]
            out[s.name]["pos"] = l["pos"] * MIRROR
            q = l["quat"]  # mirror across XZ plane: (w, x, y, z) -> (w, -x, y, -z)
            out[s.name]["quat"] = np.array([q[0], -q[1], q[2], -q[3]])
            out[s.name]["diag"] = l["diag"].copy()
    return out, geom_mass


def default_pose_qpos(skin, geoms, inertials) -> np.ndarray:
    """qpos for the default standing pose with the lowest foot/toe geom 1 mm above the ground."""
    m = mujoco.MjModel.from_xml_string(compose_model(skin, geoms, inertials, with_scene=False, n_cubes=0))
    d = mujoco.MjData(m)
    q = m.qpos0.copy()
    for i in range(m.njnt):
        jn = m.joint(i).name
        spec = JOINT_DEFAULTS.get(jn)
        if spec is not None:
            q[m.jnt_qposadr[i]] = math.radians(spec)
    q[0:2] = 0.0
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    lowest = np.inf
    for gi in range(m.ngeom):
        name = m.geom(gi).name
        if not (name.startswith("foot") or name.startswith("toe")):
            continue
        R = d.geom_xmat[gi].reshape(3, 3)
        half = m.geom_size[gi]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
        lowest = min(lowest, float((d.geom_xpos[gi] + corners @ R.T)[:, 2].min()))
    q[2] += 0.001 - lowest
    return q


JOINT_DEFAULTS: dict[str, float] = {}


def touching_pairs(skin, geoms, inertials) -> list[tuple[str, str]]:
    """Non-parent/child body pairs whose geoms are in contact at the zero pose or the default stance (self-colliding
    bodies only): those overlaps are anatomy, not collisions (e.g. upper arm against chest at the shoulder)."""
    xml = compose_model(skin, geoms, inertials, with_scene=False, n_cubes=0)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    pairs = set()
    poses = [m.qpos0.copy()]
    q = m.qpos0.copy()
    for i in range(m.njnt):
        if m.joint(i).name in JOINT_DEFAULTS:
            q[m.jnt_qposadr[i]] = math.radians(JOINT_DEFAULTS[m.joint(i).name])
    poses.append(q)
    for q in poses:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
            pair = tuple(sorted((b1, b2)))
            if pair != ("thigh_l", "thigh_r"):
                pairs.add(pair)
    return sorted(pairs)


def write_other_body(header, skin, geoms, inertials, geom_mass, qdef, scene_xml, robot_xml) -> int:
    """Phase Z: a second body writes only its own robot + training scene + report (the event meets are MATT's for now)."""
    BODY.robot_xml.write_text(header + robot_xml + "\n")
    BODY.scene_xml.write_text(header + scene_xml + "\n")
    m = mujoco.MjModel.from_xml_string(scene_xml)
    specs = body_specs()
    report = {
        "body": BODY.name, "length_scale": L, "torque_scale": BODY.torque_scale,
        "total_mass": float(sum(m.body_mass[m.body(s.name).id] for s in specs)),
        "uniform_density_geom_mass": geom_mass,
        "nq": m.nq, "nv": m.nv, "nu": m.nu, "nbody": m.nbody, "ngeom": m.ngeom,
        "sole_z_bind": skin.sole_z, "default_qpos": qdef.round(6).tolist(), "excludes": EXCLUDES,
        "gains": {k: [round(x, 5) for x in v] for k, v in GAINS.items()},
        "actuators": [m.actuator(i).name for i in range(m.nu)],
        "bodies": {s.name: {"mass": round(inertials[s.name]["mass"], 4),
                            "geoms": [{k: (v if isinstance(v, str) else np.round(np.asarray(v, float), 4).tolist())
                                       for k, v in g.items()} for g in geoms[s.name]]} for s in specs},
    }
    BODY.body_report.write_text(json.dumps(report, indent=1))
    print(f"{BODY.robot_xml.name} + {BODY.scene_xml.name} written. nq={m.nq} nv={m.nv} nu={m.nu} bodies={m.nbody} geoms={m.ngeom}")
    print(f"robot mass={report['total_mass']:.3f} kg, default standing root height z={qdef[2]:.4f} m, excludes={EXCLUDES}")
    return 0


def main() -> int:
    skin = Skin()
    geoms = fit_geoms(skin)
    inertials, geom_mass = compute_inertials(skin, geoms)

    for s in body_specs():
        for js in s.joints:
            JOINT_DEFAULTS[f"{js.name}_{s.side}" if s.side else js.name] = js.default_deg

    for jn, deg in BODY.joint_defaults.items():        # body-specific default stance (both sides)
        for key in (jn, f"{jn}_l", f"{jn}_r"):
            if key in JOINT_DEFAULTS:
                JOINT_DEFAULTS[key] = deg
    if BODY.self_collision:
        EXCLUDES.extend(touching_pairs(skin, geoms, inertials))
    qdef = default_pose_qpos(skin, geoms, inertials)
    robot_xml = compose_model(skin, geoms, inertials, with_scene=False, n_cubes=0)
    scene_xml = compose_model(skin, geoms, inertials, with_scene=True, n_cubes=N_CUBES_TRAINING, default_qpos=np.concatenate(
        [qdef, np.concatenate([np.concatenate([cube_park_pos(i), [1, 0, 0, 0]]) for i in range(N_CUBES_TRAINING)])]))
    header = f"<!-- GENERATED by training/tools/build_mjcf.py from {BODY.glb.relative_to(ROOT.parent).as_posix()}. Do not edit. -->\n"
    if BODY.name != "matt":
        return write_other_body(header, skin, geoms, inertials, geom_mass, qdef, scene_xml, robot_xml)
    (ASSETS / "cube.xml").write_text(header + cube_entity_xml() + "\n")
    (ASSETS / "matt.xml").write_text(header + robot_xml + "\n")
    (ASSETS / "scene_matt.xml").write_text(header + scene_xml + "\n")
    pedestal_xml = compose_model(skin, geoms, inertials, with_scene=True, n_cubes=N_CUBES_TRAINING, pedestal_h=PEDESTAL_H,
                                 default_qpos=np.concatenate([qdef, np.concatenate(
                                     [np.concatenate([cube_park_pos(i) - [0, 0, PEDESTAL_H], [1, 0, 0, 0]])
                                      for i in range(N_CUBES_TRAINING)])]))
    (ASSETS / "scene_pedestal.xml").write_text(header + pedestal_xml + "\n")
    meet_xml, meet_layout = compose_meet(skin, geoms, inertials, qdef)
    # Event scenes are crowd scenes (athletes of different lanes collide: crowd_bits); scene_meet8 (G6) stays isolated.
    # 1 Iron Pedestal: all 8 shoulder to shoulder (0.7 m) on one iron beam (venue "beam")
    ped_origins = venue_lane_origins(1, reference_lane=3)
    ped8_xml, ped8_layout = compose_meet(skin, geoms, inertials, qdef, origins=ped_origins, pedestal_h=PEDESTAL_H,
                                         model="pedestal8", beam=venue_box(1, "beam"), crowd=True)
    (ASSETS / "scene_pedestal8.xml").write_text(header + ped8_xml + "\n")
    (ASSETS / "pedestal8_layout.json").write_text(json.dumps(ped8_layout, indent=1) + "\n")
    # straight-track races (9 Inverted, 13 Steeplechase, 19 Terminal Velocity, 22 Emergency Brake): flat lanes 1.22 m
    # apart, from the home-straight venue of event 22 (the back straight has the same layout in the runner's frame)
    # cube pool parked 30 m to the side: the default park line (x = 50..80, y = 0) is lane 4's running line
    track_xml, track_layout = compose_meet(skin, geoms, inertials, qdef, origins=venue_lane_origins(22, reference_lane=3),
                                           model="track8", park_offset=(0.0, -30.0, 0.0), crowd=True)
    (ASSETS / "scene_track8.xml").write_text(header + track_xml + "\n")
    (ASSETS / "track8_layout.json").write_text(json.dumps(track_layout, indent=1) + "\n")
    # 8 30m All Fours: crawlers 1.1 m apart on the home straight (venue 08)
    crawl_xml, crawl_layout = compose_meet(skin, geoms, inertials, qdef, origins=venue_lane_origins(8, reference_lane=3),
                                           model="crawl8", park_offset=(0.0, -30.0, 0.0), crowd=True)
    (ASSETS / "scene_crawl8.xml").write_text(header + crawl_xml + "\n")
    (ASSETS / "crawl8_layout.json").write_text(json.dumps(crawl_layout, indent=1) + "\n")
    # 12 The 360 Turntable: 8 spin spots on the venue's ring, neighbours 0.75 m apart; the cube park line is clear of it
    turn_xml, turn_layout = compose_meet(skin, geoms, inertials, qdef, origins=venue_lane_origins(12, reference_lane=3),
                                         model="turntable8", crowd=True)
    (ASSETS / "scene_turntable8.xml").write_text(header + turn_xml + "\n")
    (ASSETS / "turntable8_layout.json").write_text(json.dumps(turn_layout, indent=1) + "\n")
    # 3 Deep Squat Endurance: 2 x 4 station grid (3 m x 4 m apart, no contact), flat floor, Rung S brain
    squat_xml, squat_layout = compose_meet(skin, geoms, inertials, qdef, origins=venue_lane_origins(3, reference_lane=3),
                                           model="squat8", park_offset=(0.0, -30.0, 0.0), crowd=True)
    (ASSETS / "scene_squat8.xml").write_text(header + squat_xml + "\n")
    (ASSETS / "squat8_layout.json").write_text(json.dumps(squat_layout, indent=1) + "\n")
    # 10 Crab Shuffle: athletes turned to face the event's left (they side-step to their right = down the course);
    # lanes are then 1.22 m apart along x, the steel rails on the lane lines are real (shin-height) obstacles
    crab_xml, crab_layout = compose_meet(skin, geoms, inertials, qdef, origins=venue_lane_origins(10, 3, 90.0),
                                         model="crab8", park_offset=(0.0, -30.0, 0.0), props=crab_rails(), crowd=True)
    (ASSETS / "scene_crab8.xml").write_text(header + crab_xml + "\n")
    (ASSETS / "crab8_layout.json").write_text(json.dumps(crab_layout, indent=1) + "\n")
    # 5 Gust Gauntlet: all 8 on ONE spring-mounted shaker floor (venue "floor", 0.1 m high, top = z 0), 2 x 4 grid 0.8 m
    # apart (SHAKER_FLOOR: 2.0 Hz, damping ratio 0.2 with everyone aboard; +-0.25 m travel)
    shaker8_xml, shaker8_layout = compose_meet(skin, geoms, inertials, qdef, origins=venue_lane_origins(5, 3),
                                               model="shaker8", park_offset=(0.0, -30.0, 0.0), shaker=shaker_floor(), crowd=True)
    (ASSETS / "scene_shaker8.xml").write_text(header + shaker8_xml + NL)
    (ASSETS / "shaker8_layout.json").write_text(json.dumps(shaker8_layout, indent=1) + NL)
    # 11 Slalom Sprint: 7 physical poles on every lane's centre line (venues.json "poles"), 1.4 m mirror-slalom lanes
    slalom_xml, slalom_layout = compose_meet(skin, geoms, inertials, qdef, origins=venue_lane_origins(11, 3),
                                             model="slalom8", park_offset=(0.0, -30.0, 0.0), props=slalom_poles(), crowd=True)
    (ASSETS / "scene_slalom8.xml").write_text(header + slalom_xml + "\n")
    (ASSETS / "slalom8_layout.json").write_text(json.dumps(slalom_layout, indent=1) + "\n")
    # 23 The Trench Crawl: 8 crawl lanes (1.22 m) under a 12 m ceiling on posts (trench_props), crawl brains
    trench_xml, trench_layout = compose_meet(skin, geoms, inertials, qdef, origins=venue_lane_origins(23, 3),
                                             model="trench8", park_offset=(0.0, -30.0, 0.0), props=trench_props(), crowd=True)
    (ASSETS / "scene_trench8.xml").write_text(header + trench_xml + "\n")
    (ASSETS / "trench8_layout.json").write_text(json.dumps(trench_layout, indent=1) + "\n")
    (ASSETS / f"scene_meet{N_LANES}.xml").write_text(header + meet_xml + "\n")
    (ASSETS / f"meet{N_LANES}_layout.json").write_text(json.dumps(meet_layout, indent=1) + "\n")

    m = mujoco.MjModel.from_xml_string(scene_xml)
    report = {
        "total_mass": float(sum(m.body_mass[m.body(s.name).id] for s in body_specs())),
        "uniform_density_geom_mass": geom_mass,
        "nq": m.nq, "nv": m.nv, "nu": m.nu, "nbody": m.nbody, "ngeom": m.ngeom,
        "sole_z_bind": skin.sole_z,
        "default_qpos": qdef.round(6).tolist(),
        "actuators": [m.actuator(i).name for i in range(m.nu)],
        "bodies": {s.name: {"mass": round(inertials[s.name]["mass"], 4),
                            "geoms": [{k: (v if isinstance(v, str) else np.round(np.asarray(v, float), 4).tolist())
                                       for k, v in g.items()} for g in geoms[s.name]]} for s in body_specs()},
    }
    (DERIVED / "body_report.json").write_text(json.dumps(report, indent=1))
    print(f"matt.xml + scene_matt.xml written. nq={m.nq} nv={m.nv} nu={m.nu} bodies={m.nbody} geoms={m.ngeom}")
    print(f"robot mass={report['total_mass']:.3f} kg (uniform-density geoms would give {geom_mass:.1f} kg)")
    print(f"default standing root height z={qdef[2]:.4f} m")
    mm = mujoco.MjModel.from_xml_string(meet_xml)
    print(f"scene_meet{N_LANES}.xml written. nq={mm.nq} nv={mm.nv} nu={mm.nu} bodies={mm.nbody} geoms={mm.ngeom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
