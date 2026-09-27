"""A5/A6 — Derive MATT's MuJoCo physics body from the extracted skeleton + skin (DESIGN.md §2).

Inputs : assets/derived/skeleton_matt.json, assets/derived/skin_matt.npz  (from extract_skeleton.py)
Outputs: assets/matt.xml         robot only (for mjlab)
         assets/scene_matt.xml   flattened: options + ground + robot + cube pool + keyframe (CPU eval, Unity import)
         assets/scene_meet8.xml  8 lane-isolated athletes (names prefixed L<k>_) + 16-cube pool (G6, events)
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
ASSETS = ROOT / "assets"
DERIVED = ASSETS / "derived"

TOTAL_MASS = 80.0
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


def lane_bits(lane: int, leg: bool) -> tuple[int, int]:
    """(contype, conaffinity). Upper body: collides with ground/cubes only. Legs: also leg<->leg within lane."""
    if leg:
        b = 1 << (lane + 8)
        return b, b
    return 1 << lane, 0


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


GAINS = {  # kp (Nm/rad), kv (Nm s/rad), force cap (Nm), armature
    "hip": (300.0, 30.0, 280.0, 0.02),
    "knee": (300.0, 30.0, 280.0, 0.02),
    "ankle": (200.0, 20.0, 220.0, 0.01),
    "abdomen": (300.0, 30.0, 200.0, 0.02),
    "shoulder": (60.0, 6.0, 80.0, 0.01),
    "elbow": (50.0, 5.0, 70.0, 0.01),
}
TOE_PASSIVE = dict(stiffness=20.0, damping=1.0, armature=0.005)

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
        sk = json.loads((DERIVED / "skeleton_matt.json").read_text())
        self.joint_pos = {j["name"]: np.array(j["pos"]) for j in sk["joints"]}
        z = np.load(DERIVED / "skin_matt.npz")
        names = list(z["joint_names"])
        mesh_names = list(z["mesh_names"])
        keep = np.array([("hair" not in mesh_names[m]) for m in z["mesh"]])  # hair is visual only
        self.verts = z["verts"][keep]
        self.dom = np.array(names)[z["dominant"][keep]]
        self.sole_z = float(self.verts[np.isin(self.dom, ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]), 2].min())

    def of(self, bones: list[str]) -> np.ndarray:
        return self.verts[np.isin(self.dom, bones)]


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
    g["pelvis"] = [fit_trunk_capsule(skin.of(["Hips"]), J["Hips"][2] - 0.08, J["Spine"][2] + 0.02)]
    g["torso"] = [fit_trunk_capsule(skin.of(["Spine", "Spine1"]))]
    g["chest"] = [fit_trunk_capsule(skin.of(["Spine2"]), J["Spine2"][2], J["LeftArm"][2] - 0.03)]
    neck = fit_limb_capsule(J["Neck"], J["Head"], skin.of(["Neck"]), pct=60, shrink=0.0)
    head = fit_sphere(skin.of(["Head"]))
    g["head"] = [neck, head]
    # Left limbs (mirrored later)
    g["upper_arm_l"] = [fit_limb_capsule(J["LeftArm"], J["LeftForeArm"], skin.of(["LeftArm"]))]
    g["forearm_l"] = [fit_limb_capsule(J["LeftForeArm"], J["LeftHandMiddle2"], skin.of(["LeftForeArm"]), pct=70)]
    g["thigh_l"] = [fit_limb_capsule(J["LeftUpLeg"], J["LeftLeg"], skin.of(["LeftUpLeg"]), pct=60)]
    g["shin_l"] = [fit_limb_capsule(J["LeftLeg"], J["LeftFoot"], skin.of(["LeftLeg"]), pct=60)]
    foot_v = skin.of(["LeftFoot", "LeftToeBase"])
    toe_x = J["LeftToeBase"][0]
    g["foot_l"] = [fit_box(foot_v, float(np.percentile(foot_v[:, 0], 0.5)), toe_x, skin.sole_z, 0.06)]
    g["toe_l"] = [fit_box(foot_v[foot_v[:, 0] > toe_x - 0.01], toe_x, float(np.percentile(foot_v[:, 0], 99.5)),
                          skin.sole_z, 0.035)]
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


def build_robot(skin: Skin, geoms, inertials=None, lane: int = LANE) -> tuple[ET.Element, list, list]:
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
        ct, ca = lane_bits(lane, s.leg)
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
        ET.SubElement(act, "position", {"name": jname, "joint": jname, "kp": f(kp), "kv": f(kv),
                                        "forcelimited": "true", "forcerange": f"{f(-cap)} {f(cap)}"})


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


def compose_model(skin, geoms, inertials, with_scene: bool, n_cubes: int, default_qpos=None) -> str:
    root = ET.Element("mujoco", {"model": "matt_scene" if with_scene else "matt"})
    option_block(root)
    wb = ET.SubElement(root, "worldbody")
    if with_scene:
        ET.SubElement(wb, "light", {"name": "sun", "pos": "0 0 5", "dir": "0 0 -1", "directional": "true"})
        ET.SubElement(wb, "geom", {"name": "ground", "type": "plane", "size": "0 0 0.05", "contype": str(ALL_BITS),
                                   "conaffinity": str(ALL_BITS), "condim": "3", "friction": vec(GROUND_FRICTION)})
    pelvis, actuators, joints = build_robot(skin, geoms, inertials)
    # Declare actuators in joint-tree (depth-first) order == MuJoCo joint id order. mjlab's XmlActuator pairs joint
    # targets (joint order) with ctrl slots (declaration order); any other order silently scrambles the action wiring.
    tree_order = [j.get("name") for j in pelvis.iter("joint")]
    actuators = sorted(actuators, key=lambda a: tree_order.index(a[0]))
    wb.append(pelvis)
    if with_scene:
        inertia = CUBE_MASS * (2 * CUBE_HALF) ** 2 / 6.0
        for i in range(n_cubes):
            cb = ET.SubElement(wb, "body", {"name": f"cube{i}", "pos": vec(cube_park_pos(i))})
            ET.SubElement(cb, "inertial", {"pos": "0 0 0", "mass": f(CUBE_MASS), "diaginertia": vec([inertia] * 3)})
            ET.SubElement(cb, "freejoint", {"name": f"cube{i}_free"})
            ET.SubElement(cb, "geom", {"name": f"cube{i}_geom", "type": "box", "size": vec([CUBE_HALF] * 3),
                                       "contype": str(ALL_BITS), "conaffinity": str(ALL_BITS), "condim": "3",
                                       "friction": f"{f(CUBE_FRICTION)} 0.005 0.0001"})
    # Inner thighs overlap at rest (hip pivots 0.172 m apart, thigh radius ~0.089 m): exclude that pair only;
    # shin/foot/toe still collide across legs (leg-crossing detection).
    contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", {"body1": "thigh_l", "body2": "thigh_r"})
    actuator_block(root, actuators)
    if with_scene and default_qpos is not None:
        kf = ET.SubElement(root, "keyframe")
        ET.SubElement(kf, "key", {"name": "default", "qpos": vec(default_qpos)})
    return indent(root)


def compose_meet(skin, geoms, inertials, default_qpos: np.ndarray, n_lanes: int = N_LANES,
                 n_cubes: int = N_CUBES_MEET) -> tuple[str, dict]:
    """Multi-athlete scene (G6 / events): lane k = the training athlete with every name prefixed `L<k>_`, its own
    collision bits (lane isolation) and its pelvis shifted to lane_origin(k). Ground + cube pool as in the solo
    scene. Returns (xml, layout) — layout is the lane table Unity and the evaluators share."""
    root = ET.Element("mujoco", {"model": f"meet{n_lanes}"})
    option_block(root)
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"name": "sun", "pos": "0 0 5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "ground", "type": "plane", "size": "0 0 0.05", "contype": str(ALL_BITS),
                               "conaffinity": str(ALL_BITS), "condim": "3", "friction": vec(GROUND_FRICTION)})
    contact = ET.Element("contact")
    all_actuators, key_qpos, lanes = [], [], []
    for k in range(n_lanes):
        p, o = lane_prefix(k), lane_origin(k)
        pelvis, actuators, _ = build_robot(skin, geoms, inertials, lane=k)
        tree_order = [j.get("name") for j in pelvis.iter("joint")]
        actuators = sorted(actuators, key=lambda a: tree_order.index(a[0]))
        for el in pelvis.iter():
            if "name" in el.attrib:
                el.set("name", p + el.get("name"))
        pelvis.set("pos", vec(np.array([float(x) for x in pelvis.get("pos").split()]) + o))
        wb.append(pelvis)
        ET.SubElement(contact, "exclude", {"body1": p + "thigh_l", "body2": p + "thigh_r"})
        all_actuators += [(p + j, g) for j, g in actuators]
        q = default_qpos.copy()
        q[0:3] += o
        key_qpos.append(q)
        lanes.append({"lane": k, "prefix": p, "origin": o.tolist(), "cubes": [2 * k, 2 * k + 1]})
    inertia = CUBE_MASS * (2 * CUBE_HALF) ** 2 / 6.0
    for i in range(n_cubes):
        cb = ET.SubElement(wb, "body", {"name": f"cube{i}", "pos": vec(cube_park_pos(i))})
        ET.SubElement(cb, "inertial", {"pos": "0 0 0", "mass": f(CUBE_MASS), "diaginertia": vec([inertia] * 3)})
        ET.SubElement(cb, "freejoint", {"name": f"cube{i}_free"})
        ET.SubElement(cb, "geom", {"name": f"cube{i}_geom", "type": "box", "size": vec([CUBE_HALF] * 3),
                                   "contype": str(ALL_BITS), "conaffinity": str(ALL_BITS), "condim": "3",
                                   "friction": f"{f(CUBE_FRICTION)} 0.005 0.0001"})
        key_qpos.append(np.concatenate([cube_park_pos(i), [1, 0, 0, 0]]))
    root.append(contact)
    actuator_block(root, all_actuators)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", {"name": "default", "qpos": vec(np.concatenate(key_qpos))})
    layout = {"n_lanes": n_lanes, "lane_width": LANE_WIDTH, "n_cubes": n_cubes, "lanes": lanes,
              "note": "lane k: names prefixed L<k>_, pelvis shifted by origin; solo-scene cube i -> meet cube cubes[i]"}
    return indent(root), layout


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


def main() -> int:
    skin = Skin()
    geoms = fit_geoms(skin)
    inertials, geom_mass = compute_inertials(skin, geoms)

    for s in body_specs():
        for js in s.joints:
            JOINT_DEFAULTS[f"{js.name}_{s.side}" if s.side else js.name] = js.default_deg

    qdef = default_pose_qpos(skin, geoms, inertials)
    robot_xml = compose_model(skin, geoms, inertials, with_scene=False, n_cubes=0)
    scene_xml = compose_model(skin, geoms, inertials, with_scene=True, n_cubes=N_CUBES_TRAINING, default_qpos=np.concatenate(
        [qdef, np.concatenate([np.concatenate([cube_park_pos(i), [1, 0, 0, 0]]) for i in range(N_CUBES_TRAINING)])]))
    header = "<!-- GENERATED by training/tools/build_mjcf.py from SourceArt/test_MATT_Avaturn.glb. Do not edit. -->\n"
    (ASSETS / "cube.xml").write_text(header + cube_entity_xml() + "\n")
    (ASSETS / "matt.xml").write_text(header + robot_xml + "\n")
    (ASSETS / "scene_matt.xml").write_text(header + scene_xml + "\n")
    meet_xml, meet_layout = compose_meet(skin, geoms, inertials, qdef)
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
