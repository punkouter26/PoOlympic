"""Phase Z7 — multi-body meets: compose an 8-lane event scene where every lane can hold a different athlete body
(MATT, zombie, …) from the generated per-body robot MJCFs (assets/<body>.xml) and their default keyframes
(assets/scene_<body>.xml). Same scene layout as build_mjcf.compose_meet (lane prefixes L<k>_, lane collision bits,
lane origins from the stadium venue, pedestals / shaker platforms / props, cube pool), so an all-MATT lineup compiles to
the identical model (checked by `--verify`).

Usage:  uv run python tools/compose_mixed.py pedestal8 matt,zombie,matt,zombie,matt,zombie,matt,zombie
        uv run python tools/compose_mixed.py --verify          (all-MATT pedestal8 == scene_pedestal8.xml)
        uv run python tools/compose_mixed.py track8 roster     (roster scene: MATT L<k>_ + zombie Z<k>_ in every lane)
        uv run python tools/compose_mixed.py --roster-all      (roster scenes for every event scene)
Roster scenes back the Unity menu: every lane holds every roster body (same lane collision bits, shared pedestal / props /
cubes); Unity's LaneLineup switches off the bodies not picked before MuJoCo compiles, leaving compose(scene, lineup) up to
name prefixes (checked by --verify).
Output: assets/scene_<scene>_<tag>.xml + <scene>_<tag>_layout.json   (tag = lineup, e.g. "mz" for alternating)
"""

from __future__ import annotations

import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_mjcf as B  # noqa: E402  (MATT profile: only the scene helpers / constants are used here)

ASSETS = ROOT / "assets"
ROSTER = ["matt", "zombie"]      # = Unity MeetLineup.Roster (lower case)
SCENES = {  # scene -> the build_mjcf.compose_meet arguments of that event scene
    "pedestal8": dict(event=1, pedestal_h=B.PEDESTAL_H),
    "track8": dict(event=8, park_offset=(0.0, -30.0, 0.0)),
    "turntable8": dict(event=12),
    "crab8": dict(event=10, yaw=90.0, park_offset=(0.0, -30.0, 0.0), props=B.crab_rails),
    "shaker8": dict(event=5, park_offset=(0.0, -30.0, 0.0), shaker=B.SHAKER),
    "slalom8": dict(event=11, park_offset=(0.0, -30.0, 0.0), props=B.slalom_poles),
}


def shift_bits(v: int, k: int) -> int:
    """Lane-0 collision bits -> lane k: upper-body byte (bit 0) and leg byte (bit 8) both move by k."""
    return ((v & 0xFF) << k) | (((v >> 8) & 0xFF) << (k + 8))


def body_parts(body: str):
    """(pelvis element, actuator elements, contact excludes, athlete default qpos) of a generated robot MJCF."""
    robot = ET.parse(ASSETS / f"{body}.xml").getroot()
    pelvis = robot.find("worldbody/body")
    acts = list(robot.find("actuator"))
    excl = list(robot.find("contact")) if robot.find("contact") is not None else []
    scene = mujoco.MjModel.from_xml_path(str(ASSETS / f"scene_{body}.xml"))
    root = scene.joint("root").id
    n = sum(1 for j in range(scene.njnt) if not scene.joint(j).name.startswith("cube"))
    last = [j for j in range(scene.njnt) if not scene.joint(j).name.startswith("cube")][-1]
    nq = scene.jnt_qposadr[last] + 1 - scene.jnt_qposadr[root]
    qdef = scene.key("default").qpos[scene.jnt_qposadr[root]: scene.jnt_qposadr[root] + nq].copy()
    return pelvis, acts, excl, qdef


def athlete_prefix(lane: int, body: str, roster: bool) -> str:
    """L<k>_ for MATT and for any body in a one-body-per-lane scene; <B><k>_ (Z<k>_ …) for the other roster bodies."""
    return B.lane_prefix(lane) if not roster or body == "matt" else f"{body[0].upper()}{lane}_"


def compose(scene: str, lineup: list) -> tuple[str, dict]:
    """lineup: one body per lane, or a list of bodies per lane (roster scene)."""
    sc = SCENES[scene]
    ped_h, shaker = sc.get("pedestal_h", 0.0), sc.get("shaker")
    props = sc["props"]() if "props" in sc else []
    park_offset = np.asarray(sc.get("park_offset", (0.0, 0.0, 0.0)), float)
    origins = B.venue_lane_origins(sc["event"], 3, sc.get("yaw", 0.0))
    root = ET.Element("mujoco", {"model": f"{scene}_mixed"})
    B.option_block(root)
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"name": "sun", "pos": "0 0 5", "dir": "0 0 -1", "directional": "true"})
    ground = {"name": "ground", "type": "plane", "size": "0 0 0.05", "contype": str(B.ALL_BITS),
              "conaffinity": str(B.ALL_BITS), "condim": "3", "friction": B.vec(B.GROUND_FRICTION)}
    drop = ped_h or (shaker["h"] if shaker else 0.0)
    if drop:
        ground["pos"] = B.vec([0, 0, -drop])
    ET.SubElement(wb, "geom", ground)
    for pr in props:
        ET.SubElement(wb, "geom", {"name": pr["name"], "type": "box", "pos": B.vec(pr["pos"]), "size": B.vec(pr["size"]),
                                   "contype": str(B.ALL_BITS), "conaffinity": str(B.ALL_BITS), "condim": "3",
                                   "friction": B.vec(B.GROUND_FRICTION)})
    contact = ET.Element("contact")
    actuator = ET.Element("actuator")
    key_qpos, lanes = [], []
    for k, bodies in enumerate(lineup):
        roster = not isinstance(bodies, str)          # roster scene: every roster body in this lane
        bodies = list(bodies) if roster else [bodies]
        p, o = B.lane_prefix(k), np.asarray(origins[k], float)
        if ped_h:
            ET.SubElement(wb, "geom", {"name": p + "pedestal", "type": "box", "pos": B.vec(o + [0, 0, -ped_h / 2]),
                                       "size": B.vec([B.PEDESTAL_HALF, B.PEDESTAL_HALF, ped_h / 2]),
                                       "contype": str(B.ALL_BITS), "conaffinity": str(B.ALL_BITS), "condim": "3",
                                       "friction": B.vec(B.GROUND_FRICTION)})
        if shaker:                                   # = build_mjcf.compose_meet's spring-mounted platform
            sh = shaker
            plat = ET.SubElement(wb, "body", {"name": p + "shaker", "pos": B.vec(o + [0, 0, -sh["h"] / 2])})
            half = [sh["half"], sh["half"], sh["h"] / 2]
            ET.SubElement(plat, "inertial", {"pos": "0 0 0", "mass": B.f(sh["mass"]), "diaginertia": B.vec(
                [sh["mass"] * (half[1] ** 2 + half[2] ** 2) / 3, sh["mass"] * (half[0] ** 2 + half[2] ** 2) / 3,
                 sh["mass"] * (half[0] ** 2 + half[1] ** 2) / 3])})
            for ax, axis in (("x", "1 0 0"), ("y", "0 1 0")):
                ET.SubElement(plat, "joint", {"name": f"{p}shaker_{ax}", "type": "slide", "axis": axis,
                                              "stiffness": B.f(sh["stiffness"]), "damping": B.f(sh["damping"]),
                                              "limited": "true", "range": B.vec([-sh["range"], sh["range"]])})
            ET.SubElement(plat, "geom", {"name": p + "shaker", "type": "box", "size": B.vec(half),
                                         "contype": str(B.ALL_BITS), "conaffinity": str(B.ALL_BITS), "condim": "3",
                                         "friction": B.vec(B.GROUND_FRICTION)})
            key_qpos.append(np.zeros(2))
        for body in bodies:
            p = athlete_prefix(k, body, roster)
            pelvis, acts, excl, qdef = body_parts(body)
            pelvis = copy.deepcopy(pelvis)
            for el in pelvis.iter():
                if "name" in el.attrib:
                    el.set("name", p + el.get("name"))
                if el.tag == "geom":
                    el.set("contype", str(shift_bits(int(el.get("contype")), k)))
                    el.set("conaffinity", str(shift_bits(int(el.get("conaffinity")), k)))
            pelvis.set("pos", B.vec(np.array([float(x) for x in pelvis.get("pos").split()]) + o))
            wb.append(pelvis)
            for e in excl:
                ET.SubElement(contact, "exclude", {"body1": p + e.get("body1"), "body2": p + e.get("body2")})
            for a in acts:
                a = copy.deepcopy(a)
                a.set("name", p + a.get("name"))
                a.set("joint", p + a.get("joint"))
                actuator.append(a)
            q = qdef.copy()
            q[0:3] += o
            key_qpos.append(q)
            lanes.append({"lane": k, "prefix": p, "body": body, "origin": o.tolist(), "cubes": [2 * k, 2 * k + 1]})
    inertia = B.CUBE_MASS * (2 * B.CUBE_HALF) ** 2 / 6.0
    n_cubes = B.N_CUBES_MEET
    for i in range(n_cubes):
        park = B.cube_park_pos(i) - [0, 0, drop] + park_offset
        cb = ET.SubElement(wb, "body", {"name": f"cube{i}", "pos": B.vec(park)})
        ET.SubElement(cb, "inertial", {"pos": "0 0 0", "mass": B.f(B.CUBE_MASS), "diaginertia": B.vec([inertia] * 3)})
        ET.SubElement(cb, "freejoint", {"name": f"cube{i}_free"})
        ET.SubElement(cb, "geom", {"name": f"cube{i}_geom", "type": "box", "size": B.vec([B.CUBE_HALF] * 3),
                                   "contype": str(B.ALL_BITS), "conaffinity": str(B.ALL_BITS), "condim": "3",
                                   "friction": f"{B.f(B.CUBE_FRICTION)} 0.005 0.0001"})
        key_qpos.append(np.concatenate([park, [1, 0, 0, 0]]))
    root.append(contact)
    root.append(actuator)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", {"name": "default", "qpos": B.vec(np.concatenate(key_qpos))})
    layout = {"n_lanes": len(lineup), "lane_width": B.LANE_WIDTH, "n_cubes": n_cubes, "lanes": lanes, "pedestal_h": ped_h,
              "lineup": lineup,
              "note": "lane k: names prefixed L<k>_, athlete body per lane (assets/<body>.xml), pelvis shifted by origin"}
    if shaker:
        layout["shaker"] = dict(shaker)
    if props:
        layout["props"] = [{"name": pr["name"], "pos": np.asarray(pr["pos"], float).tolist(), "size": list(pr["size"])}
                           for pr in props]
    return B.indent(root), layout


def tag_of(lineup: list) -> str:
    if not isinstance(lineup[0], str):
        return "roster"
    return "".join(b[0] for b in lineup) if len(set(lineup)) > 1 else lineup[0]


def verify() -> int:
    """Every all-MATT composition must compile to the model of the existing scene_<scene>.xml (per-lane fingerprints,
    world props, keyframe, sizes)."""
    from poolympic.fingerprint import canonical_bytes, fingerprint
    ok_all = True
    for scene in SCENES:
        xml, _ = compose(scene, ["matt"] * 8)
        a = mujoco.MjModel.from_xml_string(xml)
        b = mujoco.MjModel.from_xml_path(str(ASSETS / f"scene_{scene}.xml"))
        same = all(canonical_bytes(fingerprint(a, f"L{k}_", f"cube{2 * k}")) == canonical_bytes(fingerprint(b, f"L{k}_", f"cube{2 * k}"))
                   for k in range(8))
        same &= np.array_equal(a.key("default").qpos, b.key("default").qpos) and (a.nq, a.nu, a.ngeom) == (b.nq, b.nu, b.ngeom)
        world = lambda m: sorted((m.geom(g).name, tuple(np.round(m.geom_pos[g], 9)), tuple(np.round(m.geom_size[g], 9)))
                                 for g in range(m.ngeom) if m.geom_bodyid[g] == 0)
        same &= world(a) == world(b)
        print(f"all-MATT compose == scene_{scene}.xml:", "PASS" if same else "FAIL")
        ok_all &= same
    return 0 if ok_all else 1


def verify_roster() -> int:
    """Every athlete of every roster scene == the same body in that lane of a one-body-per-lane composition
    (per-lane fingerprint incl. its cube slot): switching the other bodies off leaves exactly compose(scene, lineup)."""
    from poolympic.fingerprint import canonical_bytes, fingerprint
    ok_all = True
    for scene in SCENES:
        xml, _ = compose(scene, [list(ROSTER)] * 8)
        r = mujoco.MjModel.from_xml_string(xml)
        for body in ROSTER:
            m = mujoco.MjModel.from_xml_string(compose(scene, [[body]] * 8)[0])   # same prefixes, one body per lane
            same = all(canonical_bytes(fingerprint(r, athlete_prefix(k, body, True), f"cube{2 * k}"))
                       == canonical_bytes(fingerprint(m, athlete_prefix(k, body, True), f"cube{2 * k}")) for k in range(8))
            print(f"roster {scene} {body}:", "PASS" if same else "FAIL")
            ok_all &= same
    return 0 if ok_all else 1


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--verify":
        return verify() | verify_roster()
    if argv and argv[0] == "--roster-all":
        return max(main([sc, "roster"]) for sc in SCENES)
    scene = argv[0]
    lineup = [list(ROSTER)] * 8 if argv[1] == "roster" else argv[1].split(",")
    assert len(lineup) == 8, "8 lanes"
    xml, layout = compose(scene, lineup)
    tag = tag_of(lineup)
    header = f"<!-- GENERATED by training/tools/compose_mixed.py ({scene}, lineup {tag_of(lineup) if tag_of(lineup) == 'roster' else ','.join(lineup)}). Do not edit. -->\n"
    out = ASSETS / f"scene_{scene}_{tag}.xml"
    out.write_text(header + xml + "\n")
    (ASSETS / f"{scene}_{tag}_layout.json").write_text(json.dumps(layout, indent=1) + "\n")
    m = mujoco.MjModel.from_xml_path(str(out))
    print(f"{out.name}: nq={m.nq} nu={m.nu} ngeom={m.ngeom}  lineup {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
