"""Phase Z7 — multi-body meets: compose an 8-lane event scene where every lane can hold a different athlete body
(MATT, zombie, …) from the generated per-body robot MJCFs (assets/<body>.xml) and their default keyframes
(assets/scene_<body>.xml). Same scene layout as build_mjcf.compose_meet (lane prefixes L<k>_, lane collision bits,
lane origins from the stadium venue, pedestals / shaker platforms / props, cube pool), so an all-MATT lineup compiles to
the identical model (checked by `--verify`).

Usage:  uv run python tools/compose_mixed.py pedestal8 matt,zombie,matt,zombie,matt,zombie,matt,zombie
        uv run python tools/compose_mixed.py --verify          (all-MATT pedestal8 == scene_pedestal8.xml)
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
SCENES = {  # scene -> (venue event, reference lane, extra yaw, pedestal_h)
    "pedestal8": (1, 3, 0.0, B.PEDESTAL_H),
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


def compose(scene: str, lineup: list[str]) -> tuple[str, dict]:
    event, ref, yaw, ped_h = SCENES[scene]
    origins = B.venue_lane_origins(event, ref, yaw)
    root = ET.Element("mujoco", {"model": f"{scene}_mixed"})
    B.option_block(root)
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"name": "sun", "pos": "0 0 5", "dir": "0 0 -1", "directional": "true"})
    ground = {"name": "ground", "type": "plane", "size": "0 0 0.05", "contype": str(B.ALL_BITS),
              "conaffinity": str(B.ALL_BITS), "condim": "3", "friction": B.vec(B.GROUND_FRICTION)}
    if ped_h:
        ground["pos"] = B.vec([0, 0, -ped_h])
    ET.SubElement(wb, "geom", ground)
    contact = ET.Element("contact")
    actuator = ET.Element("actuator")
    key_qpos, lanes = [], []
    for k, body in enumerate(lineup):
        p, o = B.lane_prefix(k), np.asarray(origins[k], float)
        if ped_h:
            ET.SubElement(wb, "geom", {"name": p + "pedestal", "type": "box", "pos": B.vec(o + [0, 0, -ped_h / 2]),
                                       "size": B.vec([B.PEDESTAL_HALF, B.PEDESTAL_HALF, ped_h / 2]),
                                       "contype": str(B.ALL_BITS), "conaffinity": str(B.ALL_BITS), "condim": "3",
                                       "friction": B.vec(B.GROUND_FRICTION)})
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
        park = B.cube_park_pos(i) - [0, 0, ped_h]
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
    return B.indent(root), layout


def tag_of(lineup: list[str]) -> str:
    return "".join(b[0] for b in lineup) if len(set(lineup)) > 1 else lineup[0]


def verify() -> int:
    """An all-MATT pedestal8 must compile to the model of the existing scene_pedestal8.xml."""
    from poolympic.fingerprint import canonical_bytes, fingerprint
    xml, _ = compose("pedestal8", ["matt"] * 8)
    a = mujoco.MjModel.from_xml_string(xml)
    b = mujoco.MjModel.from_xml_path(str(ASSETS / "scene_pedestal8.xml"))
    same = all(canonical_bytes(fingerprint(a, f"L{k}_", f"cube{2 * k}")) == canonical_bytes(fingerprint(b, f"L{k}_", f"cube{2 * k}"))
               for k in range(8))
    same &= np.array_equal(a.key("default").qpos, b.key("default").qpos) and a.nq == b.nq and a.nu == b.nu
    print("all-MATT compose == scene_pedestal8.xml:", "PASS" if same else "FAIL")
    return 0 if same else 1


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--verify":
        return verify()
    scene, lineup = argv[0], argv[1].split(",")
    assert len(lineup) == 8, "8 lanes"
    xml, layout = compose(scene, lineup)
    tag = tag_of(lineup)
    header = f"<!-- GENERATED by training/tools/compose_mixed.py ({scene}, lineup {','.join(lineup)}). Do not edit. -->\n"
    out = ASSETS / f"scene_{scene}_{tag}.xml"
    out.write_text(header + xml + "\n")
    (ASSETS / f"{scene}_{tag}_layout.json").write_text(json.dumps(layout, indent=1) + "\n")
    m = mujoco.MjModel.from_xml_path(str(out))
    print(f"{out.name}: nq={m.nq} nu={m.nu} ngeom={m.ngeom}  lineup {lineup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
