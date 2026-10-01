"""AGENTS.md "Self-collision" audit for every athlete body (assets/scene_<body>.xml), CPU MuJoCo.

  shapes   every athlete collider is a capsule / box / sphere (no mesh geoms)
  pairs    every pair of body parts collides, except parent-child pairs and pairs that overlap in the default standing
           pose (those may be filtered or <exclude>d); anything else that cannot collide is a violation
  rest     no self-contact in the T-pose (qpos = 0), the default stance, or a normal arm and leg swing (legs: hip flexion
           -25..+40 deg with the knee bending on the forward swing; arms: +-40 deg about the shoulder flexion axis, opposite
           to the legs; 21 phases of the stride)
Usage: uv run python tools/check_self_collision.py [body ...]      (default: matt mattbio zombie grandma)
Writes parity/self_collision.json; exit code 1 on any violation.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
BODIES = ["matt", "mattbio", "zombie", "grandma"]
SIMPLE = {mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere"}


def athlete_geoms(m: mujoco.MjModel) -> list[int]:
    pelvis = m.body("pelvis").id
    return [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == pelvis]


def can_collide(m, a: int, b: int, excl: set) -> bool:
    bits = bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
    return bits and tuple(sorted((int(m.geom_bodyid[a]), int(m.geom_bodyid[b])))) not in excl


def self_contacts(m, d, geoms: set) -> set[tuple[str, str]]:
    """Penetrating athlete-athlete geom pairs at the current qpos (root lifted 2 m: no ground contact)."""
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    out = set()
    for c in d.contact[: d.ncon]:
        if int(c.geom1) in geoms and int(c.geom2) in geoms and c.dist < 0:
            out.add(tuple(sorted((m.geom(int(c.geom1)).name, m.geom(int(c.geom2)).name))))
    return out


def overlap_all_pairs(m, key_qpos: np.ndarray, pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Which of `pairs` geometrically overlap at key_qpos, whatever the collision filter says (mj_geomDistance)."""
    d = mujoco.MjData(m)
    d.qpos[:] = key_qpos
    mujoco.mj_kinematics(m, d)
    out = set()
    fromto = np.zeros(6)
    for a, b in pairs:
        if mujoco.mj_geomDistance(m, d, a, b, 0.5, fromto) < 0:
            out.add((a, b))
    return out


def swing_poses(m, key_qpos: np.ndarray):
    """A normal stride about the default stance: (label, qpos) for 21 phases."""
    def adr(name):
        return m.jnt_qposadr[m.joint(name).id]
    for i in range(21):
        ph = 2 * math.pi * i / 21
        q = key_qpos.copy()
        s = math.sin(ph)
        for side, sign in (("l", 1.0), ("r", -1.0)):
            leg = sign * s                                         # +1 = this leg fully forward
            q[adr(f"hip_flex_{side}")] += math.radians(32.5 * leg + 7.5)      # -25 .. +40 deg about the stance
            q[adr(f"knee_{side}")] += math.radians(45.0 * max(0.0, leg))      # the knee bends on the forward swing
            q[adr(f"shoulder_flex_{side}")] += math.radians(-40.0 * leg)      # arms swing opposite to the legs
        yield f"swing phase {i}/21", q


def check(body: str) -> dict:
    m = mujoco.MjModel.from_xml_path(str(ASSETS / f"scene_{body}.xml"))
    d = mujoco.MjData(m)
    geoms = athlete_geoms(m)
    gset = set(geoms)
    excl = {tuple(sorted((int(s) >> 16, int(s) & 0xFFFF))) for s in m.exclude_signature}
    key = m.key("default").qpos.copy()
    root = m.jnt_qposadr[m.joint("root").id]
    key[root + 2] += 2.0                                            # off the ground: only self-contacts remain
    res = {"body": body, "parts": len(geoms)}

    # shapes
    res["shape_types"] = sorted({SIMPLE.get(int(m.geom_type[g]), f"type{int(m.geom_type[g])}") for g in geoms})
    res["non_simple"] = [m.geom(g).name for g in geoms if int(m.geom_type[g]) not in SIMPLE]

    # pairs
    pairs = [(a, b) for i, a in enumerate(geoms) for b in geoms[i + 1:] if m.geom_bodyid[a] != m.geom_bodyid[b]]
    parent_child = {(a, b) for a, b in pairs
                    if m.body_parentid[m.geom_bodyid[a]] == m.geom_bodyid[b] or m.body_parentid[m.geom_bodyid[b]] == m.geom_bodyid[a]}
    off = [(a, b) for a, b in pairs if (a, b) not in parent_child and not can_collide(m, a, b, excl)]
    touching = overlap_all_pairs(m, key, off)
    res["pairs"] = len(pairs)
    res["parent_child"] = len(parent_child)
    res["colliding"] = len(pairs) - len(parent_child) - len(off)
    res["off_overlapping_at_rest"] = sorted(f"{m.geom(a).name} / {m.geom(b).name}" for a, b in off if (a, b) in touching)
    res["off_violations"] = sorted(f"{m.geom(a).name} / {m.geom(b).name}" for a, b in off if (a, b) not in touching)

    # rest poses
    rest = {}
    d.qpos[:] = m.qpos0
    d.qpos[root + 2] += 2.0
    rest["T-pose"] = sorted(self_contacts(m, d, gset))
    d.qpos[:] = key
    rest["default stance"] = sorted(self_contacts(m, d, gset))
    swing = set()
    for _, q in swing_poses(m, key):
        d.qpos[:] = q
        swing |= self_contacts(m, d, gset)
    rest["arm and leg swing"] = sorted(swing)
    res["rest_contacts"] = {k: [" / ".join(p) for p in v] for k, v in rest.items()}
    res["ok"] = not res["non_simple"] and not res["off_violations"] and not any(rest.values())
    return res


def main(argv: list[str]) -> int:
    report = [check(b) for b in (argv or BODIES)]
    for r in report:
        print(f"{r['body']:8s} {'PASS' if r['ok'] else 'FAIL'}  shapes {'/'.join(r['shape_types'])}  "
              f"pairs {r['pairs']}: {r['colliding']} collide, {r['parent_child']} parent-child, "
              f"{len(r['off_overlapping_at_rest'])} off (overlap at rest), {len(r['off_violations'])} off without a reason")
        if r["off_overlapping_at_rest"]:
            print("   off, overlapping at rest:", ", ".join(r["off_overlapping_at_rest"]))
        if r["off_violations"]:
            v = r["off_violations"]
            print(f"   VIOLATION, cannot collide ({len(v)}):", ", ".join(v[:8]) + (" …" if len(v) > 8 else ""))
        for pose, c in r["rest_contacts"].items():
            if c:
                print(f"   VIOLATION, self-contact in the {pose}:", ", ".join(c[:8]) + (" …" if len(c) > 8 else ""))
    out = ROOT.parent / "parity" / "self_collision.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")
    return 0 if all(r["ok"] for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
