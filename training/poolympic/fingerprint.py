"""A10 — Model fingerprint (parity gate G0).

Dumps the physics-relevant fields of a compiled mjModel, keyed by element name, so the Unity-side dump
(same schema, produced by C# from the plug-in's mjModel) can be compared field by field.
Scope: global options, ground, one athlete (lane prefix stripped) and one pooled cube.

Usage: uv run python -m poolympic.fingerprint          -> parity/fingerprint_python.json (+ .sha256)
       uv run python -m poolympic.fingerprint a.json b.json   -> compare two dumps (G0)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT.parent / "parity"
FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-6


def _l(x) -> list:
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def fingerprint(m: mujoco.MjModel, athlete_prefix: str = "", cube_name: str = "cube0") -> dict:
    o = m.opt
    fp: dict = {
        "schema": 1,
        "option": {
            "timestep": float(o.timestep), "gravity": _l(o.gravity), "integrator": int(o.integrator),
            "cone": int(o.cone), "jacobian": int(o.jacobian), "solver": int(o.solver),
            "iterations": int(o.iterations), "tolerance": float(o.tolerance),
            "ls_iterations": int(o.ls_iterations), "ls_tolerance": float(o.ls_tolerance),
            "impratio": float(o.impratio), "disableflags": int(o.disableflags), "enableflags": int(o.enableflags),
        },
        "bodies": {}, "joints": {}, "geoms": {}, "actuators": {},
    }

    def keep(name: str) -> str | None:
        if athlete_prefix:
            return name[len(athlete_prefix):] if name.startswith(athlete_prefix) else None
        return name

    for b in range(1, m.nbody):
        name = m.body(b).name
        is_cube = name == cube_name
        key = "cube" if is_cube else keep(name)
        if key is None or (name.startswith("cube") and not is_cube):
            continue
        parent = m.body(m.body_parentid[b]).name
        fp["bodies"][key] = {
            "parent": "world" if m.body_parentid[b] == 0 else (keep(parent) or parent),
            "pos": _l(m.body_pos[b]), "quat": _l(m.body_quat[b]), "mass": float(m.body_mass[b]),
            "inertia": _l(m.body_inertia[b]), "ipos": _l(m.body_ipos[b]), "iquat": _l(m.body_iquat[b]),
            "gravcomp": float(m.body_gravcomp[b]),
        }
        if is_cube:
            fp["bodies"][key]["pos"] = None  # parked position differs per pool slot / scene
    for j in range(m.njnt):
        body = m.body(m.jnt_bodyid[j]).name
        if body.startswith("cube"):
            continue
        key = keep(m.joint(j).name)
        if key is None:
            continue
        d = m.jnt_dofadr[j]
        fp["joints"][key] = {
            "type": int(m.jnt_type[j]), "body": keep(body), "pos": _l(m.jnt_pos[j]), "axis": _l(m.jnt_axis[j]),
            "limited": int(m.jnt_limited[j]), "range": _l(m.jnt_range[j]), "stiffness": float(m.jnt_stiffness[j]),
            "springref": float(m.qpos_spring[m.jnt_qposadr[j]]) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else None,
            "armature": float(m.dof_armature[d]), "damping": float(m.dof_damping[d]),
            "frictionloss": float(m.dof_frictionloss[d]),
            "solref": _l(m.jnt_solref[j]), "solimp": _l(m.jnt_solimp[j]),
        }
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        name = m.geom(g).name
        if name == "ground":
            key = "ground"
        elif body == cube_name:
            key = "cube_geom"
        elif body.startswith("cube") or body == "world":
            continue
        else:
            key = keep(name)
            if key is None:
                continue
        fp["geoms"][key] = {
            "type": int(m.geom_type[g]), "body": "world" if body == "world" else (keep(body) or body),
            "size": _l(m.geom_size[g]) if name != "ground" else None,  # plane extent is visual only
            "pos": _l(m.geom_pos[g]) if body != cube_name else _l(m.geom_pos[g]),
            "quat": _l(m.geom_quat[g]), "contype": int(m.geom_contype[g]), "conaffinity": int(m.geom_conaffinity[g]),
            "condim": int(m.geom_condim[g]), "friction": _l(m.geom_friction[g]), "solref": _l(m.geom_solref[g]),
            "solimp": _l(m.geom_solimp[g]), "solmix": float(m.geom_solmix[g]), "margin": float(m.geom_margin[g]),
            "gap": float(m.geom_gap[g]), "priority": int(m.geom_priority[g]),
        }
    for a in range(m.nu):
        key = keep(m.actuator(a).name)
        if key is None:
            continue
        fp["actuators"][key] = {
            "joint": keep(m.joint(m.actuator_trnid[a, 0]).name), "dyntype": int(m.actuator_dyntype[a]),
            "gaintype": int(m.actuator_gaintype[a]), "biastype": int(m.actuator_biastype[a]),
            "gainprm": _l(m.actuator_gainprm[a, :3]), "biasprm": _l(m.actuator_biasprm[a, :3]),
            "gear": float(m.actuator_gear[a, 0]), "ctrllimited": int(m.actuator_ctrllimited[a]),
            "forcelimited": int(m.actuator_forcelimited[a]), "forcerange": _l(m.actuator_forcerange[a]),
        }
    fp["counts"] = {"athlete_bodies": sum(1 for k in fp["bodies"] if k != "cube"),
                    "athlete_joints": len(fp["joints"]), "athlete_actuators": len(fp["actuators"])}
    return fp


def excludes(m: mujoco.MjModel, athlete_prefix: str = "") -> str:
    """<contact><exclude> body pairs of one athlete as "a|b,c|d" (prefix stripped, names sorted) — the format of the
    "excludes" key in Unity's fingerprint (ModelFingerprint.Dump). NOT part of fingerprint() / its hash: schema 1 predates
    it and adding it would re-key every brain, contract and reference; G0 compares it separately (excludes_mismatch).
    Which body parts may collide is physics (AGENTS.md "Self-collision"), so it must match Unity like everything else."""
    out = []
    for sig in m.exclude_signature:
        a, b = m.body(int(sig) >> 16).name, m.body(int(sig) & 0xFFFF).name
        if athlete_prefix and not (a.startswith(athlete_prefix) and b.startswith(athlete_prefix)):
            continue
        out.append("|".join(sorted((a[len(athlete_prefix):], b[len(athlete_prefix):]))))
    return ",".join(sorted(out))


def excludes_mismatch(m: mujoco.MjModel, unity_fp: dict, athlete_prefix: str = "") -> list[str]:
    """Pops "excludes" from a Unity fingerprint and compares it with the Python model's; [] when equal (or when the
    Unity dump predates the key)."""
    uni = unity_fp.pop("excludes", None)
    if uni is None:
        return []
    py = excludes(m, athlete_prefix)
    return [] if py == uni else [f"/excludes: {py!r} != {uni!r}"]


def canonical_bytes(fp: dict) -> bytes:
    return json.dumps(fp, sort_keys=True, separators=(",", ":")).encode()


def write_python_fingerprint(scene_xml: Path | None = None) -> tuple[Path, str]:
    """Fingerprint of the current body's scene ($POOLYMPIC_BODY): parity/fingerprint_python[_<body>].json + .sha256."""
    from . import bodies
    body = bodies.current()
    m = mujoco.MjModel.from_xml_path(str(scene_xml or body.scene_xml))
    fp = fingerprint(m)
    PARITY.mkdir(exist_ok=True)
    path = body.fingerprint_json
    data = canonical_bytes(fp)
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    path.with_suffix(".sha256").write_text(sha + "\n")
    return path, sha


def _canonical_quat(q: list) -> list:
    """q and -q are the same rotation: make the first non-negligible component positive."""
    for x in q:
        if abs(x) > 1e-9:
            return q if x > 0 else [-v for v in q]
    return q


_GEOM_SPHERE, _GEOM_CAPSULE, _GEOM_CYLINDER = 2, 3, 5


def normalize_for_compare(fp: dict) -> dict:
    """Replace representation-dependent geom frames by their physical meaning.

    Capsules/cylinders are symmetric about their axis and about their mid-plane, so only the axis direction up to
    sign matters (the Unity plug-in rebuilds capsule frames from a Transform + height and picks an equivalent one);
    sphere orientation is irrelevant.
    """
    out = json.loads(json.dumps(fp))
    for g in out.get("geoms", {}).values():
        if g["type"] == _GEOM_SPHERE:
            g["quat"] = None
        elif g["type"] in (_GEOM_CAPSULE, _GEOM_CYLINDER):
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, np.asarray(g.pop("quat"), float))
            axis = mat.reshape(3, 3)[:, 2]
            g["axis_unsigned"] = [float(v) for v in _canonical_quat(list(axis))]
    return out


def compare(a: dict, b: dict, path: str = "") -> list[str]:
    """G0 comparison: ints/strings exact, floats within abs 1e-6 + rel 1e-6. Returns list of mismatches."""
    errs: list[str] = []
    if path.endswith(("/quat", "/iquat")) and isinstance(a, list) and isinstance(b, list):
        a, b = _canonical_quat(a), _canonical_quat(b)
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                errs.append(f"{path}/{k}: missing on {'python' if k not in a else 'unity'} side")
            else:
                errs += compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            errs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                errs += compare(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        x, y = float(a), float(b)
        if abs(x - y) > FLOAT_ABS_TOL + FLOAT_REL_TOL * max(abs(x), abs(y)):
            errs.append(f"{path}: {x!r} != {y!r}")
    elif a != b:
        errs.append(f"{path}: {a!r} != {b!r}")
    return errs


def main(argv: list[str]) -> int:
    if len(argv) == 2:
        from . import bodies
        unity = json.loads(Path(argv[1]).read_text())
        errs = excludes_mismatch(mujoco.MjModel.from_xml_path(str(bodies.current().scene_xml)), unity)
        errs += compare(normalize_for_compare(json.loads(Path(argv[0]).read_text())), normalize_for_compare(unity))
        for e in errs[:200]:
            print("MISMATCH", e)
        print(f"G0 {'PASS' if not errs else 'FAIL'} ({len(errs)} mismatches)")
        return 0 if not errs else 1
    path, sha = write_python_fingerprint()
    print(f"wrote {path}  sha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
