"""A4 — Extract MATT's bind-pose skeleton + skinned vertices into the MuJoCo frame.

Outputs (training/assets/derived/):
  skeleton_matt.json  joint names, parents, world bind positions/rotations (MuJoCo frame)
  skin_matt.npz       bind-pose vertex positions (MuJoCo frame) + per-vertex dominant joint + weights

Usage:  uv run python tools/extract_skeleton.py [path/to/matt.glb]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic.gltf_io import Glb, rot_to_mj, to_mj  # noqa: E402

DEFAULT_GLB = ROOT.parent / "SourceArt" / "test_MATT_Avaturn.glb"
OUT_DIR = ROOT / "assets" / "derived"


def mirror_name(name: str) -> str | None:
    if name.startswith("Left"):
        return "Right" + name[4:]
    if name.startswith("Right"):
        return "Left" + name[5:]
    return None


def main(glb_path: Path) -> int:
    glb = Glb(glb_path)
    skins = glb.skins()
    if len(skins) != 1:
        raise SystemExit(f"expected exactly one skin, found {len(skins)}")
    skin = skins[0]
    jn = skin.joint_nodes
    names = [glb.name(n) for n in jn]
    jset = set(jn)

    world = np.stack([glb.world(n) for n in jn])  # (J, 4, 4) glTF
    # Bind-pose consistency: joint_world @ inverse_bind must be the same (mesh bind) matrix for every joint.
    bind_mats = world @ skin.inverse_bind
    bind_dev = float(np.abs(bind_mats - bind_mats[0]).max())

    joints = []
    for i, n in enumerate(jn):
        p = glb.parent.get(n)
        r = world[i, :3, :3]
        r = r / np.linalg.norm(r, axis=0, keepdims=True)  # strip ~1e-7 scale noise
        joints.append({
            "name": names[i],
            "parent": glb.name(p) if p in jset else None,
            "pos": to_mj(world[i, :3, 3]).round(6).tolist(),
            "rot": rot_to_mj(r).round(6).tolist(),
        })

    # Skinned vertices at bind pose, in world (MuJoCo) frame.
    verts, dom, wts, jidx, mesh_ids = [], [], [], [], []
    for mi, m in enumerate(skin.meshes):
        skin_mats = bind_mats[m.joints]  # (V, 4, 4, 4)
        homo = np.concatenate([m.positions, np.ones((len(m.positions), 1))], axis=1)
        v = np.einsum("vk,vkij,vj->vi", m.weights, skin_mats, homo)[:, :3]
        verts.append(to_mj(v))
        dom.append(m.joints[np.arange(len(m.joints)), m.weights.argmax(1)])
        wts.append(m.weights)
        jidx.append(m.joints)
        mesh_ids.append(np.full(len(v), mi))
    verts = np.concatenate(verts)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "skin_matt.npz",
        verts=verts, dominant=np.concatenate(dom), weights=np.concatenate(wts),
        joints=np.concatenate(jidx), mesh=np.concatenate(mesh_ids),
        mesh_names=np.array([m.name for m in skin.meshes]), joint_names=np.array(names),
    )

    # Checks
    pos = {j["name"]: np.array(j["pos"]) for j in joints}
    mirror_err = 0.0
    for name, p in pos.items():
        mn = mirror_name(name)
        if mn:
            q = pos[mn] * np.array([1, -1, 1])
            mirror_err = max(mirror_err, float(np.linalg.norm(p - q)))
    body_mesh = [i for i, m in enumerate(skin.meshes) if "body" in m.name]
    body_verts = verts[np.isin(np.concatenate(mesh_ids), body_mesh)] if body_mesh else verts
    height_body = float(body_verts[:, 2].max() - body_verts[:, 2].min())
    height_all = float(verts[:, 2].max() - verts[:, 2].min())

    report = {
        "source": str(glb_path.relative_to(ROOT.parent)).replace("\\", "/"),
        "frame": "MuJoCo: x forward, y left, z up; (x,y,z)_mj = (z,x,y)_gltf",
        "num_joints": len(joints),
        "bind_pose_consistency_max_abs": bind_dev,
        "lr_mirror_max_error_m": mirror_err,
        "height_body_mesh_m": height_body,
        "height_with_hair_m": height_all,
        "ground_min_z_m": float(verts[:, 2].min()),
        "joints": joints,
    }
    (OUT_DIR / "skeleton_matt.json").write_text(json.dumps(report, indent=1))

    print(f"joints={len(joints)}  bind-consistency={bind_dev:.2e}  L/R mirror err={mirror_err*1000:.3f} mm")
    print(f"height body={height_body:.3f} m  with hair={height_all:.3f} m  min z={verts[:, 2].min():+.4f} m  verts={len(verts)}")
    ok = len(joints) == 52 and mirror_err < 1e-3 and bind_dev < 1e-4 and 1.80 < height_all < 1.90
    print("A4 CHECKS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GLB))
