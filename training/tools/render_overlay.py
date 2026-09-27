"""A8 — Render MATT's physics geoms overlaid on the skinned mesh (bind pose) + the default standing pose.

Writes parity/a8_overlay_{front,side}.png and parity/a8_default_{front,side}.png
Usage: uv run python tools/render_overlay.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic.gltf_io import Glb, to_mj  # noqa: E402

GLB = ROOT.parent / "SourceArt" / "test_MATT_Avaturn.glb"
OUT = ROOT.parent / "parity"
TMP = ROOT / "assets" / "derived"


def export_bind_mesh(path: Path) -> None:
    """Bind-pose body+outfit mesh (MuJoCo frame) as OBJ."""
    glb = Glb(GLB)
    skin = glb.skins()[0]
    bind = np.stack([glb.world(n) for n in skin.joint_nodes]) @ skin.inverse_bind
    lines, base = [], 1
    for ni, node in enumerate(glb.nodes):
        if "mesh" not in node:
            continue
        mesh = glb.gltf["meshes"][node["mesh"]]
        if "hair" in mesh.get("name", ""):
            continue
        for prim in mesh["primitives"]:
            a = prim["attributes"]
            pos = glb.accessor(a["POSITION"])
            j = glb.accessor(a["JOINTS_0"])
            w = glb.accessor(a["WEIGHTS_0"])
            homo = np.concatenate([pos, np.ones((len(pos), 1))], 1)
            v = to_mj(np.einsum("vk,vkij,vj->vi", w, bind[j], homo)[:, :3])
            idx = glb.accessor(prim["indices"]).reshape(-1, 3)
            lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in v]
            lines += [f"f {a_ + base} {b + base} {c + base}" for a_, b, c in idx]
            base += len(v)
    path.write_text("\n".join(lines))


def overlay_xml(with_mesh: bool) -> str:
    tree = ET.parse(ROOT / "assets" / "scene_matt.xml")
    root = tree.getroot()
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", {"offwidth": "1080", "offheight": "1920"})
    if with_mesh:
        asset = root.find("asset") or ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", {"name": "matt_bind", "file": str(TMP / "matt_bind.obj")})
        wb = root.find("worldbody")
        ET.SubElement(wb, "geom", {"name": "skin_overlay", "type": "mesh", "mesh": "matt_bind", "contype": "0",
                                   "conaffinity": "0", "group": "1", "rgba": "0.9 0.75 0.6 0.35"})
    for g in root.iter("geom"):
        if g.get("name", "").endswith(("_geom0", "_geom1")):
            g.set("rgba", "0.1 0.5 0.9 0.8")
    return ET.tostring(root, encoding="unicode")


def render(xml: str, qpos: np.ndarray | None, azimuth: float, name: str, lookat_z: float) -> None:
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    if qpos is not None:
        d.qpos[: len(qpos)] = qpos
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, lookat_z]
    cam.distance, cam.azimuth, cam.elevation = 3.2, azimuth, -5
    with mujoco.Renderer(m, height=1920, width=1080) as r:
        r.update_scene(d, camera=cam)
        img = r.render()
    import imageio.v3 as iio

    iio.imwrite(OUT / name, img)
    print("wrote", OUT / name)


def main():
    OUT.mkdir(exist_ok=True)
    export_bind_mesh(TMP / "matt_bind.obj")
    xml = overlay_xml(with_mesh=True)
    render(xml, None, 180, "a8_overlay_front.png", 0.9)  # camera in front (+x) looking back at MATT
    render(xml, None, 90, "a8_overlay_side.png", 0.9)
    scene = mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "scene_matt.xml"))
    q_default = scene.key("default").qpos
    xml2 = overlay_xml(with_mesh=False)
    render(xml2, q_default, 180, "a8_default_front.png", 0.9)
    render(xml2, q_default, 90, "a8_default_side.png", 0.9)


if __name__ == "__main__":
    main()
