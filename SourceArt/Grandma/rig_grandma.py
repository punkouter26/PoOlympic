"""GRANDMA — unrigged scan mesh -> skinned T-pose GLB for the physics-body pipeline (training/tools/extract_skeleton.py).

Run inside Blender (MCP or Text Editor) on an EMPTY file (it builds its own scene; never on stadium.blend):
    exec(open(r"<repo>/SourceArt/Grandma/rig_grandma.py").read())

Input : SourceArt/test_GRANDMA_riggedTenCent.glb (despite the name: mesh + texture only, no skin; 31k verts, 50k tris,
        1.114 m tall, arms out in a near T-pose, faces -Y in Blender)
Output: SourceArt/Grandma/grandma.glb    skinned mesh, T-pose rest, metres, 22 Mixamo bones (the names the body
                                          pipeline expects), glTF convention (+Y up, faces +Z, left = +X)
        SourceArt/Grandma/rig_report.json

User decisions (2026-09-30): Claude rigs it (no AccuRig / Mixamo), 1.60 m tall, "frail but steady".

Steps:
 1. Import, apply transforms, scale uniformly to HEIGHT_M (feet on z = 0), centre the hips on x = y = 0.
 2. Joint landmarks from mesh cross-sections (leg / arm slice centres, torso slice centres; fractions of the height
    read off the front + side views), mirrored to an exactly symmetric skeleton.
 3. 22-bone armature, automatic (bone heat) weights.
 4. The scan stands with the feet ~12 deg behind the hips and soft knees: the legs are posed straight and vertical, the
    feet flat, and that pose is baked in as the rest pose (as convert_zombie.py does) -> qpos = 0 = straight legs.
"""

import json
import math
import os

import bpy
import mathutils
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"C:\Users\punko\Downloads\PoOlympic\SourceArt\Grandma"
SRC = os.path.join(os.path.dirname(HERE), "test_GRANDMA_riggedTenCent.glb")
OUT = os.path.join(HERE, "grandma.glb")
REPORT = os.path.join(HERE, "rig_report.json")
HEIGHT_M = 1.60

# landmark heights as fractions of the scan height (front / side views + slice tables, 2026-09-30)
F_HIP, F_KNEE, F_ANKLE, F_SOLE = 0.422, 0.256, 0.058, 0.018
F_SPINE = (0.494, 0.566, 0.646)          # Spine, Spine1, Spine2 heads
F_NECK, F_HEAD, F_HEAD_TOP = 0.795, 0.840, 0.970
F_CLAV = 0.772
# arm landmarks along x (fractions of the height, from the body midline)
X_SHOULDER, X_ELBOW, X_WRIST, X_HAND_END = 0.148, 0.278, 0.381, 0.466
X_CLAV = 0.022


def clean_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images, bpy.data.cameras):
        for b in list(coll):
            if b.users == 0:
                coll.remove(b)


def import_mesh():
    bpy.ops.import_scene.gltf(filepath=SRC)
    mesh = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
    for o in list(bpy.context.scene.objects):
        if o is not mesh and o.type == "EMPTY":
            bpy.data.objects.remove(o, do_unlink=True)
    mesh.parent = None
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mesh.name = mesh.data.name = "Grandma"
    # the scan is split along its UV seams (451 islands, 6236 duplicate vertices): bone heat finds no solution on it.
    # Welding keeps the UVs (they live on the face corners).
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    n0 = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.to_mesh(mesh.data)
    bm.free()
    mesh.data.validate()
    mesh.data.update()
    mesh["welded_vertices"] = n0 - len(mesh.data.vertices)
    return mesh


def verts(mesh) -> np.ndarray:
    co = np.empty(len(mesh.data.vertices) * 3)
    mesh.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def set_verts(mesh, v: np.ndarray):
    mesh.data.vertices.foreach_set("co", v.reshape(-1).astype(np.float64))
    mesh.data.update()


def slice_centre(V, axis: int, value: float, mask=None, tol: float = 0.006) -> np.ndarray:
    """Bounding-box centre of the vertices within tol of `value` along `axis` (and `mask`)."""
    sel = np.abs(V[:, axis] - value) < tol
    if mask is not None:
        sel &= mask
    q = V[sel]
    if len(q) < 5:
        raise RuntimeError(f"empty slice axis {axis} = {value:.3f}")
    return (q.min(0) + q.max(0)) / 2


def landmarks(V) -> dict:
    """Joint positions (Blender frame, z up, faces -y, left = +x) of the LEFT side + the midline; mirrored later."""
    H = V[:, 2].max()
    lm = {}
    legs = np.abs(V[:, 0]) < 0.3 * H
    for side, sgn in (("L", 1.0), ("R", -1.0)):
        m = legs & (sgn * V[:, 0] > 0.005 * H)
        lm[f"ankle_{side}"] = slice_centre(V, 2, F_ANKLE * H, m)
        lm[f"knee_{side}"] = slice_centre(V, 2, F_KNEE * H, m)
        thigh = slice_centre(V, 2, 0.31 * H, m)
        lm[f"hip_{side}"] = np.array([thigh[0], 0.0, F_HIP * H])
        foot = V[m & (V[:, 2] < 0.04 * H)]
        toe_y, heel_y = foot[:, 1].min(), foot[:, 1].max()
        lm[f"ball_{side}"] = np.array([lm[f"ankle_{side}"][0], toe_y + 0.30 * (heel_y - toe_y), F_SOLE * H])
        lm[f"toe_{side}"] = np.array([lm[f"ankle_{side}"][0], toe_y, F_SOLE * H])
        arm = (V[:, 2] > 0.6 * H)
        for name, xf in (("shoulder", X_SHOULDER), ("elbow", X_ELBOW), ("wrist", X_WRIST), ("hand_end", X_HAND_END)):
            lm[f"{name}_{side}"] = slice_centre(V, 0, sgn * xf * H, arm)
    torso = np.abs(V[:, 0]) < 0.10 * H
    for i, f in enumerate(F_SPINE):
        lm[f"spine{i}"] = slice_centre(V, 2, f * H, torso)
    lm["neck"] = slice_centre(V, 2, F_NECK * H, torso)
    lm["head"] = slice_centre(V, 2, F_HEAD * H, torso)
    lm["head_top"] = slice_centre(V, 2, F_HEAD_TOP * H, np.abs(V[:, 0]) < 0.08 * H)
    lm["clav"] = slice_centre(V, 2, F_CLAV * H, torso)
    # hips: midline between the hip joints, at the hip-joint height, y of the pelvis slice
    pelvis = slice_centre(V, 2, F_HIP * H, torso)
    for side in "LR":
        lm[f"hip_{side}"][1] = pelvis[1]
    lm["hips"] = np.array([0.0, pelvis[1], F_HIP * H])
    return lm


def symmetric(lm: dict) -> dict:
    """Average L/R (mirror x) so the skeleton is exactly symmetric; the midline gets x = 0."""
    out = {}
    for k, v in lm.items():
        if k.endswith("_L"):
            r = lm[k[:-2] + "_R"]
            a = (np.asarray(v) + np.asarray(r) * [-1, 1, 1]) / 2
            out[k], out[k[:-2] + "_R"] = a, a * [-1, 1, 1]
        elif not k.endswith("_R"):
            out[k] = np.array([0.0, v[1], v[2]])
    return out


def build_armature(lm: dict):
    arm = bpy.data.armatures.new("GrandmaRig")
    rig = bpy.data.objects.new("Armature", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones

    def bone(name, head, tail, parent=None, connect=False):
        b = eb.new(name)
        b.head, b.tail = mathutils.Vector(head), mathutils.Vector(tail)
        b.roll = 0.0
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        return b

    bone("Hips", lm["hips"], lm["spine0"])
    bone("Spine", lm["spine0"], lm["spine1"], "Hips", True)
    bone("Spine1", lm["spine1"], lm["spine2"], "Spine", True)
    bone("Spine2", lm["spine2"], lm["neck"], "Spine1", True)
    bone("Neck", lm["neck"], lm["head"], "Spine2", True)
    bone("Head", lm["head"], lm["head_top"], "Neck", True)
    for s, S in (("L", "Left"), ("R", "Right")):
        sx = 1.0 if s == "L" else -1.0
        clav = lm["clav"] + [sx * X_CLAV * HEIGHT_M, 0.0, 0.0]
        bone(f"{S}Shoulder", clav, lm[f"shoulder_{s}"], "Spine2")
        bone(f"{S}Arm", lm[f"shoulder_{s}"], lm[f"elbow_{s}"], f"{S}Shoulder", True)
        bone(f"{S}ForeArm", lm[f"elbow_{s}"], lm[f"wrist_{s}"], f"{S}Arm", True)
        bone(f"{S}Hand", lm[f"wrist_{s}"], lm[f"hand_end_{s}"], f"{S}ForeArm", True)
        bone(f"{S}UpLeg", lm[f"hip_{s}"], lm[f"knee_{s}"], "Hips")
        bone(f"{S}Leg", lm[f"knee_{s}"], lm[f"ankle_{s}"], f"{S}UpLeg", True)
        bone(f"{S}Foot", lm[f"ankle_{s}"], lm[f"ball_{s}"], f"{S}Leg", True)
        bone(f"{S}ToeBase", lm[f"ball_{s}"], lm[f"toe_{s}"], f"{S}Foot", True)
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def skin(mesh, rig):
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    names = {g.index: g.name for g in mesh.vertex_groups}
    unweighted = sum(1 for v in mesh.data.vertices if not any(g.weight > 1e-4 for g in v.groups))
    return unweighted, sorted(set(names.values()))


def straighten_legs(mesh, rig) -> dict:
    """Pose: thigh + shin vertical (knee straight), foot level; then bake that pose as the new rest pose."""
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    down = mathutils.Vector((0.0, 0.0, -1.0))
    deg = {}

    def aim(pb, want: mathutils.Vector):
        bpy.context.view_layer.update()
        cur = (rig.matrix_world @ pb.tail - rig.matrix_world @ pb.head).normalized()
        q = cur.rotation_difference(want.normalized())
        deg[pb.name] = round(math.degrees(q.angle), 2)
        # world-space rotation about the bone head -> pose-space (bone local) rotation
        mw = rig.matrix_world @ pb.matrix
        loc = mw.to_translation()
        new = mathutils.Matrix.Translation(loc) @ q.to_matrix().to_4x4() @ mathutils.Matrix.Translation(-loc) @ mw
        pb.matrix = rig.matrix_world.inverted() @ new
        bpy.context.view_layer.update()

    for S in ("Left", "Right"):
        pbs = rig.pose.bones
        foot = pbs[f"{S}Foot"]
        foot_rot = (rig.matrix_world @ foot.matrix).to_3x3().normalized()     # the scan's flat-sole foot orientation
        aim(pbs[f"{S}UpLeg"], down)
        aim(pbs[f"{S}Leg"], down)
        # the foot keeps its world orientation (sole flat on the floor); the toes follow it unchanged
        loc = (rig.matrix_world @ foot.matrix).to_translation()
        foot.matrix = rig.matrix_world.inverted() @ (mathutils.Matrix.Translation(loc) @ foot_rot.to_4x4())
        bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")
    # apply the pose to the mesh, then make it the rest pose and re-skin with the same weights
    mod = next(m for m in mesh.modifiers if m.type == "ARMATURE")
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    m = mesh.modifiers.new("Armature", "ARMATURE")
    m.object = rig
    return deg


def main():
    """Two passes: straightening the knees lengthens the legs and drops the soles below z = 0, so the first pass
    measures that and the second scales / lifts the scan so the straightened GRANDMA is HEIGHT_M with the soles on 0."""
    scale, lift = None, 0.0
    for _ in range(2):
        report = build(scale, lift)
        s0 = report["scale"]
        scale = s0 * HEIGHT_M / report["height_m"]
        lift = (lift - report["min_z_m"]) * scale / s0
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("bone_heads", "vertex_groups")}))
    return report


def build(scale: float | None, lift: float) -> dict:
    clean_scene()
    mesh = import_mesh()
    V = verts(mesh)
    H0 = float(V[:, 2].max() - V[:, 2].min())
    scale = scale or HEIGHT_M / H0
    V[:, 2] -= V[:, 2].min()
    V *= scale
    lm = symmetric(landmarks(V))                    # landmark fractions are of the scan standing on z = 0
    shift = np.array([0.0, lm["hips"][1], -lift])   # hips on x = y = 0 (x is already the midline), scan lifted
    V -= shift
    set_verts(mesh, V)
    lm = {k: v - shift for k, v in lm.items()}
    rig = build_armature(lm)
    unweighted, groups = skin(mesh, rig)
    deg = straighten_legs(mesh, rig)
    V2 = verts(mesh)
    # export: armature + mesh, glTF +Y up
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    rig.select_set(True)
    bpy.ops.export_scene.gltf(filepath=OUT, use_selection=True, export_format="GLB", export_yup=True, export_skins=True,
                              export_animations=False, export_apply=False)
    heads = {b.name: [round(x, 4) for x in (rig.matrix_world @ b.head_local)] for b in rig.data.bones}
    mirror = max(np.linalg.norm(np.array(heads[n]) - np.array(heads["Right" + n[4:]]) * [-1, 1, 1])
                 for n in heads if n.startswith("Left"))
    report = {"source": os.path.basename(SRC), "scan_height_m": round(H0, 4), "height_m": round(float(V2[:, 2].max() - V2[:, 2].min()), 4),
              "min_z_m": round(float(V2[:, 2].min()), 4), "scale": scale, "lift_m": round(lift, 4), "vertices": len(V2),
              "welded_vertices": int(mesh["welded_vertices"]),
              "bones": len(rig.data.bones), "vertex_groups": groups, "unweighted_vertices": unweighted,
              "leg_straightening_deg": deg, "lr_mirror_error_m": round(float(mirror), 5), "bone_heads": heads}
    return report


main()
