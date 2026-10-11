"""KIM — rigged scan GLB -> clean T-pose GLB for the physics-body pipeline (training/tools/extract_skeleton.py).

Run headless (the Store build of Blender only exposes the launcher):
    blender-launcher.exe -b -P SourceArt/Kim/convert_kim.py

Input : SourceArt/Kim/Kim_source.glb (user's file "Kim.glb", 2026-10-10: one skinned mesh, 32k verts, 21 bones named
        Hips / Spine / Chest / UpperArm.L ..., T-pose, 1.062 m tall, faces -Y in Blender, one walk action)
Output: SourceArt/Kim/kim.glb        skinned mesh, T-pose rest, metres, 22 Mixamo bones (the names the body pipeline
                                     expects), glTF convention (+Y up, faces +Z, left = +X)
        SourceArt/Kim/convert_report.json

Decisions (user 2026-10-10: "give her unique characteristics" -> Claude's pick): 1.55 m, "light and quick".

Steps:
 1. Import, drop the walk action.
 2. The auto-rig is 13-20 mm asymmetric (hips +56 / -43 mm, ankles +42 / -22 mm): left and mirrored right joints are
    averaged (rest bones only; the skin does not move), centre bones go to x = 0, the hip joints to HIP_HALF_FRAC.
 3. Legs posed vertical under the hip joints (thigh, shin; feet keep their orientation) and baked as rest, as
    convert_zombie.py does: qpos = 0 = straight legs, feet hip-width apart (the scan stands with the feet together,
    which would switch the shin and foot pairs off as "touching at rest").
 4. Chest is split into Spine1 + Spine2 (weights blended over SPLIT_BAND around the middle of the bone).
 5. Mixamo names, uniform scale to HEIGHT_M, soles on z = 0, hips on x = y = 0.
"""

import json
import math
import os

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "Kim_source.glb")
OUT = os.path.join(HERE, "kim.glb")
HEIGHT_M = 1.55
HIP_HALF_FRAC = 0.053        # hip joint half-spacing / height (16.4 cm apart at 1.55 m; the rig's left hip is 0.053)
SPLIT_BAND = 0.02            # m (source scale): Spine1 <-> Spine2 weight blend half-width

BONE_MAP = {"Hips": "Hips", "Spine": "Spine", "Chest": "Spine1", "Neck": "Neck", "Head": "Head"}
for s, S in (("L", "Left"), ("R", "Right")):
    BONE_MAP.update({f"Shoulder.{s}": f"{S}Shoulder", f"UpperArm.{s}": f"{S}Arm", f"LowerArm.{s}": f"{S}ForeArm",
                     f"Hand.{s}": f"{S}Hand", f"UpperLeg.{s}": f"{S}UpLeg", f"LowerLeg.{s}": f"{S}Leg",
                     f"Foot.{s}": f"{S}Foot", f"Toe.{s}": f"{S}ToeBase"})


def symmetrize(arm) -> float:
    """Average left and mirrored right rest joints; returns the largest correction (m)."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for b in eb:
        b.use_connect = False
    worst = 0.0
    mir = Vector((-1.0, 1.0, 1.0))
    for b in list(eb):
        if b.name.endswith(".L"):
            r = eb[b.name[:-2] + ".R"]
            for end in ("head", "tail"):
                l_, r_ = getattr(b, end).copy(), getattr(r, end).copy()
                avg = (l_ + r_ * mir) / 2
                worst = max(worst, (l_ - avg).length)
                setattr(b, end, avg)
                setattr(r, end, avg * mir)
        elif not b.name.endswith(".R"):
            b.head.x = b.tail.x = 0.0
    zs = [v.co.z for o in bpy.data.objects if o.type == "MESH" and o.modifiers for v in o.data.vertices]
    half = HIP_HALF_FRAC * (max(zs) - min(zs))
    for s, sign in ((".L", 1.0), (".R", -1.0)):
        eb["UpperLeg" + s].head.x = sign * half
    bpy.ops.object.mode_set(mode="OBJECT")
    return worst


def vertical_legs(arm) -> dict:
    """Pose thigh and shin straight down from the hip joint, feet keep their world orientation. Returns deg applied."""
    bpy.context.view_layer.update()
    down = (arm.matrix_world.to_3x3().inverted() @ Vector((0.0, 0.0, -1.0))).normalized()
    applied = {}
    P = arm.pose.bones
    for s in ("L", "R"):
        thigh, calf, foot = P[f"UpperLeg.{s}"], P[f"LowerLeg.{s}"], P[f"Foot.{s}"]
        foot_rot = foot.matrix.to_3x3().copy()
        for seg, child in ((thigh, calf), (calf, foot)):
            rot = (child.head - seg.head).normalized().rotation_difference(down)
            applied[seg.name] = round(math.degrees(rot.angle), 2)
            h = seg.head.copy()
            seg.matrix = Matrix.Translation(h) @ rot.to_matrix().to_4x4() @ Matrix.Translation(-h) @ seg.matrix
            bpy.context.view_layer.update()
        h = foot.head.copy()
        foot.matrix = Matrix.Translation(h) @ foot_rot.to_4x4()
        bpy.context.view_layer.update()
    return applied


def bake_pose_as_rest(arm, mesh):
    bpy.context.view_layer.objects.active = mesh
    mod = next(m for m in mesh.modifiers if m.type == "ARMATURE")
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    m = mesh.modifiers.new("Armature", "ARMATURE")
    m.object = arm


def split_chest(arm, mesh) -> int:
    """Chest (renamed Spine1 by BONE_MAP later) keeps the lower half; a new Spine2 takes the upper half, the neck and
    the shoulders. Returns the number of vertices that received Spine2 weight."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    chest = eb["Chest"]
    mid, top = (chest.head + chest.tail) / 2, chest.tail.copy()
    kids = [c for c in chest.children]
    up = eb.new("Spine2")
    up.head, up.tail, up.roll, up.parent = mid, top, chest.roll, chest
    chest.tail = mid
    for c in kids:
        c.parent = up
    mid_z = (arm.matrix_world @ mid).z
    bpy.ops.object.mode_set(mode="OBJECT")
    lo, hi = mesh.vertex_groups["Chest"], mesh.vertex_groups.new(name="Spine2")
    n = 0
    for v in mesh.data.vertices:
        w = next((g.weight for g in v.groups if g.group == lo.index), 0.0)
        if w <= 0.0:
            continue
        t = min(1.0, max(0.0, ((mesh.matrix_world @ v.co).z - mid_z + SPLIT_BAND) / (2 * SPLIT_BAND)))
        t = t * t * (3 - 2 * t)
        if t > 0.0:
            hi.add([v.index], w * t, "REPLACE")
            lo.add([v.index], w * (1 - t), "REPLACE")
            n += 1
    return n


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=SRC)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers))
    report = {"source": "Kim/Kim_source.glb", "actions_dropped": [a.name for a in bpy.data.actions]}
    if arm.animation_data:
        arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.location, pb.rotation_quaternion, pb.rotation_euler, pb.scale = (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (1, 1, 1)
    zs = [(mesh.matrix_world @ v.co).z for v in mesh.data.vertices]
    report["scan_height_m"] = round(max(zs) - min(zs), 4)

    report["symmetrize_max_shift_m"] = round(symmetrize(arm), 4)
    report["leg_straightening_deg"] = vertical_legs(arm)
    bake_pose_as_rest(arm, mesh)
    report["spine2_vertices"] = split_chest(arm, mesh)
    for b in arm.data.bones:
        if b.name in BONE_MAP:
            b.name = BONE_MAP[b.name]              # Blender renames the matching vertex groups too

    # metres at HEIGHT_M, soles on z = 0, hips on x = y = 0
    zs = [(mesh.matrix_world @ v.co).z for v in mesh.data.vertices]
    scale = HEIGHT_M / (max(zs) - min(zs))
    hips = arm.matrix_world @ arm.data.bones["Hips"].head_local
    arm.matrix_world = Matrix.Scale(scale, 4) @ Matrix.Translation((-hips.x, -hips.y, -min(zs))) @ arm.matrix_world
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    arm.name, mesh.name = "KimArmature", "kim_body"

    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh.evaluated_get(dg)
    zs = [(ev.matrix_world @ v.co).z for v in ev.data.vertices]
    heads = {b.name: arm.matrix_world @ b.head_local for b in arm.data.bones}
    mirror = max(math.dist(heads[n], (-heads["Right" + n[4:]].x, heads["Right" + n[4:]].y, heads["Right" + n[4:]].z))
                 for n in heads if n.startswith("Left"))
    report.update({"scale": scale, "bones": len(arm.data.bones), "vertices": len(mesh.data.vertices),
                   "unweighted_vertices": sum(1 for v in mesh.data.vertices if not any(g.weight > 1e-4 for g in v.groups)),
                   "height_m": round(max(zs) - min(zs), 4), "min_z_m": round(min(zs), 4),
                   "lr_mirror_error_m": round(mirror, 5),
                   "bone_heads": {n: [round(c, 4) for c in p] for n, p in heads.items()}})
    bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, export_animations=False,
                              export_cameras=False, export_lights=False, export_apply=False, export_skins=True,
                              export_morph=False)
    report["glb_bytes"] = os.path.getsize(OUT)
    with open(os.path.join(HERE, "convert_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)


try:
    main()
except Exception:
    import traceback
    with open(os.path.join(HERE, "convert_error.txt"), "w") as fh:
        traceback.print_exc(file=fh)
    raise
