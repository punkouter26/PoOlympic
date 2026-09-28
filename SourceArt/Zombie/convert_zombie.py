"""ZOMBIE — AccuRig FBX -> clean T-pose GLB for the physics-body pipeline (training/tools/extract_skeleton.py).

Run inside Blender (MCP or Text Editor) on an EMPTY file (it builds its own scene; never on stadium.blend):
    exec(open(r"<repo>/SourceArt/Zombie/convert_zombie.py").read())

Input : SourceArt/test_ZOMBIE_RiggedAccurig.fbx (AccuRig / Character Creator "CC_Base" rig, 71 bones, 50k tris)
Output: SourceArt/Zombie/zombie.glb      skinned mesh, T-pose rest, metres, Mixamo bone names (the names the
                                          body pipeline expects), glTF convention (+Y up, faces +Z, left = +X)
        SourceArt/Zombie/convert_report.json

Fixes, in order (see rl_optimization_log.md 2026-09-28 zombie analysis):
 1. Blender's FBX import leaves the mesh object at -90 deg X relative to its armature, so skin and bones disagree
    (a 60 deg forearm bend threw vertices 0.87 m). Zeroing the mesh object's rotation re-aligns them exactly.
 2. The file's rest (bind) pose is a hunched, splayed zombie stance; its "0_T-Pose" action is a clean T-pose. The
    physics body is defined at a T-pose (qpos = 0 = bind pose, like MATT), so the T-pose is baked in as the new rest:
    shape keys dropped (Basis + unused V_None), armature modifier applied, pose applied as rest, modifier re-added.
 3. Twist / share bones carry skin weights of their limb: their vertex groups are merged into the parent limb bone,
    then every bone that is not in BONE_MAP is removed (face, jaw, tongue, eyes, toes, breasts, ribs).
 4. Kept bones are renamed to Mixamo names; the armature transform (0.01 scale, +90 deg X) is applied -> metres, Z up.
"""

import json
import math
import os

import bpy

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"C:\Users\punko\Downloads\PoOlympic\SourceArt\Zombie"
SRC = os.path.join(os.path.dirname(HERE), "test_ZOMBIE_RiggedAccurig.fbx")
OUT = os.path.join(HERE, "zombie.glb")

# CC_Base -> Mixamo (the names body_specs() / fit_geoms() use). Hands have no finger bones on this rig.
BONE_MAP = {
    "CC_Base_Hip": "Hips",
    "CC_Base_Waist": "Spine", "CC_Base_Spine01": "Spine1", "CC_Base_Spine02": "Spine2",
    "CC_Base_NeckTwist01": "Neck", "CC_Base_Head": "Head",
}
for s, S in (("L", "Left"), ("R", "Right")):
    BONE_MAP.update({
        f"CC_Base_{s}_Clavicle": f"{S}Shoulder", f"CC_Base_{s}_Upperarm": f"{S}Arm",
        f"CC_Base_{s}_Forearm": f"{S}ForeArm", f"CC_Base_{s}_Hand": f"{S}Hand",
        f"CC_Base_{s}_Thigh": f"{S}UpLeg", f"CC_Base_{s}_Calf": f"{S}Leg",
        f"CC_Base_{s}_Foot": f"{S}Foot", f"CC_Base_{s}_ToeBase": f"{S}ToeBase",
    })
# weights of these bones are merged into the mapped bone (before renaming)
MERGE = {"CC_Base_Pelvis": "CC_Base_Hip", "CC_Base_NeckTwist02": "CC_Base_NeckTwist01"}
for s in ("L", "R"):
    MERGE.update({
        f"CC_Base_{s}_ThighTwist01": f"CC_Base_{s}_Thigh", f"CC_Base_{s}_ThighTwist02": f"CC_Base_{s}_Thigh",
        f"CC_Base_{s}_CalfTwist01": f"CC_Base_{s}_Calf", f"CC_Base_{s}_CalfTwist02": f"CC_Base_{s}_Calf",
        f"CC_Base_{s}_KneeShareBone": f"CC_Base_{s}_Calf",
        f"CC_Base_{s}_UpperarmTwist01": f"CC_Base_{s}_Upperarm", f"CC_Base_{s}_UpperarmTwist02": f"CC_Base_{s}_Upperarm",
        f"CC_Base_{s}_ForearmTwist01": f"CC_Base_{s}_Forearm", f"CC_Base_{s}_ForearmTwist02": f"CC_Base_{s}_Forearm",
        f"CC_Base_{s}_ElbowShareBone": f"CC_Base_{s}_Forearm",
        f"CC_Base_{s}_ToeBaseShareBone": f"CC_Base_{s}_ToeBase",
        f"CC_Base_{s}_RibsTwist": "CC_Base_Spine02", f"CC_Base_{s}_Breast": "CC_Base_Spine02",
    })
    for t in ("Pinky", "Ring", "Mid", "Index", "Big"):
        MERGE[f"CC_Base_{s}_{t}Toe1"] = f"CC_Base_{s}_ToeBase"
for b in ("CC_Base_FacialBone", "CC_Base_JawRoot", "CC_Base_Tongue01", "CC_Base_Tongue02", "CC_Base_Tongue03",
          "CC_Base_Teeth01", "CC_Base_Teeth02", "CC_Base_UpperJaw", "CC_Base_L_Eye", "CC_Base_R_Eye"):
    MERGE[b] = "CC_Base_Head"


def merge_groups(mesh, src, dst):
    gs, gd = mesh.vertex_groups.get(src), mesh.vertex_groups.get(dst)
    if gs is None:
        return 0
    if gd is None:
        gd = mesh.vertex_groups.new(name=dst)
    n = 0
    for v in mesh.data.vertices:
        for g in v.groups:
            if g.group == gs.index and g.weight > 0:
                gd.add([v.index], g.weight, "ADD")
                n += 1
    mesh.vertex_groups.remove(gs)
    return n


def straighten_legs(arm, action):
    """The file's T-pose keeps the zombie crouch (knees ~36 deg, thighs ~18 deg forward). The physics body's zero pose
    must have straight legs (joint ranges are anatomical, measured from straight - DESIGN §2), so thigh and shin are
    turned vertical in the T-pose frame and the foot keeps its world orientation (flat). The crouch returns as the
    zombie's default stance / style reward, held by its own muscles. Returns the rotations applied (deg)."""
    from mathutils import Matrix, Vector
    arm.animation_data.action = None            # pose is edited directly; keep the T-pose values as the starting point
    bpy.context.view_layer.update()
    down = (arm.matrix_world.to_3x3().inverted() @ Vector((0.0, 0.0, -1.0))).normalized()
    applied = {}

    def rotate_about_head(pb, rot):
        h = pb.head.copy()
        pb.matrix = Matrix.Translation(h) @ rot.to_matrix().to_4x4() @ Matrix.Translation(-h) @ pb.matrix
        bpy.context.view_layer.update()

    for s in ("L", "R"):
        P = arm.pose.bones
        thigh, calf, foot = P[f"CC_Base_{s}_Thigh"], P[f"CC_Base_{s}_Calf"], P[f"CC_Base_{s}_Foot"]
        foot_rot = foot.matrix.to_3x3().copy()
        for seg, child in ((thigh, calf), (calf, foot)):
            rot = (child.head - seg.head).normalized().rotation_difference(down)
            applied[seg.name] = round(math.degrees(rot.angle), 2)
            rotate_about_head(seg, rot)
        h = foot.head.copy()
        foot.matrix = Matrix.Translation(h) @ foot_rot.to_4x4()
        bpy.context.view_layer.update()
    return applied


def main():
    import contextlib, io, logging
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bpy.ops.import_scene.fbx(filepath=SRC)
    logging.disable(logging.NOTSET)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    mesh = next(o for o in bpy.data.objects if o.type == "MESH")
    report = {"source": os.path.relpath(SRC, os.path.dirname(HERE)).replace("\\", "/")}

    # 1. skin <-> skeleton alignment
    mesh.rotation_euler = (0.0, 0.0, 0.0)

    # 2. bake the T-pose as rest
    action = arm.animation_data.action
    bpy.context.scene.frame_set(int(action.frame_range[0]))
    bpy.context.view_layer.update()
    report["leg_straightening_deg"] = straighten_legs(arm, action)
    if mesh.data.shape_keys:
        mesh.shape_key_clear()
    bpy.context.view_layer.objects.active = mesh
    mod = next(m for m in mesh.modifiers if m.type == "ARMATURE")
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.animation_data.action = None
    for pb in arm.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.scale = (1, 1, 1)
    m = mesh.modifiers.new("Armature", "ARMATURE")
    m.object = arm

    # 3. merge twist/share/face weights, drop unmapped bones
    merged = {src: merge_groups(mesh, src, dst) for src, dst in MERGE.items()}
    report["merged_vertex_weights"] = {k: v for k, v in merged.items() if v}
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    removed = [b.name for b in list(eb) if b.name not in BONE_MAP]
    for name in removed:
        eb.remove(eb[name])
    bpy.ops.object.mode_set(mode="OBJECT")
    report["removed_bones"] = len(removed)
    leftover = [g.name for g in mesh.vertex_groups if g.name not in BONE_MAP]
    for name in leftover:
        mesh.vertex_groups.remove(mesh.vertex_groups[name])
    report["dropped_unmapped_groups"] = leftover

    # 4. Mixamo names (bones + vertex groups), apply the armature transform (metres, Z up)
    for b in arm.data.bones:
        b.name = BONE_MAP[b.name]          # Blender renames the matching vertex groups too
    for o in (arm, mesh):
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    mesh.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    arm.name, mesh.name = "ZombieArmature", "zombie_body"

    # checks: unweighted vertices, height, L/R symmetry of the skeleton
    unweighted = sum(1 for v in mesh.data.vertices if not any(g.weight > 1e-4 for g in v.groups))
    zs = [(mesh.matrix_world @ v.co).z for v in mesh.data.vertices]
    heads = {b.name: arm.matrix_world @ b.head_local for b in arm.data.bones}
    def mirror_err(n):   # left side = +x (character faces -y), so the mirror of a left joint flips x
        l, r = heads[n], heads["Right" + n[4:]]
        return math.dist((l.x, l.y, l.z), (-r.x, r.y, r.z))
    mirror = max(mirror_err(n) for n in heads if n.startswith("Left"))
    report.update({"bones": len(arm.data.bones), "vertices": len(mesh.data.vertices),
                   "unweighted_vertices": unweighted, "height_m": round(max(zs) - min(zs), 4),
                   "min_z_m": round(min(zs), 4), "lr_mirror_error_m": round(mirror, 5),
                   "bone_heads": {n: [round(c, 4) for c in p] for n, p in heads.items()}})

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        logging.disable(logging.CRITICAL)
        bpy.ops.object.select_all(action="DESELECT")
        arm.select_set(True)
        mesh.select_set(True)
        bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=True, export_animations=False,
                                  export_cameras=False, export_lights=False, export_apply=False, export_skins=True,
                                  export_morph=False)
        logging.disable(logging.NOTSET)
    report["glb_bytes"] = os.path.getsize(OUT)
    with open(os.path.join(HERE, "convert_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "bone_heads"}, indent=1))


main()
