"""Event props (GFX pass 2026-10-11): detailed visuals for the small MuJoCo props that the event scenes draw as bare
boxes — pool cube, slalom pole, trench post, crab rail. Run in Blender (5.x):

    blender --python SourceArt/Props/build_props.py        (or exec() it through MCP for Blender)

Writes SourceArt/Props/event_props.blend and Assets/PoOlympic/Art/Props/EventProps.glb. Unity side:
BroadcastFx.DressProps hangs each mesh under its MjGeom (visual only; the collision box is untouched).

Rules: every prop is modelled in the MuJoCo geom frame (x forward, y left, z up, origin at the geom centre) at the
size of its collision box (SIZES below = MJCF half extents), and is mirror-symmetric in x and y, so the glTF → Unity
handedness flip cannot turn it. Nothing a foot can catch on sticks out of the collision box by more than 3 cm.
"""
import math
import os

import bmesh
import bpy

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BLEND = os.path.join(ROOT, "SourceArt", "Props", "event_props.blend")
GLB = os.path.join(ROOT, "Assets", "PoOlympic", "Art", "Props", "EventProps.glb")

# MJCF half extents (training/assets/cube.xml, scene_slalom8.xml, scene_trench8.xml, scene_crab8.xml)
SIZES = {"PoolCube": (0.1, 0.1, 0.1), "SlalomPole": (0.02, 0.02, 0.6), "TrenchPost": (0.025, 0.025, 0.36), "CrabRail": (0.02, 10.0, 0.02)}


def material(name, rgb, metallic, roughness, emission=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 1.5
    return m


def finish(name, bm, mats, bevel=0.0):
    """bmesh → object named `name` (mesh data too: Unity finds it by that name), smooth by angle, optional bevel."""
    mesh = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for m in mats:
        mesh.materials.append(m)
    if bevel > 0.0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        mod.limit_method = "ANGLE"
        mod.angle_limit = math.radians(40)
        mod.harden_normals = True
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for mod in list(obj.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.ops.object.shade_auto_smooth(angle=math.radians(40))
    for mod in list(obj.modifiers):                       # the smooth-by-angle node modifier
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.select_set(False)
    return obj


def box(bm, half, centre=(0, 0, 0), mat=0):
    r = bmesh.ops.create_cube(bm, size=1.0)
    for v in r["verts"]:
        v.co.x = v.co.x * 2 * half[0] + centre[0]
        v.co.y = v.co.y * 2 * half[1] + centre[1]
        v.co.z = v.co.z * 2 * half[2] + centre[2]
    faces = {f for v in r["verts"] for f in v.link_faces}
    for f in faces:
        f.material_index = mat
    return faces


def tube(bm, radius, half_len, centre=(0, 0, 0), axis="z", mat=0, segments=20):
    r = bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=2 * half_len)
    for v in r["verts"]:
        if axis == "y":
            v.co.y, v.co.z = v.co.z, v.co.y
        v.co.x += centre[0]
        v.co.y += centre[1]
        v.co.z += centre[2]
    for f in {f for v in r["verts"] for f in v.link_faces}:
        f.material_index = mat


def pool_cube(paint, dark):
    """A 20 cm steel crate: bevelled frame, a recessed dark panel on each face."""
    bm = bmesh.new()
    faces = list(box(bm, SIZES["PoolCube"]))
    inner = bmesh.ops.inset_individual(bm, faces=faces, thickness=0.028, depth=-0.008)
    for f in faces:                                   # after an individual inset the original faces are the panels
        f.material_index = 1
    del inner
    return finish("PoolCube", bm, [paint, dark], bevel=0.008)


def slalom_pole(red, white, rubber):
    """A 1.2 m gate pole: six red / white bands, a rounded top, a low rubber foot."""
    bm = bmesh.new()
    r, half = SIZES["SlalomPole"][0], SIZES["SlalomPole"][2]
    bands = 6
    for i in range(bands):
        h = (2 * half - 0.03) / bands
        tube(bm, r, h / 2, centre=(0, 0, -half + 0.03 + h * (i + 0.5)), mat=i % 2)
    cap = bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=8, radius=r)
    for v in cap["verts"]:
        v.co.z = max(v.co.z, 0.0) * 0.6 + half
    for f in {f for v in cap["verts"] for f in v.link_faces}:
        f.material_index = 0
    tube(bm, 0.045, 0.015, centre=(0, 0, -half + 0.015), mat=2, segments=24)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    return finish("SlalomPole", bm, [red, white, rubber], bevel=0.003)


def trench_post(steel, dark):
    """A 72 cm square steel post with a foot plate and a head plate (it carries the glass roof)."""
    bm = bmesh.new()
    hx, hy, hz = SIZES["TrenchPost"]
    box(bm, (hx, hy, hz - 0.012), mat=0)
    box(bm, (hx + 0.03, hy + 0.03, 0.006), centre=(0, 0, -hz + 0.006), mat=1)
    box(bm, (hx + 0.02, hy + 0.02, 0.006), centre=(0, 0, hz - 0.006), mat=1)
    return finish("TrenchPost", bm, [steel, dark], bevel=0.004)


def crab_rail(steel, dark):
    """A 20 m lane rail: a round tube with a collar every 2 m."""
    bm = bmesh.new()
    r, half = SIZES["CrabRail"][0], SIZES["CrabRail"][1]
    tube(bm, r, half, axis="y", mat=0)
    for i in range(11):
        tube(bm, r + 0.008, 0.03, centre=(0, -half + 2.0 * i + (0.03 if i == 0 else -0.03 if i == 10 else 0), 0), axis="y", mat=1)
    return finish("CrabRail", bm, [steel, dark], bevel=0.002)


def build():
    # clear the file by hand (read_factory_settings would also unload the add-on this may be running through)
    for block in (bpy.data.objects, bpy.data.meshes, bpy.data.materials):
        for item in list(block):
            block.remove(item)
    steel = material("PropSteel", (0.55, 0.57, 0.6), 1.0, 0.35)
    dark = material("PropDarkSteel", (0.08, 0.09, 0.11), 1.0, 0.5)
    paint = material("PropCratePaint", (0.95, 0.42, 0.08), 0.6, 0.4)
    red = material("PropPoleRed", (0.85, 0.07, 0.06), 0.0, 0.45, emission=(0.25, 0.01, 0.01))
    white = material("PropPoleWhite", (0.92, 0.92, 0.9), 0.0, 0.45)
    rubber = material("PropRubber", (0.03, 0.03, 0.03), 0.0, 0.9)
    objs = [pool_cube(paint, dark), slalom_pole(red, white, rubber), trench_post(steel, dark), crab_rail(steel, dark)]
    for i, o in enumerate(objs):                      # laid out side by side in the .blend; the export resets them
        o.location = (0.6 * i, 0, 0)
    os.makedirs(os.path.dirname(BLEND), exist_ok=True)
    os.makedirs(os.path.dirname(GLB), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=BLEND)
    for o in objs:
        o.location = (0, 0, 0)
    bpy.ops.export_scene.gltf(filepath=GLB, export_format="GLB", export_apply=True, export_yup=True)
    for i, o in enumerate(objs):
        o.location = (0.6 * i, 0, 0)
    return {o.name: (len(o.data.polygons), tuple(round(d, 3) for d in o.dimensions)) for o in objs}


if __name__ == "__main__":
    print(build())
