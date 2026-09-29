"""PoOlympics stadium — realism / game-ready pass (render-only). Run inside Blender after build_venues.py and
build_dressing.py, on SourceArt/Stadium/stadium.blend (textures: python SourceArt/Stadium/fetch_textures.py first):
    exec(open(r"<repo>/SourceArt/Stadium/build_realism.py").read())

Game-ready checklist (user request, "top 20"), what this script does:
  units        metric, unit scale 1.0; mesh objects with rotation/scale get them applied (empties = the E##_L# lane
               anchors are NEVER touched: Unity places athletes by them)
  PBR          CC0 Poly Haven sets (1K): base colour (sRGB) + OpenGL normal + ARM (R AO / G rough / B metal = glTF
               occlusion + metallicRoughness, i.e. channel-packed ORM) on concrete, track, grass, paving, steel, wood,
               rock, slate, rubber, parkour plaster; generated turf albedo with mowing stripes
  UVs          world-scale box-projected tiling UVs ("UVMap") on the faces of textured materials only (crowd / seat UVs
               untouched), random offset per object against visible repetition; second UV set "UVLightmap"
               (lightmap pack, per object, 0-1, padded) on every mesh for baked lighting in the engine
  normals      bevel (angle-limited, 1 segment, hardened normals) + Weighted Normal on hard-surface props
  albedo       every base colour clamped to the physically plausible 0.04..0.9 range
  glass/metal  facade ribbon = alpha-blended glass; steel = textured metal
  decals       rubber wear in front of the sprint start lines (between the track surface and the lane lines)
  ground       kerb ring between plaza and park, soil discs under the trees, facade plinth
  hygiene      orphan data purged; export (done by the caller) with tangents, both UV sets, modifiers applied
Scene-level AO comes from Unity's SSAO (baking AO textures for ~1500 objects would multiply the GLB); LODs are not
used because the surroundings are merged, stadium-sized objects (one draw call each) that a distance LOD cannot switch.
"""

import math
import os
import random

import bmesh
import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"C:\Users\punko\Downloads\PoOlympic\SourceArt\Stadium"
exec(open(os.path.join(HERE, "stadium_helpers.py")).read())
TEX = os.path.join(HERE, "textures")
ROOT_COL = bpy.data.collections["Stadium"]
FACADE_R, ROOF_Z, ROOF_OUT_R = 94.5, 33.4, 97.5          # = build_dressing.py
RNG = random.Random(42)

# material -> (texture set, metres per texture tile, tint or None, downscale to 512)
PBR = {
    "Concrete": ("concrete_floor_worn_001", 3.0, None, False),
    "Stair_Concrete": ("concrete_floor_worn_001", 2.0, (0.85, 0.8, 0.74), True),
    "Roof_Under": ("concrete_floor_worn_001", 4.0, (0.8, 0.82, 0.86), True),
    "Kerb_Grey": ("concrete_floor_worn_001", 1.5, None, True),
    "Plaza": ("hexagonal_concrete_paving", 1.6, None, False),
    "Track_Red": ("rubberized_track", 2.0, None, False),
    "Steel": ("metal_plate_02", 2.0, None, True),
    "Gate_Frame": ("metal_plate_02", 2.0, None, True),
    "Wood": ("brown_planks_03", 1.0, None, True),
    "Rock_A": ("rock_face_03", 2.7, None, True),
    "Rock_B": ("rock_face_03", 2.7, (1.15, 1.1, 1.0), True),
    "Rock_C": ("rock_face_03", 2.7, (0.75, 0.75, 0.78), True),
    "Slate": ("slab_tiles", 2.4, (0.55, 0.6, 0.7), True),
    "Ramp_Rubber": ("rubber_tiles", 2.0, (0.35, 0.55, 1.0), True),
    "Parkour_Wall": ("painted_plaster_wall", 2.0, (1.0, 0.55, 0.2), True),
    "Grass_Infield": ("TURF_STRIPES", 10.0, None, False),
    "Grass_Outfield": ("TURF", 6.0, None, False),
    "Park_Grass": ("TURF", 6.0, None, False),
}


# ------------------------------------------------------------------------------------------------------------ images
def load_image(name, path, colorspace):
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.load(path)
        img.name = name
    img.colorspace_settings.name = colorspace
    return img


def tinted(base, tint, name, half):
    """A packed copy of `base` multiplied by `tint` (glTF has no texture x factor we can rely on in every engine
    importer, so the tint is baked into the pixels); optionally downscaled to 512."""
    img = bpy.data.images.get(name)
    if img is not None:
        return img
    img = base.copy()
    img.name = name
    if half:
        img.scale(512, 512)
    if tint:
        px = np.empty(len(img.pixels), np.float32)
        img.pixels.foreach_get(px)
        px = px.reshape(-1, 4)
        px[:, :3] = np.clip(px[:, :3] * np.asarray(tint, np.float32), 0, 1)
        img.pixels.foreach_set(px.ravel())
    img.pack()
    return img


def turf(name, stripes):
    """Generated stadium turf albedo (1024^2): fine green noise; with mowing stripes (2 per tile) for the infield."""
    img = bpy.data.images.get(name)
    if img is not None:
        return img
    rng = np.random.default_rng(3)
    n = 1024
    base = np.array([0.16, 0.36, 0.13], np.float32)
    noise = rng.normal(0, 1, (n, n)).astype(np.float32)
    k = np.ones(5, np.float32) / 5                               # cheap blur -> blades clump
    noise = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, noise)
    noise = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, noise)
    shade = 1.0 + 0.10 * noise + 0.05 * rng.normal(0, 1, (n, n))
    if stripes:
        x = np.arange(n) / n
        shade = shade * (1.0 + 0.09 * np.sign(np.sin(2 * np.pi * 2 * x)))[None, :]
    px = np.ones((n, n, 4), np.float32)
    px[..., :3] = np.clip(base[None, None, :] * shade[..., None], 0, 1)
    img = bpy.data.images.new(name, n, n)
    img.pixels.foreach_set(px.ravel())
    img.pack()
    img.colorspace_settings.name = "sRGB"
    return img


def texture_set(key, tint, half, mat_name):
    if key.startswith("TURF"):
        diff = turf(f"T_{key}", stripes=key.endswith("STRIPES"))
        src = "grass_ground"
    else:
        diff = load_image(f"{key}_diff", os.path.join(TEX, f"{key}_diff_1k.jpg"), "sRGB")
        src = key
        if tint or half:
            diff = tinted(diff, tint, f"{key}_diff_{mat_name}", half)
    nor = load_image(f"{src}_nor", os.path.join(TEX, f"{src}_nor_gl_1k.jpg"), "Non-Color")
    arm = load_image(f"{src}_arm", os.path.join(TEX, f"{src}_arm_1k.jpg"), "Non-Color")
    if half and not key.startswith("TURF"):
        nor = tinted(nor, None, f"{src}_nor_512", True)
        arm = tinted(arm, None, f"{src}_arm_512", True)
        nor.colorspace_settings.name = arm.colorspace_settings.name = "Non-Color"
    return diff, nor, arm


def gltf_output_group():
    """The 'glTF Material Output' node group the Blender glTF exporter reads the occlusion texture from."""
    g = bpy.data.node_groups.get("glTF Material Output")
    if g is None:
        g = bpy.data.node_groups.new("glTF Material Output", "ShaderNodeTree")
        g.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
        g.interface.new_socket("Thickness", in_out="INPUT", socket_type="NodeSocketFloat")
    return g


def pbr_material(m, diff, nor, arm):
    """Rebuild material `m` as base colour + normal + ARM (glTF-exportable node pattern)."""
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    emission = (tuple(bsdf.inputs["Emission Color"].default_value), bsdf.inputs["Emission Strength"].default_value)
    for n in list(nt.nodes):
        if n not in (bsdf,) and n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.uv_map = "UVMap"; uv.location = (-1000, 0)
    t_d = nt.nodes.new("ShaderNodeTexImage"); t_d.image = diff; t_d.location = (-700, 300)
    t_a = nt.nodes.new("ShaderNodeTexImage"); t_a.image = arm; t_a.location = (-700, 0)
    t_n = nt.nodes.new("ShaderNodeTexImage"); t_n.image = nor; t_n.location = (-700, -300)
    sep = nt.nodes.new("ShaderNodeSeparateColor"); sep.location = (-400, 0)
    nmap = nt.nodes.new("ShaderNodeNormalMap"); nmap.location = (-400, -300); nmap.uv_map = "UVMap"
    for t in (t_d, t_a, t_n):
        nt.links.new(uv.outputs["UV"], t.inputs["Vector"])
    nt.links.new(t_d.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(t_a.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    nt.links.new(t_n.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    grp = nt.nodes.new("ShaderNodeGroup"); grp.node_tree = gltf_output_group(); grp.location = (200, -300)
    nt.links.new(sep.outputs["Red"], grp.inputs["Occlusion"])
    bsdf.inputs["Emission Color"].default_value, bsdf.inputs["Emission Strength"].default_value = emission


# --------------------------------------------------------------------------------------------------------------- UVs
def stadium_meshes():
    return [o for o in ROOT_COL.all_objects if o.type == "MESH"]


def box_project(ob, textured: dict):
    """World-scale box projection into "UVMap" for the faces whose material is in `textured` (name -> metres per
    tile); other faces keep their UVs. Random per-object offset breaks repetition."""
    me = ob.data
    idx = {i for i, m in enumerate(me.materials) if m and m.name in textured}
    if not idx:
        return 0
    if "UVMap" not in me.uv_layers:
        me.uv_layers.new(name="UVMap")
    bm = bmesh.new()
    bm.from_mesh(me)
    uvl = bm.loops.layers.uv["UVMap"]
    M = ob.matrix_world
    R = M.to_3x3()
    ou, ov = RNG.random(), RNG.random()
    n = 0
    for f in bm.faces:
        if f.material_index not in idx:
            continue
        tile = textured[me.materials[f.material_index].name]
        nrm = (R @ f.normal)
        ax = max(range(3), key=lambda i: abs(nrm[i]))
        for loop in f.loops:
            p = M @ loop.vert.co
            if ax == 2:
                u, v = p.x, p.y
            elif ax == 0:
                u, v = p.y, p.z
            else:
                u, v = p.x, p.z
            loop[uvl].uv = (u / tile + ou, v / tile + ov)
        n += 1
    bm.to_mesh(me)
    bm.free()
    return n


def lightmap_uvs(obs):
    """Second UV set for baked lighting: per-object lightmap pack in 0-1 with padding."""
    for ob in obs:
        me = ob.data
        if "UVLightmap" not in me.uv_layers:
            me.uv_layers.new(name="UVLightmap")
        me.uv_layers["UVMap" if "UVMap" in me.uv_layers else me.uv_layers[0].name].active_render = True
        me.uv_layers.active = me.uv_layers["UVLightmap"]
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for ob in obs:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES", PREF_PACK_IN_ONE=False, PREF_NEW_UVLAYER=False,
                             PREF_BOX_DIV=12, PREF_MARGIN_DIV=0.2)
    bpy.ops.object.mode_set(mode="OBJECT")
    for ob in obs:                                               # the tiling UVs stay the first / active set
        ob.data.uv_layers.active = ob.data.uv_layers[0]


# ------------------------------------------------------------------------------------------------------ geometry
def apply_mesh_transforms():
    bpy.ops.object.select_all(action="DESELECT")
    todo = [o for o in stadium_meshes()
            if any(abs(a) > 1e-6 for a in o.rotation_euler) or any(abs(s - 1) > 1e-6 for s in o.scale)]
    for o in todo:
        o.select_set(True)
    if todo:
        bpy.context.view_layer.objects.active = todo[0]
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return len(todo)


HARD_SURFACE = ("E01_Pedestal", "E05_Shaker", "E14_", "E16_", "E17_", "E24_Crate", "E25_Bench", "E26_Stone",
                "E30_", "E02_Mast", "E04_Reach", "E18_Pole", "E19_Trap", "Scoreboard_", "Roof_Column")


def bevel_hard_surfaces():
    n = 0
    for ob in stadium_meshes():
        if not ob.name.startswith(HARD_SURFACE) or len(ob.data.vertices) < 8:
            continue
        dims = ob.dimensions
        if min(dims) < 0.03:                                     # flat decals / thin strips: nothing to bevel
            continue
        for mod in [m for m in ob.modifiers if m.name in ("RealismBevel", "RealismWN")]:
            ob.modifiers.remove(mod)
        bev = ob.modifiers.new("RealismBevel", "BEVEL")
        bev.width = min(0.03, 0.15 * min(dims))
        bev.segments = 1
        bev.limit_method = "ANGLE"
        bev.angle_limit = math.radians(30)
        bev.harden_normals = True
        wn = ob.modifiers.new("RealismWN", "WEIGHTED_NORMAL")
        wn.keep_sharp = True
        n += 1
    return n


def clamp_albedo():
    n = 0
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        bsdf = next((nd for nd in m.node_tree.nodes if nd.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None or bsdf.inputs["Base Color"].is_linked:
            continue
        c = bsdf.inputs["Base Color"].default_value
        new = [min(0.9, max(0.04, c[i])) for i in range(3)]
        if any(abs(new[i] - c[i]) > 1e-6 for i in range(3)):
            bsdf.inputs["Base Color"].default_value = (*new, c[3])
            n += 1
    return n


def glass():
    m = bpy.data.materials["Glass_Ribbon"]
    bsdf = next(nd for nd in m.node_tree.nodes if nd.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.18, 0.32, 0.45, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.05
    bsdf.inputs["Alpha"].default_value = 0.55
    bsdf.inputs["Emission Strength"].default_value = 0.15
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"


def ground_details():
    col = get_col("GroundDetail")
    for ob in list(col.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    kerb = bpy.data.materials["Kerb_Grey"]
    soil = mat("Tree_Soil", (0.22, 0.16, 0.11), 0.95)
    wear = mat("Track_Wear", (0.2, 0.07, 0.06), 0.95)
    # kerb between plaza and park + a plinth at the facade foot
    for name, r0, r1, z in (("Kerb_Plaza", 134.3, 134.7, 0.15), ("Facade_Plinth", 94.5, 95.2, 0.6)):
        ring(name + "_Top", r0, r1, z, kerb, col="GroundDetail", n_arc=48)
        wall(name + "_Face", r1, -0.03, z, kerb, col="GroundDetail", n_arc=48)
    # soil disc under every tree: replay build_dressing.build_surroundings' tree RNG (seed 7) for the trunk positions
    rng = random.Random(7)
    verts, faces = [], []
    placed = 0
    while placed < 260:
        r = rng.uniform(FACADE_R + 48, FACADE_R + 240)
        a = rng.uniform(0, 2 * math.pi)
        x0 = L / 2 * math.copysign(1, math.cos(a)) if abs(math.cos(a)) > 0.2 else 0.0
        x, y = x0 + r * math.cos(a), r * math.sin(a)
        if abs(y) < 14 and abs(x) > L / 2 or abs(x) < 14:
            continue
        rng.uniform(5, 11)
        rng.random()
        placed += 1
        k = len(verts)
        verts += [(x + 1.3 * math.cos(2 * math.pi * i / 10), y + 1.3 * math.sin(2 * math.pi * i / 10), -0.025) for i in range(10)]
        faces.append(tuple(range(k, k + 10)))
    mesh_obj("Tree_Soil_Discs", verts, faces, soil, "GroundDetail")
    # rubber wear behind the 100 m start line (events 08/20/22 share it): on the start extension (z 0.012), under its
    # lane lines (0.024)
    import json
    ev = json.load(open(os.path.join(HERE, "venues.json")))["events"]
    wv, wf = [], []
    for lane in ev["08"]["lanes"]:
        x, y, _ = lane["pos"]
        for x0, x1, w in ((x - 0.9, x + 0.1, 0.34), (x + 0.1, x + 2.5, 0.22)):   # blocks scuff, then push-off strip
            k = len(wv)
            wv += [(x0, y - w, 0.018), (x1, y - w, 0.018), (x1, y + w, 0.018), (x0, y + w, 0.018)]
            wf.append((k, k + 1, k + 2, k + 3))
    mesh_obj("Track_Wear_Starts", wv, wf, wear, "GroundDetail")


def split_animated():
    """Flags (one object per pole, pivot on the pole, subdivided so a vertex/cloth wave has vertices) and the cauldron
    flame (pivot at its base) become separate objects so Unity can animate them."""
    ident = bpy.data.objects.get("Identity")
    if ident is None:
        return 0
    me = ident.data
    names = [m.name if m else "" for m in me.materials]
    bm = bmesh.new()
    bm.from_mesh(me)
    poles = oval(ROOF_OUT_R - 1.0, 12)[::2]
    groups = {}
    for f in bm.faces:
        n = names[f.material_index]
        if n.startswith("Flame_"):
            groups.setdefault("Cauldron_Flame", []).append(f)
        elif n.startswith("Flag_") and n != "Flag_Pole":
            c = f.calc_center_median()
            i = min(range(len(poles)), key=lambda j: (poles[j][0] - c.x) ** 2 + (poles[j][1] - c.y) ** 2)
            groups.setdefault(f"Flag_{i:02d}", []).append(f)
    made = 0
    for name, faces in sorted(groups.items()):
        pivot = Vector((0.0, 78.0, ROOF_Z + 6.0)) if name == "Cauldron_Flame" else             Vector((*poles[int(name[5:])], ROOF_Z + 5.9))
        nb = bmesh.new()
        vmap = {}
        for f in faces:
            vs = []
            for v in f.verts:
                if v not in vmap:
                    vmap[v] = nb.verts.new(v.co - pivot)
                vs.append(vmap[v])
            nf = nb.faces.new(vs)
            nf.material_index = f.material_index
        if name.startswith("Flag_"):
            bmesh.ops.subdivide_edges(nb, edges=nb.edges[:], cuts=3, use_grid_fill=True)
        nme = bpy.data.meshes.new(name)
        nb.to_mesh(nme)
        nb.free()
        for m in me.materials:
            nme.materials.append(m)
        ob = mesh_obj(name, [], [], None, "Identity")
        old = ob.data
        ob.data = nme
        bpy.data.meshes.remove(old)
        ob.location = pivot
        made += 1
    bmesh.ops.delete(bm, geom=[f for fs in groups.values() for f in fs], context="FACES")
    bm.to_mesh(me)
    bm.free()
    for ob in [o for o in ROOT_COL.all_objects if o.type == "MESH" and (o.name.startswith("Flag_") or o.name == "Cauldron_Flame")]:
        used = {p.material_index for p in ob.data.polygons}          # drop unused material slots
        keep = sorted(used)
        mats = [ob.data.materials[i] for i in keep]
        remap = {old: new for new, old in enumerate(keep)}
        for p in ob.data.polygons:
            p.material_index = remap[p.material_index]
        ob.data.materials.clear()
        for m in mats:
            ob.data.materials.append(m)
    return made


def anchor_matrices():
    return {o.name: [round(v, 6) for row in o.matrix_world for v in row] for o in bpy.data.objects if o.type == "EMPTY"}


# ----------------------------------------------------------------------------------------------------------- main
ANCHORS_BEFORE = anchor_matrices()
bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.scale_length = 1.0
report = {"transforms_applied": apply_mesh_transforms(), "albedo_clamped": clamp_albedo()}
glass()
ground_details()
report["animated_split"] = split_animated()
tiles = {}
for mname, (key, tile, tint, half) in PBR.items():
    m = bpy.data.materials.get(mname)
    if m is None:
        continue
    pbr_material(m, *texture_set(key, tint, half, mname))
    tiles[mname] = tile
report["pbr_materials"] = len(tiles)
report["textured_faces"] = sum(box_project(o, tiles) for o in stadium_meshes())
report["bevelled_objects"] = bevel_hard_surfaces()
lightmap_uvs(stadium_meshes())
report["lightmap_uv_objects"] = len(stadium_meshes())
bpy.data.orphans_purge(do_recursive=True)
assert anchor_matrices() == ANCHORS_BEFORE, "lane anchors moved"
report["anchors_unchanged"] = len(ANCHORS_BEFORE)
print("realism:", report)
