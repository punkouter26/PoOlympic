"""PoOlympics stadium — polish pass ("do all 20", 2026-09-28): the items of the 20-point game-ready list that
build_realism.py did not cover. Render-only. Run inside Blender after build_realism.py (and build_indoor.py):
    exec(open(r"<repo>/SourceArt/Stadium/build_polish.py").read())
Export afterwards with export_stadium() (end of this file): vertex colours, tangents, both UV sets, modifiers applied,
glTF scene renamed "Scene" (Unity scene references depend on it).

  1  bevel         bevel + Weighted Normal extended to the hard-surface objects build_realism.py skipped (roof columns,
                   frames, blocks, …)
  2  n-gons        Triangulate modifier (n-gons only, custom normals kept) on every mesh: tangents for normal maps
  3  pivots        venue props: origin at the base centre of their bounds (move / rotate naturally in Unity);
                   transforms were already applied, anchors (empties) untouched
  4  detail        drain grating ring along the infield side of the inner kerb; rubber edging round every event pad
  5  variation     per-object tint jitter (±4 % brightness, ±1.5 % hue) in the vertex colour — the 8 copies of every
                   lane prop no longer look cloned
  8  roughness     shared tiling grunge ARM (roughness variation) on every flat-colour material
  9  wear          shared tiling grunge albedo (stains, mottling) on the same materials + vertex-colour grime at the
                   base of standing objects (darker within GRIME_H of the object's foot)
 10  metalness     binary: painted / plastic / glass = 0, bare metal = 1
 13  detail maps   the grunge pair is one shared tiling detail set (trim-sheet style) instead of unique textures
 14  AO            ambient occlusion baked into the vertex colour of standing props (Cycles, AO_DISTANCE)
 16  textures      every image ≤ 1K (JPEG on export); the grunge pair is 512
 18  naming        checked: no default / duplicate-suffix names
Items handled elsewhere: 6, 7, 11, 12, 20 (build_realism.py), 17 (Unity static batching, StadiumLook), 19 (indoor
lighting: build_indoor.py + StadiumLook), 15 (no high-poly sources: bevel geometry + weighted normals instead of
baked normal maps).
"""

import colorsys
import math
import os
import random
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"C:\Users\punko\Downloads\PoOlympic\SourceArt\Stadium"
exec(open(os.path.join(HERE, "stadium_helpers.py")).read())
TEX = os.path.join(HERE, "textures")
ROOT_COL = bpy.data.collections["Stadium"]
RNG = random.Random(20260928)
BAKE_AO = True
AO_DISTANCE = 1.5
GRIME_H, GRIME = 0.45, 0.16
GRUNGE_TILE = 1.6                        # m per grunge tile
# flat-colour (untextured) opaque materials that get the grunge detail pair; emissive screens, glass, flames, logos
# and cloth keep their clean look
GRUNGE = ["Line_White", "Marker_Orange", "Stop_Red", "GymMat_Blue", "Hazard_Yellow", "IronPedestal",
          "EventPad_P1", "EventPad_P2", "EventPad_P3", "EventPad_P4", "EventPad_P5", "EventPad_P6", "Seats_Blue",
          "Seats_White", "Foam_Blue", "Sand", "Tree_Trunk", "City_Concrete", "Tree_Soil", "Track_Wear", "Stair_Nosing",
          "Start_Block", "Start_Pad", "Cauldron", "Flag_Pole", "Net_White"]
METAL = {"IronPedestal": 1.0, "Title_Gold": 1.0, "Start_Block": 1.0, "City_Glass": 0.0, "Flag_Pole": 1.0,
         "Cauldron": 1.0, "Floodlight": 0.0}
NO_VCOL = ("Board_LED", "Scoreboard_Screen", "Floodlight", "Flame_", "Glass_Ribbon", "Crowd_", "Spot_Lens", "Rings_",
           "Title_", "Flag_")
HARD = {"Steel", "Concrete", "Stair_Concrete", "Wood", "Kerb_Grey", "IronPedestal", "Start_Block", "Cauldron",
        "Gate_Frame", "Flag_Pole", "Hazard_Yellow", "Marker_Orange", "Stop_Red", "Foam_Blue", "GymMat_Blue"}


def stadium_meshes():
    return [o for o in ROOT_COL.all_objects if o.type == "MESH"]


def mats_of(o):
    return [s.material.name for s in o.material_slots if s.material]


def bounds(o):
    ws = [o.matrix_world @ v.co for v in o.data.vertices]
    lo = Vector((min(p.x for p in ws), min(p.y for p in ws), min(p.z for p in ws)))
    hi = Vector((max(p.x for p in ws), max(p.y for p in ws), max(p.z for p in ws)))
    return lo, hi


# ------------------------------------------------------------------------------------------------- 16 textures
def periodic_noise(n, beta, seed):
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    k = np.sqrt(fx ** 2 + fy ** 2)
    k[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / k ** beta
    spec[0, 0] = 0
    x = np.real(np.fft.ifft2(spec))
    return (x - x.mean()) / (np.abs(x).max() + 1e-9)          # tileable, zero mean, in [-1, 1]


def save_image(name, rgb, colorspace):
    path = os.path.join(TEX, name + ".png")
    n = rgb.shape[0]
    img = bpy.data.images.get(name) or bpy.data.images.new(name, n, n, alpha=False)
    if img.size[0] != n:
        img.scale(n, n)
    px = np.ones((n, n, 4), np.float32)
    px[..., :3] = rgb
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    img.colorspace_settings.name = colorspace
    return img


def grunge_images():
    n = 512
    mott = periodic_noise(n, 1.1, 1)
    fine = periodic_noise(n, 0.6, 2)
    stain = np.clip(periodic_noise(n, 1.6, 3), 0, 1) ** 1.5
    # albedo multiplier (sRGB), mean ~0.97 so painted colours keep their brightness
    alb = np.clip(0.975 + 0.035 * mott + 0.015 * fine - 0.12 * stain, 0.8, 1.0)
    albedo = save_image("Grunge_Albedo", np.repeat(alb[..., None], 3, axis=2), "sRGB")
    # ARM (Non-Color): R occlusion 1, G roughness multiplier (mean 0.85), B metal multiplier 1
    g = np.clip(0.85 + 0.12 * mott + 0.05 * fine + 0.1 * stain, 0.55, 1.0)
    arm = np.stack([np.ones_like(g), g, np.ones_like(g)], axis=2)
    armimg = save_image("Grunge_ARM", arm, "Non-Color")
    return albedo, armimg


def grunge_material(m, albedo, arm):
    """Base colour = factor × grunge albedo, roughness = factor × grunge G, metallic constant (binary). The glTF
    exporter reads the multiply nodes as baseColorFactor / roughnessFactor."""
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    for n in [n for n in nt.nodes if n.get("polish")]:
        nt.nodes.remove(n)
    col = tuple(b.inputs["Base Color"].default_value)
    if b.inputs["Base Color"].is_linked:
        return False
    rough = b.inputs["Roughness"].default_value
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.uv_map = "UVMap"; uv["polish"] = 1; uv.location = (-900, 200)
    ta = nt.nodes.new("ShaderNodeTexImage"); ta.image = albedo; ta["polish"] = 1; ta.location = (-650, 300)
    tr = nt.nodes.new("ShaderNodeTexImage"); tr.image = arm; tr["polish"] = 1; tr.location = (-650, -50)
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"; mix["polish"] = 1
    mix.inputs["Factor"].default_value = 1.0
    sock = lambda socks, ident: next(x for x in socks if x.identifier == ident)
    sock(mix.inputs, "B_Color").default_value = col
    mix.location = (-350, 300)
    sep = nt.nodes.new("ShaderNodeSeparateColor"); sep["polish"] = 1; sep.location = (-400, -50)
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul["polish"] = 1; mul.location = (-200, -50)
    mul.inputs[1].default_value = min(rough / 0.85, 1.0)
    L = nt.links
    L.new(uv.outputs["UV"], ta.inputs["Vector"]); L.new(uv.outputs["UV"], tr.inputs["Vector"])
    L.new(ta.outputs["Color"], sock(mix.inputs, "A_Color"))
    L.new(sock(mix.outputs, "Result_Color"), b.inputs["Base Color"])
    L.new(tr.outputs["Color"], sep.inputs["Color"])
    L.new(sep.outputs["Green"], mul.inputs[0])
    L.new(mul.outputs["Value"], b.inputs["Roughness"])
    return True


def grunge_uvs(o, names):
    """World box projection (GRUNGE_TILE m per tile) into UVMap for the faces of grunge materials; UVMap becomes the
    first UV layer (glTF TEXCOORD_0) if the object had only its lightmap UVs."""
    me = o.data
    idx = {i for i, s in enumerate(o.material_slots) if s.material and s.material.name in names}
    if not idx:
        return 0
    if "UVMap" not in me.uv_layers:
        old = [(l.name, [d.uv.copy() for d in l.data]) for l in me.uv_layers]
        for l in list(me.uv_layers):
            me.uv_layers.remove(l)
        me.uv_layers.new(name="UVMap")
        for name, data in old:
            nl = me.uv_layers.new(name=name)
            for d, uv in zip(nl.data, data):
                d.uv = uv
    uv = me.uv_layers["UVMap"]
    ox, oy = RNG.random(), RNG.random()
    mw = o.matrix_world
    n = 0
    for poly in me.polygons:
        if poly.material_index not in idx:
            continue
        nrm = (mw.to_3x3() @ poly.normal).normalized()
        for li in poly.loop_indices:
            p = mw @ me.vertices[me.loops[li].vertex_index].co
            if abs(nrm.z) >= max(abs(nrm.x), abs(nrm.y)):
                u, v = p.x, p.y
            elif abs(nrm.x) >= abs(nrm.y):
                u, v = p.y, p.z
            else:
                u, v = p.x, p.z
            uv.data[li].uv = (u / GRUNGE_TILE + ox, v / GRUNGE_TILE + oy)
        n += 1
    me.uv_layers.active = uv
    uv.active_render = True
    return n


# ------------------------------------------------------------------------------------------------- 10 metalness
def binary_metal():
    done = {}
    for name, v in METAL.items():
        m = bpy.data.materials.get(name)
        if m and m.use_nodes:
            b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            if not b.inputs["Metallic"].is_linked:
                b.inputs["Metallic"].default_value = v
                done[name] = v
    return done


# ------------------------------------------------------------------------------------------------- 1 / 2 modifiers
def modifiers():
    bev = tri = 0
    for o in stadium_meshes():
        lo, hi = bounds(o)
        mats = set(mats_of(o))
        standing = min(hi - lo) > 0.05
        if (standing and mats and mats <= HARD and not any(m.type == "BEVEL" for m in o.modifiers)
                and len(o.data.polygons) < 3000 and not o.name.startswith(("Roof_Trusses", "Spot_"))):
            b = o.modifiers.new("PolishBevel", "BEVEL")
            b.width = 0.012; b.segments = 1; b.limit_method = "ANGLE"; b.angle_limit = math.radians(30)
            b.harden_normals = True
            w = o.modifiers.new("PolishWN", "WEIGHTED_NORMAL"); w.keep_sharp = True
            bev += 1
        if not any(m.type == "TRIANGULATE" for m in o.modifiers):
            t = o.modifiers.new("PolishTriangulate", "TRIANGULATE")
            t.min_vertices = 5
            t.keep_custom_normals = True
            tri += 1
    return bev, tri


# ------------------------------------------------------------------------------------------------- 3 pivots
def venue_pivots():
    n = 0
    for o in bpy.data.collections["Venues"].all_objects:
        if o.type != "MESH" or o.data.users > 1:
            continue
        lo, hi = bounds(o)
        base = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
        off = base - o.matrix_world.translation
        if off.length < 0.01:
            continue
        o.data.transform(Matrix.Translation(-off))
        o.matrix_world.translation += off
        n += 1
    return n


# ------------------------------------------------------------------------------------------------- 4 detail
def details():
    made = []
    steel = bpy.data.materials["Steel"]
    rubber = bpy.data.materials["Ramp_Rubber"]
    col = "GroundDetail"
    # drain grating on the infield side of the inner kerb (kerb 36.20-36.25 m, grass at 0)
    ob = ring("Infield_Drain", 35.92, 36.19, 0.004, steel, col, n_arc=128)
    made.append(ob)
    # rubber edging round every event pad (pads at Z_PAD = 0.02; lines/labels sit on the pad at 0.03-0.064)
    for pad in [o for o in ROOT_COL.all_objects if o.type == "MESH" and any(m.startswith("EventPad") for m in mats_of(o))]:
        lo, hi = bounds(pad)
        name = pad.name + "_Edge"
        bm = bmesh.new()
        e, h = 0.07, 0.045
        for (x0, y0, x1, y1) in ((lo.x - e, lo.y - e, hi.x + e, lo.y), (lo.x - e, hi.y, hi.x + e, hi.y + e),
                                 (lo.x - e, lo.y, lo.x, hi.y), (hi.x, lo.y, hi.x + e, hi.y)):
            g = bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.transform(bm, verts=g["verts"], matrix=Matrix.Translation(((x0 + x1) / 2, (y0 + y1) / 2, h / 2))
                                @ Matrix.Diagonal((x1 - x0, y1 - y0, h, 1.0)))
        old = bpy.data.objects.get(name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me); bm.free()
        o = bpy.data.objects.new(name, me)
        get_col(col).objects.link(o)
        me.materials.append(rubber)
        made.append(o)
    for o in made:                                            # textured materials: world-box UVMap + lightmap UVs
        me = o.data
        uv = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                p = me.vertices[me.loops[li].vertex_index].co
                n = poly.normal
                uv.data[li].uv = ((p.x, p.y) if abs(n.z) > 0.7 else (p.y, p.z) if abs(n.x) > abs(n.y) else (p.x, p.z))
                uv.data[li].uv = (uv.data[li].uv[0] / 1.0, uv.data[li].uv[1] / 1.0)
        me.uv_layers.new(name="UVLightmap")
    bpy.ops.object.select_all(action="DESELECT")
    for o in made:
        o.select_set(True)
        o.data.uv_layers.active = o.data.uv_layers["UVLightmap"]
    bpy.context.view_layer.objects.active = made[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES", PREF_PACK_IN_ONE=False, PREF_NEW_UVLAYER=False,
                             PREF_BOX_DIV=12, PREF_MARGIN_DIV=0.2)
    bpy.ops.object.mode_set(mode="OBJECT")
    for o in made:
        o.data.uv_layers.active = o.data.uv_layers["UVMap"]
        o.data.uv_layers["UVMap"].active_render = True
    return len(made)


# ------------------------------------------------------------------------------------------------- 5 / 9 / 14 vertex colour
def vcol_targets():
    return [o for o in stadium_meshes() if not any(m.startswith(NO_VCOL) for m in mats_of(o))]


def standing(o):
    lo, hi = bounds(o)
    return (hi.z - lo.z) > 0.08


def bake_ao(obs):
    """Cycles AO bake into the vertex colour attribute "AO" of the standing props (their contact shadows)."""
    sc = bpy.context.scene
    engine = sc.render.engine
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 32
    sc.cycles.device = "GPU" if bpy.context.preferences.addons.get("cycles") else "CPU"
    sc.world.light_settings.distance = AO_DISTANCE
    for o in obs:
        ca = o.data.color_attributes.get("AO") or o.data.color_attributes.new("AO", "BYTE_COLOR", "CORNER")
        o.data.color_attributes.active_color = ca
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    t = time.time()
    bpy.ops.object.bake(type="AO", target="VERTEX_COLORS")
    sc.render.engine = engine
    return round(time.time() - t, 1)


def vertex_colours(ao_objs):
    """Col = tint jitter × base grime × AO (linear multipliers; glTF COLOR_0 multiplies the base colour)."""
    n = 0
    for o in vcol_targets():
        me = o.data
        if "Col" in me.color_attributes:
            me.color_attributes.remove(me.color_attributes["Col"])
        col = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
        br = 1.0 + RNG.uniform(-0.04, 0.04)
        hue = RNG.uniform(-0.015, 0.015)
        r, g, b = colorsys.hsv_to_rgb((0.08 + hue) % 1.0, 0.04, br)   # a hint of warm/cool per object
        tint = np.array([r, g, b]) / max(r, g, b) * br
        lo, hi = bounds(o)
        tall = (hi.z - lo.z) > 0.08
        ao = me.color_attributes.get("AO") if o in ao_objs else None
        mw = o.matrix_world
        zs = np.array([(mw @ me.vertices[l.vertex_index].co).z for l in me.loops])
        grime = 1.0 - GRIME * np.clip(1.0 - (zs - lo.z) / GRIME_H, 0.0, 1.0) if tall else np.ones(len(zs))
        occ = np.array([d.color[0] for d in ao.data]) if ao is not None else np.ones(len(zs))
        occ = 0.45 + 0.55 * occ
        vals = np.clip(tint[None, :] * (grime * occ)[:, None], 0.0, 1.0)
        rgba = np.concatenate([vals, np.ones((len(zs), 1))], axis=1).astype(np.float32)
        col.data.foreach_set("color", rgba.ravel())
        me.color_attributes.active_color = col
        if ao is not None:
            me.color_attributes.remove(ao)
        n += 1
    return n


# ------------------------------------------------------------------------------------------------- main
def run():
    t0 = time.time()
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    anchors_before = {o.name: [round(v, 6) for row in o.matrix_world for v in row] for o in bpy.data.objects if o.type == "EMPTY"}
    rep = {}
    rep["pivots"] = venue_pivots()
    rep["details"] = details()
    albedo, arm = grunge_images()
    rep["grunge_materials"] = sum(grunge_material(bpy.data.materials[m], albedo, arm) for m in GRUNGE if m in bpy.data.materials)
    rep["grunge_faces"] = sum(grunge_uvs(o, set(GRUNGE)) for o in stadium_meshes())
    rep["metal"] = binary_metal()
    rep["bevel_tri"] = modifiers()
    ao_objs = [o for o in vcol_targets() if standing(o) and len(o.data.polygons) < 20000] if BAKE_AO else []
    rep["ao_objects"] = len(ao_objs)
    rep["ao_bake_s"] = bake_ao(ao_objs) if ao_objs else 0
    rep["vcol_objects"] = vertex_colours(set(ao_objs))
    big = [i.name for i in bpy.data.images if i.size[0] > 1024 or i.size[1] > 1024]
    for name in big:
        img = bpy.data.images[name]
        s = 1024 / max(img.size)
        img.scale(int(img.size[0] * s), int(img.size[1] * s))
    rep["images_downscaled"] = big
    generic = [o.name for o in ROOT_COL.all_objects if o.name.split(".")[0] in ("Cube", "Plane", "Cylinder", "Mesh") or "." in o.name[-4:]]
    rep["generic_names"] = generic
    after = {o.name: [round(v, 6) for row in o.matrix_world for v in row] for o in bpy.data.objects if o.type == "EMPTY"}
    assert all(after[k] == v for k, v in anchors_before.items()), "anchors moved"
    rep["anchors_unchanged"] = len(anchors_before)
    rep["seconds"] = round(time.time() - t0, 1)
    print("polish:", rep)
    return rep


if not globals().get("POLISH_NO_RUN"):
    run()


def export_stadium(path=r"C:\Users\punko\Downloads\PoOlympic\Assets\PoOlympic\Art\Stadium\Stadium.glb"):
    """Export the Stadium collection as Unity imports it: tangents, both UV sets, modifiers applied, vertex colours,
    JPEG textures, no lights / cameras / animation — then rename the glTF scene back to "Scene". A collection export
    names it after the collection, and glTFast derives the imported root's file ID from that name: a different name
    turns every event scene's Stadium into a "Missing Prefab" (seen 2026-09-28)."""
    import json, struct
    bpy.ops.export_scene.gltf(filepath=path, collection="Stadium", export_format="GLB", export_tangents=True,
                              export_texcoords=True, export_normals=True, export_image_format="JPEG",
                              export_cameras=False, export_lights=False, export_animations=False, export_extras=False,
                              export_apply=True, export_yup=True, export_materials="EXPORT",
                              export_vertex_color="ACTIVE", export_all_vertex_colors=False)
    b = open(path, "rb").read()
    n = struct.unpack_from("<I", b, 12)[0]
    j, rest = json.loads(b[20:20 + n]), b[20 + n:]
    j["scenes"][0]["name"] = "Scene"
    js = json.dumps(j, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    open(path, "wb").write(struct.pack("<III", 0x46546C67, 2, 20 + len(js) + len(rest)) + struct.pack("<I", len(js)) + b"JSON" + js + rest)
    return path
