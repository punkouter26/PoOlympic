"""PoOlympics stadium — crowd venues pass (2026-09-30, user: "more interaction and collisions between the 8 players").
Run inside Blender on SourceArt/Stadium/stadium.blend after build_showcase.py (the last full pass):
    exec(open(r"<repo>/SourceArt/Stadium/build_crowd.py").read())

The crowd events' venues are defined in build_venues.py (BEAM_PITCH … BREAK_X). This pass rebuilds just those venues in
place with build_venues' own functions (a full build_venues run would wipe the realism / polish / showcase work on every
venue) and finishes the new meshes like those passes did:
  01 Iron Pedestal     one iron beam under the row (was 8 pedestals 3 m apart), athletes 0.7 m apart
  05 Gust Gauntlet     one hazard-striped shaker floor, 2 x 4 grid 0.8 m apart (was 8 separate 1.6 m pads)
  08 30m All Fours     crawl spots 1.1 m apart (was the 1.22 m track lanes); its start blocks are removed (crawlers)
  11 Slalom Sprint     1.4 m lanes (mirror slalom: neighbours meet at every other pole)
  12 360 Turntable     8 spin spots on a ring 0.75 m apart + rosette (was a 2 x 4 grid 3 m x 4 m)
  19 Terminal Velocity green lane-break line + BREAK label 15 m after the start
Finishing of every new mesh: transforms applied, base pivot, world box UVs ("UVMap": PBR tile or grunge tile), lightmap
UVs ("UVLightmap"), angle bevel + weighted normals on standing hard surfaces, triangulate, vertex colour "Col" (height
grime); pads get the showcase floor UVs (world / 2 m). Material settings are snapshotted and restored (build_venues'
mat() calls would reset the later passes' values). venues.json (SourceArt + Unity copy) gets the rebuilt events; every
other anchor is asserted unchanged; Stadium.glb is exported with build_showcase.export_stadium.
"""

import json
import math
import os
import random

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"C:\Users\punko\Downloads\PoOlympic\SourceArt\Stadium"
UNITY_VENUES = os.path.join(HERE, "..", "..", "Assets", "PoOlympic", "Art", "Stadium", "venues.json")
CROWD = (1, 5, 8, 11, 12, 19)
PBR_TILE = {"Concrete": 3.0, "Stair_Concrete": 2.0, "Kerb_Grey": 1.5, "Track_Red": 2.0, "Steel": 2.0, "Gate_Frame": 2.0,
            "Wood": 1.0}                                         # = build_realism.PBR tiles (materials used here)
GRUNGE_TILE = 1.6                                               # = build_polish.GRUNGE_TILE
HARD = {"Steel", "Concrete", "Stair_Concrete", "Wood", "Kerb_Grey", "IronPedestal", "Start_Block", "Cauldron",
        "Gate_Frame", "Flag_Pole", "Hazard_Yellow", "Marker_Orange", "Stop_Red", "Foam_Blue", "GymMat_Blue"}   # = build_polish.HARD
NO_VCOL = ("Board_LED", "Scoreboard_Screen", "Floodlight")
GRIME_H, GRIME = 0.45, 0.16                                     # = build_polish
RNG = random.Random(20260930)


def snapshot_materials():
    snap = {}
    for m in bpy.data.materials:
        if not m.use_nodes or not m.node_tree:
            continue
        for nd in m.node_tree.nodes:
            if nd.type == "BSDF_PRINCIPLED":
                snap[(m.name, nd.name)] = {i.name: (tuple(i.default_value) if hasattr(i.default_value, "__len__") else i.default_value)
                                           for i in nd.inputs if not i.is_linked and hasattr(i, "default_value")}
    return snap


def restore_materials(snap):
    for (mname, nname), vals in snap.items():
        nd = bpy.data.materials[mname].node_tree.nodes.get(nname)
        for k, v in vals.items():
            try:
                nd.inputs[k].default_value = v
            except (KeyError, TypeError, AttributeError):
                pass


def anchors():
    return {o.name: [round(v, 6) for row in o.matrix_world for v in row] for o in bpy.data.objects if o.type == "EMPTY"}


def remove_event(num):
    tag = f"E{num:02d}_"
    keep_edge = num != 11                                        # pad size unchanged: keep its rubber edging
    for o in list(bpy.data.objects):
        if o.name.startswith(tag) and not (keep_edge and o.name.endswith("_Pad_Edge")):
            bpy.data.objects.remove(o, do_unlink=True)


def bounds(o):
    ws = [o.matrix_world @ Vector(c) for c in o.bound_box]
    return Vector([min(w[i] for w in ws) for i in range(3)]), Vector([max(w[i] for w in ws) for i in range(3)])


def finish(obs):
    """build_realism / build_polish finishing of new meshes (see module docstring)."""
    bpy.ops.object.select_all(action="DESELECT")
    rot = [o for o in obs if any(abs(a) > 1e-6 for a in o.rotation_euler) or any(abs(s - 1) > 1e-6 for s in o.scale)]
    for o in rot:
        o.select_set(True)
    if rot:
        bpy.context.view_layer.objects.active = rot[0]
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    for o in obs:                                               # base-centre pivot (build_polish.venue_pivots)
        lo, hi = bounds(o)
        base = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
        off = base - o.matrix_world.translation
        if off.length >= 0.01:
            o.data.transform(Matrix.Translation(-off))
            o.matrix_world.translation += off
    for o in obs:                                               # world box UVs
        me = o.data
        uv = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
        mw = o.matrix_world
        pad = o.name.endswith("_Pad")
        ou, ov = RNG.random(), RNG.random()
        for poly in me.polygons:
            m = me.materials[poly.material_index] if me.materials else None
            tile = 2.0 if pad else PBR_TILE.get(m.name if m else "", GRUNGE_TILE)
            nrm = (mw.to_3x3() @ poly.normal).normalized()
            for li in poly.loop_indices:
                p = mw @ me.vertices[me.loops[li].vertex_index].co
                if abs(nrm.z) >= max(abs(nrm.x), abs(nrm.y)):
                    u, v = p.x, p.y
                elif abs(nrm.x) >= abs(nrm.y):
                    u, v = p.y, p.z
                else:
                    u, v = p.x, p.z
                uv.data[li].uv = (u / tile, v / tile) if pad else (u / tile + ou, v / tile + ov)
        if "UVLightmap" not in me.uv_layers:
            me.uv_layers.new(name="UVLightmap")
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
        o.data.uv_layers.active = o.data.uv_layers["UVLightmap"]
    bpy.context.view_layer.objects.active = obs[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES", PREF_PACK_IN_ONE=False, PREF_NEW_UVLAYER=False,
                             PREF_BOX_DIV=12, PREF_MARGIN_DIV=0.2)
    bpy.ops.object.mode_set(mode="OBJECT")
    for o in obs:
        o.data.uv_layers.active = o.data.uv_layers["UVMap"]
        o.data.uv_layers["UVMap"].active_render = True
        lo, hi = bounds(o)
        mats = {m.name for m in o.data.materials if m}
        if min(hi - lo) > 0.05 and mats and mats <= HARD:        # standing hard surface: bevel + weighted normals
            b = o.modifiers.new("RealismBevel", "BEVEL")
            b.width = min(0.03, 0.15 * min(hi - lo)); b.segments = 1; b.limit_method = "ANGLE"
            b.angle_limit = math.radians(30); b.harden_normals = True
            w = o.modifiers.new("RealismWN", "WEIGHTED_NORMAL"); w.keep_sharp = True
        t = o.modifiers.new("PolishTriangulate", "TRIANGULATE"); t.min_vertices = 5; t.keep_custom_normals = True
        if any(m.startswith(NO_VCOL) for m in mats):
            continue
        me = o.data
        col = me.color_attributes.get("Col") or me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
        tall = (hi.z - lo.z) > 0.08
        for li, loop in enumerate(me.loops):
            z = (o.matrix_world @ me.vertices[loop.vertex_index].co).z
            g = 1.0 - GRIME * max(0.0, min(1.0, 1.0 - (z - lo.z) / GRIME_H)) if tall else 1.0
            col.data[li].color = (g, g, g, 1.0)
        me.color_attributes.active_color = col


def pad_edge(pad):
    """build_polish.details rubber edging round one event pad."""
    lo, hi = bounds(pad)
    name = pad.name + "_Edge"
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    bm = bmesh.new()
    e, h = 0.07, 0.045
    for (x0, y0, x1, y1) in ((lo.x - e, lo.y - e, hi.x + e, lo.y), (lo.x - e, hi.y, hi.x + e, hi.y + e),
                             (lo.x - e, lo.y, lo.x, hi.y), (hi.x, lo.y, hi.x + e, hi.y)):
        g = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.transform(bm, verts=g["verts"], matrix=Matrix.Translation(((x0 + x1) / 2, (y0 + y1) / 2, h / 2))
                            @ Matrix.Diagonal((x1 - x0, y1 - y0, h, 1.0)))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.data.collections["GroundDetail"].objects.link(o)
    me.materials.append(bpy.data.materials["Ramp_Rubber"])
    return o


def widen_pad_lines(cy, x_lo, x_hi, dy):
    """Showcase Pad_Lines (one merged mesh): move the E11 border lines outwards by dy (its pad grew by 2 dy)."""
    o = bpy.data.objects["Pad_Lines"]
    n = 0
    for v in o.data.vertices:
        w = o.matrix_world @ v.co
        if x_lo <= w.x <= x_hi and abs(w.y - cy) < 7.0:
            v.co.y += math.copysign(dy, w.y - cy)
            n += 1
    o.data.update()
    return n


def strip_start_blocks(x0, cy, half):
    """VenueDressing: remove the start blocks behind the 08 start (crawlers start face down, no blocks)."""
    o = bpy.data.objects["VenueDressing"]
    bm = bmesh.new()
    bm.from_mesh(o.data)
    mw = o.matrix_world
    kill = [v for v in bm.verts if x0 - 1.3 <= (mw @ v.co).x <= x0 - 0.2 and abs((mw @ v.co).y - cy) <= half]
    bmesh.ops.delete(bm, geom=kill, context="VERTS")
    bm.to_mesh(o.data); bm.free()
    o.data.update()
    return len(kill)


def run(export=True):
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    snap = snapshot_materials()
    before_anchors = anchors()
    V = {"__file__": os.path.join(HERE, "build_venues.py"), "VENUES_NO_RUN": True}
    exec(open(os.path.join(HERE, "build_venues.py"), encoding="utf-8").read(), V)
    old11 = V["LANE_EVENTS"][11]
    for num in CROWD:
        remove_event(num)
    names = set(bpy.data.objects.keys())
    for num in CROWD:
        if num in V["STATIONS"]:
            V["build_station"](num, *V["STATIONS"][num])
        elif num in V["LANE_EVENTS"]:
            V["build_lane_event"](num)
        else:
            V["build_track_event"](num)
    restore_materials(snap)
    new = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - names]
    meshes = [o for o in new if o.type == "MESH"]
    finish(meshes)
    rep = {"new_objects": len(new), "meshes": len(meshes)}
    rep["e11_pad_edge"] = pad_edge(bpy.data.objects["E11_Pad"]).name
    x0, cy, length = old11
    rep["pad_lines_moved"] = widen_pad_lines(cy, x0 - 1.6, x0 + length + 1.6, (V["SLALOM_LW"] - V["LW"]) * V["N"] / 2)
    t8 = V["TRACK_EVENTS"][8]
    rep["start_block_verts"] = strip_start_blocks(t8[0], t8[1], V["N"] * V["LW"] / 2 + 0.3)
    after = anchors()
    moved = [k for k, v in before_anchors.items() if k in after and after[k] != v and not k.startswith(tuple(f"E{n:02d}_" for n in CROWD))]
    assert not moved, moved
    assert len(after) == len(before_anchors), (len(after), len(before_anchors))
    for path in (os.path.join(HERE, "venues.json"), UNITY_VENUES):
        layout = json.load(open(path))
        for k, ev in V["LAYOUT"]["events"].items():
            layout["events"][k] = ev
        with open(path, "w") as f:
            json.dump(layout, f, indent=1)
    rep["venues_updated"] = sorted(V["LAYOUT"]["events"])
    if export:
        S = {"__file__": os.path.join(HERE, "build_showcase.py"), "SHOWCASE_NO_RUN": True}
        exec(open(os.path.join(HERE, "build_showcase.py"), encoding="utf-8").read(), S)
        rep["export"] = S["export_stadium"]()
    print("crowd:", rep)
    return rep


if not globals().get("CROWD_NO_RUN"):
    run()
