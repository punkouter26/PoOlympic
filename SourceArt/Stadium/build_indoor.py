"""PoOlympics stadium — indoor arena (user decision 2026-09-28: "make it an indoor scene with a roof and multiple
spotlights"; the open bowl put the whole home straight in the canopy's shadow). Render-only. Run inside Blender after
build_realism.py, on SourceArt/Stadium/stadium.blend:
    exec(open(r"<repo>/SourceArt/Stadium/build_indoor.py").read())

  ceiling      shallow dome closing the roof opening (r < 60.8 m on the stadium curve), 33.4 m at the canopy edge,
               ROOF_RISE higher over the centre line; underside = Roof_Under (textured steel)
  seal         concrete band between the upper stand's back (26.3 m) and the canopy (33.4 m) at the facade, so no sky
               shows from inside
  trusses      planar Warren trusses (Steel): 4 along the straights (y = ±11, ±33 m), 7 across (x = 0, ±30, ±60, ±90 m),
               bottom chord 29 m, top 31 m, hangers up to the ceiling at every crossing
  spots        36 fixtures: 24 on the long trusses (x = ±15, ±45, ±75 m; y = ±11, ±33 m) for track + infield, 12 crowd
               wash fixtures under the fascia ring aimed at the stands; housing (Steel) + emissive lens
               (Spot_Lens); per fixture an empty Spot_## at the lens and SpotAim_## on the floor — Unity
               (StadiumLook) puts a spot light on each Spot_## aimed at its SpotAim_## (lights are not exported)
  preview      Blender spot lights at the same fixtures (collection Preview, not exported), sun off, dim indoor world
Everything new lives in the collection Stadium/Indoor; rerunning replaces it. The E##_L# lane anchors are untouched.
"""

import math
import os

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"C:\Users\punko\Downloads\PoOlympic\SourceArt\Stadium"
exec(open(os.path.join(HERE, "stadium_helpers.py")).read())

OPEN_R = 60.8                 # roof opening radius on the stadium curve (Roof_Canopy / Roof_Fascia inner edge)
ROOF_Z = 33.4                 # canopy height
ROOF_RISE = 6.5               # dome rise over the centre
SEAL_R, SEAL_Z0 = 94.45, 26.3  # just inside the facade (94.5), from the upper stand's back edge
TRUSS_Z0, TRUSS_Z1 = 29.0, 31.0
LONG_Y = (-33.0, -11.0, 11.0, 33.0)
CROSS_X = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
SPOT_X = (-75.0, -45.0, -15.0, 15.0, 45.0, 75.0)
SPOT_Z = 28.3
PANEL = 2.5                   # truss panel length (m)
CHORD, WEB = 0.3, 0.14        # chord / diagonal box sizes (m)
COL = "Indoor"


def ceiling_z(r: float) -> float:
    return ROOF_Z + ROOF_RISE * (1.0 - (r / OPEN_R) ** 2)


def half_span_x(y: float) -> float:
    """|x| extent of the opening at lateral position y (stadium curve: straight ±L/2 plus the bend)."""
    return L / 2 + math.sqrt(max(OPEN_R ** 2 - y * y, 0.0))


def half_span_y(x: float) -> float:
    dx = max(abs(x) - L / 2, 0.0)
    return math.sqrt(max(OPEN_R ** 2 - dx * dx, 0.0))


def clear_indoor():
    col = bpy.data.collections.get(COL)
    if col:
        for o in list(col.objects):
            bpy.data.objects.remove(o, do_unlink=True)
    for o in [o for o in bpy.data.objects if o.name.startswith("SpotLight_")]:
        bpy.data.objects.remove(o, do_unlink=True)
    get_col(COL)


def planar_uv(me, tile, lightmap=True):
    """UVMap = world xy / tile (first layer), UVLightmap = per-face-projected 0-1 (filled by lightmap_pack later)."""
    uv = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            n = poly.normal
            if abs(n.z) > 0.7:
                uv.data[li].uv = (co.x / tile, co.y / tile)
            elif abs(n.x) > abs(n.y):
                uv.data[li].uv = (co.y / tile, co.z / tile)
            else:
                uv.data[li].uv = (co.x / tile, co.z / tile)
    if lightmap:
        me.uv_layers.new(name="UVLightmap")


def build_ceiling():
    rings = [OPEN_R * (1 - i / 12) for i in range(12)] + [0.6]
    verts, faces = [], []
    n = None
    for r in rings:
        pts = oval(r)
        n = len(pts)
        verts += [(x, y, ceiling_z(r)) for x, y in pts]
    for k in range(len(rings) - 1):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append((a + i, b + i, b + j, a + j))          # winding: normals point down (into the bowl)
    last = (len(rings) - 1) * n
    faces.append(tuple(last + i for i in range(n - 1, -1, -1)))
    ob = mesh_obj("Ceiling_Dome", verts, faces, bpy.data.materials["Roof_Under"], COL)
    me = ob.data
    me.update()
    if sum(p.normal.z for p in me.polygons) > 0:
        for p in me.polygons:
            p.flip()
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(me); bm.free()
    planar_uv(me, 6.0)
    return ob


def build_seal():
    ob = wall("Facade_Seal", SEAL_R, SEAL_Z0, ROOF_Z + 0.05, bpy.data.materials["Concrete"], COL)
    me = ob.data
    # normals towards the centre
    c = Vector((0, 0, (SEAL_Z0 + ROOF_Z) / 2))
    if sum((p.center - c).dot(p.normal) for p in me.polygons) > 0:
        for p in me.polygons:
            p.flip()
    uv = me.uv_layers.new(name="UVMap")
    pts = oval(SEAL_R)
    arc = [0.0]
    for i in range(1, len(pts)):
        arc.append(arc[-1] + math.dist(pts[i - 1], pts[i]))
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            k = vi % len(pts)
            uv.data[li].uv = (arc[k] / 4.0, me.vertices[vi].co.z / 4.0)
    me.uv_layers.new(name="UVLightmap")
    return ob


def add_box(bm, a: Vector, b: Vector, size: float):
    """Square beam of side `size` from a to b."""
    d = b - a
    length = d.length
    if length < 1e-6:
        return
    z = d.normalized()
    x = z.cross(Vector((0, 0, 1)))
    if x.length < 1e-3:
        x = z.cross(Vector((1, 0, 0)))
    x.normalize()
    y = z.cross(x)
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = (a + b) / 2
    geom = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.transform(bm, verts=geom["verts"], matrix=m @ Matrix.Diagonal((size, size, length, 1.0)))


def truss(bm, p0: Vector, p1: Vector):
    """Planar Warren truss between two points at the bottom-chord height (vertical plane through them)."""
    d = p1 - p0
    n = max(1, round(d.length / PANEL))
    up = Vector((0, 0, TRUSS_Z1 - TRUSS_Z0))
    add_box(bm, p0, p1, CHORD)
    add_box(bm, p0 + up, p1 + up, CHORD)
    for i in range(n):
        a, b = p0 + d * (i / n), p0 + d * ((i + 1) / n)
        if i % 2 == 0:
            add_box(bm, a, b + up, WEB)
        else:
            add_box(bm, a + up, b, WEB)
        add_box(bm, a, a + up, WEB)
    add_box(bm, p1, p1 + up, WEB)


def build_trusses():
    bm = bmesh.new()
    for y in LONG_Y:
        hx = half_span_x(y) - 0.5
        truss(bm, Vector((-hx, y, TRUSS_Z0)), Vector((hx, y, TRUSS_Z0)))
    for x in CROSS_X:
        hy = half_span_y(x) - 0.5
        truss(bm, Vector((x, -hy, TRUSS_Z0)), Vector((x, hy, TRUSS_Z0)))
    # hangers from every crossing up to the ceiling
    for y in LONG_Y:
        for x in CROSS_X:
            if abs(x) < half_span_x(y) - 1.0:
                r = math.hypot(max(abs(x) - L / 2, 0.0), y)
                add_box(bm, Vector((x, y, TRUSS_Z1)), Vector((x, y, ceiling_z(r))), 0.18)
    me = bpy.data.meshes.new("Roof_Trusses")
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new("Roof_Trusses", me)
    get_col(COL).objects.link(ob)
    me.materials.append(bpy.data.materials["Steel"])
    planar_uv(me, 1.0)
    return ob


def spot_lens_material():
    m = mat("Spot_Lens", (1.0, 0.97, 0.9), rough=0.2, emit=(1.0, 0.97, 0.9), strength=6.0)
    return m


WASH_N, WASH_R, WASH_Z = 12, 59.6, 31.3        # crowd wash: fixtures under the fascia, aimed at the stands
WASH_AIM_R, WASH_AIM_Z = 66.0, 6.0


def oval_point(r: float, t: float) -> Vector:
    """Point at parameter t (0..1, arc-length) on the stadium curve of radius r."""
    pts = oval(r, 256)
    arc = [0.0]
    for i in range(1, len(pts)):
        arc.append(arc[-1] + math.dist(pts[i - 1], pts[i]))
    s = t * arc[-1]
    i = next(k for k in range(1, len(arc)) if arc[k] >= s)
    f = (s - arc[i - 1]) / max(arc[i] - arc[i - 1], 1e-9)
    (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    return Vector((x0 + f * (x1 - x0), y0 + f * (y1 - y0), 0.0))


def spot_targets():
    """(fixture position, aim point, yoke top z) for the 36 fixtures: Spot_00-23 on the long trusses (inner rows light
    the infield, outer rows lean out to the track straights, outer columns into the bends), Spot_24-35 = crowd wash
    under the fascia ring, evenly spaced along the curve, aimed at the lower/upper stands."""
    out = []
    for y in LONG_Y:
        for x in SPOT_X:
            ty = math.copysign(42.0 if abs(y) > 20 else 16.0, y)
            tx = x * (1.12 if abs(x) > 60 else 1.0)
            out.append((Vector((x, y, SPOT_Z)), Vector((tx, ty, 0.0)), TRUSS_Z0 - CHORD / 2))
    for k in range(WASH_N):
        t = (k + 0.5) / WASH_N
        p = oval_point(WASH_R, t)
        p.z = WASH_Z
        a = oval_point(WASH_AIM_R, t)
        a.z = WASH_AIM_Z
        out.append((p, a, 31.8))                                  # yoke to the fascia's lower edge
    return out


def build_spots():
    housing, lens = bmesh.new(), bmesh.new()
    lens_mat = spot_lens_material()
    col = get_col(COL)
    preview = bpy.data.collections.get("Preview") or bpy.data.collections.new("Preview")
    if preview.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(preview)
    for k, (p, t, yoke_z) in enumerate(spot_targets()):
        aim = (t - p).normalized()
        # housing: octagonal can along the aim axis, 0.75 m long, 0.36 m radius; lens: disk at its front
        rot = aim.to_track_quat("Z", "Y").to_matrix().to_4x4()
        can = bmesh.ops.create_cone(housing, cap_ends=True, segments=8, radius1=0.36, radius2=0.3, depth=0.75)
        bmesh.ops.transform(housing, verts=can["verts"], matrix=Matrix.Translation(p - aim * 0.2) @ rot)
        disk = bmesh.ops.create_circle(lens, cap_ends=True, segments=8, radius=0.28)
        bmesh.ops.transform(lens, verts=disk["verts"], matrix=Matrix.Translation(p + aim * 0.19) @ rot)
        # yoke up to the bottom chord / the fascia
        add_box(housing, p, Vector((p.x, p.y, yoke_z)), 0.08)
        e = bpy.data.objects.new(f"Spot_{k:02d}", None)
        e.empty_display_type = "SINGLE_ARROW"
        e.location = p + aim * 0.2
        e.rotation_mode = "QUATERNION"
        e.rotation_quaternion = aim.to_track_quat("Z", "Y")        # arrow (+Z) points down the beam
        col.objects.link(e)
        a = bpy.data.objects.new(f"SpotAim_{k:02d}", None)
        a.empty_display_type = "PLAIN_AXES"
        a.location = t
        col.objects.link(a)
        ld = bpy.data.lights.get(f"SpotLight_{k:02d}") or bpy.data.lights.new(f"SpotLight_{k:02d}", "SPOT")
        wash = k >= len(LONG_Y) * len(SPOT_X)
        ld.energy = 30000.0 if wash else 50000.0  # W: ~4 W/m2 on the floor from 30 m (sun-strength 4 equivalent)
        ld.spot_size = math.radians(80 if wash else 70)
        ld.spot_blend = 0.35
        ld.color = (1.0, 0.97, 0.92)
        ld.shadow_soft_size = 0.6
        lo = bpy.data.objects.new(f"SpotLight_{k:02d}", ld)
        lo.location = p + aim * 0.25
        lo.rotation_mode = "QUATERNION"
        lo.rotation_quaternion = (-aim).to_track_quat("Z", "Y")      # Blender spots shine along local -Z
        preview.objects.link(lo)
    obs = []
    for name, bm, m in (("Spot_Fixtures", housing, bpy.data.materials["Steel"]), ("Spot_Lenses", lens, lens_mat)):
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new(name, me)
        col.objects.link(ob)
        me.materials.append(m)
        planar_uv(me, 1.0)
        obs.append(ob)
    return obs


def indoor_preview_lighting():
    sun = bpy.data.objects.get("Preview_Sun")
    if sun:
        sun.hide_render = True
        sun.hide_viewport = True
    w = bpy.context.scene.world
    if w and w.use_nodes:
        bg = next((n for n in w.node_tree.nodes if n.type == "BACKGROUND"), None)
        if bg:
            bg.inputs["Color"].default_value = (0.16, 0.16, 0.17, 1.0)   # indoor bounce stand-in
            bg.inputs["Strength"].default_value = 1.0


def lightmap_pack(obs):
    bpy.ops.object.select_all(action="DESELECT")
    for ob in obs:
        ob.select_set(True)
        ob.data.uv_layers.active = ob.data.uv_layers["UVLightmap"]
    bpy.context.view_layer.objects.active = obs[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES", PREF_PACK_IN_ONE=False, PREF_NEW_UVLAYER=False,
                             PREF_BOX_DIV=12, PREF_MARGIN_DIV=0.2)
    bpy.ops.object.mode_set(mode="OBJECT")
    for ob in obs:
        ob.data.uv_layers.active = ob.data.uv_layers["UVMap"]
        ob.data.uv_layers["UVMap"].active_render = True


anchors_before = {o.name: [round(v, 6) for row in o.matrix_world for v in row]
                  for o in bpy.data.objects if o.type == "EMPTY" and not o.name.startswith(("Spot_", "SpotAim_"))}
if bpy.context.object and bpy.context.object.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")
clear_indoor()
made = [build_ceiling(), build_seal(), build_trusses(), *build_spots()]
lightmap_pack(made)
indoor_preview_lighting()
anchors_after = {o.name: [round(v, 6) for row in o.matrix_world for v in row]
                 for o in bpy.data.objects if o.type == "EMPTY" and not o.name.startswith(("Spot_", "SpotAim_"))}
assert anchors_after == anchors_before, "lane anchors moved"
print("indoor:", {o.name: sum(len(p.vertices) - 2 for p in o.data.polygons) for o in made},
      "spots", len(spot_targets()), "anchors unchanged", len(anchors_before))
