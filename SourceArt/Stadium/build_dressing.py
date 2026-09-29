"""PoOlympics stadium — dressing pass (render-only): crowd, branding, Olympic identity, venue materials, surroundings.

Run inside Blender AFTER build_venues.py, on SourceArt/Stadium/stadium.blend:
    exec(open(r"<repo>/SourceArt/Stadium/build_dressing.py").read())

Everything here is decoration: no venue geometry moves, venues.json and the E##_L# anchors are untouched, nothing is
physical (Unity's EventScenes.PlaceStadium refuses colliders; physics props come from the event MJCFs). Idempotent:
objects it creates live in the collections Branding / Identity / Surroundings (children of Stadium) and are rebuilt on
every run; materials are created or updated by name; the crowd images are redrawn in place.

  crowd        Crowd_*_front images redrawn: supporter sections in team colours (a coherent colour per block, some
               neutral clothes, brightness jitter) instead of per-person random saturated shirts, which read as colour
               static from the broadcast camera.
  branding     Olympic rings + POOLYMPICS title on both scoreboards; POOLYMPICS repeated on the LED boards (straights).
  identity     rings on both long sides of the facade, a glass ribbon round the facade, four entrance gates with signs,
               a flag ring on the roof rim, the Olympic cauldron (flame) on the north roof.
  venues       materials for the grey venue props (stair nosing strips, rubber ramp, rocks, slate stones, parkour wall,
               foam drop) + start blocks behind the sprint start lines.
  surroundings plaza ring, park with trees, city skyline (the sky itself is a Unity skybox setting).
"""

import json
import math
import os
import random

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"C:\Users\punko\Downloads\PoOlympic\SourceArt\Stadium"
exec(open(os.path.join(HERE, "stadium_helpers.py")).read())

FACADE_R = 94.5          # facade: oval(r) around the straight segment, 0 .. 28.4 m
FACADE_TOP = 28.4
ROOF_Z = 33.4            # canopy plane, from r 60.8 (fascia) to 97.5 (outer rim)
ROOF_OUT_R = 97.5
SCREEN_X, SCREEN_W, SCREEN_Z0, SCREEN_Z1 = 115.69, 30.0, 14.85, 25.35
BOARD_R = 50.5           # LED boards: oval(50.5), 0..1 m
RNG = random.Random(2026)

# ----------------------------------------------------------------------------------------------------------- materials
M = {
    "Rings_Blue": mat("Rings_Blue", (0.0, 0.33, 0.72), 0.4, 0.0, (0.0, 0.33, 0.72), 1.5),
    "Rings_Yellow": mat("Rings_Yellow", (0.98, 0.72, 0.0), 0.4, 0.0, (0.98, 0.72, 0.0), 1.5),
    "Rings_Black": mat("Rings_Black", (0.02, 0.02, 0.02), 0.4, 0.0, (0.25, 0.25, 0.25), 1.0),
    "Rings_Green": mat("Rings_Green", (0.0, 0.62, 0.24), 0.4, 0.0, (0.0, 0.62, 0.24), 1.5),
    "Rings_Red": mat("Rings_Red", (0.93, 0.2, 0.25), 0.4, 0.0, (0.93, 0.2, 0.25), 1.5),
    "Title_Gold": mat("Title_Gold", (1.0, 0.85, 0.3), 0.3, 0.2, (1.0, 0.85, 0.3), 3.0),
    "Title_White": mat("Title_White", (0.95, 0.95, 0.95), 0.4, 0.0, (1.0, 1.0, 1.0), 2.5),
    "Glass_Ribbon": mat("Glass_Ribbon", (0.12, 0.35, 0.62), 0.1, 0.6, (0.1, 0.3, 0.6), 0.6),
    "Gate_Frame": mat("Gate_Frame", (0.85, 0.86, 0.88), 0.35, 0.3),
    "Flag_Pole": mat("Flag_Pole", (0.8, 0.8, 0.82), 0.3, 0.8),
    "Cauldron": mat("Cauldron", (0.55, 0.45, 0.3), 0.3, 0.9),
    "Flame_Core": mat("Flame_Core", (1.0, 0.85, 0.3), 0.8, 0.0, (1.0, 0.8, 0.25), 12.0),
    "Flame_Outer": mat("Flame_Outer", (1.0, 0.35, 0.05), 0.8, 0.0, (1.0, 0.3, 0.02), 8.0),
    "Stair_Concrete": mat("Stair_Concrete", (0.5, 0.47, 0.43), 0.85),
    "Scoreboard_Screen": mat("Scoreboard_Screen", (0.01, 0.02, 0.06), 0.3, 0.0, (0.02, 0.05, 0.18), 2.0),
    "Stair_Nosing": mat("Stair_Nosing", (0.95, 0.75, 0.05), 0.5),
    "Ramp_Rubber": mat("Ramp_Rubber", (0.12, 0.3, 0.55), 0.9),
    "Rock_A": mat("Rock_A", (0.42, 0.36, 0.3), 0.95),
    "Rock_B": mat("Rock_B", (0.52, 0.47, 0.4), 0.95),
    "Rock_C": mat("Rock_C", (0.33, 0.31, 0.3), 0.95),
    "Slate": mat("Slate", (0.18, 0.2, 0.24), 0.7),
    "Parkour_Wall": mat("Parkour_Wall", (0.95, 0.45, 0.08), 0.7),
    "Foam_Blue": mat("Foam_Blue", (0.1, 0.4, 0.75), 0.9),
    "Start_Block": mat("Start_Block", (0.15, 0.15, 0.17), 0.5, 0.6),
    "Start_Pad": mat("Start_Pad", (0.85, 0.1, 0.1), 0.8),
    "Plaza": mat("Plaza", (0.62, 0.6, 0.57), 0.9),
    "Park_Grass": mat("Park_Grass", (0.22, 0.42, 0.18), 0.95),
    "Tree_Trunk": mat("Tree_Trunk", (0.3, 0.2, 0.12), 0.9),
    "Tree_Leaves": mat("Tree_Leaves", (0.13, 0.36, 0.14), 0.9),
    "Tree_Leaves_B": mat("Tree_Leaves_B", (0.2, 0.42, 0.12), 0.9),
    "City_Concrete": mat("City_Concrete", (0.58, 0.6, 0.64), 0.8),
    "City_Glass": mat("City_Glass", (0.25, 0.38, 0.52), 0.15, 0.7, (0.2, 0.3, 0.4), 0.3),
}
FLAG_COLOURS = [mat(f"Flag_{n}", c, 0.8) for n, c in (
    ("Red", (0.8, 0.08, 0.1)), ("White", (0.95, 0.95, 0.95)), ("Blue", (0.05, 0.2, 0.6)), ("Green", (0.05, 0.5, 0.2)),
    ("Yellow", (0.98, 0.8, 0.05)), ("Black", (0.03, 0.03, 0.03)), ("Orange", (0.95, 0.45, 0.05)), ("Sky", (0.35, 0.65, 0.9)))]


# ------------------------------------------------------------------------------------------------------ mesh builder
class Builder:
    """Accumulates geometry with per-face materials; `emit` creates ONE object (few draw calls in Unity)."""

    def __init__(self):
        self.v, self.f, self.mi, self.mats = [], [], [], []

    def _m(self, m):
        if m not in self.mats:
            self.mats.append(m)
        return self.mats.index(m)

    def add(self, verts, faces, m):
        k, i = len(self.v), self._m(m)
        self.v += [tuple(p) for p in verts]
        self.f += [tuple(k + j for j in fc) for fc in faces]
        self.mi += [i] * len(faces)

    def box(self, c, size, m, rot=None):
        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        R = (rot or Matrix.Identity(3))
        pts = [Vector(c) + R @ Vector((sx * hx, sy * hy, sz * hz)) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
        faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
        self.add(pts, faces, m)

    def cylinder(self, c, r0, r1, h, m, n=12):
        base = Vector(c)
        verts = [base + Vector((r0 * math.cos(2 * math.pi * i / n), r0 * math.sin(2 * math.pi * i / n), 0)) for i in range(n)]
        verts += [base + Vector((r1 * math.cos(2 * math.pi * i / n), r1 * math.sin(2 * math.pi * i / n), h)) for i in range(n)]
        faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)] + [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
        self.add(verts, faces, m)

    def cone(self, c, r, h, m, n=10):
        base = Vector(c)
        verts = [base + Vector((r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0)) for i in range(n)]
        verts.append(base + Vector((0, 0, h)))
        self.add(verts, [(i, (i + 1) % n, n) for i in range(n)] + [tuple(range(n))[::-1]], m)

    def ico(self, c, r, m, sx=1.0, sz=1.0):
        t = (1 + 5 ** 0.5) / 2
        p = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
        s = r / math.sqrt(1 + t * t)
        verts = [Vector(c) + Vector((x * s * sx, y * s * sx, z * s * sz)) for x, y, z in p]
        faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
                 (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
                 (8, 6, 7), (9, 8, 1)]
        self.add(verts, faces, m)

    def torus(self, c, R, r, m, normal_yaw, seg=40, rseg=8):
        """Ring standing upright, its axis (the side you look at) horizontal at yaw `normal_yaw`."""
        rot = Matrix.Rotation(normal_yaw - math.pi / 2, 3, "Z")     # local +Y (the axis) -> normal_yaw
        verts = []
        for i in range(seg):
            a = 2 * math.pi * i / seg
            for j in range(rseg):
                b = 2 * math.pi * j / rseg
                # ring in the local XZ plane, axis along local Y
                x = (R + r * math.cos(b)) * math.cos(a)
                z = (R + r * math.cos(b)) * math.sin(a)
                y = r * math.sin(b)
                verts.append(Vector(c) + rot @ Vector((x, y, z)))
        faces = [(i * rseg + j, i * rseg + (j + 1) % rseg, ((i + 1) % seg) * rseg + (j + 1) % rseg, ((i + 1) % seg) * rseg + j)
                 for i in range(seg) for j in range(rseg)]
        self.add(verts, faces, m)

    def emit(self, name, col):
        ob = mesh_obj(name, self.v, self.f, None, col)
        for m in self.mats:
            ob.data.materials.append(m)
        for poly, i in zip(ob.data.polygons, self.mi):
            poly.material_index = i
        return ob


def clear_collection(name):
    col = get_col(name)
    for ob in list(col.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    return col


def text_mesh(builder, body, size, center, normal_yaw, m, extrude=0.05):
    """Blender text -> mesh, standing upright, facing the horizontal direction `normal_yaw` (radians, world)."""
    cu = bpy.data.curves.new("tmp_text", type="FONT")
    cu.body, cu.size, cu.extrude = body, size, extrude
    cu.resolution_u = 3                     # coarse glyph curves: invisible at stadium distances, a fraction of the verts
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    ob = bpy.data.objects.new("tmp_text", cu)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    # text lies in XY facing +Z -> stand up (+90 X: faces -Y, i.e. -90 deg) -> turn by yaw + 90 so it faces normal_yaw;
    # its reading direction (+X) then runs to the right of a viewer looking at it
    R = Matrix.Rotation(normal_yaw + math.pi / 2, 3, "Z") @ Matrix.Rotation(math.pi / 2, 3, "X")
    verts = [Vector(center) + R @ v.co for v in me.vertices]
    builder.add(verts, [tuple(p.vertices) for p in me.polygons], m)
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.curves.remove(cu)
    bpy.data.meshes.remove(me)


def olympic_rings(builder, center, R, normal_yaw):
    """Five interlaced rings: top blue-black-red, bottom yellow-green; spacing like the emblem."""
    right = Matrix.Rotation(normal_yaw + math.pi / 2, 3, "Z") @ Vector((1, 0, 0))   # viewer's right
    up = Vector((0, 0, 1))
    r = 0.12 * R
    for (dx, dz), m in zip(((-2.2, 0), (0, 0), (2.2, 0), (-1.1, -1.0), (1.1, -1.0)),
                           ("Rings_Blue", "Rings_Black", "Rings_Red", "Rings_Yellow", "Rings_Green")):
        builder.torus(Vector(center) + right * (dx * R) + up * (dz * R), R, r, M[m], normal_yaw)


# ---------------------------------------------------------------------------------------------------------- crowd
def redraw_crowd():
    """Crowd strips (1536 x 48, tiled along the stands): fans mostly in the stand's own home colour (blue / white) with
    a second home colour and sparse accents — calm from the broadcast camera (random rainbow shirts read as static and
    team-colour blocks tile into diagonal stripes)."""
    accents = [(0.8, 0.12, 0.12), (0.98, 0.78, 0.1), (0.1, 0.5, 0.22), (0.95, 0.45, 0.08)]
    neutral = [(0.15, 0.15, 0.17), (0.35, 0.35, 0.38), (0.75, 0.75, 0.75), (0.2, 0.28, 0.45), (0.45, 0.35, 0.28)]
    skins = [(0.96, 0.8, 0.68), (0.88, 0.66, 0.5), (0.72, 0.5, 0.36), (0.52, 0.35, 0.24), (0.36, 0.24, 0.17)]
    for name, seat, seat_gap, home, second in (
            ("Crowd_Blue_front", (0.1, 0.2, 0.55), (0.04, 0.07, 0.2), (0.12, 0.25, 0.6), (0.9, 0.9, 0.92)),
            ("Crowd_White_front", (0.82, 0.84, 0.88), (0.45, 0.47, 0.52), (0.88, 0.88, 0.9), (0.14, 0.28, 0.62))):
        img = bpy.data.images[name]
        W, H = img.size
        rng = np.random.default_rng(7 if "Blue" in name else 11)
        px = np.zeros((H, W, 4), np.float32)
        px[..., 3] = 1.0
        px[..., :3] = seat_gap
        pitch = 24                                     # px per seat (64 seats per strip)
        yy, xx = np.mgrid[0:H, 0:W]
        for s in range(W // pitch):
            x0 = s * pitch
            px[4:20, x0 + 2: x0 + pitch - 2, :3] = seat              # seat back (visible where empty)
            if rng.uniform() < 0.08:
                continue                                              # empty seat
            u = rng.uniform()
            if u < 0.6:
                pick = home
            elif u < 0.8:
                pick = second
            elif u < 0.93:
                pick = neutral[rng.integers(len(neutral))]
            else:
                pick = accents[rng.integers(len(accents))]
            shirt = np.array(pick, np.float32)
            shirt = np.clip(shirt * rng.uniform(0.82, 1.08), 0, 1)
            cx = x0 + pitch / 2 + rng.uniform(-2, 2)
            top = 30 + rng.integers(-2, 3)
            body = (np.abs(xx - cx) <= 7) & (yy >= 3) & (yy <= top)
            px[body, :3] = shirt
            head = (xx - cx) ** 2 + (yy - (top + 7)) ** 2 <= 30
            px[head, :3] = skins[rng.integers(len(skins))]
            if rng.uniform() < 0.06:                                  # waving a flag / raised arm in team colour
                arm = (np.abs(xx - (cx + 8)) <= 1) & (yy >= top - 4) & (yy <= H - 2)
                px[arm, :3] = shirt
        img.pixels.foreach_set(px.ravel())
        img.pack()


# ------------------------------------------------------------------------------------------------------- branding
def build_branding():
    clear_collection("Branding")
    b = Builder()
    for sx, yaw in ((1, math.pi), (-1, 0.0)):                        # screens face the infield (-x at east, +x at west)
        x = sx * (SCREEN_X - 0.08)
        zc = (SCREEN_Z0 + SCREEN_Z1) / 2
        olympic_rings(b, (x, 0.0, zc + 2.3), 1.35, yaw)
        text_mesh(b, "POOLYMPICS", 3.0, (x, 0.0, zc - 2.0), yaw, M["Title_Gold"])
        text_mesh(b, "ROBOT DECATHLON", 1.1, (x, 0.0, zc - 4.1), yaw, M["Title_White"])
    for sy, yaw in ((1, -math.pi / 2), (-1, math.pi / 2)):          # LED boards on both straights, facing the track
        y = sy * (BOARD_R - 0.06)
        for x in np.arange(-L / 2 + 7, L / 2 - 5, 14.0):
            text_mesh(b, "POOLYMPICS", 0.55, (float(x), y, 0.5), yaw, M["Title_White"], extrude=0.01)
    b.emit("Branding", "Branding")


# ------------------------------------------------------------------------------------------------------- identity
def build_identity():
    clear_collection("Identity")
    b = Builder()
    # rings on both long sides of the facade, facing out
    for sy, yaw in ((1, math.pi / 2), (-1, -math.pi / 2)):          # beside the gate on each long side
        olympic_rings(b, (-sy * 38.0, sy * (FACADE_R + 0.6), 14.5), 3.2, yaw)
    # glass ribbon round the facade, 20.5 - 24.5 m
    ob = wall("Glass_Ribbon", FACADE_R + 0.15, 20.5, 24.5, M["Glass_Ribbon"], col="Identity")
    # four gates (portal frames + sign) on the axes
    for (x, y, yaw, label) in ((0.0, -(FACADE_R + 1.5), -math.pi / 2, "GATE A"), (0.0, FACADE_R + 1.5, math.pi / 2, "GATE C"),
                               (L / 2 + FACADE_R + 1.5, 0.0, 0.0, "GATE B"), (-(L / 2 + FACADE_R + 1.5), 0.0, math.pi, "GATE D")):
        R = Matrix.Rotation(yaw, 3, "Z")
        side = R @ Vector((0, 1, 0))
        c = Vector((x, y, 0.0))
        for s in (-1, 1):
            b.box(c + side * (s * 7.0) + Vector((0, 0, 6.0)), (2.0, 1.6, 12.0), M["Gate_Frame"], R)
        b.box(c + Vector((0, 0, 12.8)), (2.2, 16.0, 1.8), M["Gate_Frame"], R)
        text_mesh(b, label, 1.1, c + R @ Vector((1.15, 0, 12.8)), yaw, M["Title_White"], extrude=0.02)
        text_mesh(b, "POOLYMPICS", 1.6, c + R @ Vector((0.9, 0, 15.2)), yaw, M["Title_Gold"])
    # flag ring on the roof rim
    pts = oval(ROOF_OUT_R - 1.0, 12)
    for i, (x, y) in enumerate(pts[::2]):
        b.cylinder((x, y, ROOF_Z), 0.12, 0.08, 6.0, M["Flag_Pole"], n=6)
        outward = Vector((x - math.copysign(min(abs(x), L / 2), x), y, 0)).normalized()
        tangent = Vector((-outward.y, outward.x, 0))
        cols = RNG.sample(FLAG_COLOURS, 3)
        for k in range(3):                                           # vertical tricolour, 2.4 x 1.6 m
            p0 = Vector((x, y, ROOF_Z + 5.9)) + tangent * (0.1 + 0.8 * k)
            quad = [p0, p0 + tangent * 0.8, p0 + tangent * 0.8 + Vector((0, 0, -1.6)), p0 + Vector((0, 0, -1.6))]
            b.add(quad, [(0, 1, 2, 3), (3, 2, 1, 0)], cols[k])
    # Olympic cauldron on the north roof, visible from the infield cameras
    cx, cy = 0.0, 78.0
    b.cylinder((cx, cy, ROOF_Z), 2.2, 1.4, 4.0, M["Cauldron"], n=16)
    b.cylinder((cx, cy, ROOF_Z + 4.0), 1.2, 4.2, 2.2, M["Cauldron"], n=24)
    for k in range(7):
        a = 2 * math.pi * k / 7
        b.cone((cx + 1.6 * math.cos(a), cy + 1.6 * math.sin(a), ROOF_Z + 6.0), 1.2, 3.5 + RNG.uniform(0, 1.5), M["Flame_Outer"], n=8)
    b.cone((cx, cy, ROOF_Z + 6.0), 1.8, 6.5, M["Flame_Core"], n=10)
    b.emit("Identity", "Identity")


# --------------------------------------------------------------------------------------------------------- venues
def dress_venues():
    clear_collection("VenueDressing")
    b = Builder()

    def objs(prefix):
        return [o for o in bpy.data.collections["Venues"].all_objects if o.name.startswith(prefix) and o.type == "MESH"]

    def setmat(obs, m):
        for o in obs:
            o.data.materials.clear()
            o.data.materials.append(m)

    for p in ("E16_Step", "E16_Platform", "E17_Step", "E17_Landing"):
        setmat(objs(p), M["Stair_Concrete"])
    for o in objs("E16_Step") + objs("E17_Step"):                    # yellow nosing strip on every step edge
        ws = [o.matrix_world @ Vector(c) for c in o.bound_box]
        mn = Vector([min(w[i] for w in ws) for i in range(3)]); mx = Vector([max(w[i] for w in ws) for i in range(3)])
        edge_x = mn.x if "E16" in o.name else mx.x    # E17 climbs +x (nosing at the far edge); E16 descends (near edge)
        b.box((edge_x, (mn.y + mx.y) / 2, mx.z + 0.006), (0.06, mx.y - mn.y, 0.012), M["Stair_Nosing"])
    setmat(objs("E14_Ramp") + objs("E14_Deck"), M["Ramp_Rubber"])
    for o in objs("E15_Rubble"):
        setmat([o], M[RNG.choice(("Rock_A", "Rock_B", "Rock_C"))])
    setmat(objs("E26_Stone"), M["Slate"])
    setmat(objs("E30_Wall"), M["Parkour_Wall"])
    setmat(objs("E30_DropDeck") + objs("E30_DropRamp"), M["Foam_Blue"])
    # start blocks just behind the start spot of each sprint lane (render-only; the athletes stand in front of them)
    ev = json.load(open(os.path.join(HERE, "venues.json")))["events"]
    for num in ("08", "20", "22", "27", "19"):
        for lane in ev[num]["lanes"]:
            x, y, z = lane["pos"]
            yaw = math.radians(lane["yaw_deg"])
            fwd = Vector((math.cos(yaw), math.sin(yaw), 0))
            R = Matrix.Rotation(yaw, 3, "Z")
            c = Vector((x, y, z)) - fwd * 0.75
            b.box(c + Vector((0, 0, 0.02)), (0.7, 0.18, 0.04), M["Start_Block"], R)
            for s in (-1, 1):
                pad = c + R @ Vector((-0.12 * s, 0.1 * s, 0.09))
                b.box(pad, (0.12, 0.14, 0.1), M["Start_Pad"], R @ Matrix.Rotation(math.radians(-40), 3, "Y"))
    b.emit("VenueDressing", "VenueDressing")


# --------------------------------------------------------------------------------------------------- surroundings
def build_surroundings():
    clear_collection("Surroundings")
    ring("Plaza", FACADE_R, FACADE_R + 40.0, -0.02, M["Plaza"], col="Surroundings", n_arc=48)
    ring("Park", FACADE_R + 40.0, FACADE_R + 260.0, -0.03, M["Park_Grass"], col="Surroundings", n_arc=48)
    b = Builder()
    rng = random.Random(7)
    placed = 0
    while placed < 260:
        r = rng.uniform(FACADE_R + 48, FACADE_R + 240)
        a = rng.uniform(0, 2 * math.pi)
        x0 = L / 2 * math.copysign(1, math.cos(a)) if abs(math.cos(a)) > 0.2 else 0.0
        x, y = x0 + r * math.cos(a), r * math.sin(a)
        if abs(y) < 14 and abs(x) > L / 2 or abs(x) < 14:            # keep the four gate avenues open
            continue
        h = rng.uniform(5, 11)
        b.cylinder((x, y, 0), 0.35, 0.25, h * 0.45, M["Tree_Trunk"], n=6)
        b.ico((x, y, h * 0.65), h * 0.35, M["Tree_Leaves" if rng.random() < 0.6 else "Tree_Leaves_B"], sz=1.25)
        placed += 1
    for _ in range(170):                                             # skyline ring
        r = rng.uniform(420, 700)
        a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a) * 1.2, r * math.sin(a)
        w, d, h = rng.uniform(18, 45), rng.uniform(18, 45), rng.choice((rng.uniform(20, 60), rng.uniform(60, 160)))
        b.box((x, y, h / 2), (w, d, h), M["City_Glass" if rng.random() < 0.45 else "City_Concrete"],
              Matrix.Rotation(a, 3, "Z"))
    b.emit("Trees_Skyline", "Surroundings")


# ---------------------------------------------------------------------------------------------------------- main
redraw_crowd()
build_branding()
build_identity()
dress_venues()
build_surroundings()
print("dressing:", {c: len(bpy.data.collections[c].objects) for c in ("Branding", "Identity", "VenueDressing", "Surroundings")})
