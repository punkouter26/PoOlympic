"""PoOlympics stadium — showcase pass (2026-09-29, GFX/sound top 10, "do all"). Render-only. Run inside Blender after
build_polish.py, on SourceArt/Stadium/stadium.blend:
    exec(open(r"<repo>/SourceArt/Stadium/build_showcase.py").read())
then export with export_stadium() (build_polish.py; this file defines it too). Idempotent: every object it makes lives in
the collection "Showcase" (child of Stadium) and is rebuilt on each run; materials / images are updated by name.

   1  lighting      crowd-wash fixtures (Spot_24-35) re-aimed higher up the stands and softened: the 12 hard ovals on the
                    far stand become one even wash (Unity StadiumLook reads the anchors; preview lights follow)
   2  screens       centre-hung video cube (4 faces) under the dome + both scoreboards on one "Screen_Live" material with
                    0-1 UVs per face; Unity (ScreenFeed) renders the live board into it. The old logo text in front of the
                    scoreboards is gone (the board shows the brand when idle)
   3  crowd         1.0 m crowd cards on every row (the fans were painted on 0.4 m risers and read as thin lines from
                    trackside) with a 2048 x 256 atlas: seated fans (top half, what the UVs show) and the SAME fans
                    cheering with arms up (bottom half); Unity's PoOlympic/Crowd shader swaps seats between the halves and
                    bobs them with the tension of the heat
   4  roof          catwalks along the four long trusses, 8 hanging speaker clusters (Speaker_## anchors), acoustic
                    ceiling panels (generated texture), cables of the video cube; light-beam cones under the 24 field
                    spots ("Light_Beam", Unity draws them with the additive PoOlympic/LightBeam shader)
   5  infield       event pads: sports-floor granule texture (greyscale from the track rubber) tinted a deep, muted phase
                    colour; the phase colour stays saturated only on a thin painted border line
   6  podium        1-2-3 podium + three flag poles on the free infield D at the end of the home straight
                    (Podium_Step_1-3 anchors on the step tops, Podium_Flag_1-3 raised by Unity, PodiumCam, PodiumLook)
   7  branding      Olympic rings removed everywhere (docs/LICENSING.md blocker): the facade rings, gates, flag ring and
                    cauldron are exterior-only (invisible under the closed roof) and move to the non-exported "Offstage"
                    collection with the city surroundings; the scoreboard emblem is replaced by the live board
   8  audio         AudioAnchor_* empties: PA speaker positions, four crowd sectors, the finish line (Unity ArenaAudio)
  10  perf          Offstage (exterior identity + surroundings, ~20k tris never seen from inside) is not exported
Anchors of the lanes (E##_L#) are asserted unchanged.
"""

import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"C:\Users\punko\Downloads\PoOlympic\SourceArt\Stadium"
exec(open(os.path.join(HERE, "stadium_helpers.py")).read())

COL = "Showcase"
OFFSTAGE = "Offstage"
OPEN_R, ROOF_Z, ROOF_RISE = 60.8, 33.4, 6.5
LONG_Y = (-33.0, -11.0, 11.0, 33.0)
CUBE_Z, CUBE_W, CUBE_H = 22.0, 16.0, 9.0          # video cube: centre height, face width / height (m)
WASH_FROM = 24                                     # Spot_24-35 = crowd wash
PODIUM = Vector((56.0, -27.0, 0.0))                # free infield D at the east end of the home straight (no pads)
PHASE_TINT = {                                     # deep, muted sports-floor colours per phase (border keeps the hue)
    "EventPad_P1": ((0.09, 0.19, 0.42), (0.25, 0.55, 1.0)),
    "EventPad_P2": ((0.06, 0.3, 0.31), (0.1, 0.85, 0.8)),
    "EventPad_P3": ((0.24, 0.13, 0.34), (0.7, 0.35, 1.0)),
    "EventPad_P4": ((0.34, 0.11, 0.12), (1.0, 0.3, 0.3)),
    "EventPad_P5": ((0.22, 0.24, 0.09), (0.85, 0.9, 0.2)),
    "EventPad_P6": ((0.15, 0.15, 0.17), (0.8, 0.8, 0.85)),
}


# ------------------------------------------------------------------------------------------------------------ helpers
def col_clear(name, parent="Stadium"):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        if parent:
            bpy.data.collections[parent].children.link(col)
        else:
            bpy.context.scene.collection.children.link(col)
    for o in list(col.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    return col


def material(name, color, rough=0.6, metal=0.0, emit=None, strength=0.0, image=None, alpha=None):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in [n for n in nt.nodes if n.type == "TEX_IMAGE"]:
        nt.nodes.remove(n)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Emission Color"].default_value = (*(emit or (0, 0, 0)), 1)
    bsdf.inputs["Emission Strength"].default_value = strength if emit else 0.0
    if image is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        if emit:
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    if alpha is not None:
        bsdf.inputs["Alpha"].default_value = alpha
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "BLENDED"
        elif hasattr(m, "blend_method"):
            m.blend_method = "BLEND"
    return m


def image(name, rgb, colorspace="sRGB"):
    """numpy (H, W, 3|4) float → Blender image saved as SourceArt/Stadium/textures/<name>.png (rows bottom-up like
    Blender). Saved to disk BEFORE the colour space is set: changing it on an unsaved generated image reloads a blank
    buffer (same order as build_polish.save_image)."""
    h, w = rgb.shape[:2]
    img = bpy.data.images.get(name)
    if img is None or tuple(img.size) != (w, h):
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(name, w, h, alpha=rgb.shape[2] == 4)
    px = np.ones((h, w, 4), np.float32)
    px[..., :rgb.shape[2]] = rgb
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = os.path.join(HERE, "textures", name + ".png")
    img.file_format = "PNG"
    img.save()
    img.colorspace_settings.name = colorspace
    return img


UNITY_CROWD = os.path.normpath(os.path.join(HERE, "..", "..", "Assets", "PoOlympic", "Art", "Stadium", "Crowd"))


def unity_png(name, rgb):
    """Write an sRGB(A) PNG straight into the Unity project (Assets/PoOlympic/Art/Stadium/Crowd/<name>.png)."""
    os.makedirs(UNITY_CROWD, exist_ok=True)
    h, w = rgb.shape[:2]
    old = bpy.data.images.get("tmp_unity_png")
    if old is not None:
        bpy.data.images.remove(old)
    img = bpy.data.images.new("tmp_unity_png", w, h, alpha=True)
    img.alpha_mode = "STRAIGHT"
    px = np.ones((h, w, 4), np.float32)
    px[..., :rgb.shape[2]] = rgb
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = os.path.join(UNITY_CROWD, name + ".png")
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


class Mesh:
    """Geometry accumulator with per-face material + explicit UVs (UVMap) → one object; UVLightmap is packed later."""

    def __init__(self):
        self.v, self.f, self.uv, self.mi, self.mats = [], [], [], [], []

    def add(self, verts, faces, m, uvs=None):
        if m not in self.mats:
            self.mats.append(m)
        k = len(self.v)
        self.v += [tuple(p) for p in verts]
        for i, fc in enumerate(faces):
            self.f.append(tuple(k + j for j in fc))
            self.uv.append(uvs[i] if uvs else [(0.0, 0.0)] * len(fc))
            self.mi.append(self.mats.index(m))

    def quad(self, a, b, c, d, m, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
        self.add([a, b, c, d], [(0, 1, 2, 3)], m, [list(uv)])

    def box(self, c, size, m, rot=None, tile=None):
        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        R = rot or Matrix.Identity(3)
        p = [Vector(c) + R @ Vector((sx * hx, sy * hy, sz * hz)) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
        faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
        uvs = []
        for fc in faces:
            pts = [p[i] for i in fc]
            e1, e2 = (pts[1] - pts[0]).length, (pts[3] - pts[0]).length
            t = tile or max(e1, e2, 1e-3)
            uvs.append([(0, 0), (e1 / t, 0), (e1 / t, e2 / t), (0, e2 / t)])
        self.add(p, faces, m, uvs)

    def emit(self, name, col, lightmap=True):
        old = bpy.data.objects.get(name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.update()
        uv = me.uv_layers.new(name="UVMap")
        for poly, uvs in zip(me.polygons, self.uv):
            for li, t in zip(poly.loop_indices, uvs):
                uv.data[li].uv = t
        if lightmap:
            me.uv_layers.new(name="UVLightmap")
        ob = bpy.data.objects.new(name, me)
        bpy.data.collections[col].objects.link(ob)
        for m in self.mats:
            me.materials.append(m)
        for poly, i in zip(me.polygons, self.mi):
            poly.material_index = i
        return ob


def empty(name, loc, col, rot=(0, 0, 0), size=0.5):
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = size
    e.location = loc
    e.rotation_euler = rot
    bpy.data.collections[col].objects.link(e)
    return e


def lightmap_pack(obs):
    obs = [o for o in obs if o.type == "MESH" and "UVLightmap" in o.data.uv_layers]
    if not obs:
        return
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


# ---------------------------------------------------------------------------------------------- 7 / 10 offstage
def offstage():
    """Exterior-only content (never visible under the closed roof) → the non-exported Offstage collection."""
    off = bpy.data.collections.get(OFFSTAGE) or bpy.data.collections.new(OFFSTAGE)
    if off.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(off)
    moved = []
    for cname in ("Identity", "Surroundings", "Branding"):
        c = bpy.data.collections.get(cname)
        st = bpy.data.collections["Stadium"]
        if c is not None and c.name in st.children:
            st.children.unlink(c)
            off.children.link(c)
            moved.append(cname)
    # the scoreboard logo (rings + title) was Branding; the LED-board lettering stays → rebuilt here without rings
    return moved


def led_lettering(col):
    """POOLYMPICS lettering on the LED boards along both straights (was part of Branding, without the rings)."""
    white = bpy.data.materials["Title_White"]
    parts = []
    for sy, yaw in ((1, -math.pi / 2), (-1, math.pi / 2)):
        y = sy * (50.5 - 0.06)
        for x in np.arange(-L / 2 + 7, L / 2 - 5, 14.0):
            cu = bpy.data.curves.new("tmp_text", type="FONT")
            cu.body, cu.size, cu.extrude, cu.resolution_u = "POOLYMPICS", 0.55, 0.01, 3
            cu.align_x, cu.align_y = "CENTER", "CENTER"
            ob = bpy.data.objects.new("tmp_text", cu)
            bpy.context.scene.collection.objects.link(ob)
            bpy.context.view_layer.update()
            me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
            R = Matrix.Rotation(yaw + math.pi / 2, 3, "Z") @ Matrix.Rotation(math.pi / 2, 3, "X")
            parts.append(([Vector((float(x), y, 0.5)) + R @ v.co for v in me.vertices], [tuple(p.vertices) for p in me.polygons]))
            bpy.data.objects.remove(ob, do_unlink=True)
            bpy.data.curves.remove(cu)
            bpy.data.meshes.remove(me)
    m = Mesh()
    for verts, faces in parts:
        m.add(verts, faces, white)
    return m.emit("LED_Lettering", col, lightmap=False)


# --------------------------------------------------------------------------------------------------------- 1 wash
def soften_wash():
    """Aim the 12 crowd-wash fixtures at the middle of the stands (upper-lower boundary, ~13 m) instead of a spot low
    on the lower tier: the cones overlap into one even wash. Preview lights follow."""
    n = 0
    for k in range(WASH_FROM, 36):
        s, a = bpy.data.objects.get(f"Spot_{k:02d}"), bpy.data.objects.get(f"SpotAim_{k:02d}")
        if s is None or a is None:
            continue
        p = s.matrix_world.translation
        d = Vector((p.x, p.y, 0.0))
        cx = math.copysign(min(abs(p.x), L / 2), p.x)
        outward = Vector((p.x - cx, p.y, 0.0)).normalized()
        target = Vector((cx, 0.0, 0.0)) + outward * 72.0          # between the tiers
        target.z = 12.5
        a.location = target
        n += 1
    return n


# ------------------------------------------------------------------------------------------------------- 2 screens
def screens(col):
    live = material("Screen_Live", (0.02, 0.04, 0.1), 0.35, 0.0, (0.03, 0.08, 0.25), 1.5)
    frame = bpy.data.materials["Steel"]
    black = material("Screen_Bezel", (0.015, 0.015, 0.02), 0.5)
    m = Mesh()
    h, w = CUBE_H, CUBE_W
    z0, z1 = CUBE_Z - h / 2, CUBE_Z + h / 2
    for k in range(4):                                             # faces look out along ±x / ±y
        yaw = k * math.pi / 2
        R = Matrix.Rotation(yaw, 3, "Z")
        out = R @ Vector((1, 0, 0))
        right = R @ Vector((0, 1, 0))
        c = out * (w / 2 + 0.05)
        a = c - right * (w / 2 - 0.25) + Vector((0, 0, z0 + 0.25))
        b = c + right * (w / 2 - 0.25) + Vector((0, 0, z0 + 0.25))
        m.quad(a, b, b + Vector((0, 0, h - 0.5)), a + Vector((0, 0, h - 0.5)), live)
    m.box((0, 0, CUBE_Z), (w, w, h), black)                        # body (bezel shows round the screens)
    m.box((0, 0, z1 + 0.6), (w + 0.6, w + 0.6, 1.2), frame, tile=2.0)   # top ring with LED ticker band below
    m.box((0, 0, z0 - 0.5), (w * 0.8, w * 0.8, 1.0), frame, tile=2.0)   # lower bowl
    for sx in (-1, 1):                                             # four cables to the trusses (dome ~ 39.9, trusses 29-31)
        for sy in (-1, 1):
            m.box((sx * w * 0.35, sy * w * 0.35, (z1 + 1.2 + 31.0) / 2), (0.08, 0.08, 31.0 - z1 - 1.2), frame, tile=1.0)
    cube = m.emit("VideoCube", col)
    # both scoreboards: the existing 1-quad screens take the live material (UVs already 0-1)
    for n in ("Scoreboard_East_Screen", "Scoreboard_West_Screen"):
        o = bpy.data.objects[n]
        o.data.materials.clear()
        o.data.materials.append(live)
    empty("ScreenAnchor_Cube", (0, 0, CUBE_Z), col)
    return cube


# --------------------------------------------------------------------------------------------------------- 3 crowd
def crowd_atlas():
    """2048 x 256 per stand colour: rows 128-255 (top, v 0.5-1) seated, rows 0-127 (bottom) the same fans cheering.
    64 seats per strip (32 px each), like the 1536 x 48 strips it replaces (face UVs keep u, v is remapped to the top
    half)."""
    W, H, P = 2048, 128, 32
    accents = [(0.8, 0.12, 0.12), (0.98, 0.78, 0.1), (0.1, 0.5, 0.22), (0.95, 0.45, 0.08)]
    neutral = [(0.15, 0.15, 0.17), (0.35, 0.35, 0.38), (0.75, 0.75, 0.75), (0.2, 0.28, 0.45), (0.45, 0.35, 0.28)]
    skins = [(0.96, 0.8, 0.68), (0.88, 0.66, 0.5), (0.72, 0.5, 0.36), (0.52, 0.35, 0.24), (0.36, 0.24, 0.17)]
    hairs = [(0.05, 0.04, 0.03), (0.2, 0.12, 0.06), (0.45, 0.3, 0.12), (0.7, 0.6, 0.4), (0.55, 0.55, 0.55)]
    out = {}
    for name, seat, gap, home, second, seed in (
            ("Crowd_Blue_atlas", (0.1, 0.2, 0.55), (0.03, 0.05, 0.14), (0.12, 0.25, 0.6), (0.9, 0.9, 0.92), 7),
            ("Crowd_White_atlas", (0.82, 0.84, 0.88), (0.35, 0.37, 0.42), (0.88, 0.88, 0.9), (0.14, 0.28, 0.62), 11)):
        rng = np.random.default_rng(seed)
        halves = []
        people = []
        for s in range(W // P):
            if rng.uniform() < 0.07:
                people.append(None)
                continue
            u = rng.uniform()
            pick = home if u < 0.6 else second if u < 0.8 else neutral[rng.integers(len(neutral))] if u < 0.93 else accents[rng.integers(len(accents))]
            people.append(dict(shirt=np.clip(np.array(pick) * rng.uniform(0.82, 1.08), 0, 1), skin=np.array(skins[rng.integers(len(skins))]),
                               hair=np.array(hairs[rng.integers(len(hairs))]), dx=rng.uniform(-3, 3), top=int(rng.integers(-6, 5)),
                               flag=rng.uniform() < 0.08, wide=rng.uniform(9, 12)))
        # a crowd card (crowd_cards) is 1.0 m tall and 0.5 m per seat: 128 px ↔ 1.0 m (128 px/m) up, 32 px ↔ 0.5 m
        # (64 px/m) across, so shapes are drawn in metres (X(m) / Y(m)); y = 0 is the tread the fan sits on
        yy, xx = np.mgrid[0:H, 0:W]
        ym, xm = yy / 128.0, xx / 64.0
        for cheer in (False, True):
            px = np.zeros((H, W, 3), np.float32)
            px[...] = gap
            px *= (0.8 + 0.35 * (yy / H))[..., None]                   # the back of the row is lit a little brighter
            bg = px.copy()
            for s, p in enumerate(people):
                c0 = (s + 0.5) * 0.5                                    # seat centre (m)
                back = (np.abs(xm - c0) <= 0.21) & (ym >= 0.0) & (ym <= 0.44)
                px[back] = seat                                        # seat back (shows where the seat is empty)
                px[back & (ym >= 0.41)] = np.array(seat) * 0.65        # top rail
                if p is None:
                    continue
                cx = c0 + p["dx"] / 64.0
                sh = 0.52 + p["top"] / 320.0                           # shoulder height above the tread (m)
                half = p["wide"] / 64.0                                # half shoulder width (0.14-0.19 m)
                taper = np.clip((ym - sh + 0.06) * 1.2, 0, 0.05)
                torso = (np.abs(xm - cx) <= half - taper) & (ym >= 0.0) & (ym <= sh)
                px[torso] = p["shirt"]
                px[(np.abs(xm - cx) <= 0.045) & (ym > sh) & (ym <= sh + 0.04)] = p["skin"]        # neck
                hc = sh + 0.15                                         # head: 0.16 m wide, 0.22 m tall
                head = ((xm - cx) / 0.08) ** 2 + ((ym - hc) / 0.11) ** 2 <= 1.0
                px[head] = p["skin"]
                px[head & (ym >= hc + 0.04)] = p["hair"]
                if cheer:                                              # both arms up, hands near the top of the card
                    for side in (-1, 1):
                        ax = cx + side * (half - 0.03)
                        lean = side * (ym - sh) * 0.3
                        arm = (np.abs(xm - (ax + lean)) <= 0.035) & (ym >= sh - 0.06) & (ym <= 0.94)
                        px[arm] = p["shirt"]
                        hx = ax + side * (0.94 - sh) * 0.3
                        px[((xm - hx) / 0.04) ** 2 + ((ym - 0.95) / 0.04) ** 2 <= 1.0] = p["skin"]
                    if p["flag"]:                                      # a small flag waved over the head
                        fx = cx + half + 0.02
                        px[(np.abs(xm - fx) <= 0.012) & (ym >= sh) & (ym <= 0.99)] = (0.9, 0.9, 0.9)
                        px[(xm >= fx) & (xm <= fx + 0.25) & (ym >= 0.84) & (ym <= 0.99)] = (
                            p["shirt"] if tuple(p["shirt"]) != tuple(home) else second)
                else:                                                  # arms down along the torso
                    for side in (-1, 1):
                        ax = cx + side * (half - 0.02)
                        px[(np.abs(xm - ax) <= 0.03) & (ym >= 0.06) & (ym <= sh - 0.05)] = np.array(p["shirt"]) * 0.8
            # alpha: 1 where a seat back or a fan is painted, 0 elsewhere (Unity clips it: the row behind shows through)
            alpha = (np.abs(px - bg).sum(-1) > 1e-4).astype(np.float32)
            halves.append(np.concatenate([px, alpha[..., None]], -1))
        atlas = np.concatenate([halves[1], halves[0]], axis=0)[..., :3]   # bottom rows first: cheering 0-127, seated 128-255
        out[name] = image(name, atlas)
        # Unity samples the halves as two textures (V clamped, U repeating): in one atlas the halves bled into each
        # other at the low mips and the distant stands flickered in horizontal bands
        for half, tag in ((halves[0], "seated"), (halves[1], "cheer")):
            unity_png(f"{name.replace('_atlas', '')}_{tag}", half)
    return out


def crowd_cards(atlas, col):
    """Fans as 1.0 m cards standing on the rows. The crowd used to be painted on the 0.4 m risers: from a trackside
    camera (below the upper rows) the treads between risers are culled and each row read as a thin line. Now every
    crowd riser (face attribute crowd_face, stored on the first run) turns into plain seat, and a card stands on its
    tread: 0.3 m behind the riser's top edge, 1.0 m tall, facing the field, UV u from the riser (64 seats per u),
    v 0.5-1 = the seated half of the atlas (Unity's PoOlympic/Crowd swaps in the cheering half)."""
    for mname, iname in (("Crowd_Blue", "Crowd_Blue_atlas"), ("Crowd_White", "Crowd_White_atlas")):
        tex = next(nd for nd in bpy.data.materials[mname].node_tree.nodes if nd.type == "TEX_IMAGE")
        tex.image = atlas[iname]
    seat_for = {1: "Seats_Blue", 2: "Seats_White"}
    crowd_mat = {1: bpy.data.materials["Crowd_Blue"], 2: bpy.data.materials["Crowd_White"]}
    up = Vector((0, 0, 1.0))
    cards = Mesh()
    n = 0
    for oname in ("Stand_Lower", "Stand_Upper"):
        o = bpy.data.objects[oname]
        me, mw = o.data, o.matrix_world
        attr = me.attributes.get("crowd_face")
        if attr is None:
            names = [mt.name if mt else "" for mt in me.materials]
            attr = me.attributes.new("crowd_face", "INT", "FACE")
            attr.data.foreach_set("value", [1 if names[p.material_index] == "Crowd_Blue" else 2 if names[p.material_index] == "Crowd_White" else 0
                                            for p in me.polygons])
        vals = [0] * len(me.polygons)
        attr.data.foreach_get("value", vals)
        slot = {mt.name: i for i, mt in enumerate(me.materials) if mt}
        uv = me.uv_layers["UVMap"].data
        for p, v in zip(me.polygons, vals):
            if not v:
                continue
            p.material_index = slot[seat_for[v]]
            co = [mw @ me.vertices[i].co for i in p.vertices]
            us = [uv[li].uv[0] for li in p.loop_indices]
            ztop = max(c.z for c in co)
            top = sorted([(c, u) for c, u in zip(co, us) if c.z > ztop - 0.05], key=lambda t: t[1])
            if len(top) < 2:
                continue
            (a, ua), (b, ub) = top[0], top[-1]
            mid = (a + b) / 2
            cx = math.copysign(min(abs(mid.x), L / 2), mid.x)
            to_field = Vector((cx - mid.x, -mid.y, 0.0)).normalized()
            A, B = a - to_field * 0.3, b - to_field * 0.3
            quad = [A, B, B + up, A + up]
            uvq = [(ua, 0.5), (ub, 0.5), (ub, 1.0), (ua, 1.0)]
            if (B - A).cross(up).dot(to_field) < 0:                  # face the field
                quad, uvq = quad[::-1], uvq[::-1]
            cards.add(quad, [(0, 1, 2, 3)], crowd_mat[v], [uvq])
            n += 1
    ob = cards.emit("Crowd_Cards", col)
    return ob, n


# ---------------------------------------------------------------------------------------------------------- 4 roof
def roof(col):
    steel = bpy.data.materials["Steel"]
    grate = material("Catwalk_Grate", (0.18, 0.19, 0.2), 0.55, 1.0)
    speaker = material("Speaker_Black", (0.03, 0.03, 0.035), 0.7)
    m = Mesh()
    for y in LONG_Y:                                               # catwalk beside each long truss, 1 m below the chord
        yc = y + math.copysign(1.4, y if y else 1)
        span = L / 2 + math.sqrt(max(OPEN_R ** 2 - yc * yc, 0.0)) - 4.0
        z = 28.2
        m.box((0, yc, z), (2 * span, 1.0, 0.06), grate, tile=1.0)
        for side in (-0.5, 0.5):
            m.box((0, yc + side, z + 1.05), (2 * span, 0.05, 0.05), steel, tile=1.0)     # hand rail
            m.box((0, yc + side, z + 0.5), (2 * span, 0.03, 0.03), steel, tile=1.0)      # knee rail
        for x in np.arange(-span, span + 0.1, 6.0):                                     # hangers
            m.box((float(x), yc, 29.6), (0.05, 0.05, 2.8), steel, tile=1.0)
    speakers = []
    for k, (x, y) in enumerate([(-13, 0), (13, 0), (0, -13), (0, 13), (-60, -20), (60, -20), (-60, 20), (60, 20)]):
        z = 26.5
        yaw = math.atan2(-y, -x) if (x or y) else 0.0
        R = Matrix.Rotation(yaw, 3, "Z")
        for j in range(4):                                         # line array: 4 boxes curving down/out
            tilt = Matrix.Rotation(math.radians(6 + j * 7), 3, "Y")
            m.box(Vector((x, y, z - j * 0.75)) + R @ Vector((j * 0.12, 0, 0)), (0.8, 1.4, 0.7), speaker, R @ tilt)
        m.box((x, y, z + 0.6), (1.0, 1.6, 0.12), steel)             # rigging frame
        m.box((x, y, (z + 0.7 + 29.0) / 2), (0.04, 0.04, 29.0 - z - 0.7), steel, tile=1.0)
        speakers.append(empty(f"Speaker_{k:02d}", (x, y, z - 1.2), col, rot=(0, 0, yaw)))
    ob = m.emit("Roof_Rigging", col)
    return ob, speakers


def ceiling_panels():
    """Acoustic ceiling: 1 panel per UV tile of the dome (planar UVs at 6 m) → 3 x 3 panels, seams, perforation."""
    N = 512
    y, x = np.mgrid[0:N, 0:N] / N
    cell = (x * 3) % 1, (y * 3) % 1
    seam = (np.minimum(np.minimum(cell[0], 1 - cell[0]), np.minimum(cell[1], 1 - cell[1])) < 0.012).astype(np.float32)
    rng = np.random.default_rng(3)
    tone = rng.normal(0, 1, (3, 3)) * 0.025
    base = 0.34 + tone[(y * 3).astype(int) % 3, (x * 3).astype(int) % 3]
    perf = (((x * N) % 8 < 2) & ((y * N) % 8 < 2)).astype(np.float32) * 0.05
    v = np.clip(base - perf - seam * 0.2, 0, 1)
    img = image("Ceiling_Panels", np.stack([v * 0.97, v * 0.98, v], -1))
    m = material("Ceiling_Panels", (1, 1, 1), 0.9, 0.0, image=img)
    dome = bpy.data.objects["Ceiling_Dome"]
    dome.data.materials.clear()
    dome.data.materials.append(m)
    return m


def beams(col):
    """One open cone per field spot (Spot_00-23): apex at the lens, 70 % of the way to the floor, radius from a
    35° half angle scaled down to the bright core. UV v = 0 at the lens → 1 at the end (Unity fades along v)."""
    m = material("Light_Beam", (1.0, 0.97, 0.9), 1.0, 0.0, (1.0, 0.97, 0.9), 0.4, alpha=0.08)
    mesh = Mesh()
    seg = 16
    for k in range(WASH_FROM):
        s, a = bpy.data.objects.get(f"Spot_{k:02d}"), bpy.data.objects.get(f"SpotAim_{k:02d}")
        if s is None or a is None:
            continue
        p0, p1 = s.matrix_world.translation, a.matrix_world.translation
        axis = (p1 - p0)
        length = axis.length * 0.7
        axis.normalize()
        r0, r1 = 0.35, math.tan(math.radians(35)) * length * 0.55
        q = axis.to_track_quat("Z", "Y").to_matrix()
        ring = lambda r, d: [p0 + q @ Vector((r * math.cos(2 * math.pi * i / seg), r * math.sin(2 * math.pi * i / seg), d)) for i in range(seg)]
        verts = ring(r0, 0.0) + ring(r1, length)
        faces, uvs = [], []
        for i in range(seg):
            j = (i + 1) % seg
            faces.append((i, j, seg + j, seg + i))
            uvs.append([(i / seg, 0), ((i + 1) / seg, 0), ((i + 1) / seg, 1), (i / seg, 1)])
        mesh.add(verts, faces, m, uvs)
    ob = mesh.emit("Light_Beams", col, lightmap=False)
    for p in ob.data.polygons:                                     # smooth normals: the fresnel edge fades evenly
        p.use_smooth = True
    return ob


# ------------------------------------------------------------------------------------------------------- 5 infield
def infield():
    src = bpy.data.images.get("rubberized_track_diff")
    arr = np.array(src.pixels[:], np.float32).reshape(src.size[1], src.size[0], 4)[::2, ::2, :3]
    lum = arr @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lum = 1.0 + (lum / max(lum.mean(), 1e-3) - 1.0) * 0.8        # granule contrast around 1
    nor = bpy.data.images.get("rubberized_track_nor")
    n = 0
    for name, (tint, _) in PHASE_TINT.items():
        m = bpy.data.materials.get(name)
        if m is None:
            continue
        # the tint is baked into a small texture per phase (the glTF exporter carries an image, not a multiply node)
        img = image(f"Pad_{name[-2:]}_Floor", np.clip(lum[..., None] * np.array(tint, np.float32), 0, 1))
        nt = m.node_tree
        for nd in [nd for nd in nt.nodes if nd.type in ("TEX_IMAGE", "MIX", "MIX_RGB", "NORMAL_MAP", "MATH", "VECT_MATH", "MAPPING", "TEX_COORD")]:
            nt.nodes.remove(nd)
        bsdf = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")
        tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.78
        bsdf.inputs["Metallic"].default_value = 0.0
        if nor is not None:
            nt_ = nt.nodes.new("ShaderNodeTexImage"); nt_.image = nor
            nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 0.6
            nt.links.new(nt_.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
        n += 1
    # a thin painted border line in the saturated phase colour, 0.25 m inside every pad edge
    lines = Mesh()
    for o in [o for o in bpy.data.objects if o.type == "MESH" and o.name.endswith("_Pad") and o.data.materials]:
        mname = o.data.materials[0].name
        if mname not in PHASE_TINT:
            continue
        uv = o.data.uv_layers.get("UVMap") or o.data.uv_layers.new(name="UVMap")
        mw = o.matrix_world
        for li, loop in enumerate(o.data.loops):                  # world-scale floor UVs: one granule tile per 2 m
            w = mw @ o.data.vertices[loop.vertex_index].co
            uv.data[li].uv = (w.x / 2.0, w.y / 2.0)
        ws = [o.matrix_world @ Vector(c) for c in o.bound_box]
        x0, x1 = min(v.x for v in ws) + 0.25, max(v.x for v in ws) - 0.25
        y0, y1 = min(v.y for v in ws) + 0.25, max(v.y for v in ws) - 0.25
        z = max(v.z for v in ws) + 0.004
        lm = material(f"PadLine_{mname[-2:]}", PHASE_TINT[mname][1], 0.5)
        w = 0.12
        for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            d = Vector((b[0] - a[0], b[1] - a[1], 0)).normalized()
            nrm = Vector((-d.y, d.x, 0)) * w / 2
            A, B = Vector((a[0], a[1], z)), Vector((b[0], b[1], z))
            lines.quad(A - nrm, B - nrm, B + nrm, A + nrm, lm)
    return n, lines.emit("Pad_Lines", COL, lightmap=False)


# -------------------------------------------------------------------------------------------------------- 6 podium
def podium(col):
    white = material("Podium_White", (0.9, 0.9, 0.92), 0.35)
    gold = material("Podium_Gold", (0.95, 0.72, 0.2), 0.3, 1.0)
    silver = material("Podium_Silver", (0.78, 0.8, 0.84), 0.3, 1.0)
    bronze = material("Podium_Bronze", (0.72, 0.42, 0.2), 0.35, 1.0)
    base = material("Podium_Base", (0.08, 0.1, 0.18), 0.6)
    pole = bpy.data.materials.get("Flag_Pole") or material("Flag_Pole", (0.8, 0.8, 0.82), 0.3, 0.8)
    flagm = material("Podium_Flag", (0.95, 0.95, 0.95), 0.8)
    m = Mesh()
    m.box(PODIUM + Vector((0, 0.5, 0.06)), (7.0, 4.0, 0.12), base, tile=1.0)          # plinth
    steps = {1: (0.0, 0.9, gold), 2: (-1.6, 0.6, silver), 3: (1.6, 0.4, bronze)}      # viewer left = 2nd, right = 3rd
    anchors = []
    for place, (dx, h, trim) in steps.items():
        c = PODIUM + Vector((dx, 0, 0.12 + h / 2))
        m.box(c, (1.6, 1.4, h), white, tile=1.0)
        m.box(c + Vector((0, -0.705, 0)), (1.6, 0.02, h * 0.9), trim)                   # front medal panel
        anchors.append(empty(f"Podium_Step_{place}", PODIUM + Vector((dx, 0, 0.12 + h)), col))
    flags = []
    for place, dx, ph in ((2, -2.2, 7.0), (1, 0.0, 8.0), (3, 2.2, 6.5)):
        foot = PODIUM + Vector((dx, 3.2, 0))
        m.box(foot + Vector((0, 0, ph / 2)), (0.12, 0.12, ph), pole, tile=1.0)
        m.box(foot + Vector((0, 0, ph + 0.06)), (0.22, 0.22, 0.12), pole, tile=1.0)
        # flag: separate object, pivot at its bottom-left on the pole, parked low (Unity raises it)
        fm = Mesh()
        fm.quad(Vector((0.08, 0, 0)), Vector((1.9, 0, 0)), Vector((1.9, 0, 1.2)), Vector((0.08, 0, 1.2)), flagm)
        fm.quad(Vector((0.08, 0, 1.2)), Vector((1.9, 0, 1.2)), Vector((1.9, 0, 0)), Vector((0.08, 0, 0)), flagm)
        fo = fm.emit(f"Podium_Flag_{place}", col, lightmap=False)
        fo.location = foot + Vector((0, -0.08, 1.4))
        flags.append(fo)
        empty(f"Podium_FlagTop_{place}", foot + Vector((0, -0.08, ph - 1.3)), col)
    ob = m.emit("Podium", col)
    empty("PodiumCam", PODIUM + Vector((0.0, -9.5, 2.2)), col)
    empty("PodiumLook", PODIUM + Vector((0.0, 0.0, 1.6)), col)
    return ob, anchors, flags


# --------------------------------------------------------------------------------------------------------- 8 audio
def audio_anchors(col):
    made = []
    for name, loc in (("AudioAnchor_Crowd_S", (0, -70, 10)), ("AudioAnchor_Crowd_N", (0, 70, 10)),
                      ("AudioAnchor_Crowd_E", (112, 0, 10)), ("AudioAnchor_Crowd_W", (-112, 0, 10)),
                      ("AudioAnchor_PA_Centre", (0, 0, CUBE_Z - 6)), ("AudioAnchor_Finish", (42.2, -42, 1.5)),
                      ("AudioAnchor_Podium", tuple(PODIUM + Vector((0, 0, 2))))):
        made.append(empty(name, loc, col, size=2.0))
    return made


# ------------------------------------------------------------------------------------------------------------- main
def run():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    lanes_before = {o.name: [round(v, 6) for row in o.matrix_world for v in row]
                    for o in bpy.data.objects if o.type == "EMPTY" and o.name[:1] == "E" and "_L" in o.name}
    col_clear(COL)
    rep = {"offstage": offstage()}
    rep["led_lettering"] = led_lettering(COL).name
    rep["wash_reaimed"] = soften_wash()
    rep["screens"] = screens(COL).name
    atlas = crowd_atlas()
    cards, rep["crowd_cards"] = crowd_cards(atlas, COL)
    rig, speakers = roof(COL)
    rep["speakers"] = len(speakers)
    ceiling_panels()
    beam = beams(COL)
    rep["pads"], lines = infield()
    pod, steps, flags = podium(COL)
    rep["podium_flags"] = len(flags)
    rep["audio_anchors"] = len(audio_anchors(COL))
    lightmap_pack([bpy.data.objects["VideoCube"], rig, pod, cards])
    # preview lights follow the re-aimed wash anchors
    for k in range(WASH_FROM, 36):
        lo, a = bpy.data.objects.get(f"SpotLight_{k:02d}"), bpy.data.objects.get(f"SpotAim_{k:02d}")
        if lo is not None and a is not None:
            d = a.matrix_world.translation - lo.matrix_world.translation
            lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
            lo.data.spot_blend = 0.8
    after = {o.name: [round(v, 6) for row in o.matrix_world for v in row]
             for o in bpy.data.objects if o.type == "EMPTY" and o.name[:1] == "E" and "_L" in o.name}
    assert after == lanes_before, "lane anchors moved"
    rep["lane_anchors_unchanged"] = len(lanes_before)
    rep["tris_export"] = sum(sum(len(p.vertices) - 2 for p in o.data.polygons)
                             for o in bpy.data.collections["Stadium"].all_objects if o.type == "MESH")
    print("showcase:", rep)
    return rep


def export_stadium(path=r"C:\Users\punko\Downloads\PoOlympic\Assets\PoOlympic\Art\Stadium\Stadium.glb"):
    """Same export as build_polish.export_stadium (glTF scene renamed "Scene" — see there)."""
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


if not globals().get("SHOWCASE_NO_RUN"):
    run()
