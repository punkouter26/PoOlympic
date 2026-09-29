"""PoOlympics stadium — one venue per Olympic event (30), each with room for 8 competitors.

Run inside Blender (MCP or Text Editor) after the stadium shell exists (stadium_helpers.py + stadium.blend):
    exec(open(r"<repo>/SourceArt/Stadium/build_venues.py").read())

Coordinates = MuJoCo world (x along the home straight, y left, z up, metres); the infield surface is z = 0.
Everything here is RENDER-ONLY. Physical props (pedestals, hurdles, ramps, stairs, …) are generated into the event MJCF
from venues.json, which this script writes — one layout drives both the art and the physics.

Per event: a coloured pad (colour = programme phase), an event label, props, and 8 lane anchors E##_L0..E##_L7
(empties at each competitor's start spot, facing the event direction). Lane 0 is on the competitor's left.
"""

import json
import math
import os
import random

import bpy

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else r"C:\Users\punko\Downloads\PoOlympic\SourceArt\Stadium"
exec(open(os.path.join(HERE, "stadium_helpers.py")).read())

N = 8
LW = LANE_W                  # 1.22 m
BLOCK_W = N * LW             # 9.76 m
STATION_PITCH = 3.0
PEDESTAL = (1.0, 1.0, 0.5)
# Decal layers above the infield (z = 0). >= 1 cm apart: at 30-60 m a 24-bit depth buffer resolves only a few mm, so the
# old 1-5 mm offsets z-fought ("flicker").
Z_PAD, Z_BORDER, Z_SURFACE, Z_LINE, Z_MARK, Z_LABEL = 0.02, 0.032, 0.03, 0.042, 0.052, 0.064   # = training/tools/build_mjcf.py PEDESTAL_HALF*2, PEDESTAL_H

PHASE_COL = {1: (0.07, 0.15, 0.36), 2: (0.04, 0.28, 0.29), 3: (0.24, 0.11, 0.34),
             4: (0.33, 0.08, 0.10), 5: (0.24, 0.26, 0.08), 6: (0.11, 0.11, 0.12)}

EVENTS = {  # number: (name, phase)
    1: ("The Iron Pedestal", 1), 2: ("Torso Archer", 1), 3: ("Deep Squat Endurance", 1), 4: ("Precision Javelin Reach", 1),
    5: ("The Gust Gauntlet", 1), 6: ("The Flamingo Classic", 2), 7: ("Cadence March", 2), 8: ("30m All Fours", 2),
    9: ("The Inverted Sprint", 2), 10: ("Crab Shuffle Relay", 2), 11: ("Slalom Sprint", 3), 12: ("The 360 Turntable", 3),
    13: ("Steeplechase Jog", 3), 14: ("The Alpine Ramp", 3), 15: ("Cross-Country Rubble", 3), 16: ("The Platform Drop", 4),
    17: ("Stadium Stair Climb", 4), 18: ("The Olympic High Jump", 4), 19: ("Terminal Velocity Sprint", 4),
    20: ("Low Hurdle Dash", 4), 21: ("The Sandpit Long Jump", 5), 22: ("Emergency Brake", 5), 23: ("The Trench Crawl", 5),
    24: ("The Courier Carry", 5), 25: ("The Bench Relay", 5), 26: ("Stepping Stones", 6), 27: ("The Resurrection Dash", 6),
    28: ("Floor Acrobatic Sprint", 6), 29: ("Striker Shootout", 6), 30: ("The Grand Parkour Vault", 6),
}

# ---------------------------------------------------------------------------------------------------- layout
# station events: 2 x 4 grid (x pitch 3 m, rows y = cy ± 2) unless 'row' (1 x 8, the Iron Pedestal showpiece)
STATIONS = {18: (-62, 0), 12: (-48, 0), 5: (-34, 0), 3: (-20, 0), 1: (0, 0), 6: (20, 0), 2: (34, 0), 4: (48, 0), 7: (62, 0)}
# lane events in the infield: (x_start, y_centre, length) — 8 lanes along +x
LANE_EVENTS = {
    13: (-70, -13, 50), 9: (-16, -13, 20), 10: (8, -13, 20), 11: (32, -13, 32),
    30: (-70, 13, 30), 14: (-36, 13, 16), 16: (-16, 13, 14), 17: (2, 13, 14), 15: (20, 13, 26), 26: (50, 13, 18),
    23: (-58, -26, 16), 24: (-38, -26, 16), 25: (-18, -26, 16), 28: (2, -26, 16), 29: (22, -26, 22),
    21: (-56, 26, 34),
}
HOME_Y = -(R0 + BLOCK_W / 2)        # home straight lanes (runners go +x); lane 0 = inner lane = +y side
BACK_Y = R0 + BLOCK_W / 2           # back straight (runners go -x); lane 0 = the runner's left = -y side
TRACK_EVENTS = {  # (x_start, y_centre, length, direction)
    8: (L / 2 - 100.0, HOME_Y, 30, +1), 20: (-24, HOME_Y, 30, +1), 22: (8, HOME_Y, 30, +1),
    19: (L / 2, BACK_Y, L, -1), 27: (-77, HOME_Y, 8, +1),
}

M = {m.name: m for m in bpy.data.materials}
line_m, iron = M["Line_White"], mat("IronPedestal", (0.23, 0.24, 0.26), 0.45, 0.85)
steel, red = M["Steel"], mat("Stop_Red", (0.80, 0.05, 0.05), 0.5)
hazard, sand = mat("Hazard_Yellow", (0.95, 0.75, 0.05), 0.5), M["Sand"]
wood, mat_blue = mat("Wood", (0.55, 0.38, 0.22), 0.7), M["GymMat_Blue"]
orange, white_net = M["Marker_Orange"], mat("Net_White", (0.95, 0.95, 0.95), 0.8)
track_red = M["Track_Red"]
PAD = {p: mat(f"EventPad_P{p}", c, 0.85) for p, c in PHASE_COL.items()}
COL = "Venues"


def clear_old():
    for cname in ("Venues", "Anchors"):
        c = bpy.data.collections.get(cname)
        if c:
            for o in list(c.objects):
                bpy.data.objects.remove(o, do_unlink=True)


def box(name, cx, cy, cz, sx, sy, sz, m, yaw=0.0):
    """Axis box centred at (cx, cy, cz) with full sizes (sx, sy, sz), optional yaw about z."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dz in (-sz / 2, sz / 2):
        for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)):
            pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy, cz + dz))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return mesh_obj(name, pts, faces, m, COL)


def flat(name, x0, y0, x1, y1, z, m):
    return mesh_obj(name, [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], [(0, 1, 2, 3)], m, COL)


def ring_mark(name, cx, cy, r, w, z, m, n=40):
    verts, faces = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        verts += [(cx + (r - w / 2) * math.cos(a), cy + (r - w / 2) * math.sin(a), z),
                  (cx + (r + w / 2) * math.cos(a), cy + (r + w / 2) * math.sin(a), z)]
    for i in range(n):
        j = (i + 1) % n
        faces.append((2 * i, 2 * j, 2 * j + 1, 2 * i + 1))
    return mesh_obj(name, verts, faces, m, COL)


def label(name, text, x, y, size, z=Z_LABEL, yaw=0.0, m=None):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = text
    cu.size = size
    cu.align_x = "LEFT"
    tmp = bpy.data.objects.new(name + "_tmp", cu)
    bpy.context.scene.collection.objects.link(tmp)
    tmp.location = (x, y, z)
    tmp.rotation_euler = (0, 0, yaw)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    ob = bpy.data.objects.new(name, me)
    ob.matrix_world = tmp.matrix_world.copy()
    bpy.data.objects.remove(tmp, do_unlink=True)
    bpy.data.curves.remove(cu)
    me.materials.append(m or line_m)
    get_col(COL).objects.link(ob)
    return ob


def anchor(name, x, y, z=0.0, yaw=0.0):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = "SINGLE_ARROW"
    e.empty_display_size = 0.8
    e.location = (x, y, z)
    e.rotation_euler = (0, 0, yaw)
    get_col("Anchors").objects.link(e)
    return e


def pad_with_border(tag, x0, y0, x1, y1, phase, z=Z_PAD):
    flat(f"{tag}_Pad", x0, y0, x1, y1, z, PAD[phase])
    w = 0.08
    for i, (a, b, c, d) in enumerate(((x0, y0, x1, y0 + w), (x0, y1 - w, x1, y1), (x0, y0, x0 + w, y1), (x1 - w, y0, x1, y1))):
        flat(f"{tag}_Border{i}", a, b, c, d, z + (Z_BORDER - Z_PAD), line_m)


LAYOUT = {"units": "m", "frame": "MuJoCo world: x along the home straight, y left, z up; infield z = 0",
          "unity": "Stadium.glb root yaw 180 deg; EventScenes.PlaceStadium snaps an anchor (E##_L#) onto the MuJoCo origin",
          "events": {}}


def record(num, kind, lanes, extra=None):
    name, phase = EVENTS[num]
    LAYOUT["events"][f"{num:02d}"] = {"name": name, "phase": phase, "kind": kind,
                                     "lanes": [{"lane": k, "pos": [round(x, 4), round(y, 4), round(z, 4)], "yaw_deg": round(math.degrees(yaw), 3)}
                                               for k, (x, y, z, yaw) in enumerate(lanes)], **(extra or {})}
    for k, (x, y, z, yaw) in enumerate(lanes):
        anchor(f"E{num:02d}_L{k}", x, y, z, yaw)


# ---------------------------------------------------------------------------------------------------- stations
def build_station(num, cx, cy):
    name, phase = EVENTS[num]
    tag = f"E{num:02d}"
    if num == 1:   # the Iron Pedestal showpiece: 8 pedestals in one row, facing -y (towards the main camera side)
        xs = [cx + (k - 3.5) * STATION_PITCH for k in range(N)]
        spots = [(x, cy) for x in xs]
        pad_with_border(tag, cx - 13.0, cy - 4.2, cx + 13.0, cy + 4.2, phase)
    else:          # 2 x 4 grid
        spots = [(cx + (c - 1.5) * STATION_PITCH, cy + (2.0 if r == 0 else -2.0)) for r in range(2) for c in range(4)]
        pad_with_border(tag, cx - 6.5, cy - 4.2, cx + 6.5, cy + 4.2, phase)
    label(f"{tag}_Label", f"{num:02d} {name.upper()}", cx - (12.6 if num == 1 else 6.2), cy - 4.0, 0.55)
    lanes = []
    for k, (x, y) in enumerate(spots):
        top = 0.0
        if num == 1:
            box(f"{tag}_Pedestal_{k}", x, y, PEDESTAL[2] / 2, PEDESTAL[0], PEDESTAL[1], PEDESTAL[2], iron)
            label(f"{tag}_No_{k}", str(k + 1), x - 0.25, y - 2.1, 0.8)
            top = PEDESTAL[2]
        elif num == 5:   # shaker platforms (hazard striped)
            box(f"{tag}_Shaker_{k}", x, y, 0.05, 1.6, 1.6, 0.1, hazard)
            top = 0.1
        elif num == 12:
            ring_mark(f"{tag}_Spot_{k}", x, y, 1.1, 0.07, Z_MARK, line_m)
            flat(f"{tag}_Dot_{k}", x - 0.1, y - 0.1, x + 0.1, y + 0.1, Z_LABEL, orange)
        elif num == 18:  # measuring pole with height stripes behind each station
            box(f"{tag}_Pole_{k}", x + 0.9, y, 1.5, 0.08, 0.08, 3.0, steel)
            for h in range(1, 6):
                box(f"{tag}_Mark_{k}_{h}", x + 0.9, y, 0.5 * h, 0.2, 0.2, 0.03, red)
            ring_mark(f"{tag}_Spot_{k}", x, y, 0.6, 0.06, Z_MARK, line_m)
        elif num == 2:   # archer: foot box + overhead target mast
            flat(f"{tag}_FootBox_{k}", x - 0.5, y - 0.4, x + 0.5, y + 0.4, Z_MARK, line_m)
            box(f"{tag}_Mast_{k}", x, y + (1.3 if y > cy else -1.3), 2.5, 0.08, 0.08, 5.0, steel)
            box(f"{tag}_Target_{k}", x, y + (1.3 if y > cy else -1.3), 5.1, 0.7, 0.05, 0.7, red)
        elif num == 4:   # javelin reach: target pole at arm's length
            flat(f"{tag}_FootBox_{k}", x - 0.4, y - 0.4, x + 0.4, y + 0.4, Z_MARK, line_m)
            box(f"{tag}_Reach_{k}", x + 1.1, y, 0.8, 0.05, 0.05, 1.6, steel)
            box(f"{tag}_Reach_Target_{k}", x + 1.1, y, 1.45, 0.18, 0.18, 0.18, orange)
        elif num == 3:
            flat(f"{tag}_Mat_{k}", x - 0.6, y - 0.6, x + 0.6, y + 0.6, Z_SURFACE, mat_blue)
        elif num == 6:
            ring_mark(f"{tag}_Spot_{k}", x, y, 0.5, 0.06, Z_MARK, line_m)
        elif num == 7:
            ring_mark(f"{tag}_Spot_{k}", x, y, 0.7, 0.06, Z_MARK, line_m)
        lanes.append((x, y, top, -math.pi / 2 if num == 1 else 0.0))
    if num == 7:     # stadium metronome tower at the grid centre
        box(f"{tag}_Metronome", cx, cy, 2.5, 0.4, 0.4, 5.0, steel)
        box(f"{tag}_Metronome_Arm", cx, cy, 5.3, 0.1, 1.6, 0.1, orange, yaw=0.0)
    if num == 1:
        # judges' plinth + event board behind the row
        box(f"{tag}_Board", cx, cy + 4.0, 0.45, 18.0, 0.12, 0.9, M["Board_LED"])   # low LED strip: must not wall off the broadcast view
    record(num, "row" if num == 1 else "grid", lanes, {"pedestal_size_m": list(PEDESTAL)} if num == 1 else None)


# ---------------------------------------------------------------------------------------------------- lane events
def lane_block(num, x0, cy, length, direction=+1, pad=True, lines=True):
    name, phase = EVENTS[num]
    tag = f"E{num:02d}"
    x1 = x0 + direction * length
    lo, hi = min(x0, x1), max(x0, x1)
    if pad:
        pad_with_border(tag, lo - 1.5, cy - BLOCK_W / 2 - 0.4, hi + 1.5, cy + BLOCK_W / 2 + 0.4, phase)
    if lines:
        for j in range(N + 1):
            y = cy - BLOCK_W / 2 + j * LW
            flat(f"{tag}_Lane_{j}", lo, y - 0.025, hi, y + 0.025, Z_LINE, line_m)
    for nm, x in (("Start", x0), ("Finish", x1)):
        flat(f"{tag}_{nm}", x - 0.04, cy - BLOCK_W / 2, x + 0.04, cy + BLOCK_W / 2, Z_MARK, red if nm == "Finish" else line_m)
    label(f"{tag}_Label", f"{num:02d} {name.upper()}", lo - 1.2, cy + BLOCK_W / 2 + 0.6, 0.6)
    yaw = 0.0 if direction > 0 else math.pi
    lanes = []
    for k in range(N):   # lane 0 on the competitor's left
        y = cy + direction * (3.5 - k) * LW
        lanes.append((x0, y, 0.0, yaw))
        label(f"{tag}_No_{k}", str(k + 1), x0 - direction * 1.1 - 0.2, y - 0.3, 0.6, yaw=0.0)
    record(num, "lanes", lanes, {"length_m": length, "direction": "+x" if direction > 0 else "-x"})
    return tag, lo, hi


def lane_ys(cy):
    return [cy + (3.5 - k) * LW for k in range(N)]


def build_lane_events():
    for num, (x0, cy, length) in LANE_EVENTS.items():
        tag, lo, hi = lane_block(num, x0, cy, length)
        ys = lane_ys(cy)
        if num == 10:    # crab shuffle rails along every lane boundary
            for j in range(N + 1):
                y = cy - BLOCK_W / 2 + j * LW
                box(f"{tag}_Rail_{j}", (lo + hi) / 2, y, 0.3, hi - lo, 0.04, 0.04, steel)
        elif num == 11:  # slalom poles on every lane's centre line every 4 m (weave: left of pole 1, right of pole 2, …)
            poles = []
            for k, y in enumerate(ys):
                poles.append([])
                for g in range(7):
                    gx = lo + 3.0 + 4.0 * g
                    box(f"{tag}_Gate_{k}_{g}", gx, y, 0.6, 0.04, 0.04, 1.2, orange)
                    poles[-1].append([round(gx, 4), round(y, 4)])
            LAYOUT["events"]["11"]["poles"] = poles    # physical poles in scene_slalom8.xml
        elif num == 13:  # steeplechase: distance boards every 10 m
            for d in range(10, int(hi - lo), 10):
                flat(f"{tag}_Mark_{d}", lo + d - 0.03, cy - BLOCK_W / 2, lo + d + 0.03, cy + BLOCK_W / 2, Z_MARK, orange)
        elif num == 14:  # 15 deg ramp (10 m) + top deck
            rise = 10.0 * math.tan(math.radians(15))
            xa, xb = lo + 2.0, lo + 12.0
            yl, yh = cy - BLOCK_W / 2, cy + BLOCK_W / 2
            mesh_obj(f"{tag}_Ramp", [(xa, yl, 0), (xb, yl, 0), (xb, yl, rise), (xa, yh, 0), (xb, yh, 0), (xb, yh, rise)],
                     [(0, 1, 2), (3, 5, 4), (0, 2, 5, 3), (1, 4, 5, 2), (0, 3, 4, 1)], M["Concrete"], COL)
            box(f"{tag}_Deck", xb + 1.5, cy, rise / 2, 3.0, BLOCK_W, rise, M["Concrete"])
            box(f"{tag}_Sensor", xb + 2.5, cy, rise + 1.2, 0.15, BLOCK_W, 0.15, red)
        elif num in (16, 17):  # 16: platform + 6 steps down; 17: 20 steps up + landing
            steps, rise, run = (6, 0.2, 0.3) if num == 16 else (20, 0.17, 0.30)
            if num == 17:
                for s in range(steps):
                    box(f"{tag}_Step_{s}", lo + 3.0 + s * run + run / 2, cy, (s + 1) * rise / 2, run, BLOCK_W, (s + 1) * rise, M["Concrete"])
                box(f"{tag}_Landing", lo + 3.0 + steps * run + 1.5, cy, steps * rise / 2, 3.0, BLOCK_W, steps * rise, M["Concrete"])
            else:
                top = steps * rise
                box(f"{tag}_Platform", lo + 2.5, cy, top / 2, 3.0, BLOCK_W, top, M["Concrete"])
                for s in range(steps):
                    h = top - (s + 1) * rise
                    if h > 0:
                        box(f"{tag}_Step_{s}", lo + 4.0 + s * run + run / 2, cy, h / 2, run, BLOCK_W, h, M["Concrete"])
        elif num == 15:  # seeded rubble: mounds and ruts
            rnd = random.Random(15)
            for i in range(70):
                x = rnd.uniform(lo + 2, hi - 1); y = rnd.uniform(cy - BLOCK_W / 2 + 0.3, cy + BLOCK_W / 2 - 0.3)
                h = rnd.uniform(0.04, 0.18)
                box(f"{tag}_Rubble_{i}", x, y, h / 2, rnd.uniform(0.3, 1.2), rnd.uniform(0.3, 1.0), h, M["Concrete"], yaw=rnd.uniform(0, math.pi))
        elif num == 26:  # stepping stones: 8 lanes x 8 pads on 0.5 m posts
            for k, y in enumerate(ys):
                for s in range(8):
                    x = lo + 2.0 + s * 1.8
                    yy = y + (0.15 if s % 2 else -0.15)
                    box(f"{tag}_Post_{k}_{s}", x, yy, 0.25, 0.12, 0.12, 0.5, steel)
                    box(f"{tag}_Stone_{k}_{s}", x, yy, 0.52, 0.4, 0.4, 0.04, M["Concrete"])
        elif num == 30:  # parkour: wall vault, drop platform, hurdles
            box(f"{tag}_Wall", lo + 8.0, cy, 0.5, 0.4, BLOCK_W, 1.0, M["Concrete"])
            box(f"{tag}_DropDeck", lo + 14.0, cy, 0.75, 4.0, BLOCK_W, 1.5, M["Concrete"])
            box(f"{tag}_DropRamp", lo + 11.0, cy, 0.4, 2.0, BLOCK_W, 0.8, M["Concrete"])
            for i, hx in enumerate((lo + 20.0, lo + 23.5, lo + 27.0)):
                for k, y in enumerate(ys):
                    box(f"{tag}_Hurdle_{i}_{k}", hx, y, 0.45, 0.05, 1.0, 0.05, M["Line_White"])
        elif num == 23:  # trench crawl: 0.6 m ceiling over 12 m
            box(f"{tag}_Ceiling", lo + 8.0, cy, 0.62, 12.0, BLOCK_W, 0.04, M["Roof_Under"])
            for j in range(N + 1):
                y = cy - BLOCK_W / 2 + j * LW
                for p in range(4):
                    box(f"{tag}_Post_{j}_{p}", lo + 2.0 + p * 4.0, y, 0.3, 0.05, 0.05, 0.6, steel)
        elif num == 24:  # crates at the start, finish at 15 m
            for k, y in enumerate(ys):
                box(f"{tag}_Crate_{k}", lo + 1.0, y, 0.2, 0.4, 0.4, 0.4, wood)
        elif num == 25:  # benches at 8 m
            for k, y in enumerate(ys):
                box(f"{tag}_Bench_{k}", lo + 8.0, y, 0.225, 0.45, 1.0, 0.45, wood)
        elif num == 28:  # gymnastics mats
            for k, y in enumerate(ys):
                box(f"{tag}_Mat_{k}", (lo + hi) / 2, y, 0.05, hi - lo - 2.0, 1.1, 0.1, mat_blue)
        elif num == 29:  # 8 mini goals + ball spots
            for k, y in enumerate(ys):
                gx = hi + 0.8
                box(f"{tag}_GoalL_{k}", gx, y - 0.55, 0.4, 0.05, 0.05, 0.8, white_net)
                box(f"{tag}_GoalR_{k}", gx, y + 0.55, 0.4, 0.05, 0.05, 0.8, white_net)
                box(f"{tag}_GoalBar_{k}", gx, y, 0.8, 0.05, 1.15, 0.05, white_net)
                ring_mark(f"{tag}_Ball_{k}", lo + 8.0, y, 0.12, 0.04, Z_MARK, line_m, n=16)
        elif num == 21:  # 8 runways + take-off boards + wide sand pit
            for k, y in enumerate(ys):
                flat(f"{tag}_Runway_{k}", lo, y - LW / 2 + 0.03, lo + 22.0, y + LW / 2 - 0.03, Z_SURFACE, track_red)
                flat(f"{tag}_Board_{k}", lo + 21.8, y - LW / 2 + 0.03, lo + 22.0, y + LW / 2 - 0.03, Z_LINE, line_m)
            flat(f"{tag}_Pit", lo + 23.0, cy - BLOCK_W / 2, lo + 33.0, cy + BLOCK_W / 2, Z_SURFACE, sand)


def build_track_events():
    for num, (x0, cy, length, direction) in TRACK_EVENTS.items():
        tag, lo, hi = lane_block(num, x0, cy, length, direction, pad=False, lines=False)
        ys = lane_ys(cy) if direction > 0 else [cy - (3.5 - k) * LW for k in range(N)]
        if num == 20:    # 0.3 m low hurdles every 6 m
            for i in range(4):
                hx = x0 + 6.0 * (i + 1)
                for k, y in enumerate(ys):
                    box(f"{tag}_Hurdle_{i}_{k}", hx, y, 0.3, 0.05, 1.0, 0.05, M["Line_White"])
                    for s in (-0.5, 0.5):
                        box(f"{tag}_HurdleLeg_{i}_{k}_{int(s > 0)}", hx, y + s, 0.15, 0.04, 0.04, 0.3, steel)
        elif num == 22:  # red stop zone at the end of the run-in
            flat(f"{tag}_StopZone", hi - 0.3, cy - BLOCK_W / 2, hi, cy + BLOCK_W / 2, Z_SURFACE, red)
        elif num == 19:  # speed-trap gantries at both ends of the back straight
            for i, gx in enumerate((lo + 2.0, hi - 2.0)):
                for s in (-1, 1):
                    box(f"{tag}_TrapPost_{i}_{s}", gx, cy + s * (BLOCK_W / 2 + 0.4), 2.0, 0.2, 0.2, 4.0, steel)
                box(f"{tag}_TrapBeam_{i}", gx, cy, 4.1, 0.3, BLOCK_W + 1.0, 0.3, M["Board_LED"])
        elif num == 27:  # get-up mats at the start
            for k, y in enumerate(ys):
                box(f"{tag}_Mat_{k}", x0 + 1.0, y, 0.03, 2.2, 1.1, 0.06, mat_blue)


# ---------------------------------------------------------------------------------------------------- main
clear_old()
for n, (cx, cy) in STATIONS.items():
    build_station(n, cx, cy)
build_lane_events()
build_track_events()
missing = sorted(set(EVENTS) - {int(k) for k in LAYOUT["events"]})
assert not missing, missing
out = os.path.join(HERE, "venues.json")
LAYOUT["track"] = {"straight_m": L, "bend_radius_m": R0, "lanes": N, "lane_width_m": LW, "finish_line_x": L / 2,
                   "start_100m_x": L / 2 - 100.0}
with open(out, "w") as f:
    json.dump(LAYOUT, f, indent=1)
print(f"built {len(LAYOUT['events'])} event venues, {len(bpy.data.collections['Anchors'].objects)} lane anchors -> {out}")
