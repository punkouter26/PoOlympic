"""12 The 360 Turntable — 8 athletes on the venue's spin spots (assets/scene_turntable8.xml), Rung 2 brain.
Mirror of Unity TurntableEvent.

On GO every athlete is commanded a pure in-place yaw rate (WZ, the top of the trained envelope) in the heat's seeded
direction until its pelvis heading has turned TURNS full circles, then zero command (hold still on the spot).
  score  = time to complete the turns + DRIFT_PENALTY x the largest pelvis drift from the spot (s per m)
  out    = a fall (FELL) or the pelvis leaving the painted ring (DQ, drift > RING_R)
Rank by score; fallers / DQs / unfinished after them. Traits as in the Iron Pedestal heat.
"""

from __future__ import annotations

import json
from pathlib import Path
import math
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np
import onnxruntime as ort

from .. import contract as C
from .iron_pedestal import Traits, _Lane, make_lanes

SCENE = C.ROOT / "assets" / "scene_turntable8.xml"
LAYOUT = C.ROOT / "assets" / "turntable8_layout.json"
WZ = 3.0               # rad/s — Rung 2 training envelope top (the G1 turntable drill uses 2.5)
TURNS = 3
DRIFT_PENALTY = 2.0    # s per metre of max drift
RING_R = 1.1           # m — the painted ring around each spot (build_venues.py)
START_S = 0.5          # s of zero command before GO (same as the Unity countdown hand-off)
MAX_S = 15.0


@dataclass
class SpinLane:
    lane: int
    traits: Traits
    time_s: float | None = None      # time to complete TURNS turns
    max_drift_m: float = 0.0
    score: float | None = None
    status: str = ""                 # DONE / FELL / DQ / DNF
    place: int = 0


@dataclass
class SpinResult:
    seed: int
    direction: int
    duration_s: float
    lanes: list[SpinLane] = field(default_factory=list)


def heat_direction(seed: int) -> int:
    """+1 = anticlockwise (left), -1 = clockwise, drawn per heat from the heat seed (Unity draws it from its own heat RNG)."""
    return 1 if np.random.default_rng([seed, 3000]).uniform() < 0.5 else -1


def run_heat(onnx, seed: int, traits: list[Traits] | None = None, scene=None, layout_path=None, brains: dict | None = None) -> SpinResult:
    m = mujoco.MjModel.from_xml_path(str(scene or SCENE))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path or LAYOUT).read_text())
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, onnx, brains)
    mujoco.mj_forward(m, d)
    direction = heat_direction(seed)
    res = [SpinLane(ln.k, ln.traits) for ln in lanes]
    turned = [0.0] * len(lanes)
    prev_yaw = [C.yaw_of(d.qpos[ln.ath.root_qposadr + 3: ln.ath.root_qposadr + 7]) for ln in lanes]
    dt = m.opt.timestep * C.DECIMATION
    goal = TURNS * 2 * math.pi
    tick = 0
    while tick * dt < START_S + MAX_S:
        t = tick * dt
        for i, ln in enumerate(lanes):
            spinning = t >= START_S and not res[i].status
            ln.control(ln.sess, d, np.array([0.0, 0.0, direction * WZ]) if spinning else np.zeros(3))
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt
        for i, ln in enumerate(lanes):
            r = res[i]
            ra = ln.ath.root_qposadr
            yaw = C.yaw_of(d.qpos[ra + 3: ra + 7])
            step = (yaw - prev_yaw[i] + math.pi) % (2 * math.pi) - math.pi
            prev_yaw[i] = yaw
            if r.status in ("FELL", "DQ"):
                continue
            drift = math.hypot(d.qpos[ra] - ln.origin[0], d.qpos[ra + 1] - ln.origin[1])
            r.max_drift_m = max(r.max_drift_m, drift)
            if ln.eliminated(m, d) == "FELL":
                r.status = "FELL"
                continue
            if drift > RING_R:
                r.status = "DQ"
                continue
            if r.status == "DONE":
                continue
            turned[i] += direction * step
            if turned[i] >= goal:
                r.status, r.time_s = "DONE", t - START_S
        if all(r.status for r in res):
            break
    for r in res:
        r.status = r.status or "DNF"
        if r.status == "DONE":
            r.score = r.time_s + DRIFT_PENALTY * r.max_drift_m
    order = sorted(res, key=lambda r: (0, r.score) if r.status == "DONE" else (1, 0))
    for p, r in enumerate(order, 1):
        r.place = p
    return SpinResult(seed, direction, tick * dt, res)


def to_json(res: SpinResult) -> dict:
    return {"seed": res.seed, "direction": res.direction, "duration_s": res.duration_s,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}
