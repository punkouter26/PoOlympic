"""10 Crab Shuffle — 8 athletes side-step 20 m between steel rails (assets/scene_crab8.xml), Rung 2 brain.
Mirror of Unity CrabShuffleEvent.

The athletes stand turned to face the course's left, so the course runs along their right (-y in the athlete frame);
lanes are 1.22 m apart along x with a 4 cm steel rail 0.3 m up on every lane line (real MuJoCo geoms).
Command every control tick (crab_command, = C# CrabShuffleEvent.Steer): vy = -VY; heading held square to the course
(wz = clip(HEADING_GAIN * -yaw)); drift towards a rail corrected with vx = clip(-X_GAIN * x_offset).
  crossing  a leg cross = the left foot passing to the right of the right foot (in the pelvis frame) — every crossing
            adds CROSS_PENALTY s
  rail      every new contact of the athlete with a rail adds RAIL_PENALTY s
  out       a fall
Rank by finish time + penalties; fallers / unfinished after them.
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

SCENE = C.ROOT / "assets" / "scene_crab8.xml"
LAYOUT = C.ROOT / "assets" / "crab8_layout.json"
DISTANCE = 20.0
VY = 1.2                # m/s side-step command = top of the Rung 2 training envelope (probe: no crossings, rail touches
                        # or falls from 0.8 to 1.8 m/s — over-commanding adds no risk, so the event stays in range)
HEADING_GAIN, WZ_LIMIT = 2.0, 0.5
X_GAIN, VX_LIMIT = 1.0, 0.3
CROSS_PENALTY = 1.0     # s per leg crossing
RAIL_PENALTY = 1.0      # s per rail touch
MAX_S = 40.0


def crab_command(quat: np.ndarray, x_offset: float, vy: float = VY) -> np.ndarray:
    yaw = C.yaw_of(quat)
    wz = float(np.clip(HEADING_GAIN * ((-yaw + math.pi) % (2 * math.pi) - math.pi), -WZ_LIMIT, WZ_LIMIT))
    vx = float(np.clip(-X_GAIN * x_offset, -VX_LIMIT, VX_LIMIT))
    return np.array([vx, -vy, wz])


def feet_crossed(d, ln: _Lane) -> bool:
    """Left foot to the right of the right foot, measured along the pelvis' left axis."""
    left_axis = d.xmat[ln.pelvis].reshape(3, 3)[:, 1]
    return float(np.dot(d.geom_xpos[ln.feet[0]] - d.geom_xpos[ln.feet[2]], left_axis)) < 0.0


@dataclass
class CrabLane:
    lane: int
    traits: Traits
    finish_s: float | None = None
    crossings: int = 0
    rail_touches: int = 0
    max_x_drift_m: float = 0.0
    score: float | None = None
    status: str = ""                 # FINISHED / FELL / DNF
    place: int = 0


@dataclass
class CrabResult:
    seed: int
    duration_s: float
    lanes: list[CrabLane] = field(default_factory=list)


def run_heat(onnx, seed: int, traits: list[Traits] | None = None, vy: float = VY, scene=None, layout_path=None, brains: dict | None = None) -> CrabResult:
    m = mujoco.MjModel.from_xml_path(str(scene or SCENE))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path or LAYOUT).read_text())
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, onnx, brains)
    mujoco.mj_forward(m, d)
    rails = {m.geom(p["name"]).id for p in layout["props"]}
    res = [CrabLane(ln.k, ln.traits) for ln in lanes]
    crossed = [False] * len(lanes)
    touching = [False] * len(lanes)
    dt = m.opt.timestep * C.DECIMATION
    tick = 0
    while tick * dt < MAX_S:
        for i, ln in enumerate(lanes):
            ra = ln.ath.root_qposadr
            if res[i].status:
                ln.control(ln.sess, d, np.zeros(3))
            else:
                ln.control(ln.sess, d, crab_command(d.qpos[ra + 3: ra + 7], d.qpos[ra] - ln.origin[0], vy))
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt
        on_rail = set()
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 in rails else g1 if g2 in rails else None
            if other is not None:
                on_rail.add(int(m.body_rootid[m.geom_bodyid[other]]))
        for i, ln in enumerate(lanes):
            r = res[i]
            if r.status:
                continue
            ra = ln.ath.root_qposadr
            r.max_x_drift_m = max(r.max_x_drift_m, abs(d.qpos[ra] - ln.origin[0]))
            if ln.eliminated(m, d) == "FELL":
                r.status = "FELL"
                continue
            now = feet_crossed(d, ln)
            if now and not crossed[i]:
                r.crossings += 1
            crossed[i] = now
            now = ln.pelvis in on_rail
            if now and not touching[i]:
                r.rail_touches += 1
            touching[i] = now
            if -(d.qpos[ra + 1] - ln.origin[1]) >= DISTANCE:
                r.status, r.finish_s = "FINISHED", t
        if all(r.status for r in res):
            break
    for r in res:
        r.status = r.status or "DNF"
        if r.status == "FINISHED":
            r.score = r.finish_s + CROSS_PENALTY * r.crossings + RAIL_PENALTY * r.rail_touches
    order = sorted(res, key=lambda r: (0, r.score) if r.status == "FINISHED" else (1, 0))
    for p, r in enumerate(order, 1):
        r.place = p
    return CrabResult(seed, tick * dt, res)


def to_json(res: CrabResult) -> dict:
    return {"seed": res.seed, "duration_s": res.duration_s,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}
