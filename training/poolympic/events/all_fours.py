"""Event 8 — 30m All Fours (replaces The 30m Dash; user request 2026-09-29: race on all fours, falls do not eliminate,
only natural falls, MATT + zombie). 8 runners on the home straight (assets/scene_track8*.xml), crawl brains per body
(tasks/crawl_env.py). Mirror of Unity AllFoursRace.

  start     every runner lies face down on its lane line, head towards the finish (pelvis at 0.22 m × λ)
  setup     SETUP_S of zero command: get onto all fours
  race      GO: crawl command VX (MATT units; other bodies × √λ in float32 like PolicyRunner.BodyCommand) with lane
            keeping on the crawl heading (tools/crawl_probe.crawl_steer: the contract law on mdp.crawl_heading)
  finish    pelvis DISTANCE past the start; rank by time
  falls     a tumble (pelvis drops below the crawl band) is counted and costs time, never eliminates
  stand-up  pelvis above the band with the torso within 40° of vertical for more than STAND_DQ_S = DQ (all fours!)
Traits as in every event (strength, latency, sensor noise).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .. import bodies
from .. import contract as C
from .iron_pedestal import Traits, make_lanes

SCENE = C.ROOT / "assets" / "scene_track8.xml"
LAYOUT = C.ROOT / "assets" / "track8_layout.json"
DISTANCE, VX, SETUP_S, MAX_S = 30.0, 1.2, 3.0, 60.0
STAND_DQ_S = 1.0
BAND = (0.25, 0.8)          # crawl pelvis band (m) × λ
START_Z = 0.22              # prone start pelvis height (m) × λ


@dataclass
class CrawlLane:
    lane: int
    body: str
    traits: Traits
    finish_s: float | None = None
    status: str = ""        # FINISHED / DQ / DNF
    tumbles: int = 0
    place: int = 0


@dataclass
class CrawlResult:
    seed: int
    duration_s: float
    lanes: list[CrawlLane] = field(default_factory=list)


def crawl_yaw(q: np.ndarray) -> float:
    """Heading of mdp.crawl_heading (horizontal projection of pelvis x + z axes): valid on all fours."""
    w, x, y, z = q
    hx = (1 - 2 * (y * y + z * z)) + 2 * (x * z + w * y)
    hy = 2 * (x * y + w * z) + 2 * (y * z - w * x)
    return math.atan2(hy, hx)


def crawl_steer(q: np.ndarray, lane_offset_y: float) -> float:
    """The contract lane-keeping law on the crawl heading."""
    target = math.atan(-C.LANE_GAIN * lane_offset_y)
    err = (target - crawl_yaw(q) + math.pi) % (2 * math.pi) - math.pi
    return float(np.clip(C.HEADING_GAIN * err, -C.STEER_WZ_LIMIT, C.STEER_WZ_LIMIT))


def body_command(cmd: np.ndarray, body: str) -> np.ndarray:
    """PolicyRunner.BodyCommand in float32: k = (float)(0.8 / gait_hz_base); (x·k, y·k, z / k). Identity for MATT."""
    if body == "matt":
        return cmd
    ct = json.loads(bodies.BODIES[body].contract_json.read_text())
    k = np.float32(0.8 / ct["gait_hz_base"])
    c = np.asarray(cmd, np.float32)
    return np.array([c[0] * k, c[1] * k, c[2] / k], dtype=np.float64)


def prone_start(m, d, ln) -> None:
    """Face down on the lane line, head towards +x (the finish): pitch +90° about the lane's y axis."""
    lam = bodies.BODIES[ln.body].length_scale
    r, dv = ln.ath.root_qposadr, ln.ath.root_dofadr
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0.0, math.pi / 2, 0.0]), "XYZ")
    d.qpos[r:r + 3] = [ln.origin[0], ln.origin[1], START_Z * lam]
    d.qpos[r + 3:r + 7] = q
    d.qvel[dv:dv + 6] = 0.0


def run_race(brains: dict[str, Path], seed: int, traits: list[Traits] | None = None, scene: Path = SCENE,
             layout_path: Path = LAYOUT) -> CrawlResult:
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path).read_text())
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, brains.get("matt"), brains)
    for ln in lanes:
        prone_start(m, d, ln)
    mujoco.mj_forward(m, d)
    res = [CrawlLane(ln.k, ln.body, ln.traits) for ln in lanes]
    torso = [m.body(ln.prefix + "torso").id for ln in lanes]
    lam = [bodies.BODIES[ln.body].length_scale for ln in lanes]
    was_up = [False] * len(lanes)
    stand_t = [0.0] * len(lanes)
    dt = m.opt.timestep * C.DECIMATION
    tick = 0
    while tick * dt < SETUP_S + MAX_S:
        t = tick * dt
        for i, ln in enumerate(lanes):
            r = ln.ath.root_qposadr
            if t < SETUP_S or res[i].status:
                cmd = np.zeros(3)
            else:
                cmd = np.array([VX, 0.0, crawl_steer(d.qpos[r + 3:r + 7], d.qpos[r + 1] - ln.origin[1])])
            ln.control(ln.sess, d, body_command(cmd, ln.body))
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt
        if t < SETUP_S:
            continue
        for i, ln in enumerate(lanes):
            rr = res[i]
            if rr.status:
                continue
            r = ln.ath.root_qposadr
            z = d.qpos[r + 2]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, d.xmat[torso[i]][8]))))
            on4 = BAND[0] * lam[i] < z < BAND[1] * lam[i] and tilt > 50.0
            if was_up[i] and z < BAND[0] * lam[i]:
                rr.tumbles += 1
            was_up[i] = on4
            stand_t[i] = stand_t[i] + dt if (z > BAND[1] * lam[i] and tilt < 40.0) else 0.0
            if stand_t[i] > STAND_DQ_S:
                rr.status = "DQ"
                continue
            if d.qpos[r] - ln.origin[0] >= DISTANCE:
                rr.status, rr.finish_s = "FINISHED", round(t - SETUP_S, 2)
        if all(r.status for r in res):
            break
    for r in res:
        r.status = r.status or "DNF"
    order = sorted(res, key=lambda r: (0, r.finish_s) if r.status == "FINISHED" else (1 if r.status == "DNF" else 2, 0))
    for p, r in enumerate(order, 1):
        r.place = p
    return CrawlResult(seed, tick * dt, res)


def to_json(res: CrawlResult) -> dict:
    return {"seed": res.seed, "duration_s": res.duration_s,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}
