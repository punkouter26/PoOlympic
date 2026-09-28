"""Straight-track races, 8 runners, Rung 2 brain (assets/scene_track8.xml). Mirror of Unity TrackRaceEvent.

  dash      (8  The 30m Dash)          sprint 30 m from standstill; rank by finish time
  terminal  (19 Terminal Velocity)     open 84.39 m sprint (back straight) at the maximum trained command; rank by peak 1 s speed
  brake     (22 Emergency Brake)       run in at BRAKE_VX, each runner brakes (command 0) when its pelvis is BRAKE_TRIGGER
                                       metres before the red line; must stop without crossing it (toe past the line =
                                       DQ); rank by the gap left to the line
Runners steer with the contract's lane keeping (contract.steer_yaw_rate); traits as in the Iron Pedestal heat.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from .. import contract as C
from .iron_pedestal import Traits, _Lane

SCENE = C.ROOT / "assets" / "scene_track8.xml"
LAYOUT = C.ROOT / "assets" / "track8_layout.json"
MODES = {
    "dash": {"distance": 30.0, "vx": 3.8, "max_s": 25.0},
    "terminal": {"distance": 84.39, "vx": 4.0, "max_s": 45.0},   # the whole back straight (stadium venue E19)
    "brake": {"distance": 30.0, "vx": 3.0, "max_s": 25.0},
}
BRAKE_NERVE = (1.4, 2.2)   # m before the line at which a runner hits the brakes — per-runner "nerve" (seeded); the
                           # brain stops from 3 m/s with the toe ~1.6 m past its trigger point → late = DQ, early = big gap
TOE_AHEAD = 0.25      # m — the toe tip is ahead of the pelvis while standing
STOPPED = 0.1         # m/s


@dataclass
class RaceLane:
    lane: int
    traits: Traits
    finish_s: float | None = None
    peak_mps: float = 0.0
    gap_m: float | None = None       # brake: distance left between toe and line (negative = crossed)
    status: str = ""                 # FINISHED / FELL / DQ / STOPPED / DNF
    nerve_m: float | None = None     # brake: trigger distance before the line
    place: int = 0


@dataclass
class RaceResult:
    mode: str
    seed: int
    duration_s: float
    lanes: list[RaceLane] = field(default_factory=list)


def run_race(onnx: Path, mode: str, seed: int, traits: list[Traits] | None = None) -> RaceResult:
    cfg = MODES[mode]
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    layout = json.loads(LAYOUT.read_text())
    defaults = json.loads(C.CONTRACT_JSON.read_text())["default_joint_qpos"]
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = [_Lane(m, d, l["lane"], l["prefix"], np.asarray(l["origin"], float), traits[i], np.random.default_rng([seed, 1000 + i]))
             for i, l in enumerate(layout["lanes"])]
    for ln in lanes:
        ln.reset(m, d, defaults)
    mujoco.mj_forward(m, d)
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    res = [RaceLane(ln.k, ln.traits) for ln in lanes]
    nerve = [float(np.random.default_rng([seed, 2000 + i]).uniform(*BRAKE_NERVE)) for i in range(len(lanes))]
    braking = [False] * len(lanes)
    speed_hist = [[] for _ in lanes]
    dt = m.opt.timestep * C.DECIMATION
    tick = 0
    while tick * dt < cfg["max_s"]:
        t = tick * dt
        for i, ln in enumerate(lanes):
            r = res[i]
            if r.status in ("FELL", "FINISHED") and mode != "brake":
                ln.control(sess, d, np.zeros(3))
                continue
            ra = ln.ath.root_qposadr
            x = d.qpos[ra] - ln.origin[0]
            if mode == "brake" and not braking[i] and x >= cfg["distance"] - nerve[i]:
                braking[i] = True
            if braking[i] or r.status == "FELL":
                cmd = np.zeros(3)
            else:
                q = d.qpos[ra + 3: ra + 7]
                cmd = np.array([cfg["vx"], 0.0, C.steer_yaw_rate(q, d.qpos[ra + 1] - ln.origin[1], cfg["vx"])])
            ln.control(sess, d, cmd)
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt
        for i, ln in enumerate(lanes):
            r = res[i]
            if r.status in ("FELL", "FINISHED", "DQ", "STOPPED"):
                continue
            ra, da = ln.ath.root_qposadr, ln.ath.root_dofadr
            x = d.qpos[ra] - ln.origin[0]
            vx = float(d.qvel[da])
            speed_hist[i].append(vx)
            if len(speed_hist[i]) >= 50:
                r.peak_mps = max(r.peak_mps, float(np.mean(speed_hist[i][-50:])))
            if ln.eliminated(m, d) == "FELL" or (ln.eliminated(m, d) and mode != "brake"):
                r.status = "FELL"
                continue
            if mode in ("dash", "terminal") and x >= cfg["distance"]:
                r.status, r.finish_s = "FINISHED", t
            if mode == "brake":
                toe = x + TOE_AHEAD
                if toe > cfg["distance"]:
                    r.status, r.gap_m = "DQ", cfg["distance"] - toe
                elif braking[i] and abs(vx) < STOPPED and t > 1.0:
                    r.status, r.gap_m = "STOPPED", cfg["distance"] - toe
        if all(r.status for r in res):
            break
    for i, r in enumerate(res):
        r.status = r.status or "DNF"
        if mode == "brake":
            r.nerve_m = nerve[i]
    # ranking
    if mode == "dash":
        key = lambda r: (0, r.finish_s) if r.status == "FINISHED" else (1, 0)
    elif mode == "terminal":
        key = lambda r: (0 if r.status != "FELL" else 1, -r.peak_mps)
    else:
        key = lambda r: (0, r.gap_m) if r.status == "STOPPED" else (1, 0)
    order = sorted(res, key=key)
    for p, r in enumerate(order, 1):
        r.place = p
    return RaceResult(mode, seed, tick * dt, sorted(res, key=lambda r: r.lane))


def to_json(res: RaceResult) -> dict:
    return {"mode": res.mode, "seed": res.seed, "duration_s": res.duration_s,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}
