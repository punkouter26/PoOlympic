"""Straight-track races, 8 runners, Rung 2 brain (assets/scene_track8.xml). Mirror of Unity TrackRaceEvent.

  dash      (8  The 30m Dash)          sprint 30 m from standstill; rank by finish time
  terminal  (19 Terminal Velocity)     open 84.39 m sprint (back straight) at the maximum trained command; rank by peak 1 s speed.
                                       Lane break (crowd rule, 2026-09-30): past BREAK_X the field squeezes to the
                                       inside (lane 1 = the runner's left): every runner's line moves to SQUEEZE x its
                                       distance from the inside lane (1.22 -> 0.61 m apart: shoulder to shoulder),
                                       merging at most MERGE_DEG off straight (break_target)
  brake     (22 Emergency Brake)       run in at BRAKE_VX, each runner brakes (command 0) when its pelvis is BRAKE_TRIGGER
                                       metres before the red line; must stop without crossing it (toe past the line =
                                       DQ); rank by the gap left to the line
  inverted  (9  The Inverted Sprint)   20 m backwards (runners face away from the finish: the finish is at x = -20 in the
                                       runner's frame; Unity turns the stadium 180 deg); leaving the lane (|y| > half a
                                       lane) = DQ, a fall = out; rank by finish time
  steeple   (13 Steeplechase Jog)      50 m at 3.5 m/s with the flight brain (MATT: r2f_v3_it100); scored on GROUND
                                       TIME = finish time - hang time (total of the flights, both feet off the ground
                                       for >= MIN_FLIGHT): the clock only runs while a foot touches the track; a fall =
                                       out; rank by ground time, then finish time
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
from .iron_pedestal import Traits, _Lane, make_lanes

SCENE = C.ROOT / "assets" / "scene_track8.xml"
LAYOUT = C.ROOT / "assets" / "track8_layout.json"
MODES = {
    "dash": {"distance": 30.0, "vx": 3.8, "max_s": 25.0},
    "terminal": {"distance": 84.39, "vx": 4.0, "max_s": 45.0},   # the whole back straight (stadium venue E19)
    "brake": {"distance": 30.0, "vx": 3.0, "max_s": 25.0},
    "inverted": {"distance": 20.0, "vx": -1.5, "max_s": 30.0},   # vx = the Rung 2 envelope's backward limit
    "steeple": {"distance": 50.0, "vx": 3.5, "max_s": 30.0},
}
BREAK_X = 15.0        # m — terminal: the green lane-break line
MERGE_DEG = 12.0      # terminal: largest heading off straight while merging (lateral ~0.8 m/s at 4 m/s)
SQUEEZE = 0.5         # terminal: lines after the break = inside + SQUEEZE x (lane line - inside). Tuned on CPU: one
                      # shared inside line gave rear-end pile-ups (2-6 of 8 down, slower zombies run over); 0.5 = steady
                      # shoulder contact, 1-3 MATTs down; 0.45 = 2-5 down
MIN_FLIGHT = 0.02     # s — steeple: both feet off the ground at least this long = a flight (contact chatter ignored)
LANE_HALF = 0.61      # m — inverted sprint: pelvis further than this from the lane centre line = DQ (lane drift)
BRAKE_NERVE = (1.4, 2.2)   # m before the line at which a runner hits the brakes — per-runner "nerve" (seeded); the
                           # brain stops from 3 m/s with the toe ~1.6 m past its trigger point → late = DQ, early = big gap
TOE_AHEAD = 0.25      # m — the toe tip is ahead of the pelvis while standing
STOPPED = 0.1         # m/s


def break_target(lane_y: float, inside_y: float) -> float:
    """Terminal lane break: the line a runner steers for after BREAK_X (race frame y, +y = inside)."""
    return inside_y - SQUEEZE * (inside_y - lane_y)


def merge_offset(y: float, target: float) -> float:
    """Lane offset fed to the contract steering law, capped so the merge heading stays within MERGE_DEG."""
    cap = math.tan(math.radians(MERGE_DEG)) / C.LANE_GAIN
    return float(np.clip(y - target, -cap, cap))


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
    body: str = "matt"
    flights: int = 0                 # steeple: flights >= MIN_FLIGHT
    hang_s: float = 0.0              # steeple: total airborne time (flights >= MIN_FLIGHT)
    longest_ms: float = 0.0          # steeple: longest flight
    score_s: float | None = None     # steeple: ground time = finish time - hang time


class FootGait:
    """Steeple: per-substep foot contact of one lane (foot + toe geoms of both legs vs the ground) → flights (both feet
    off for >= MIN_FLIGHT): count, total hang time, longest. Mirror of Unity TrackRaceEvent.FootGait."""

    def __init__(self, m, prefix: str, dt: float):
        self.feet = {m.geom(prefix + n).id for n in ("foot_l_geom0", "toe_l_geom0", "foot_r_geom0", "toe_r_geom0")}
        self.ground = m.geom("ground").id
        self.dt = dt
        self.air = 0                 # substeps airborne in the current flight
        self.flights = 0
        self.hang = self.longest = 0.0

    def step(self, contacts) -> None:
        on = any((g1 == self.ground and g2 in self.feet) or (g2 == self.ground and g1 in self.feet) for g1, g2 in contacts)
        if not on:
            self.air += 1
            return
        if self.air >= round(MIN_FLIGHT / self.dt):
            self.flights += 1
            self.hang += self.air * self.dt
            self.longest = max(self.longest, self.air * self.dt)
        self.air = 0


@dataclass
class RaceResult:
    mode: str
    seed: int
    duration_s: float
    lanes: list[RaceLane] = field(default_factory=list)


def run_race(onnx: Path, mode: str, seed: int, traits: list[Traits] | None = None, scene=None, layout_path=None, brains: dict | None = None) -> RaceResult:
    cfg = MODES[mode]
    m = mujoco.MjModel.from_xml_path(str(scene or SCENE))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path or LAYOUT).read_text())
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, onnx, brains)
    mujoco.mj_forward(m, d)
    res = [RaceLane(ln.k, ln.traits, body=ln.body) for ln in lanes]
    gait = [FootGait(m, ln.prefix, m.opt.timestep) for ln in lanes] if mode == "steeple" else None
    nerve = [float(np.random.default_rng([seed, 2000 + i]).uniform(*BRAKE_NERVE)) for i in range(len(lanes))]
    braking = [False] * len(lanes)
    speed_hist = [[] for _ in lanes]
    dt = m.opt.timestep * C.DECIMATION
    tick = 0
    inside_y = max(float(ln.origin[1]) for ln in lanes)        # lane 1 (the runner's left) = the inside
    while tick * dt < cfg["max_s"]:
        t = tick * dt
        for i, ln in enumerate(lanes):
            r = res[i]
            if r.status and mode != "brake":
                ln.control(ln.sess, d, np.zeros(3))
                continue
            ra = ln.ath.root_qposadr
            x = d.qpos[ra] - ln.origin[0]
            if mode == "brake" and not braking[i] and x >= cfg["distance"] - nerve[i]:
                braking[i] = True
            if braking[i] or r.status == "FELL":
                cmd = np.zeros(3)
            else:
                q = d.qpos[ra + 3: ra + 7]
                off = d.qpos[ra + 1] - ln.origin[1]
                if mode == "terminal" and x >= BREAK_X:
                    off = merge_offset(d.qpos[ra + 1], break_target(float(ln.origin[1]), inside_y))
                cmd = np.array([cfg["vx"], 0.0, C.steer_yaw_rate(q, off, cfg["vx"])])
            ln.control(ln.sess, d, cmd)
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
            if gait is not None:
                pairs = [(int(c.geom1), int(c.geom2)) for c in d.contact[:d.ncon]]
                for i, g in enumerate(gait):
                    if not res[i].status:
                        g.step(pairs)
        tick += 1
        t = tick * dt
        for i, ln in enumerate(lanes):
            r = res[i]
            if r.status in ("FELL", "FINISHED", "DQ", "STOPPED"):
                continue
            ra, da = ln.ath.root_qposadr, ln.ath.root_dofadr
            sign = -1.0 if cfg["vx"] < 0 else 1.0
            x = sign * (d.qpos[ra] - ln.origin[0])       # progress along the race direction
            vx = sign * float(d.qvel[da])
            speed_hist[i].append(vx)
            if len(speed_hist[i]) >= 50:
                r.peak_mps = max(r.peak_mps, float(np.mean(speed_hist[i][-50:])))
            if ln.eliminated(m, d) == "FELL" or (ln.eliminated(m, d) and mode != "brake"):
                r.status = "FELL"
                continue
            if mode == "inverted" and abs(d.qpos[ra + 1] - ln.origin[1]) > LANE_HALF:
                r.status = "DQ"
                continue
            if mode in ("dash", "terminal", "inverted", "steeple") and x >= cfg["distance"]:
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
        if gait is not None:
            g = gait[i]
            r.flights, r.hang_s, r.longest_ms = g.flights, round(g.hang, 3), round(1000 * g.longest, 1)
            if r.status == "FINISHED":
                r.score_s = round(r.finish_s - g.hang, 3)
    # ranking
    if mode in ("dash", "inverted"):
        key = lambda r: (0, r.finish_s) if r.status == "FINISHED" else (1, 0)
    elif mode == "steeple":
        key = lambda r: (0, r.score_s, r.finish_s) if r.status == "FINISHED" else (1, 0)
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
