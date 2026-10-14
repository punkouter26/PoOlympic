"""Straight-track races, 8 runners, Rung 2 brain (assets/scene_track8.xml). Mirror of Unity TrackRaceEvent.

  dash      (8  The 30m Dash)          sprint 30 m from standstill; rank by finish time
  terminal  (19 Terminal Velocity)     open 84.39 m sprint (back straight) at the maximum trained command; rank by peak 1 s speed.
                                       Lane break (crowd rule, 2026-09-30; pack rule 2026-10-12): past BREAK_X the
                                       field squeezes to the inside (lane 1 = the runner's left): every runner lines up
                                       shoulder to shoulder with the runners alongside it (pack_target; two MATTs
                                       0.61 m apart), merging at most MERGE_DEG off straight
  brake     (22 Emergency Brake)       run in at BRAKE_VX, each runner brakes (command 0) when its pelvis is BRAKE_TRIGGER
                                       metres before the red line; must stop without crossing it (toe past the line =
                                       DQ); rank by the gap left to the line
  inverted  (9  The Inverted Sprint)   20 m backwards (runners face away from the finish: the finish is at x = -20 in the
                                       runner's frame; Unity turns the stadium 180 deg); leaving the lane (|y| > half a
                                       lane) = DQ, a fall = out; rank by finish time
  resurrection (27 The Resurrection Dash) start flat on the back (feet towards the finish), rise and sprint 5 m; the
                                       get-up brain drives from the gun (`recover`: a body without one stays down = DNF);
                                       falling again = down again, not out; rank by finish time
  steeple   (13 Steeplechase Jog)      50 m at 3.5 m/s with the flight brain (MATT: r2f_v3_it100); scored on GROUND
                                       TIME = finish time - hang time (total of the flights, both feet off the ground
                                       for >= MIN_FLIGHT): the clock only runs while a foot touches the track; a fall =
                                       out; rank by ground time, then finish time
  ramp      (14 The Alpine Ramp)       10 m up a 15 deg ramp with the slope brain (Rung4-Slope15); the ramp is the flat
                                       track with gravity tilted by 15 deg (x = distance along the ramp; Unity draws the
                                       scene tilted); a fall = out; rank by finish time
Runners steer with the contract's lane keeping (contract.steer_yaw_rate); traits as in the Iron Pedestal heat.

Fall recovery (opt-in, `recover` = body -> get-up brain; Unity TrackRaceEvent.recoverFromFalls): a runner whose body has
a get-up brain is not out when it falls. It first lies still in the default joint pose for RELAX_S (the tumble ends
and the body is in the pose the get-up brain was trained from), then the get-up brain runs (zero command) until the
pelvis is back above UP_Z x its standing height with the torso within UP_TILT_DEG for UP_HOLD_S, then the race brain
runs on. Still down RETRY_S after lying still: lie still and try again (the brain can lock into a propped pose), each
time RELAX_STEP_S longer (a retry from the same pose repeats the same failure), up to RELAX_MAX_STEPS steps.
Falls are counted (RaceLane.falls); a runner cannot finish while down. Not in the brake event. Shared with the slalom.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from .. import action_log as AL
from .. import contract as C
from .iron_pedestal import FALL_Z, Traits, _Lane, make_lanes

SCENE = C.ROOT / "assets" / "scene_track8.xml"
LAYOUT = C.ROOT / "assets" / "track8_layout.json"
MODES = {
    "dash": {"distance": 30.0, "vx": 3.8, "max_s": 25.0},
    "terminal": {"distance": 84.39, "vx": 4.0, "max_s": 45.0},   # the whole back straight (stadium venue E19)
    "brake": {"distance": 30.0, "vx": 3.0, "max_s": 25.0},
    "inverted": {"distance": 20.0, "vx": -1.5, "max_s": 30.0},   # vx = the Rung 2 envelope's backward limit
    "steeple": {"distance": 50.0, "vx": 3.5, "max_s": 30.0},
    "resurrection": {"distance": 5.0, "vx": 3.8, "max_s": 20.0},
    # the stadium's ramp (venue E14) is 10 m of run = 10.35 m of slope; vx = the slope brains' top command (half the flat envelope)
    "ramp": {"distance": 10.0, "vx": 2.0, "max_s": 30.0, "slope_deg": 15.0},
}
SUPINE_Z = 0.22       # m x the body's height scale — resurrection: pelvis height lying on the back (tools/getup_probe.py)
BREAK_X = 15.0        # m — terminal: the green lane-break line
MERGE_DEG = 12.0      # terminal: largest heading off straight while merging (lateral ~0.8 m/s at 4 m/s)
MIN_FLIGHT = 0.02     # s — steeple: both feet off the ground at least this long = a flight (contact chatter ignored)
LANE_HALF = 0.61      # m — inverted sprint: pelvis further than this from the lane centre line = DQ (lane drift)
BRAKE_NERVE = (1.4, 2.2)   # m before the line at which a runner hits the brakes — per-runner "nerve" (seeded); the
                           # brain stops from 3 m/s with the toe ~1.6 m past its trigger point → late = DQ, early = big gap
TOE_AHEAD = 0.25      # m — the toe tip is ahead of the pelvis while standing
STOPPED = 0.1         # m/s
UP_Z, UP_TILT_DEG, UP_HOLD_S = 0.85 / 0.55, 20.0, 0.5   # recovery: "up" = pelvis above UP_Z x the fall line (tools/getup_probe.py)
RELAX_S, RETRY_S = 1.0, 5.0                              # recovery: lie still, then the get-up brain for RETRY_S - RELAX_S
RELAX_STEP_S, RELAX_MAX_STEPS = 0.5, 4                   # recovery: every retry lies still this much longer


class Recovery:
    """Fall recovery of the lanes whose body has a get-up brain (module docstring). Mirror of Unity PolicyRunner
    (SetRecovering / relax) + the events' "up" check. Per control tick: control(i) drives a downed lane and returns
    True; after the physics: fall(i) when the judge says FELL (False = no get-up brain, the runner is out), and
    still_down(i) for a downed lane."""

    def __init__(self, lanes, recover: dict | None, dt: float):
        self.lanes, self.dt = lanes, dt
        self.sess = {b: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for b, p in (recover or {}).items()}
        n = len(lanes)
        self.down, self.up_ticks, self.cycle_tick, self.attempt = [False] * n, [0] * n, [0] * n, [0] * n
        self.relax, self.retry, self.step = round(RELAX_S / dt), round(RETRY_S / dt), round(RELAX_STEP_S / dt)

    def can(self, i: int) -> bool:
        return self.lanes[i].body in self.sess

    def relax_ticks(self, i: int) -> int:
        return self.relax + self.step * min(self.attempt[i], RELAX_MAX_STEPS)

    def fall(self, i: int, skip_relax: bool = False) -> bool:
        if not self.can(i):
            return False
        self.down[i], self.up_ticks[i], self.attempt[i] = True, 0, 0
        self.cycle_tick[i] = self.relax if skip_relax else 0
        return True

    def control(self, i: int, d) -> bool:
        if not self.down[i]:
            return False
        ln = self.lanes[i]
        if self.cycle_tick[i] < self.relax_ticks(i):          # lie still = Unity PolicyRunner relax
            ln.ctrl_prev, ln.ctrl_now = ln.ctrl_now, ln.ath.default_pos.copy()
            ln.last[:] = 0.0
        else:
            ln.control(self.sess[ln.body], d, np.zeros(3))
        self.cycle_tick[i] += 1
        if self.cycle_tick[i] >= self.relax_ticks(i) + self.retry - self.relax:
            self.cycle_tick[i], self.attempt[i] = 0, self.attempt[i] + 1
        return True

    def still_down(self, i: int, d) -> bool:
        ln = self.lanes[i]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, d.xmat[ln.torso][8]))))
        up = d.qpos[ln.ath.root_qposadr + 2] > UP_Z * ln.fall_z and tilt < UP_TILT_DEG
        self.up_ticks[i] = self.up_ticks[i] + 1 if up else 0
        self.down[i] = self.up_ticks[i] * self.dt < UP_HOLD_S
        return self.down[i]


PACK = 0.61 / 0.64    # terminal, pack rule: lines of two runners side by side = PACK x the sum of their half-widths apart
                      # (two MATTs, 0.32 m each: 0.61 m = half a 1.22 m lane. Tuned on CPU 2026-09-30 as fixed
                      # lines: one shared inside line gave rear-end pile-ups, 2-6 of 8 down; 0.61 m = steady
                      # shoulder contact, 1-3 MATTs down; 0.55 m = 2-5 down)
PACK_X = 1.5          # m — "side by side": within this distance along the track


def half_width(m, d, ln) -> float:
    """Half the width of a standing athlete: its widest collider (the forearms) from the pelvis line, at the start pose."""
    y0 = d.qpos[ln.ath.root_qposadr + 1]
    return max(abs(d.geom_xpos[g][1] - y0) + (m.geom_size[g][1] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else m.geom_size[g][0])
               for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == ln.pelvis)


def pack_target(i: int, xs: list[float], racing: list[bool], hw: list[float], inside_y: float) -> float:
    """Terminal lane break, pack rule (2026-10-12; = C# TrackRaceEvent.PackTarget): the field closes up on the inside
    line shoulder to shoulder — but only with the runners actually alongside. Runner i lines up outside every racing
    runner of a lower lane (lane 0 = the inside) within PACK_X of it along the track, PACK x their half-widths apart.
    A level field of 8 MATTs gets lines 0.61 m apart (the rule before: every line at half its distance to lane 1); a
    mixed field, which splits up by speed (MATT 4.0, KIM 3.7, zombie 2.9, GRANDMA 2.8 m/s) and ran 2.4 m apart under
    that rule, runs in shoulder-to-shoulder groups."""
    y, prev = inside_y, None
    for j in range(i + 1):
        if j == i or (racing[j] and abs(xs[j] - xs[i]) < PACK_X):
            if prev is not None:
                y -= PACK * (hw[prev] + hw[j])
            prev = j
    return y


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
    falls: int = 0                   # recovery: falls the runner got up from (or is still down from)


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


@AL.logged("event_track")
def run_race(onnx: Path, mode: str, seed: int, traits: list[Traits] | None = None, scene=None, layout_path=None, brains: dict | None = None,
             recover: dict | None = None) -> RaceResult:
    cfg = MODES[mode]
    m = mujoco.MjModel.from_xml_path(str(scene or SCENE))
    if "slope_deg" in cfg:       # the ramp = the flat track with gravity tilted by the slope (tasks/matt_env.add_slope), uphill = +x
        a = math.radians(cfg["slope_deg"])
        m.opt.gravity[:] = (-9.81 * math.sin(a), 0.0, -9.81 * math.cos(a))
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
    rec = Recovery(lanes, None if mode == "brake" else recover, dt)
    if mode == "resurrection":
        for i, ln in enumerate(lanes):      # on the back, head away from the finish: pitch -90 deg (PolicyRunner.startSupine)
            ra = ln.ath.root_qposadr
            d.qpos[ra + 2] = SUPINE_Z * ln.fall_z / FALL_Z
            d.qpos[ra + 3: ra + 7] = [math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0]
            rec.fall(i, skip_relax=True)
        mujoco.mj_forward(m, d)
    tick = 0
    inside_y = max(float(ln.origin[1]) for ln in lanes)        # lane 1 (the runner's left) = the inside
    assert [ln.k for ln in lanes] == sorted(ln.k for ln in lanes) and inside_y == float(lanes[0].origin[1])
    hw = [half_width(m, d, ln) for ln in lanes]
    while tick * dt < cfg["max_s"]:
        t = tick * dt
        xs = [float(d.qpos[ln.ath.root_qposadr] - ln.origin[0]) for ln in lanes]       # positions at the tick start
        alongside = [not res[i].status and not rec.down[i] for i in range(len(lanes))]
        for i, ln in enumerate(lanes):
            r = res[i]
            if rec.control(i, d):
                continue
            if mode == "resurrection" and not rec.can(i) and not r.status:     # no get-up brain: lies there
                ln.ctrl_prev, ln.ctrl_now = ln.ctrl_now, ln.ath.default_pos.copy()
                continue
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
                    target = pack_target(i, xs, alongside, hw, inside_y)
                    off = merge_offset(d.qpos[ra + 1], target)
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
            if rec.down[i]:
                rec.still_down(i, d)
                continue
            if mode == "resurrection" and not rec.can(i):
                continue
            if ln.eliminated(m, d) == "FELL" or (ln.eliminated(m, d) and mode != "brake"):
                if rec.fall(i):
                    r.falls += 1
                    continue
                r.status = "FELL"
                continue
            if mode == "inverted" and abs(d.qpos[ra + 1] - ln.origin[1]) > LANE_HALF:
                r.status = "DQ"
                continue
            if mode in ("dash", "terminal", "inverted", "steeple", "resurrection", "ramp") and x >= cfg["distance"]:
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
    if mode in ("dash", "inverted", "resurrection", "ramp"):
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
