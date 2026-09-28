"""11 Slalom Sprint — 8 runners weave through 7 poles on their lane's centre line (assets/scene_slalom8.xml), Rung 2
brain. Mirror of Unity SlalomEvent.

Course (runner frame, lane origin = start): poles at x = POLE_X0 + g * POLE_DX (g = 0..N_POLES-1) on y = 0, finish at
DISTANCE. Pole g must be passed on the left (y > 0) for even g, on the right for odd g.
Racing line = y*(x) = A cos(pi (x - POLE_X0) / POLE_DX) between half a pole gap before the first pole and after the
last one (0 outside): A is the runner's seeded "line" (how wide it swings round the poles, LINE range).
Command every control tick (slalom_command, = C# SlalomEvent.Steer): vx = VX; heading towards the line —
    psi* = atan(y*'(x + LOOKAHEAD)) + atan(-Y_GAIN (y - y*(x)));  wz = clip(HEADING_GAIN wrap(psi* - yaw), +-WZ_LIMIT)
Penalties: a pole passed on the wrong side = MISS_PENALTY s; every new contact with a pole = CLIP_PENALTY s; a fall =
out. Rank by time + penalties.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np
import onnxruntime as ort

from .. import contract as C
from .iron_pedestal import Traits, _Lane

SCENE = C.ROOT / "assets" / "scene_slalom8.xml"
LAYOUT = C.ROOT / "assets" / "slalom8_layout.json"
DISTANCE = 32.0
POLE_X0, POLE_DX, N_POLES = 3.0, 4.0, 7      # = venues.json "poles" (checked against the layout in run_heat)
# Tuned on CPU (see rl_optimization_log.md): at 2.5-3 m/s the alternating yaw commands topple the Rung 2 brain even
# without pole contact; 2.2 m/s with the training lateral-acceleration cap finishes 16/16.
VX = 2.2
LINE = (0.40, 0.56)       # m — racing-line amplitude range (seeded per runner). At VX: 0.40 ~5 clips, 0.45 ~2-3,
                          # 0.50 ~1, 0.55 no clips but ~1 in 2 falls — the runner's line is its risk
LOOKAHEAD = 0.8           # m
Y_GAIN, HEADING_GAIN = 1.0, 2.0
WZ_LIMIT = 4.0 / VX       # rad/s — v * wz <= 4 m/s^2, the Rung 2 training cap (AthleteCommandCfg.max_lateral_accel)
MISS_PENALTY = 2.0        # s per pole on the wrong side
CLIP_PENALTY = 0.5        # s per pole contact
MAX_S = 30.0


def line_y(x: float, a: float) -> tuple[float, float]:
    """Racing line y*(x) and its slope."""
    lo, hi = POLE_X0 - POLE_DX / 2, POLE_X0 + (N_POLES - 1) * POLE_DX + POLE_DX / 2
    if x < lo or x > hi:
        return 0.0, 0.0
    w = math.pi / POLE_DX
    return a * math.cos(w * (x - POLE_X0)), -a * w * math.sin(w * (x - POLE_X0))


def slalom_command(quat: np.ndarray, x: float, y: float, a: float, vx: float = VX) -> np.ndarray:
    yaw = C.yaw_of(quat)
    ys, _ = line_y(x, a)
    _, slope = line_y(x + LOOKAHEAD, a)
    target = math.atan(slope) + math.atan(-Y_GAIN * (y - ys))
    err = (target - yaw + math.pi) % (2 * math.pi) - math.pi
    return np.array([vx, 0.0, float(np.clip(HEADING_GAIN * err, -WZ_LIMIT, WZ_LIMIT))])


def runner_line(seed: int, i: int) -> float:
    return float(np.random.default_rng([seed, 4000 + i]).uniform(*LINE))


@dataclass
class SlalomLane:
    lane: int
    traits: Traits
    line_m: float = 0.0
    finish_s: float | None = None
    misses: int = 0
    clips: int = 0
    score: float | None = None
    status: str = ""                 # FINISHED / FELL / DNF
    place: int = 0


@dataclass
class SlalomResult:
    seed: int
    duration_s: float
    lanes: list[SlalomLane] = field(default_factory=list)


def run_heat(onnx, seed: int, traits: list[Traits] | None = None, lines: list[float] | None = None, vx: float = VX) -> SlalomResult:
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    layout = json.loads(LAYOUT.read_text())
    defaults = json.loads(C.CONTRACT_JSON.read_text())["default_joint_qpos"]
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = [_Lane(m, d, l["lane"], l["prefix"], np.asarray(l["origin"], float), traits[i], np.random.default_rng([seed, 1000 + i]))
             for i, l in enumerate(layout["lanes"])]
    poles = [{m.geom(f"pole{ln.k}_{g}").id for g in range(N_POLES)} for ln in lanes]
    for ln in lanes:        # the course constants must match the physical poles
        for g in range(N_POLES):
            p = np.asarray(m.geom_pos[m.geom(f"pole{ln.k}_{g}").id]) - ln.origin
            assert abs(p[0] - (POLE_X0 + g * POLE_DX)) < 1e-6 and abs(p[1]) < 1e-6, (ln.k, g, p)
    for ln in lanes:
        ln.reset(m, d, defaults)
    mujoco.mj_forward(m, d)
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    lines = lines or [runner_line(seed, i) for i in range(len(lanes))]
    res = [SlalomLane(ln.k, ln.traits, lines[i]) for i, ln in enumerate(lanes)]
    next_pole = [0] * len(lanes)
    touching = [False] * len(lanes)
    dt = m.opt.timestep * C.DECIMATION
    tick = 0
    while tick * dt < MAX_S:
        for i, ln in enumerate(lanes):
            ra = ln.ath.root_qposadr
            if res[i].status:
                ln.control(sess, d, np.zeros(3))
            else:
                ln.control(sess, d, slalom_command(d.qpos[ra + 3: ra + 7], d.qpos[ra] - ln.origin[0],
                                                   d.qpos[ra + 1] - ln.origin[1], lines[i], vx))
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt
        clipped = set()
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            for i in range(len(lanes)):
                if g1 in poles[i] or g2 in poles[i]:
                    other = g2 if g1 in poles[i] else g1
                    if m.body_rootid[m.geom_bodyid[other]] == lanes[i].pelvis:
                        clipped.add(i)
        for i, ln in enumerate(lanes):
            r = res[i]
            if r.status:
                continue
            ra = ln.ath.root_qposadr
            x, y = d.qpos[ra] - ln.origin[0], d.qpos[ra + 1] - ln.origin[1]
            if ln.eliminated(m, d) == "FELL":
                r.status = "FELL"
                continue
            now = i in clipped
            if now and not touching[i]:
                r.clips += 1
            touching[i] = now
            while next_pole[i] < N_POLES and x >= POLE_X0 + next_pole[i] * POLE_DX:
                if (y > 0) != (next_pole[i] % 2 == 0):
                    r.misses += 1
                next_pole[i] += 1
            if x >= DISTANCE:
                r.status, r.finish_s = "FINISHED", t
        if all(r.status for r in res):
            break
    for r in res:
        r.status = r.status or "DNF"
        if r.status == "FINISHED":
            r.score = r.finish_s + MISS_PENALTY * r.misses + CLIP_PENALTY * r.clips
    order = sorted(res, key=lambda r: (0, r.score) if r.status == "FINISHED" else (1, 0))
    for p, r in enumerate(order, 1):
        r.place = p
    return SlalomResult(seed, tick * dt, res)


def to_json(res: SlalomResult) -> dict:
    return {"seed": res.seed, "duration_s": res.duration_s,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}
