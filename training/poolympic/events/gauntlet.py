"""5 The Gust Gauntlet — 8 athletes on spring-mounted shaker platforms (assets/scene_shaker8.xml), Rung 2 brain.
Mirror of Unity GustGauntletEvent. Crowd stage (2026-09-30): all 8 share ONE shaker floor (joints shaker_x/_y) in a
tight 2 x 4 grid — the wind bursts push the two rows into each other, a floor jolt throws everyone at once (one seeded
direction per round, FLOOR_SEED); scenes with a platform per lane (L<k>_shaker) keep their own per-lane jolts.

Every athlete homes on its spot (the platform's rest centre = lane origin): each control tick (homing_command, = C#
GustGauntletEvent.Steer) the offset in the pelvis frame becomes a walk command back, vxy = clip(-HOME_GAIN * offset,
+-HOME_V) outside a HOME_DEADBAND, and the heading is held (wz = clip(-2 yaw, +-0.5)).
Rounds every ROUND_S from START_S: a lateral wind burst for everyone — root velocity kick of the same magnitude
(GUST_START + GUST_STEP * round), each athlete to its own seeded side (+-90 deg +- 30 deg from its facing) — and on every
SHAKE_EVERY-th round also a floor shake (platform velocity kick SHAKE_V in a seeded direction).
  recovery  per round: time from the burst until the pelvis is within CALM_R of the spot and slower than CALM_V for
            CALM_S; not recovered by the next burst = ROUND_S
  out       a fall, or a foot off the platform (STEPPED OFF) — every remaining round then counts ROUND_S
Rank: athletes still in by total recovery time, then the eliminated (later = better).
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

SCENE = C.ROOT / "assets" / "scene_shaker8.xml"
LAYOUT = C.ROOT / "assets" / "shaker8_layout.json"
START_S = 1.0
ROUND_S = 4.0
N_ROUNDS = 10
GUST_START, GUST_STEP = 0.5, 0.05       # m/s — 0.5 recovers in ~1.2-2 s, 0.9 pushes ~1 in 4 off the 1.6 m platform
GUST_SPREAD = math.radians(30)
SHAKE_EVERY, SHAKE_V = 2, 2.0           # every 2nd round a 2 m/s platform jolt (~16 cm travel at 2 Hz)
HOME_GAIN, HOME_V, HOME_DEADBAND = 1.5, 0.6, 0.08
CALM_R, CALM_V, CALM_S = 0.15, 0.2, 0.5
FLOOR_SEED = 5999                       # shared floor: jolt direction stream


def homing_command(quat: np.ndarray, x: float, y: float) -> np.ndarray:
    yaw = C.yaw_of(quat)
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = c * x + s * y, -s * x + c * y
    if math.hypot(bx, by) > HOME_DEADBAND:
        vx, vy = float(np.clip(-HOME_GAIN * bx, -HOME_V, HOME_V)), float(np.clip(-HOME_GAIN * by, -HOME_V, HOME_V))
    else:
        vx = vy = 0.0
    return np.array([vx, vy, float(np.clip(-2.0 * yaw, -0.5, 0.5))])


def burst(rng: np.random.Generator) -> tuple[float, float]:
    """Unit direction of one lateral wind burst in the world frame (athletes face +x)."""
    a = (math.pi / 2 if rng.uniform() < 0.5 else -math.pi / 2) + rng.uniform(-GUST_SPREAD, GUST_SPREAD)
    return math.cos(a), math.sin(a)


@dataclass
class GauntletLane:
    lane: int
    traits: Traits
    recoveries: list[float] = field(default_factory=list)
    total_s: float = 0.0
    out_at_s: float | None = None
    reason: str = ""
    place: int = 0


@dataclass
class GauntletResult:
    seed: int
    duration_s: float
    lanes: list[GauntletLane] = field(default_factory=list)


def run_heat(onnx, seed: int, traits: list[Traits] | None = None, scene=None, layout_path=None, brains: dict | None = None) -> GauntletResult:
    m = mujoco.MjModel.from_xml_path(str(scene or SCENE))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path or LAYOUT).read_text())
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, onnx, brains)
    mujoco.mj_forward(m, d)
    shared = any(m.joint(j).name == "shaker_x" for j in range(m.njnt))
    shaker = [m.jnt_dofadr[m.joint("shaker_x" if shared else ln.prefix + "shaker_x").id] for ln in lanes]
    floor = np.random.default_rng([seed, FLOOR_SEED])
    res = [GauntletLane(ln.k, ln.traits) for ln in lanes]
    dirs = [np.random.default_rng([seed, 5000 + i]) for i in range(len(lanes))]
    dt = m.opt.timestep * C.DECIMATION
    end_tick = round((START_S + N_ROUNDS * ROUND_S) / dt)
    round_ticks = {round((START_S + r * ROUND_S) / dt): r for r in range(N_ROUNDS)}
    burst_t = [None] * len(lanes)      # time of the current round's burst while not yet recovered
    calm = [0.0] * len(lanes)
    for tick in range(end_tick):
        for i, ln in enumerate(lanes):
            ra = ln.ath.root_qposadr
            cmd = np.zeros(3) if res[i].out_at_s is not None else homing_command(
                d.qpos[ra + 3: ra + 7], d.qpos[ra] - ln.origin[0], d.qpos[ra + 1] - ln.origin[1])
            ln.control(ln.sess, d, cmd)
        if tick in round_ticks:
            r = round_ticks[tick]
            dv = GUST_START + GUST_STEP * r
            fa = floor.uniform(0, 2 * math.pi)       # shared floor jolt direction (drawn every round)
            if shared and r % SHAKE_EVERY == SHAKE_EVERY - 1 and any(x.out_at_s is None for x in res):
                d.qvel[shaker[0]: shaker[0] + 2] += SHAKE_V * np.array([math.cos(fa), math.sin(fa)])
            for i, ln in enumerate(lanes):
                g = dirs[i]
                ux, uy = burst(g)
                sa = g.uniform(0, 2 * math.pi)       # drawn every round so the sequence does not depend on SHAKE_EVERY
                if res[i].out_at_s is not None:
                    continue
                if burst_t[i] is not None:           # previous round never recovered
                    res[i].recoveries.append(ROUND_S)
                da = ln.ath.root_dofadr
                d.qvel[da: da + 2] += dv * np.array([ux, uy])
                if not shared and r % SHAKE_EVERY == SHAKE_EVERY - 1:
                    d.qvel[shaker[i]: shaker[i] + 2] += SHAKE_V * np.array([math.cos(sa), math.sin(sa)])
                burst_t[i], calm[i] = tick * dt, 0.0
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        t = (tick + 1) * dt
        for i, ln in enumerate(lanes):
            rr = res[i]
            if rr.out_at_s is not None:
                continue
            why = ln.eliminated(m, d)
            if why:
                rr.out_at_s, rr.reason = t, why
                continue
            if burst_t[i] is None:
                continue
            ra, da = ln.ath.root_qposadr, ln.ath.root_dofadr
            off = math.hypot(d.qpos[ra] - ln.origin[0], d.qpos[ra + 1] - ln.origin[1])
            spd = math.hypot(d.qvel[da], d.qvel[da + 1])
            calm[i] = calm[i] + dt if (off < CALM_R and spd < CALM_V) else 0.0
            if calm[i] >= CALM_S - 1e-9:
                rr.recoveries.append(min(ROUND_S, t - CALM_S - burst_t[i]))
                burst_t[i] = None
    for i, rr in enumerate(res):
        if rr.out_at_s is None and burst_t[i] is not None:
            rr.recoveries.append(ROUND_S)
        rr.recoveries += [ROUND_S] * (N_ROUNDS - len(rr.recoveries))
        rr.total_s = float(sum(rr.recoveries))
    order = sorted(res, key=lambda r: (0, r.total_s) if r.out_at_s is None else (1, -r.out_at_s))
    for p, r in enumerate(order, 1):
        r.place = p
    return GauntletResult(seed, end_tick * dt, res)


def to_json(res: GauntletResult) -> dict:
    return {"seed": res.seed, "duration_s": res.duration_s,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}
