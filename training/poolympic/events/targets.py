"""2 Torso Archer and 4 Precision Javelin Reach — 8 athletes on the venue's 2 x 4 station grid (assets/scene_archer8 /
reach8_mattbio.xml, stations 3 m x 4 m apart: no contact; all-mattbio lineup), Rung S brains (contract v4 torso-aim /
hand-target commands). Mirror of Unity TargetEvent. Both are a string of ROUNDS timed targets that get faster
(round r lasts max(T_MIN, T_START − T_STEP x r) s); every lane gets the same targets.

  archer  a flight target crosses the sky: the aim (chest yaw, pitch) moves in a straight line from the last aim to a new
          one over the first MOVE_FRAC of the round, then hangs there. Error = angle between the chest aim and the target,
          averaged over the WHOLE round (tracking). Aims: yaw ±ARCHER_YAW (the spine twists 45 deg; the contract's ±60 deg
          cannot be reached), pitch ARCHER_PITCH, each at least ARCHER_MIN_MOVE from the last.
  reach   a target appears at the reaching arm's side (arms alternate), farther out every round: REACH_FRAC[0] ..
          REACH_FRAC[1] of the reach radius from the shoulder (0.78 = the straight arm; the contract's 0.95 is past it).
          Error = distance forearm tip - target, averaged over the last SAMPLE_FRAC of the round (touch and hold).
  round points = ROUND_POINTS x clamp(1 − error / TOL, 0, 1)
  out          = a fall (FELL; the tilt line gives way by the commanded forward pitch) or a foot sliding more than
                 STEP_TOL from its start spot (STEPPED)
Finishing all rounds in balance earns BALANCE_BONUS. Rank by total points; ties: the later exit first.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .. import action_log as AL
from .. import contract as C
from .. import skills as K
from .iron_pedestal import Traits, make_lanes
from .squat import STEP_TOL, _SquatJudge

KINDS = ("archer", "reach")
ROUNDS = 12
T_START, T_STEP, T_MIN = 2.0, 0.1, 1.0
ROUND_POINTS = 10.0
BALANCE_BONUS = 10.0
START_S = 1.0

ARCHER_YAW = math.radians(45)
ARCHER_PITCH = (math.radians(-15), math.radians(35))
ARCHER_MIN_MOVE = math.radians(30)
MOVE_FRAC = 0.6
ARCHER_TOL = 0.25                  # rad: zero points this far off on average

REACH_FRAC = (0.50, 0.78)          # of the reach radius: round 0 .. the last round
REACH_AZ = K.REACH_AZ
REACH_EL = (K.REACH_EL[0], math.radians(45))
SAMPLE_FRAC = 0.4
REACH_TOL = 0.15                   # m


def round_s(r: int) -> float:
    return max(T_MIN, T_START - T_STEP * r)


def total_s() -> float:
    return sum(round_s(r) for r in range(ROUNDS))


def scene_paths(kind: str) -> tuple[Path, Path]:
    return C.ROOT / "assets" / f"scene_{kind}8_mattbio.xml", C.ROOT / "assets" / f"{kind}8_mattbio_layout.json"


def archer_aims(rng: np.random.Generator) -> list[tuple[float, float]]:
    """ROUNDS + 1 aims (yaw, pitch): straight ahead, then one per round."""
    aims = [(0.0, 0.0)]
    for _ in range(ROUNDS):
        while True:
            a = (float(rng.uniform(-ARCHER_YAW, ARCHER_YAW)), float(rng.uniform(*ARCHER_PITCH)))
            if math.hypot(a[0] - aims[-1][0], a[1] - aims[-1][1]) >= ARCHER_MIN_MOVE:
                break
        aims.append(a)
    return aims


def archer_target(aims, r: int, frac: float) -> tuple[float, float]:
    """Target aim `frac` (0..1) through round r."""
    s = min(1.0, frac / MOVE_FRAC)
    a, b = aims[r], aims[r + 1]
    return a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s


def reach_dirs(rng: np.random.Generator) -> list[tuple[int, float, float, float]]:
    """Per round: (arm −1 left / +1 right, azimuth, elevation, fraction of the reach radius)."""
    first = -1 if rng.uniform() < 0.5 else 1
    return [(first if r % 2 == 0 else -first, float(rng.uniform(*REACH_AZ)), float(rng.uniform(*REACH_EL)),
             REACH_FRAC[0] + (REACH_FRAC[1] - REACH_FRAC[0]) * r / (ROUNDS - 1)) for r in range(ROUNDS)]


def reach_target(shoulder: np.ndarray, arm: int, az: float, el: float, dist: float) -> np.ndarray:
    u = np.array([math.cos(el) * math.cos(az), -arm * math.cos(el) * math.sin(az), math.sin(el)])
    return shoulder + u * dist


@dataclass
class TargetLane:
    lane: int
    traits: Traits
    points: float = 0.0
    rounds: list[float] = field(default_factory=list)   # points per completed round
    errs: list[float] = field(default_factory=list)     # error per round (rad / m)
    out_at_s: float | None = None
    status: str = ""                                     # DONE / FELL / STEPPED
    place: int = 0


@dataclass
class TargetResult:
    kind: str
    seed: int
    duration_s: float
    lanes: list[TargetLane] = field(default_factory=list)


@AL.logged("event_targets")
def run_heat(onnx, seed: int, kind: str = "archer", traits: list[Traits] | None = None, scene=None,
             layout_path=None) -> TargetResult:
    assert kind in KINDS
    sc, lay = scene_paths(kind)
    m = mujoco.MjModel.from_xml_path(str(scene or sc))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path or lay).read_text())
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, onnx)
    mujoco.mj_forward(m, d)
    judges = [_SquatJudge(m, d, ln) for ln in lanes]
    sbs = [K.SkillBodies.bind(m, ln.prefix, keyframe=None) for ln in lanes]
    res = [TargetLane(ln.k, ln.traits) for ln in lanes]
    for ln in lanes:
        ln.skill = np.zeros(C.SKILL_DIM)
    aims = archer_aims(rng) if kind == "archer" else None
    dirs = reach_dirs(rng) if kind == "reach" else None
    reach = K.body_ranges()["hand_reach"]
    shoulders: list[dict] = [{} for _ in lanes]          # per lane: arm -> shoulder in the heading frame at GO
    starts = np.concatenate([[0.0], np.cumsum([round_s(r) for r in range(ROUNDS)])])
    dt = m.opt.timestep * C.DECIMATION
    end = START_S + total_s()
    buf: list[list[float]] = [[] for _ in lanes]
    rnd, tick = -1, 0
    while tick * dt < end - 1e-9:
        live = tick * dt - START_S
        new = int(np.searchsorted(starts, live + 1e-9, side="right")) - 1 if live >= -1e-9 else -1
        new = min(new, ROUNDS - 1)
        if new != rnd:
            if rnd >= 0:
                _score(kind, res, buf)
            rnd = new
            buf = [[] for _ in lanes]
            if rnd == 0:
                for i, (j, ln) in enumerate(zip(judges, lanes)):
                    j.mark_feet(d)
                    shoulders[i] = {a: K.shoulder_in_heading(d, m, a, ln.prefix) for a in (-1, 1)}
        frac = (live - starts[rnd]) / round_s(rnd) if rnd >= 0 else 0.0
        tilt_extra = 0.0
        for i, ln in enumerate(lanes):
            ln.skill[:] = 0.0
            if rnd >= 0 and not res[i].status:
                if kind == "archer":
                    yaw, pitch = archer_target(aims, rnd, frac)
                    ln.skill[5], ln.skill[6] = yaw, pitch
                    tilt_extra = math.degrees(max(0.0, pitch))
                else:
                    arm, az, el, f = dirs[rnd]
                    ln.skill[7:10] = reach_target(shoulders[i][arm], arm, az, el, reach * f)
                    ln.skill[10] = arm
            ln.control(ln.sess, d, np.zeros(3))
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt - START_S
        for i, (ln, j) in enumerate(zip(lanes, judges)):
            r = res[i]
            if r.status:
                continue
            why = j.out(m, d, 0.0, tilt_extra)
            if why:
                r.status, r.out_at_s = why, t
                continue
            if rnd < 0:
                continue
            if kind == "archer":
                yaw, pitch = K.torso_aim(d, sbs[i])
                buf[i].append(math.hypot(K.wrap(yaw - ln.skill[5]), pitch - ln.skill[6]))
            elif t > starts[rnd] + round_s(rnd) * (1 - SAMPLE_FRAC):
                arm = int(ln.skill[10])
                buf[i].append(float(np.linalg.norm(K.hand_in_heading(d, sbs[i], arm) - ln.skill[7:10])))
    if rnd >= 0:
        _score(kind, res, buf)
    for r in res:
        r.status = r.status or "DONE"
        r.points = round(sum(r.rounds) + (BALANCE_BONUS if r.status == "DONE" else 0.0), 3)
    order = sorted(res, key=lambda r: (-r.points, -(r.out_at_s if r.out_at_s is not None else 1e9)))
    for p, r in enumerate(order, 1):
        r.place = p
    return TargetResult(kind, seed, tick * dt, res)


def _score(kind: str, res: list[TargetLane], buf: list[list[float]]):
    tol = ARCHER_TOL if kind == "archer" else REACH_TOL
    for r, b in zip(res, buf):
        if r.status or not b:          # out during this round: no points for it
            continue
        err = float(np.mean(b))
        r.errs.append(round(err, 4))
        r.rounds.append(ROUND_POINTS * min(1.0, max(0.0, 1.0 - err / tol)))


def to_json(res: TargetResult) -> dict:
    return {"kind": res.kind, "seed": res.seed, "duration_s": res.duration_s,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Torso Archer / Javelin Reach heats on CPU MuJoCo")
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    runs = []
    for s in range(a.first_seed, a.first_seed + a.seeds):
        t0 = time.time()
        res = run_heat(a.onnx, s, a.kind)
        runs.append(to_json(res))
        line = " | ".join(f"L{l.lane + 1} {l.points:5.1f} {l.status}{'' if l.out_at_s is None else f'@{l.out_at_s:.1f}'}"
                          for l in sorted(res.lanes, key=lambda l: l.place))
        print(f"seed {s} ({res.duration_s:.1f} s sim, {time.time() - t0:.0f} s wall): {line}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(runs, indent=1))
