"""3 Deep Squat Endurance — 8 athletes on the venue's 2 x 4 station grid (assets/scene_squat8.xml, stations 3 m x 4 m
apart: no contact), Rung S brain (contract v4 pelvis-height command). Mirror of Unity DeepSquatEvent.

A metronome calls REPS squat reps that get deeper and faster: rep r goes down to depth(r) = min(DEPTH_MAX,
DEPTH_START + DEPTH_STEP x r) below the standing pelvis height for down_s(r), then back up to standing for up_s(r).
  rep points = REP_POINTS x clamp(1 − |pelvis drop − depth| / DEPTH_TOL, 0, 1), the drop averaged over the last
               SAMPLE_FRAC of the down phase (on time and on depth)
  out        = a fall (FELL: pelvis below the fall line lowered by the squat target − 0.10 m like training, torso tilt,
               non-foot ground contact) or a foot sliding more than STEP_TOL from its start spot (STEPPED)
Rank by total points; ties: the later exit first. Traits as in the Iron Pedestal heat.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .. import contract as C
from .iron_pedestal import FALL_TILT_DEG, Traits, _Lane, make_lanes

SCENE = C.ROOT / "assets" / "scene_squat8.xml"
LAYOUT = C.ROOT / "assets" / "squat8_layout.json"
REPS = 12
DEPTH_START, DEPTH_STEP, DEPTH_MAX = 0.25, 0.02, 0.45    # m below standing (Rung S range: 0 .. 0.45)
DOWN_START, UP_START, TEMPO_STEP = 2.5, 2.0, 0.1         # s; each rep 0.1 s faster down and up
DOWN_MIN, UP_MIN = 1.2, 1.0
SAMPLE_FRAC = 0.4
REP_POINTS = 10.0
DEPTH_TOL = 0.10       # m: zero points this far off the called depth
STEP_TOL = 0.20        # m: a foot this far from its start spot = STEPPED
START_S = 1.0          # settle before the first rep (Unity countdown hand-off)


def depth(r: int) -> float:
    return min(DEPTH_MAX, DEPTH_START + DEPTH_STEP * r)


def down_s(r: int) -> float:
    return max(DOWN_MIN, DOWN_START - TEMPO_STEP * r)


def up_s(r: int) -> float:
    return max(UP_MIN, UP_START - TEMPO_STEP * r)


def schedule() -> list[tuple[float, float, float, int]]:
    """(start s after GO, duration s, pelvis-height command, rep index; down phases carry the rep, up phases −1)."""
    out, t = [], 0.0
    for r in range(REPS):
        out.append((t, down_s(r), -depth(r), r))
        t += down_s(r)
        out.append((t, up_s(r), 0.0, -1))
        t += up_s(r)
    return out


def total_s() -> float:
    return sum(dur for _, dur, _, _ in schedule())


@dataclass
class SquatLane:
    lane: int
    traits: Traits
    points: float = 0.0
    reps: list[float] = field(default_factory=list)     # points per completed rep
    errs_m: list[float] = field(default_factory=list)   # |drop − depth| per rep
    out_at_s: float | None = None
    status: str = ""                                     # DONE / FELL / STEPPED
    place: int = 0


@dataclass
class SquatResult:
    seed: int
    duration_s: float
    lanes: list[SquatLane] = field(default_factory=list)


class _SquatJudge:
    """Per lane: standing pelvis height, foot start spots, the squat-aware fall rule."""

    def __init__(self, m, d, ln: _Lane):
        self.ln = ln
        self.z0 = float(d.qpos[ln.ath.root_qposadr + 2])            # reset pose = the body's default standing height
        self.feet = [m.body(ln.prefix + n).id for n in ("foot_l", "foot_r")]
        self.start = None

    def drop(self, d) -> float:
        return self.z0 - float(d.qpos[self.ln.ath.root_qposadr + 2])

    def mark_feet(self, d):
        self.start = np.array([d.xpos[b][:2] for b in self.feet])

    def out(self, m, d, target: float) -> str | None:
        ln = self.ln
        r = ln.ath.root_qposadr
        if d.qpos[r + 2] < ln.fall_z + min(0.0, target) - (0.10 if target < 0 else 0.0):
            return "FELL"
        if math.degrees(math.acos(max(-1.0, min(1.0, d.xmat[ln.torso][8])))) > FALL_TILT_DEG:
            return "FELL"
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 in ln.support else g1 if g2 in ln.support else None
            if other is not None and other not in ln.feet and m.body_rootid[m.geom_bodyid[other]] == ln.pelvis:
                return "FELL"
        if self.start is not None:
            now = np.array([d.xpos[b][:2] for b in self.feet])
            if float(np.max(np.linalg.norm(now - self.start, axis=1))) > STEP_TOL:
                return "STEPPED"
        return None


def run_heat(onnx, seed: int, traits: list[Traits] | None = None, scene=None, layout_path=None) -> SquatResult:
    m = mujoco.MjModel.from_xml_path(str(scene or SCENE))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path or LAYOUT).read_text())
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, onnx)
    mujoco.mj_forward(m, d)
    judges = [_SquatJudge(m, d, ln) for ln in lanes]
    res = [SquatLane(ln.k, ln.traits) for ln in lanes]
    for ln in lanes:
        ln.skill = np.zeros(C.SKILL_DIM)
    dt = m.opt.timestep * C.DECIMATION
    sched = schedule()
    end = START_S + total_s()
    buf: list[list[float]] = [[] for _ in lanes]
    seg_i, tick = -1, 0
    while tick * dt < end - 1e-9:
        t = tick * dt
        # metronome: which segment is live (ticks are 20 ms; segment edges are multiples of 0.1 s)
        live = t - START_S
        new_seg = max((i for i, s in enumerate(sched) if s[0] <= live + 1e-9), default=-1) if live >= -1e-9 else -1
        if new_seg != seg_i:
            if seg_i >= 0 and sched[seg_i][3] >= 0:
                _score_rep(sched[seg_i], res, buf)
            seg_i = new_seg
            buf = [[] for _ in lanes]
            if seg_i == 0:
                for j in judges:
                    j.mark_feet(d)
        target = sched[seg_i][2] if seg_i >= 0 else 0.0
        for i, ln in enumerate(lanes):
            if not res[i].status:
                ln.skill[0] = target
            ln.control(ln.sess, d, np.zeros(3))
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt
        for i, (ln, j) in enumerate(zip(lanes, judges)):
            r = res[i]
            if r.status:
                continue
            why = j.out(m, d, ln.skill[0])
            if why:
                r.status, r.out_at_s = why, t - START_S
                ln.skill[0] = 0.0
                continue
            if seg_i >= 0:
                st, dur, _, rep = sched[seg_i]
                if rep >= 0 and t - START_S > st + dur * (1 - SAMPLE_FRAC):
                    buf[i].append(j.drop(d))
    if seg_i >= 0 and sched[seg_i][3] >= 0:
        _score_rep(sched[seg_i], res, buf)
    for r in res:
        r.status = r.status or "DONE"
        r.points = round(sum(r.reps), 3)
    order = sorted(res, key=lambda r: (-r.points, -(r.out_at_s if r.out_at_s is not None else 1e9)))
    for p, r in enumerate(order, 1):
        r.place = p
    return SquatResult(seed, tick * dt, res)


def _score_rep(seg, res: list[SquatLane], buf: list[list[float]]):
    _, _, cmd, rep = seg
    for r, b in zip(res, buf):
        if r.status or not b:          # out during this rep: no points for it
            continue
        err = abs(float(np.mean(b)) + cmd)       # cmd = −depth
        r.errs_m.append(round(err, 4))
        r.reps.append(REP_POINTS * min(1.0, max(0.0, 1.0 - err / DEPTH_TOL)))


def to_json(res: SquatResult) -> dict:
    return {"seed": res.seed, "duration_s": res.duration_s,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Deep Squat Endurance heats on CPU MuJoCo")
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    runs = []
    for s in range(a.first_seed, a.first_seed + a.seeds):
        t0 = time.time()
        res = run_heat(a.onnx, s)
        runs.append(to_json(res))
        line = " | ".join(f"L{l.lane + 1} {l.points:5.1f} {l.status}{'' if l.out_at_s is None else f'@{l.out_at_s:.1f}'}"
                          for l in sorted(res.lanes, key=lambda l: l.place))
        print(f"seed {s} ({res.duration_s:.1f} s sim, {time.time() - t0:.0f} s wall): {line}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(runs, indent=1))
