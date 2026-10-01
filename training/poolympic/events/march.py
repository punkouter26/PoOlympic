"""7 Cadence March — 8 athletes march in place on the venue's 2 x 4 station grid to a rising metronome, Rung S brain
(contract v4 march command: cadence drives the gait clock at zero velocity, knee lift in metres). Mirror of Unity
CadenceMarchEvent.

Scene: assets/scene_march8_mattbio.xml (tools/compose_mixed.py march8 mattbio x 8) — venue 7's 2 x 4 station grid
(3 m x 4 m apart, the same spacing as venue 3), all-mattbio lineup = the Rung S training body, and the stadium's
metronome tower at the grid centre as a physical box (compose_mixed.march_props: 2 m from every station row).

The metronome calls STAGES stages of STAGE_S seconds; stage s marches at cadence(s) = HZ_START + HZ_STEP x s strides per
second (one stride = one lift of each knee) with the knees called LIFT metres above their standing height.
  a lift       = one excursion of a knee above half the called lift; it counts for the stage in which it ends
  stage points = STAGE_POINTS x rhythm x height
                 rhythm = clamp(1 − |lifts − expected| / (RHYTHM_TOL x expected), 0, 1), expected = 2 x cadence x STAGE_S
                 height = clamp(mean peak knee rise of the stage's lifts / LIFT, 0, 1)
  out          = a fall (FELL: DESIGN §1 fall rule) or the pelvis drifting more than DRIFT_OUT from its spot (WANDERED)
Finishing every stage on the spot earns FINISH_BONUS. Rank by total points; ties: the later exit first. Traits as in the
Iron Pedestal heat.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .. import contract as C
from .iron_pedestal import Traits, _Lane, make_lanes

SCENE = C.ROOT / "assets" / "scene_march8_mattbio.xml"
LAYOUT = C.ROOT / "assets" / "march8_mattbio_layout.json"
STAGES = 9
STAGE_S = 4.0
HZ_START, HZ_STEP = 1.0, 0.125        # strides / s: 1.0 -> 2.0 (Rung S range 0.8 .. 2.0); multiples of 1/8 are exact in float32
LIFT = 0.25                           # m knee rise called (Rung S range 0.10 .. 0.35)
RHYTHM_TOL = 0.25                     # zero rhythm points when a quarter of the lifts are missing (or extra)
STAGE_POINTS = 10.0
FINISH_BONUS = 10.0
DRIFT_OUT = 0.60                      # m from the spot = WANDERED
START_S = 1.0                         # settle before the first stage (Unity countdown hand-off)


def cadence(s: int) -> float:
    return HZ_START + HZ_STEP * s


def expected_lifts(s: int) -> float:
    return 2.0 * cadence(s) * STAGE_S


def stage_points(lifts: int, mean_peak: float, s: int) -> float:
    exp = expected_lifts(s)
    rhythm = min(1.0, max(0.0, 1.0 - abs(lifts - exp) / (RHYTHM_TOL * exp)))
    height = min(1.0, max(0.0, mean_peak / LIFT))
    return STAGE_POINTS * rhythm * height


@dataclass
class MarchLane:
    lane: int
    traits: Traits
    points: float = 0.0
    stages: list[float] = field(default_factory=list)     # points per completed stage
    lifts: list[int] = field(default_factory=list)        # knee lifts counted per stage (both knees)
    peaks_m: list[float] = field(default_factory=list)    # mean peak knee rise per stage
    max_drift_m: float = 0.0
    out_at_s: float | None = None
    status: str = ""                                       # DONE / FELL / WANDERED
    place: int = 0


@dataclass
class MarchResult:
    seed: int
    duration_s: float
    lanes: list[MarchLane] = field(default_factory=list)


class KneeCounter:
    """One knee: excursions above half the called lift, with the peak rise of each (state machine per sample)."""

    def __init__(self):
        self.up, self.peak = False, 0.0
        self.done: list[float] = []          # peaks of the lifts that ended since the last take()

    def sample(self, rise: float):
        if rise > 0.5 * LIFT:
            self.up, self.peak = True, max(self.peak, rise) if self.up else rise
        elif self.up:
            self.done.append(self.peak)
            self.up, self.peak = False, 0.0

    def take(self) -> list[float]:
        out, self.done = self.done, []
        return out


class _MarchJudge:
    """Per lane: standing knee heights + spot (marked at GO), the two knee counters."""

    def __init__(self, m, ln: _Lane):
        self.ln = ln
        self.shins = [m.body(ln.prefix + n).id for n in ("shin_l", "shin_r")]
        self.knee0 = None
        self.spot = None
        self.knees = [KneeCounter(), KneeCounter()]

    def mark(self, d):
        self.knee0 = [float(d.xpos[b][2]) for b in self.shins]
        r = self.ln.ath.root_qposadr
        self.spot = d.qpos[r: r + 2].copy()

    def drift(self, d) -> float:
        r = self.ln.ath.root_qposadr
        return float(np.linalg.norm(d.qpos[r: r + 2] - self.spot))

    def sample(self, d):
        for k, b in enumerate(self.shins):
            self.knees[k].sample(float(d.xpos[b][2]) - self.knee0[k])

    def take(self) -> list[float]:
        return self.knees[0].take() + self.knees[1].take()


def run_heat(onnx, seed: int, traits: list[Traits] | None = None, scene=None, layout_path=None) -> MarchResult:
    m = mujoco.MjModel.from_xml_path(str(scene or SCENE))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path or LAYOUT).read_text())
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, onnx)
    mujoco.mj_forward(m, d)
    judges = [_MarchJudge(m, ln) for ln in lanes]
    res = [MarchLane(ln.k, ln.traits) for ln in lanes]
    for ln in lanes:
        ln.skill = np.zeros(C.SKILL_DIM)
    dt = m.opt.timestep * C.DECIMATION
    end = START_S + STAGES * STAGE_S
    stage, tick = -1, 0
    while tick * dt < end - 1e-9:
        live = tick * dt - START_S
        new_stage = int((live + 1e-9) // STAGE_S) if live >= -1e-9 else -1
        if new_stage != stage:
            if stage >= 0:
                _score_stage(stage, res, judges)
            if stage < 0:
                for j in judges:
                    j.mark(d)
            stage = new_stage
        for i, ln in enumerate(lanes):
            if stage >= 0 and not res[i].status:
                ln.skill[3], ln.skill[4] = cadence(stage), LIFT
            ln.control(ln.sess, d, np.zeros(3))
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt
        if stage < 0:
            continue
        for i, (ln, j) in enumerate(zip(lanes, judges)):
            r = res[i]
            if r.status:
                continue
            drift = j.drift(d)
            r.max_drift_m = max(r.max_drift_m, round(drift, 3))
            why = "FELL" if ln.eliminated(m, d) else "WANDERED" if drift > DRIFT_OUT else None
            if why:
                r.status, r.out_at_s = why, round(t - START_S, 2)
                ln.skill[3] = ln.skill[4] = 0.0
                continue
            j.sample(d)
    _score_stage(stage, res, judges)
    for r in res:
        r.status = r.status or "DONE"
        r.points = round(sum(r.stages) + (FINISH_BONUS if r.status == "DONE" else 0.0), 3)
    order = sorted(res, key=lambda r: (-r.points, -(r.out_at_s if r.out_at_s is not None else 1e9)))
    for p, r in enumerate(order, 1):
        r.place = p
    return MarchResult(seed, tick * dt, res)


def _score_stage(stage: int, res: list[MarchLane], judges: list[_MarchJudge]):
    for r, j in zip(res, judges):
        peaks = j.take()
        if r.status:                   # out during this stage: no points for it
            continue
        mean_peak = float(np.mean(peaks)) if peaks else 0.0
        r.lifts.append(len(peaks))
        r.peaks_m.append(round(mean_peak, 3))
        r.stages.append(round(stage_points(len(peaks), mean_peak, stage), 3))


def to_json(res: MarchResult) -> dict:
    return {"seed": res.seed, "duration_s": res.duration_s,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Cadence March heats on CPU MuJoCo")
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("-v", action="store_true", help="per-stage lifts / peaks / points of every lane")
    a = ap.parse_args()
    runs = []
    for s in range(a.first_seed, a.first_seed + a.seeds):
        t0 = time.time()
        res = run_heat(a.onnx, s)
        runs.append(to_json(res))
        line = " | ".join(f"L{l.lane + 1} {l.points:5.1f} {l.status}{'' if l.out_at_s is None else f'@{l.out_at_s:.1f}'}"
                          for l in sorted(res.lanes, key=lambda l: l.place))
        print(f"seed {s} ({res.duration_s:.1f} s sim, {time.time() - t0:.0f} s wall): {line}", flush=True)
        if a.v:
            for l in res.lanes:
                print(f"   L{l.lane + 1} str {l.traits.strength:.2f} lat {l.traits.latency_substeps} drift {l.max_drift_m:.2f} m | lifts {l.lifts} "
                      f"(expected {[round(expected_lifts(k), 1) for k in range(STAGES)]}) | peaks {l.peaks_m} | pts {l.stages}")
    if a.out:
        a.out.write_text(json.dumps(runs, indent=1))
