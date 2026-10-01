"""6 The Flamingo Classic — 8 athletes stand on one leg on the venue's 2 x 4 station grid, Rung S brain (contract v4
lift-foot command: stand on the other leg). Mirror of Unity FlamingoEvent.

Scene: assets/scene_flamingo8_mattbio.xml (tools/compose_mixed.py flamingo8 mattbio x 8) — venue 6's 2 x 4 station grid
(3 m x 4 m apart, the same spacing as venues 3 and 7), all-mattbio lineup = the Rung S training body, no props.

At GO everyone lifts the same foot (seeded per heat) and has LIFT_S to get it off the ground; the stance foot's spot is
marked at LIFT_S. From then on the wind rises: every ROUND_S each athlete still up gets a gust of the same magnitude
(GUST_START + GUST_STEP per round) in its own seeded direction.
  out = the lifted foot touching the ground (TOUCHDOWN), the stance foot more than HOP_TOL from its spot (HOPPED), or a
        fall (FELL: DESIGN §1 fall rule)
The mark is the time on one leg (s after GO). Rank by it, longest first; the heat runs until the last flamingo is down
(anyone still up at MAX_S shares first place). Traits as in the Iron Pedestal heat.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .. import contract as C
from .iron_pedestal import Traits, _Lane, make_lanes

SCENE = C.ROOT / "assets" / "scene_flamingo8_mattbio.xml"
LAYOUT = C.ROOT / "assets" / "flamingo8_mattbio_layout.json"
LIFT_S = 2.0                          # s after GO to get the foot off the ground; touches before do not count
ROUND_S = 2.0                         # a gust every round, the first one ROUND_S after LIFT_S
GUST_START, GUST_STEP = 0.04, 0.012   # m/s pelvis Δv — tuned on CPU heats, see rl_optimization_log.md
HOP_TOL = 0.30                        # m: the stance foot this far from its spot = HOPPED
MAX_S = 120.0
START_S = 1.0                         # settle before GO (Unity countdown hand-off)


def gust(r: int) -> float:
    return GUST_START + GUST_STEP * r


@dataclass
class FlamingoLane:
    lane: int
    traits: Traits
    out_at_s: float | None = None                          # time on one leg (s after GO); None = still up at MAX_S
    status: str = ""                                       # TOUCHDOWN / HOPPED / FELL / UP
    gusts: int = 0                                         # gusts taken while still up
    max_hop_m: float = 0.0
    place: int = 0


@dataclass
class FlamingoResult:
    seed: int
    duration_s: float
    foot: str                                              # the lifted foot, "l" or "r"
    lanes: list[FlamingoLane] = field(default_factory=list)


class _FlamingoJudge:
    """Per lane: the lifted foot's geoms, the stance foot and its spot (marked at LIFT_S)."""

    def __init__(self, m, ln: _Lane, foot: int):
        self.ln = ln
        side = "lr"[foot]
        self.lifted = {m.geom(ln.prefix + f"{p}_{side}_geom0").id for p in ("foot", "toe")}
        self.stance = m.body(ln.prefix + f"foot_{'rl'[foot]}").id
        self.ground = m.geom("ground").id
        self.spot = None

    def mark(self, d):
        self.spot = d.xpos[self.stance][:2].copy()

    def hop(self, d) -> float:
        return float(np.linalg.norm(d.xpos[self.stance][:2] - self.spot))

    def touchdown(self, d) -> bool:
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == self.ground and g2 in self.lifted) or (g2 == self.ground and g1 in self.lifted):
                return True
        return False


def run_heat(onnx, seed: int, traits: list[Traits] | None = None, scene=None, layout_path=None,
             max_s: float = MAX_S) -> FlamingoResult:
    m = mujoco.MjModel.from_xml_path(str(scene or SCENE))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path or LAYOUT).read_text())
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, onnx)
    mujoco.mj_forward(m, d)
    foot = int(rng.integers(0, 2))
    judges = [_FlamingoJudge(m, ln, foot) for ln in lanes]
    res = [FlamingoLane(ln.k, ln.traits) for ln in lanes]
    for ln in lanes:
        ln.skill = np.zeros(C.SKILL_DIM)
    dt = m.opt.timestep * C.DECIMATION
    go, lift, rnd_ticks = round(START_S / dt), round(LIFT_S / dt), round(ROUND_S / dt)
    tick, rnd = 0, 0
    next_round = go + lift + rnd_ticks
    while tick < go + round(max_s / dt) and (tick < go or any(not r.status for r in res)):
        for i, ln in enumerate(lanes):
            if tick >= go and not res[i].status:
                ln.skill[1 + foot] = 1.0
            ln.control(ln.sess, d, np.zeros(3))
        if tick == go + lift:
            for j in judges:
                j.mark(d)
        if tick >= next_round:                   # gust round: same magnitude, own direction
            dv = gust(rnd)
            for i, ln in enumerate(lanes):
                if res[i].status:
                    continue
                a = rng.uniform(0, 2 * math.pi)
                da = m.jnt_dofadr[m.joint(ln.prefix + "root").id]
                d.qvel[da: da + 2] += dv * np.array([math.cos(a), math.sin(a)])
                res[i].gusts += 1
            rnd += 1
            next_round += rnd_ticks
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        if tick <= go:
            continue
        for i, (ln, j) in enumerate(zip(lanes, judges)):
            r = res[i]
            if r.status:
                continue
            why = "FELL" if ln.eliminated(m, d) else None
            if why is None and tick >= go + lift:
                hop = j.hop(d) if j.spot is not None else 0.0
                r.max_hop_m = max(r.max_hop_m, round(hop, 3))
                why = "TOUCHDOWN" if j.touchdown(d) else "HOPPED" if hop > HOP_TOL else None
            if why:
                r.status, r.out_at_s = why, round((tick - go) * dt, 2)
                ln.skill[1 + foot] = 0.0
    for r in res:
        r.status = r.status or "UP"
    order = sorted(res, key=lambda r: -(r.out_at_s if r.out_at_s is not None else math.inf))
    place = 1
    for i, r in enumerate(order):
        if i > 0 and r.out_at_s != order[i - 1].out_at_s:
            place = i + 1
        r.place = place
    return FlamingoResult(seed, tick * dt, "lr"[foot], res)


def to_json(res: FlamingoResult) -> dict:
    return {"seed": res.seed, "duration_s": res.duration_s, "foot": res.foot,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Flamingo Classic heats on CPU MuJoCo")
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
        line = " | ".join(f"L{l.lane + 1} {'-' if l.out_at_s is None else f'{l.out_at_s:.1f}'} {l.status}"
                          for l in sorted(res.lanes, key=lambda l: l.place))
        print(f"seed {s} foot {res.foot} ({res.duration_s:.1f} s sim, {time.time() - t0:.0f} s wall): {line}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(runs, indent=1))
