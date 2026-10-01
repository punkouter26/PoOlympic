"""Gate G1 for Rung S (contract v4 stance skills; docs/CONTRACT_V4_STANCE_PROPOSAL.md) — one drill per event, CPU MuJoCo.

Every seed runs all five drills from the default stance (1 s settle, zero velocity command). Bars (MATT; a seed passes
when every drill passes, the rung when 10/10 seeds pass):

  E3 squat     3 reps, depth U(0.25, 0.40) m: down 2.5 s, up 2.0 s; pelvis within SQUAT_TOL of the target (mean of each
               hold's last 1 s); no fall (fall line lowered by the target − 0.10 m, like training)
  E6 flamingo  extra settle U(0, 1) s, then lift one foot 8 s, both feet 1.5 s, lift the other 8 s (random order): after 1 s the lifted foot never touches the ground,
               the stance foot moves < FLAMINGO_SLIP; no fall
  E7 march     cadence U(1.0, 1.6) Hz, knee lift U(0.15, 0.30) m, 10 s: stride cadence (left-knee peaks) within ±5 %,
               mean peak knee rise ≥ 0.75 x lift, pelvis drift < MARCH_DRIFT; no fall
  E2 torso     5 aims (yaw, pitch) in the approved ranges, 2 s each: |yaw err|, |pitch err| < TORSO_TOL (mean of the
               last 1 s); feet slip < FEET_SLIP; no fall
  E4 reach     6 targets (alternating arms) sampled like training, 2 s each: forearm tip within REACH_TOL (mean of the
               last 0.75 s); feet slip < FEET_SLIP; no fall
All measurements: poolympic/skills.py (the same quantities the Rung S rewards use).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import contract as C
from . import evaluate as E
from . import skills as K

SQUAT_TOL = 0.05
FLAMINGO_SLIP = 0.10
MARCH_CADENCE_TOL = 0.05
MARCH_LIFT_FRAC = 0.75
MARCH_DRIFT = 0.30
TORSO_TOL = 0.15
REACH_TOL = 0.08
FEET_SLIP = 0.05
SETTLE_S = 1.0
TICK = C.DECIMATION * 0.005


class SkillSim(E.Sim):
    """E.Sim with the v4 skill block (95-obs brain) and the march cadence driving the clock."""

    def __init__(self, onnx_path: Path, scene_xml: Path = C.SCENE_XML):
        super().__init__(onnx_path, scene_xml)
        self.sb = K.SkillBodies.bind(self.m)
        self.skill = np.zeros(C.SKILL_DIM)
        self.feet_bodies = [self.m.body(n).id for n in ("foot_l", "foot_r")]

    def reset(self):
        super().reset()
        self.skill = np.zeros(C.SKILL_DIM)

    def control_tick(self, command=np.zeros(3), pre_step=None):
        d = self.d
        self.phase = C.advance_phase(self.phase, command, C.skill_cadence(self.skill))
        obs = C.build_obs(self.ath, d.qpos, d.qvel, command, self.phase, self.last_action, self.skill)
        ctrl, action = self.sess.run(None, {"obs": obs[None]})
        d.ctrl[self.ath.actuator_ids] = ctrl[0].astype(np.float64)
        for _ in range(C.DECIMATION):
            mujoco.mj_step(self.m, d)
        self.last_action = action[0].astype(np.float64)
        self.tick += 1

    def run(self, seconds: float, sample=None) -> str | None:
        """Tick for `seconds` (zero velocity command); sample(t_in_segment) each tick; returns a fall reason or None."""
        n = int(round(seconds / TICK))
        for k in range(n):
            self.control_tick()
            if sample is not None:
                sample((k + 1) * TICK)
            why = self.fell_skill()
            if why:
                return why
        return None

    def fell_skill(self) -> str | None:
        line = E.FALL_PELVIS_Z + min(0.0, self.skill[0]) - (0.10 if self.skill[0] < 0 else 0.0)
        if self.pelvis()[2] < line:
            return "pelvis_low"
        if self.torso_tilt_deg() > E.FALL_TILT_DEG + (math.degrees(self.skill[6]) if self.skill[6] > 0 else 0.0):
            return "tilt"
        bad = self.illegal_ground_contact()
        return f"contact:{bad}" if bad else None

    def feet_xy(self) -> np.ndarray:
        return np.array([self.d.xpos[b][:2] for b in self.feet_bodies])


@dataclass
class StanceResult:
    seed: int
    drills: dict = field(default_factory=dict)       # name -> {"pass": bool, ...details}

    @property
    def passed(self) -> bool:
        return all(v["pass"] for v in self.drills.values())


def _settle(sim: SkillSim) -> str | None:
    sim.reset()
    return sim.run(SETTLE_S)


def drill_squat(sim: SkillSim, rng: np.random.Generator) -> dict:
    fall = _settle(sim)
    errs = []
    for _ in range(3):
        for target, hold in ((-rng.uniform(0.25, 0.40), 2.5), (0.0, 2.0)):
            sim.skill = C.SkillCommand(pelvis_height=target).to_array()
            buf = []
            fall = fall or sim.run(hold, lambda t: buf.append(K.pelvis_height_rel(sim.d, sim.sb)) if t > hold - 1.0 else None)
            errs.append(abs(float(np.mean(buf)) - target) if buf else 9.9)
    worst = max(errs)
    return {"pass": fall is None and worst < SQUAT_TOL, "fall": fall, "worst_err_m": round(worst, 4)}


def drill_flamingo(sim: SkillSim, rng: np.random.Generator) -> dict:
    fall = _settle(sim)
    # 2026-09-30 (after rs_v6): the drill drew nothing from rng, so all seeds were ONE trial ("10/10" at it2550, "0/10"
    # at the next three gates). Seeds now differ by an extra settle of 0-1 s and by which foot goes first.
    fall = fall or sim.run(float(rng.uniform(0.0, 1.0)))
    touch, slip = 0, 0.0
    for foot in (("l", "r") if rng.uniform() < 0.5 else ("r", "l")):
        sim.skill = C.SkillCommand(lift_foot=foot).to_array()
        k = 0 if foot == "l" else 1
        stance = 1 - k
        start = {}

        def sample(t):
            nonlocal touch, slip
            if t < 1.0:
                return
            if "xy" not in start:
                start["xy"] = sim.feet_xy()[stance].copy()
            if K.foot_on_ground(sim.d, sim.sb)[k]:
                touch += 1
            slip = max(slip, float(np.linalg.norm(sim.feet_xy()[stance] - start["xy"])))

        fall = fall or sim.run(8.0, sample)
        sim.skill = np.zeros(C.SKILL_DIM)
        fall = fall or sim.run(1.5)
    return {"pass": fall is None and touch == 0 and slip < FLAMINGO_SLIP, "fall": fall, "touch_ticks": touch,
            "stance_slip_m": round(slip, 4)}


def drill_march(sim: SkillSim, rng: np.random.Generator) -> dict:
    fall = _settle(sim)
    hz, lift = rng.uniform(1.0, 1.6), rng.uniform(0.15, 0.30)
    sim.skill = C.SkillCommand(march_hz=hz, knee_lift=lift).to_array()
    p0 = sim.pelvis()[:2].copy()
    rise, drift = [], 0.0

    def sample(t):
        nonlocal drift
        if t >= 2.0:                       # after 2 s of marching
            rise.append(K.knee_rise(sim.d, sim.sb)[0])
        drift = max(drift, float(np.linalg.norm(sim.pelvis()[:2] - p0)))

    fall = fall or sim.run(10.0, sample)
    r = np.array(rise)
    thr = 0.5 * lift
    peaks, i = [], 0
    while i < len(r):                      # one peak per excursion above half the lift
        if r[i] > thr:
            j = i
            while j < len(r) and r[j] > thr:
                j += 1
            peaks.append(float(r[i:j].max()))
            i = j
        else:
            i += 1
    span = len(r) * TICK
    cadence = len(peaks) / span if span else 0.0
    mean_peak = float(np.mean(peaks)) if peaks else 0.0
    ok = (fall is None and abs(cadence - hz) <= MARCH_CADENCE_TOL * hz and mean_peak >= MARCH_LIFT_FRAC * lift
          and drift < MARCH_DRIFT)
    return {"pass": bool(ok), "fall": fall, "cadence_cmd_hz": round(hz, 3), "cadence_hz": round(cadence, 3),
            "lift_cmd_m": round(lift, 3), "mean_peak_m": round(mean_peak, 3), "drift_m": round(drift, 3)}


def drill_torso(sim: SkillSim, rng: np.random.Generator) -> dict:
    fall = _settle(sim)
    r = K.body_ranges()
    feet0 = sim.feet_xy().copy()
    worst = 0.0
    for _ in range(5):
        yaw, pitch = rng.uniform(*r["torso_yaw"]), rng.uniform(*r["torso_pitch"])
        sim.skill = C.SkillCommand(torso_yaw=yaw, torso_pitch=pitch).to_array()
        buf = []
        fall = fall or sim.run(2.0, lambda t: buf.append(K.torso_aim(sim.d, sim.sb)) if t > 1.0 else None)
        if buf:
            a = np.mean(buf, axis=0)
            worst = max(worst, abs(K.wrap(a[0] - yaw)), abs(a[1] - pitch))
        else:
            worst = 9.9
    slip = float(np.linalg.norm(sim.feet_xy() - feet0, axis=1).max())
    return {"pass": fall is None and worst < TORSO_TOL and slip < FEET_SLIP, "fall": fall,
            "worst_err_rad": round(worst, 4), "feet_slip_m": round(slip, 4)}


def drill_reach(sim: SkillSim, rng: np.random.Generator) -> dict:
    fall = _settle(sim)
    reach = K.body_ranges()["hand_reach"]
    feet0 = sim.feet_xy().copy()
    worst = 0.0
    for i in range(6):
        arm = -1 if i % 2 == 0 else 1
        target = K.sample_reach_target(rng, K.shoulder_in_heading(sim.d, sim.m, arm), arm, reach)
        sim.skill = C.SkillCommand(hand=tuple(target), arm=arm).to_array()
        buf = []
        fall = fall or sim.run(2.0, lambda t: buf.append(K.hand_in_heading(sim.d, sim.sb, arm)) if t > 1.25 else None)
        worst = max(worst, float(np.linalg.norm(np.mean(buf, axis=0) - target)) if buf else 9.9)
    slip = float(np.linalg.norm(sim.feet_xy() - feet0, axis=1).max())
    return {"pass": fall is None and worst < REACH_TOL and slip < FEET_SLIP, "fall": fall,
            "worst_err_m": round(worst, 4), "feet_slip_m": round(slip, 4)}


DRILLS = {"E3_squat": drill_squat, "E6_flamingo": drill_flamingo, "E7_march": drill_march,
          "E2_torso": drill_torso, "E4_reach": drill_reach}


def stance_episode(onnx_path: Path, seed: int, sim: SkillSim | None = None) -> StanceResult:
    sim = sim or SkillSim(onnx_path)
    res = StanceResult(seed)
    for k, (name, drill) in enumerate(DRILLS.items()):
        res.drills[name] = drill(sim, np.random.default_rng([seed, k]))
    return res


def stance_verdict(results: list[StanceResult]) -> dict:
    per = {n: sum(r.drills[n]["pass"] for r in results) for n in DRILLS}
    passed = sum(r.passed for r in results)
    return {"seeds": len(results), "passed_seeds": passed, "per_drill": per, "PASS": passed == len(results)}


def to_json(r: StanceResult) -> dict:
    return {**asdict(r), "passed": r.passed}
