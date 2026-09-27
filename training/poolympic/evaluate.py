"""C2 — Gate G1: evaluate an exported ONNX brain in CPU MuJoCo (float64, same C library as Unity)
against the rung pass bars of DESIGN.md §1 (10/10 consecutive seeds).

Rollouts follow the contract tick order exactly (see reference.py) so a brain that passes here is the brain the
Unity parity gates then check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from . import contract as C

FALL_PELVIS_Z = 0.55
FALL_TILT_DEG = 60.0
RECOVER_TILT_DEG = 10.0
RECOVER_WINDOW_S = 1.5
JOINT_VEL_LIMIT = 18.0
JOINT_VEL_MAX_FRACTION = 0.05


@dataclass
class EpisodeResult:
    seed: int
    fell: bool
    fall_time: float | None
    fall_reason: str | None
    hits: int
    recovered_hits: int
    worst_recovery_s: float | None
    max_foot_excursion_m: float
    joint_vel_over_fraction: float
    max_tilt_after_hit_deg: float = 0.0
    notes: list[str] = field(default_factory=list)


class Sim:
    """CPU MuJoCo scene + ONNX brain driven with the contract tick order."""

    def __init__(self, onnx_path: Path, scene_xml: Path = C.SCENE_XML):
        self.m = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.d = mujoco.MjData(self.m)
        self.ath = C.Athlete.bind(self.m)
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        m = self.m
        self.ground = m.geom("ground").id
        self.torso = m.body("torso").id
        self.cube_geoms = {m.geom(f"cube{i}_geom").id for i in range(4)}
        self.foot_geoms = [m.geom(n).id for n in ("foot_l_geom0", "toe_l_geom0", "foot_r_geom0", "toe_r_geom0")]
        self.foot_geom_set = set(self.foot_geoms)
        self.athlete_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and g not in self.cube_geoms}
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("default").id)
        mujoco.mj_forward(self.m, self.d)
        self.phase = 0.0
        self.last_action = np.zeros(C.NUM_ACTIONS)
        self.tick = 0

    def control_tick(self, command: np.ndarray, pre_step=None):
        d = self.d
        self.phase = C.advance_phase(self.phase, command)
        obs = C.build_obs(self.ath, d.qpos, d.qvel, command, self.phase, self.last_action)
        ctrl, action = self.sess.run(None, {"obs": obs[None]})
        d.ctrl[self.ath.actuator_ids] = ctrl[0].astype(np.float64)
        if pre_step is not None:
            pre_step(self)
        for _ in range(C.DECIMATION):
            mujoco.mj_step(self.m, d)
        self.last_action = action[0].astype(np.float64)
        self.tick += 1

    # ---- measurements
    def torso_tilt_deg(self) -> float:
        z = self.d.xmat[self.torso].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(max(-1.0, min(1.0, z[2]))))

    def pelvis(self) -> np.ndarray:
        return self.d.qpos[self.ath.root_qposadr : self.ath.root_qposadr + 3].copy()

    def illegal_ground_contact(self) -> str | None:
        for c in self.d.contact[: self.d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 == self.ground else g1 if g2 == self.ground else None
            if other is not None and other in self.athlete_geoms and other not in self.foot_geom_set:
                return self.m.geom(other).name
        return None

    def cube_touching_athlete(self) -> bool:
        for c in self.d.contact[: self.d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.cube_geoms and g2 in self.athlete_geoms) or (g2 in self.cube_geoms and g1 in self.athlete_geoms):
                return True
        return False

    def foot_corners_xy(self) -> np.ndarray:
        pts = []
        for g in self.foot_geoms:
            R = self.d.geom_xmat[g].reshape(3, 3)
            h = self.m.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    pts.append((self.d.geom_xpos[g] + R @ np.array([sx * h[0], sy * h[1], -h[2]]))[:2])
        return np.array(pts)

    def fell(self) -> str | None:
        if self.pelvis()[2] < FALL_PELVIS_Z:
            return "pelvis_low"
        if self.torso_tilt_deg() > FALL_TILT_DEG:
            return "tilt"
        bad = self.illegal_ground_contact()
        return f"contact:{bad}" if bad else None


def rung0_episode(onnx_path: Path, seed: int, seconds: float = 20.0, shove_dv: float = 0.5,
                  cube_drop_m: float = 1.5, sim: Sim | None = None) -> EpisodeResult:
    """Rung 0: stand under random 0.5 m/s shoves + 2 kg cube drops (independent 3–5 s schedules)."""
    rng = np.random.default_rng(seed)
    sim = sim or Sim(onnx_path)
    sim.reset()
    dt = C.DECIMATION * sim.m.opt.timestep
    n_ticks = int(round(seconds / dt))
    shove_ticks, t = [], rng.uniform(3, 5)
    while t < seconds - 1.5:
        shove_ticks.append(int(t / dt)); t += rng.uniform(3, 5)
    cube_ticks, t = [], rng.uniform(3, 5)
    while t < seconds - 2.0:
        cube_ticks.append(int(t / dt)); t += rng.uniform(3, 5)
    shove_set, cube_set = set(shove_ticks), set(cube_ticks)

    foot0 = sim.foot_corners_xy().mean(axis=0)
    hit_times: list[float] = []
    cube_pending: list[float] = []
    next_cube = 0
    tilt_log: list[float] = []
    joint_over = 0
    max_exc = 0.0
    fell_at, reason = None, None

    for k in range(n_ticks):
        now = k * dt

        def pre(s: Sim, k=k):
            nonlocal next_cube
            if k in shove_set:
                a = rng.uniform(0, 2 * math.pi)
                s.d.qvel[s.ath.root_dofadr : s.ath.root_dofadr + 2] += shove_dv * np.array([math.cos(a), math.sin(a)])
                hit_times.append(k * dt)
            if k in cube_set:
                p = s.pelvis()
                j = s.m.joint(f"cube{next_cube % 4}_free")
                qa, da = s.m.jnt_qposadr[j.id], s.m.jnt_dofadr[j.id]
                off = rng.uniform(-0.15, 0.15, 2)
                s.d.qpos[qa : qa + 7] = [p[0] + off[0], p[1] + off[1], p[2] + 0.55 + cube_drop_m, 1, 0, 0, 0]
                s.d.qvel[da : da + 6] = 0
                next_cube += 1
                cube_pending.append(k * dt)

        sim.control_tick(np.zeros(3), pre)
        if cube_pending and sim.cube_touching_athlete():
            hit_times.append(now)
            cube_pending.clear()

        tilt_log.append(sim.torso_tilt_deg())
        qv = np.abs(sim.d.qvel[sim.ath.joint_dofadr])
        joint_over += int(qv.max() > JOINT_VEL_LIMIT)
        max_exc = max(max_exc, float(np.abs(sim.foot_corners_xy() - foot0).max()))
        why = sim.fell()
        if why:
            fell_at, reason = now, why
            break

    tl = np.array(tilt_log)
    hs = sorted(hit_times)
    rec = [recovery_time(tl, h, dt, hs[i + 1] if i + 1 < len(hs) else None) for i, h in enumerate(hs)]
    peak = max((float(tl[int(round(h / dt)): int(round((h + RECOVER_WINDOW_S) / dt))].max(initial=0.0)) for h in hit_times), default=0.0)
    ok = [r for r in rec if r is not None and r <= RECOVER_WINDOW_S]
    return EpisodeResult(seed=seed, fell=fell_at is not None, fall_time=fell_at, fall_reason=reason,
                         hits=len(hit_times), recovered_hits=len(ok),
                         worst_recovery_s=max((r for r in rec if r is not None), default=None) if all(r is not None for r in rec) else None,
                         max_foot_excursion_m=max_exc, joint_vel_over_fraction=joint_over / max(1, len(tilt_log)),
                         max_tilt_after_hit_deg=peak)


def recovery_time(tilt: np.ndarray, hit_t: float, dt: float, next_hit_t: float | None = None) -> float | None:
    """Seconds from the hit until the LAST tick with torso tilt >= 10° before the next hit (0 if never exceeded).
    None if the athlete is still above 10° when the log ends (or the next hit arrives)."""
    start = int(round(hit_t / dt))
    end = len(tilt) if next_hit_t is None else min(len(tilt), int(round(next_hit_t / dt)))
    seg = tilt[start:end]
    over = np.nonzero(seg >= RECOVER_TILT_DEG)[0]
    if len(over) == 0:
        return 0.0
    if over[-1] == len(seg) - 1:
        return None
    return float((over[-1] + 1) * dt)


def rung0_verdict(results: list[EpisodeResult], foot_box_half_m: float = 0.5) -> dict:
    per_seed = []
    for r in results:
        ok = (not r.fell and r.hits > 0 and r.recovered_hits == r.hits and r.max_foot_excursion_m <= foot_box_half_m
              and r.joint_vel_over_fraction <= JOINT_VEL_MAX_FRACTION)
        per_seed.append(ok)
    return {"rung": 0, "seeds": len(results), "passed_seeds": int(sum(per_seed)), "PASS": all(per_seed),
            "per_seed_pass": per_seed}


# ---------------------------------------------------------------------------------------------------------------
# Rung 1 — forward velocity: 30 m dash with heading hold and 0.3 m/s shoves
# ---------------------------------------------------------------------------------------------------------------
HEADING_STIFFNESS = 0.5
HEADING_WZ_LIMIT = 0.5


def heading_yaw(q: np.ndarray) -> float:
    """mjlab heading: yaw of the pelvis x-axis. Unity's heading controller uses exactly this."""
    w, x, y, z = q
    return math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def heading_command(yaw: float, target: float) -> float:
    err = (target - yaw + math.pi) % (2 * math.pi) - math.pi
    return float(np.clip(HEADING_STIFFNESS * err, -HEADING_WZ_LIMIT, HEADING_WZ_LIMIT))


@dataclass
class DashResult:
    seed: int
    speed_cmd: float
    fell: bool
    fall_time: float | None
    fall_reason: str | None
    distance_m: float
    finished: bool
    time_s: float
    vel_rms_err: float
    lateral_drift_m: float
    joint_vel_over_fraction: float


def rung1_episode(onnx_path: Path, seed: int, speed: float | None = None, distance: float = 30.0,
                  shove_dv: float = 0.3, warmup_s: float = 2.0, sim: Sim | None = None) -> DashResult:
    rng = np.random.default_rng(seed)
    speed = float(rng.uniform(0.5, 3.0)) if speed is None else speed
    sim = sim or Sim(onnx_path)
    sim.reset()
    dt = C.DECIMATION * sim.m.opt.timestep
    max_ticks = int((distance / speed + 12.0) / dt)
    start = sim.pelvis()[:2].copy()
    next_shove = rng.uniform(3, 5)
    vel_err, joint_over, lat = [], 0, 0.0
    fell_at, reason = None, None
    k = 0
    for k in range(max_ticks):
        now = k * dt
        q = sim.d.qpos[sim.ath.root_qposadr + 3: sim.ath.root_qposadr + 7]
        cmd = np.array([speed, 0.0, heading_command(heading_yaw(q), 0.0)])

        def pre(s: Sim, now=now):
            nonlocal next_shove
            if now >= next_shove:
                a = rng.uniform(0, 2 * math.pi)
                s.d.qvel[s.ath.root_dofadr: s.ath.root_dofadr + 2] += shove_dv * np.array([math.cos(a), math.sin(a)])
                next_shove = now + rng.uniform(3, 5)

        sim.control_tick(cmd, pre)
        p = sim.pelvis()
        lat = max(lat, abs(p[1] - start[1]))
        if now >= warmup_s:
            qv = sim.d.qvel
            yaw = heading_yaw(sim.d.qpos[sim.ath.root_qposadr + 3: sim.ath.root_qposadr + 7])
            vx_h = math.cos(yaw) * qv[sim.ath.root_dofadr] + math.sin(yaw) * qv[sim.ath.root_dofadr + 1]
            vel_err.append(vx_h - speed)
        joint_over += int(np.abs(sim.d.qvel[sim.ath.joint_dofadr]).max() > JOINT_VEL_LIMIT)
        why = sim.fell()
        if why:
            fell_at, reason = now, why
            break
        if p[0] - start[0] >= distance:
            break
    d = float(sim.pelvis()[0] - start[0])
    return DashResult(seed=seed, speed_cmd=speed, fell=fell_at is not None, fall_time=fell_at, fall_reason=reason,
                      distance_m=d, finished=d >= distance and fell_at is None, time_s=(k + 1) * dt,
                      vel_rms_err=float(np.sqrt(np.mean(np.square(vel_err)))) if vel_err else float("nan"),
                      lateral_drift_m=lat, joint_vel_over_fraction=joint_over / max(1, k + 1))


def rung1_verdict(results: list[DashResult]) -> dict:
    per = [r.finished and not r.fell and r.vel_rms_err < 0.15 and r.lateral_drift_m < 0.5
           and r.joint_vel_over_fraction <= JOINT_VEL_MAX_FRACTION for r in results]
    return {"rung": 1, "seeds": len(results), "passed_seeds": int(sum(per)), "PASS": all(per), "per_seed_pass": per}
