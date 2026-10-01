"""Event 1 — The Iron Pedestal, 8-runner heat (tasks.md Phase E / D4). Mirror of Unity IronPedestalHeat.

Rules: 8 runners on 1 m x 1 m pedestals (assets/scene_pedestal8.xml, lanes from the stadium venue). Every round
(ROUND_S) each runner gets a gust — same magnitude for everyone, its own seeded direction — and every
CUBE_EVERY-th round a 2 kg cube dropped from 1.5 m above the shoulder. Gust magnitude escalates each round. A runner is
out when it falls (DESIGN §1 fall rule) or steps off (a foot below the pedestal top). Last one standing wins; ranking =
elimination order (later = better); anyone still standing at MAX_S shares first place.

Athlete traits (contract.TRAIT_RANGES): strength scales the lane's actuator force limits, latency delays its ctrl by
0-4 physics substeps (mjlab XmlActuator delay semantics), obs_noise scales the training observation noise.
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

SCENE = C.ROOT / "assets" / "scene_pedestal8.xml"
LAYOUT = C.ROOT / "assets" / "pedestal8_layout.json"
ROUND_S = 3.0
GUST_START = 0.3      # m/s — tuned for ~35 s heats (first out ~20 s), see rl_optimization_log.md
GUST_STEP = 0.05      # m/s per round
CUBE_EVERY = 3
CUBE_DROP_M = 1.5
COUNTDOWN_S = 1.0     # settle before the first gust
MAX_S = 120.0
FALL_Z, FALL_TILT_DEG, STEPPED_OFF_Z = 0.55, 60.0, -0.06


@dataclass
class Traits:
    strength: float = 1.0
    latency_substeps: int = 0
    obs_noise: float = 0.0

    @staticmethod
    def sample(rng: np.random.Generator) -> "Traits":
        r = C.TRAIT_RANGES
        return Traits(float(rng.uniform(*r["strength"])), int(rng.integers(r["latency_substeps"][0], r["latency_substeps"][1] + 1)),
                      float(rng.uniform(*r["obs_noise"])))


@dataclass
class LaneResult:
    lane: int
    traits: Traits
    out_at_s: float | None
    reason: str | None
    gusts_survived: int
    place: int = 0
    body: str = "matt"


@dataclass
class HeatResult:
    seed: int
    duration_s: float
    lanes: list[LaneResult] = field(default_factory=list)

    @property
    def winner(self) -> list[int]:
        return [l.lane for l in self.lanes if l.place == 1]


class _Lane:
    def __init__(self, m, d, k: int, prefix: str, origin: np.ndarray, traits: Traits, rng: np.random.Generator,
                 fall_z: float = FALL_Z, gait: tuple[float, float, float] | None = None, body: str = "matt"):
        self.k, self.prefix, self.origin, self.traits, self.rng = k, prefix, origin, traits, rng
        self.fall_z = fall_z          # per body: the same fraction of its standing pelvis height as MATT's 0.55 m
        self.body = body
        # gait clock (base Hz, Hz per m/s, yaw weight) of this lane's body; default = the process body's contract
        self.gait = gait or (C.GAIT_HZ_BASE, C.GAIT_HZ_PER_MPS, C.GAIT_HZ_YAW_WEIGHT)
        self.sess = None              # this lane's brain (make_lanes)
        self.ath = C.Athlete.bind(m, prefix)
        self.torso = m.body(prefix + "torso").id
        self.pelvis = m.body(prefix + "pelvis").id
        self.feet = [m.geom(prefix + n).id for n in ("foot_l_geom0", "toe_l_geom0", "foot_r_geom0", "toe_r_geom0")]
        names = {m.geom(g).name for g in range(m.ngeom)}
        # own pedestal / shaker platform, or the one every lane shares (crowd events: beam `pedestal`, floor `shaker`)
        self.support = {m.geom("ground").id} | {m.geom(n).id for n in (prefix + "pedestal", prefix + "shaker", "pedestal", "shaker")
                                               if n in names}
        self.phase = 0.0
        self.skill: np.ndarray | None = None   # contract v4 skill block (Rung S brains, event 3); None = v3 obs
        self.last = np.zeros(C.NUM_ACTIONS)
        self.ctrl_now = self.ath.default_pos.copy()
        self.ctrl_prev = self.ath.default_pos.copy()
        self.out_at, self.reason, self.gusts = None, None, 0
        lo = m.actuator_forcerange[self.ath.actuator_ids]
        m.actuator_forcerange[self.ath.actuator_ids] = lo * traits.strength
        self.noise = np.zeros(C.OBS_DIM)
        layout = json.loads(C.CONTRACT_JSON.read_text())["obs_layout"]
        for term, amp in C.OBS_NOISE.items():
            o = layout[term]
            self.noise[o["offset"]: o["offset"] + o["size"]] = amp * traits.obs_noise

    def reset(self, m, d, defaults):
        for jq in defaults:
            if jq["joint"].startswith("cube"):
                continue
            j = m.joint(self.prefix + jq["joint"]).id
            q = np.asarray(jq["qpos"], float).copy()
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                q[0:3] += self.origin
            d.qpos[m.jnt_qposadr[j]: m.jnt_qposadr[j] + len(q)] = q
            nv = 6 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
            d.qvel[m.jnt_dofadr[j]: m.jnt_dofadr[j] + nv] = 0

    def control(self, sess, d, cmd=None):
        """cmd in MATT units: scaled to this lane's body like Unity PolicyRunner.BodyCommand (identity for MATT)."""
        cmd = np.zeros(3) if cmd is None else body_command(np.asarray(cmd, float), self.body)
        self.phase = self.advance_phase(cmd)
        obs = C.build_obs(self.ath, d.qpos, d.qvel, cmd, self.phase, self.last, self.skill)
        if self.traits.obs_noise > 0:     # the skill block is a command: no noise (as in training)
            obs = obs.copy()
            obs[:C.OBS_DIM] = (obs[:C.OBS_DIM] + self.rng.uniform(-1, 1, C.OBS_DIM) * self.noise).astype(np.float32)
        ctrl, act = sess.run(None, {"obs": obs[None]})
        self.ctrl_prev, self.ctrl_now = self.ctrl_now, ctrl[0].astype(np.float64)
        self.last = act[0].astype(np.float64)

    def advance_phase(self, cmd: np.ndarray) -> float:
        """contract.advance_phase with this lane's body clock (same arithmetic: MATT lanes are bit-identical). v4: a march
        cadence > 0 in the skill block drives the clock whatever the command (event 7)."""
        if self.skill is not None and self.skill[3] > 0.0:
            return (self.phase + float(self.skill[3]) * C.DECIMATION * 0.005) % 1.0
        if float(np.linalg.norm(cmd)) < C.PHASE_CMD_THRESHOLD:
            return 0.0
        base, per_mps, yaw_w = self.gait
        hz = base + per_mps * (math.hypot(float(cmd[0]), float(cmd[1])) + yaw_w * abs(float(cmd[2])))
        return (self.phase + hz * C.DECIMATION * 0.005) % 1.0

    def write_ctrl(self, d, substep_in_tick: int):
        # ctrl computed at a tick reaches the actuators `latency` substeps later (mjlab delay semantics)
        use_now = substep_in_tick >= self.traits.latency_substeps
        d.ctrl[self.ath.actuator_ids] = self.ctrl_now if use_now else self.ctrl_prev

    def eliminated(self, m, d) -> str | None:
        r = self.ath.root_qposadr
        if d.qpos[r + 2] < self.fall_z:
            return "FELL"
        if math.degrees(math.acos(max(-1.0, min(1.0, d.xmat[self.torso][8])))) > FALL_TILT_DEG:
            return "FELL"
        if any(d.geom_xpos[g][2] < STEPPED_OFF_Z for g in self.feet):
            return "STEPPED OFF"
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 in self.support else g1 if g2 in self.support else None
            if other is not None and other not in self.feet and m.body_rootid[m.geom_bodyid[other]] == self.pelvis:
                return "FELL"
        return None


def body_command(cmd: np.ndarray, body: str) -> np.ndarray:
    """PolicyRunner.BodyCommand in float32: k = (float)(0.8 / gait_hz_base); (min(x·k, vx_max), y·k, z / k).
    Identity for MATT."""
    if body == "matt":
        return cmd
    ct = body_contract(body)
    k = np.float32(0.8 / ct["gait_hz_base"])
    c = np.asarray(cmd, np.float32)
    vx = c[0] * k
    if "vx_max" in ct:                 # the body's own top speed (GRANDMA 2.8 m/s)
        vx = min(vx, np.float32(ct["vx_max"]))
    return np.array([vx, c[1] * k, c[2] / k], dtype=np.float64)


def body_contract(body: str) -> dict:
    from .. import bodies
    return json.loads(bodies.BODIES[body].contract_json.read_text())


def make_lanes(m, d, layout: dict, seed: int, traits: list[Traits], onnx, brains: dict | None = None) -> list["_Lane"]:
    """The 8 lanes of an event scene, each with its own body (layout lane "body", default matt), brain (brains[body],
    default `onnx`), default pose, fall line and gait clock; reset to the default pose."""
    lanes, sessions = [], {}
    for i, l in enumerate(layout["lanes"]):
        body = l.get("body", "matt")
        ct = body_contract(body)
        ln = _Lane(m, d, l["lane"], l["prefix"], np.asarray(l["origin"], float), traits[i],
                   np.random.default_rng([seed, 1000 + i]), fall_z=body_fall_z(body),
                   gait=(ct["gait_hz_base"], ct["gait_hz_per_mps"], ct["gait_hz_yaw_weight"]), body=body)
        ln.reset(m, d, ct["default_joint_qpos"])
        if body not in sessions:
            sessions[body] = ort.InferenceSession(str((brains or {}).get(body, onnx)), providers=["CPUExecutionProvider"])
        ln.sess = sessions[body]
        lanes.append(ln)
    return lanes


def body_fall_z(body: str) -> float:
    """MATT's 0.55 m pelvis fall line at the same fraction of the body's standing pelvis height."""
    from .. import bodies
    if body == "matt":
        return FALL_Z
    z = float(mujoco.MjModel.from_xml_path(str(bodies.BODIES[body].scene_xml)).key("default").qpos[2])
    matt = float(mujoco.MjModel.from_xml_path(str(bodies.BODIES["matt"].scene_xml)).key("default").qpos[2])
    return FALL_Z * z / matt


def run_heat(onnx: Path, seed: int, traits: list[Traits] | None = None, max_s: float = MAX_S,
             scene: Path = SCENE, layout_path: Path = LAYOUT, brains: dict[str, Path] | None = None) -> HeatResult:
    """brains: body -> ONNX for multi-body heats (layout lanes carry "body", tools/compose_mixed.py); default: `onnx`
    for every lane."""
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    layout = json.loads(Path(layout_path).read_text())
    lane_body = [l.get("body", "matt") for l in layout["lanes"]]
    rng = np.random.default_rng(seed)
    traits = traits or [Traits.sample(rng) for _ in range(len(layout["lanes"]))]
    lanes = make_lanes(m, d, layout, seed, traits, onnx, brains)
    mujoco.mj_forward(m, d)
    dt_tick = m.opt.timestep * C.DECIMATION
    n_cubes = layout["n_cubes"]
    next_cube, rnd = 0, 0
    tick, t = 0, 0.0
    next_round = COUNTDOWN_S
    while t < max_s and sum(ln.out_at is None for ln in lanes) > 1:
        for ln in lanes:
            if ln.out_at is None:
                ln.control(ln.sess, d)
        if t >= next_round:                      # gust round: same magnitude, own direction, cube every other round
            dv = GUST_START + GUST_STEP * rnd
            for ln in lanes:
                if ln.out_at is not None:
                    continue
                a = rng.uniform(0, 2 * math.pi)
                da = m.jnt_dofadr[m.joint(ln.prefix + "root").id]
                d.qvel[da: da + 2] += dv * np.array([math.cos(a), math.sin(a)])
                ln.gusts += 1
                if rnd % CUBE_EVERY == CUBE_EVERY - 1:
                    j = m.joint(f"cube{next_cube % n_cubes}_free").id
                    next_cube += 1
                    p = d.xpos[ln.pelvis]
                    d.qpos[m.jnt_qposadr[j]: m.jnt_qposadr[j] + 7] = [p[0], p[1] + 0.2, p[2] + 0.6 + CUBE_DROP_M, 1, 0, 0, 0]
                    d.qvel[m.jnt_dofadr[j]: m.jnt_dofadr[j] + 6] = 0
            rnd += 1
            next_round += ROUND_S
        for s in range(C.DECIMATION):
            for ln in lanes:
                ln.write_ctrl(d, s)
            mujoco.mj_step(m, d)
        tick += 1
        t = tick * dt_tick
        for ln in lanes:
            if ln.out_at is None and t >= COUNTDOWN_S:
                why = ln.eliminated(m, d)
                if why:
                    ln.out_at, ln.reason = t, why
    res = HeatResult(seed=seed, duration_s=t)
    order = sorted(lanes, key=lambda ln: -(ln.out_at if ln.out_at is not None else math.inf))
    place = 1
    for i, ln in enumerate(order):
        if i > 0 and not (ln.out_at is None and order[i - 1].out_at is None) and ln.out_at != order[i - 1].out_at:
            place = i + 1
        res.lanes.append(LaneResult(ln.k, ln.traits, ln.out_at, ln.reason, ln.gusts, place, lane_body[ln.k]))
    res.lanes.sort(key=lambda r: r.lane)
    return res


def to_json(res: HeatResult) -> dict:
    return {"seed": res.seed, "duration_s": res.duration_s, "winner": res.winner,
            "lanes": [{**asdict(l), "traits": asdict(l.traits)} for l in res.lanes]}
