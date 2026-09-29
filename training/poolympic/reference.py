"""A12 — Golden reference rollouts in CPU MuJoCo (float64, same C library version as Unity). DESIGN.md §4.

The rollout follows the contract tick order exactly:
  read state -> advance phase -> build obs -> infer -> write ctrl -> apply disturbance -> mj_step x DECIMATION
Each frame records the state *read* at the start of the tick plus everything derived from it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from . import bodies
from . import contract as C

PARITY = C.ROOT.parent / "parity"


@dataclass
class Disturbance:
    tick: int
    kind: str  # "shove" (Δqvel on root linear dofs) | "cube" (teleport pooled cube: qpos7 + qvel6)
    target: str  # joint name: "root" or "cube<i>_free"
    dqvel: list[float] | None = None
    qpos: list[float] | None = None
    qvel: list[float] | None = None

    def apply(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        j = m.joint(self.target).id
        qa, da = m.jnt_qposadr[j], m.jnt_dofadr[j]
        if self.kind == "shove":
            d.qvel[da : da + 3] += np.asarray(self.dqvel)
        elif self.kind == "cube":
            d.qpos[qa : qa + 7] = np.asarray(self.qpos)
            d.qvel[da : da + 6] = np.asarray(self.qvel)
        else:
            raise ValueError(self.kind)


def default_disturbances(body: bodies.Body | None = None) -> list[Disturbance]:
    """Standard parity script: lateral 0.5 m/s shove at 1.0 s (other bodies: Froude-scaled), 2 kg cube dropped from
    z = 3 m at 2.0 s."""
    dv = 0.5 if body is None or body.name == "matt" else round(0.5 * body.speed_scale, 4)
    return [
        Disturbance(50, "shove", "root", dqvel=[0.0, dv, 0.0]),
        Disturbance(100, "cube", "cube0_free", qpos=[0.0, 0.2, 3.0, 1.0, 0.0, 0.0, 0.0], qvel=[0.0] * 6),
    ]


@dataclass
class Rollout:
    frames: list[dict] = field(default_factory=list)


def rollout(onnx_path: Path, seconds: float = 5.0, disturbances: list[Disturbance] | None = None,
            command=(0.0, 0.0, 0.0), scene_xml: Path | None = None, body: str | None = None) -> dict:
    """body: athlete body (bodies.BODIES key) to roll out solo — its scene, gait clock and fingerprint; default = the
    process body ($POOLYMPIC_BODY)."""
    b = bodies.BODIES[body] if body else C.BODY
    scene_xml = scene_xml or b.scene_xml
    gait = C.body_gait(json.loads(b.contract_json.read_text())) if b.name != C.BODY.name else None
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
    mujoco.mj_forward(m, d)
    ath = C.Athlete.bind(m)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    disturbances = default_disturbances(b) if disturbances is None else disturbances
    by_tick = {}
    for dist in disturbances:
        by_tick.setdefault(dist.tick, []).append(dist)

    command = np.asarray(command, dtype=float)
    phase = 0.0
    last_action = np.zeros(C.NUM_ACTIONS)
    frames = []
    n_ticks = int(round(seconds / (m.opt.timestep * C.DECIMATION)))
    for tick in range(n_ticks):
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        phase = C.advance_phase(phase, command) if gait is None else C.advance_phase_clock(phase, command, gait)
        obs = C.build_obs(ath, qpos, qvel, command, phase, last_action)
        ctrl, action_raw = sess.run(None, {"obs": obs[None]})
        ctrl64 = ctrl[0].astype(np.float64)
        d.ctrl[ath.actuator_ids] = ctrl64
        for dist in by_tick.get(tick, []):
            dist.apply(m, d)
        for _ in range(C.DECIMATION):
            mujoco.mj_step(m, d)
        frames.append({
            "tick": tick, "t": tick * m.opt.timestep * C.DECIMATION, "phase": phase,
            "qpos": qpos.tolist(), "qvel": qvel.tolist(),
            "obs": obs.astype(np.float64).tolist(), "action_raw": action_raw[0].astype(np.float64).tolist(),
            "ctrl": ctrl64.tolist(), "actuator_force": d.actuator_force[ath.actuator_ids].tolist(),
        })
        last_action = action_raw[0].astype(np.float64)

    fp_sha = b.fingerprint_json.with_suffix(".sha256").read_text().strip()
    onnx_sha = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()
    return {
        "meta": {
            **({} if b.name == "matt" else {"body": b.name}),
            "schema": 1, "mujoco_version": mujoco.__version__, "timestep": m.opt.timestep, "decimation": C.DECIMATION,
            "fingerprint_sha256": fp_sha, "onnx": Path(onnx_path).name, "onnx_sha256": onnx_sha,
            "scene": scene_xml.name, "keyframe": "default", "command": command.tolist(), "seconds": seconds,
            "n_frames": len(frames), "actuators": list(ath.actuator_names),
            "nq": int(m.nq), "nv": int(m.nv),
            "joints": [{"name": m.joint(j).name, "type": int(m.jnt_type[j]), "qposadr": int(m.jnt_qposadr[j]),
                        "dofadr": int(m.jnt_dofadr[j])} for j in range(m.njnt)],
            "note": "qpos/qvel are the full scene state read at the START of each tick (before ctrl/disturbance/steps). "
                    "actuator_force is after the tick's last mj_step.",
        },
        "disturbances": [dist.__dict__ for dist in disturbances],
        "frames": frames,
    }


def write(ref: dict, name: str) -> Path:
    PARITY.mkdir(exist_ok=True)
    path = PARITY / f"reference_trajectory_{name}.json"
    path.write_text(json.dumps(ref))
    return path
