"""C6 / G6 — the 8-lane meet: lane table, lane-local <-> meet mapping, CPU meet rollout (DESIGN.md §4).

The meet scene (assets/scene_meet8.xml) holds 8 copies of the training athlete, each with its own collision bits.
Gate G6: every lane must behave like the same athlete running alone. Plans, disturbances and references are all
written in lane-local (solo-scene) terms; entering the meet maps them by
    root -> L<k>_root,  <hinge> -> L<k>_<hinge>,  cube<i>_free -> cube<cubes[i]>_free,  free-joint xyz += origin(k)
(C# mirror: PolicyRunner.ToLane / AthleteBinding). Solo = the identity lane (prefix "", origin 0, cubes 0..3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from . import contract as C
from .reference import PARITY, Disturbance

MEET_XML = C.ROOT / "assets" / "scene_meet8.xml"
LAYOUT_JSON = C.ROOT / "assets" / "meet8_layout.json"
CUBE_DROP_LOOKAHEAD_TICKS = 25  # 2 kg cube released at z = 3 m reaches head/shoulder height ~0.5 s later


@dataclass(frozen=True)
class Lane:
    lane: int
    prefix: str
    origin: np.ndarray
    cubes: tuple[int, ...]

    def joint(self, solo_name: str) -> str | None:
        """Meet joint name for a solo-scene joint name (None: a solo cube slot this lane does not own)."""
        if solo_name.startswith("cube"):
            i = int(solo_name[4:-len("_free")])
            return f"cube{self.cubes[i]}_free" if i < len(self.cubes) else None
        return self.prefix + solo_name

    def disturbance(self, dist: Disturbance) -> Disturbance:
        target = self.joint(dist.target)
        if target is None:
            raise ValueError(f"lane {self.lane} owns no slot for {dist.target}")
        if dist.kind == "cube":
            q = list(dist.qpos)
            q[0:3] = (np.asarray(q[0:3]) + self.origin).tolist()
            return replace(dist, target=target, qpos=q)
        return replace(dist, target=target)


SOLO = Lane(-1, "", np.zeros(3), (0, 1, 2, 3))


def load_layout() -> list[Lane]:
    lay = json.loads(LAYOUT_JSON.read_text())
    return [Lane(l["lane"], l["prefix"], np.asarray(l["origin"], float), tuple(l["cubes"])) for l in lay["lanes"]]


def default_joint_qpos() -> list[dict]:
    """The contract's default state keyed by solo joint name (what Unity resets from)."""
    return json.loads(C.CONTRACT_JSON.read_text())["default_joint_qpos"]


def reset_lane(m: mujoco.MjModel, d: mujoco.MjData, lane: Lane, defaults: list[dict]) -> None:
    """Mirror of PolicyRunner.ResetToDefault: write this athlete's default joints (root shifted to the lane origin),
    zero its velocities, mj_forward. Cubes are left alone (they start parked at qpos0)."""
    for jq in defaults:
        if jq["joint"].startswith("cube"):
            continue
        j = m.joint(lane.prefix + jq["joint"]).id
        qa, da = m.jnt_qposadr[j], m.jnt_dofadr[j]
        q = np.asarray(jq["qpos"], float)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            q = q.copy()
            q[0:3] += lane.origin
        d.qpos[qa: qa + len(q)] = q
        nv = 6 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        d.qvel[da: da + nv] = 0.0
    mujoco.mj_forward(m, d)


def recorded_joints(m: mujoco.MjModel, lane: Lane) -> list[int]:
    """Joints a lane records: its athlete (bodies under its pelvis) + its cube slots."""
    pelvis = m.body(lane.prefix + "pelvis").id
    cubes = {f"cube{c}_free" for c in lane.cubes}
    return [j for j in range(m.njnt)
            if m.body_rootid[m.jnt_bodyid[j]] == pelvis or m.joint(j).name in cubes]


def _dims(t: int) -> tuple[int, int]:
    return (7, 6) if t == 0 else (4, 3) if t == 1 else (1, 1)


def rollout_meet(onnx_path: Path, plan: list[dict], seconds: float = 5.0) -> dict[int, dict]:
    """All lanes of `plan` (entries: lane, command, disturbances in lane-local terms) in one CPU meet scene.
    Returns per-lane recordings in the Unity run format (PolicyRunner: state at tick start, previous tick's force)."""
    m = mujoco.MjModel.from_xml_path(str(MEET_XML))
    d = mujoco.MjData(m)
    lanes = {l.lane: l for l in load_layout()}
    defaults = default_joint_qpos()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    st = []
    for p in plan:
        lane = lanes[p["lane"]]
        reset_lane(m, d, lane, defaults)
        by_tick: dict[int, list[Disturbance]] = {}
        for x in p["disturbances"]:
            dist = lane.disturbance(Disturbance(**x))
            by_tick.setdefault(dist.tick, []).append(dist)
        rj = recorded_joints(m, lane)
        st.append({"lane": lane, "ath": C.Athlete.bind(m, lane.prefix), "cmd": np.asarray(p["command"], float),
                   "phase": 0.0, "last": np.zeros(C.NUM_ACTIONS), "by_tick": by_tick, "joints": rj, "frames": []})
    n_ticks = int(round(seconds / (m.opt.timestep * C.DECIMATION)))
    for tick in range(n_ticks):
        for s in st:  # control step per lane (lanes only read/write their own state)
            ath = s["ath"]
            qpos, qvel = d.qpos.copy(), d.qvel.copy()
            force_prev = d.actuator_force[ath.actuator_ids].copy()
            s["phase"] = C.advance_phase(s["phase"], s["cmd"])
            obs = C.build_obs(ath, qpos, qvel, s["cmd"], s["phase"], s["last"])
            ctrl, action_raw = sess.run(None, {"obs": obs[None]})
            d.ctrl[ath.actuator_ids] = ctrl[0].astype(np.float64)
            for dist in s["by_tick"].get(tick, []):
                dist.apply(m, d)
            qi = [a for j in s["joints"] for a in range(m.jnt_qposadr[j], m.jnt_qposadr[j] + _dims(m.jnt_type[j])[0])]
            vi = [a for j in s["joints"] for a in range(m.jnt_dofadr[j], m.jnt_dofadr[j] + _dims(m.jnt_type[j])[1])]
            s["frames"].append({"tick": tick, "t": tick * m.opt.timestep * C.DECIMATION, "phase": s["phase"],
                                "qpos": qpos[qi].tolist(), "qvel": qvel[vi].tolist(),
                                "actuator_force_prev": force_prev.tolist(), "obs": obs.astype(np.float64).tolist(),
                                "action_raw": action_raw[0].astype(np.float64).tolist(),
                                "ctrl": ctrl[0].astype(np.float64).tolist()})
            s["last"] = action_raw[0].astype(np.float64)
        for _ in range(C.DECIMATION):
            mujoco.mj_step(m, d)
    out = {}
    for s in st:
        lane, qa, da, joints = s["lane"], 0, 0, []
        for j in s["joints"]:
            nq, nv = _dims(int(m.jnt_type[j]))
            joints.append({"name": m.joint(j).name, "type": int(m.jnt_type[j]), "qposadr": qa, "dofadr": da})
            qa, da = qa + nq, da + nv
        out[lane.lane] = {"meta": {"source": "python_meet", "timestep": m.opt.timestep, "decimation": C.DECIMATION,
                                   "prefix": lane.prefix, "origin": lane.origin.tolist(), "cube_slots": list(lane.cubes),
                                   "joints": joints}, "frames": s["frames"]}
    return out


def write_run(run: dict, name: str) -> Path:
    path = PARITY / f"meet_run_{name}.json"
    path.write_text(json.dumps(run))
    return path
