"""Bio-realism audit of a brain (CPU MuJoCo, contract tick order) — how human is the motion, not whether it passes G1.

Per drill (stand / walk / run / sprint / spin, steady state after 3 s):
  grf_peak_bw     p95 and max of the per-foot vertical ground reaction, in body weights (human: walk ~1.2, run ~2.5-3)
  power_w / cot   mean mechanical power Σ|τ·q̇| and cost of transport P / (m g v)
  hill_viol       % of actuator-frames doing concentric work above the Hill torque-velocity envelope (mdp.HILL_W_MAX)
  cap_sat         % of actuator-frames at the torque cap
  qd_peak         peak joint speed per group (rad/s); vel18 = % of frames with any joint > 18 rad/s
  bio_cap_over    % of frames a joint exceeds the staged mattbio caps (bodies.BIO_TORQUE_CAPS)
  clip_cm / clip% worst arm-through-body penetration (arms do not collide with the trunk / legs on MATT today) and the
                  share of frames with > 1 cm of it — should be 0 for a realistic athlete
Usage: uv run python tools/bio_probe.py <brain.onnx> [--seconds 8] [--out file.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import contract as C  # noqa: E402
from poolympic.bodies import BIO_TORQUE_CAPS  # noqa: E402
from poolympic.tasks.mdp import HILL_W_MAX  # noqa: E402

STEADY_S = 3.0
DRILLS = {"stand": (0.0, 0.0, 0.0), "walk_1.2": (1.2, 0.0, 0.0), "run_3.0": (3.0, 0.0, 0.0),
          "sprint_3.8": (3.8, 0.0, 0.0), "spin_2.5": (0.0, 0.0, 2.5)}
ARMS = ("upper_arm_l", "forearm_l", "upper_arm_r", "forearm_r")
TRUNK_LEGS = ("pelvis", "torso", "chest", "head", "thigh_l", "thigh_r", "shin_l", "shin_r")


def geoms_of(m, body: str) -> list[int]:
    b = m.body(body).id
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]


def probe(onnx: str, cmd: tuple, seconds: float) -> dict:
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
    mujoco.mj_forward(m, d)
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    ath = C.Athlete.bind(m)
    bw = float(sum(m.body_mass[m.body(n).id] for n in ath_bodies(m))) * 9.81
    mass = bw / 9.81
    ground = m.geom("ground").id
    foot = {side: {m.geom(f"foot_{side}_geom0").id, m.geom(f"toe_{side}_geom0").id} for side in "lr"}
    cap = np.array([max(abs(m.actuator_forcerange[a][0]), abs(m.actuator_forcerange[a][1])) for a in ath.actuator_ids])
    wmax = np.array([HILL_W_MAX[next(g for g in HILL_W_MAX if m.actuator(a).name.startswith(g))] for a in ath.actuator_ids])
    dof = np.array([m.jnt_dofadr[m.actuator_trnid[a][0]] for a in ath.actuator_ids])
    groups = {g: [i for i, a in enumerate(ath.actuator_ids) if m.actuator(a).name.startswith(g)] for g in HILL_W_MAX}
    pairs = [(ga, gb) for a in ARMS for b in TRUNK_LEGS if not (a.startswith("upper_arm") and b == "chest")
             for ga in geoms_of(m, a) for gb in geoms_of(m, b)]
    names = [m.actuator(a).name for a in ath.actuator_ids]
    base = [n[:-2] if n.endswith(("_l", "_r")) else n for n in names]
    bio_lo = np.array([-BIO_TORQUE_CAPS[b][0] for b in base])
    bio_hi = np.array([BIO_TORQUE_CAPS[b][1] for b in base])
    over = np.zeros(len(names))
    fromto = np.zeros(6)
    phase, last = 0.0, np.zeros(C.NUM_ACTIONS)
    grf, power, hill, sat, qdmax, v18, clip = [], [], [], [], np.zeros(len(dof)), [], []
    x0 = fell = None
    cmdv = np.array(cmd, float)
    n_ticks = int(seconds / (m.opt.timestep * C.DECIMATION))
    for tick in range(n_ticks):
        t = tick * m.opt.timestep * C.DECIMATION
        phase = C.advance_phase(phase, cmdv)
        obs = C.build_obs(ath, d.qpos, d.qvel, cmdv, phase, last)
        ctrl, act = sess.run(None, {"obs": obs[None]})
        d.ctrl[ath.actuator_ids] = ctrl[0]
        last = act[0].astype(np.float64)
        if t >= STEADY_S and x0 is None:
            x0 = d.qpos[:2].copy()
        for _ in range(C.DECIMATION):
            mujoco.mj_step(m, d)
            if t < STEADY_S:
                continue
            fz = {"l": 0.0, "r": 0.0}
            f6 = np.zeros(6)
            for i, c in enumerate(d.contact[:d.ncon]):
                other = c.geom2 if c.geom1 == ground else c.geom1 if c.geom2 == ground else None
                for side in "lr":
                    if other in foot[side]:
                        mujoco.mj_contactForce(m, d, i, f6)
                        fz[side] += abs(f6[0] * c.frame[2])      # normal force, vertical part
            grf.extend([fz["l"] / bw, fz["r"] / bw])
            tau = d.actuator_force[ath.actuator_ids]
            qd = d.qvel[dof]
            power.append(float(np.abs(tau * qd).sum()))
            allowed = cap * np.clip(1 - np.abs(qd) / wmax, 0, None)
            hill.append(np.mean((np.abs(tau) > allowed + 1e-6) & (tau * qd > 0)))
            sat.append(np.mean(np.abs(tau) >= cap - 1e-6))
            qdmax = np.maximum(qdmax, np.abs(qd))
            v18.append(bool((np.abs(qd) > 18.0).any()))
            over += (tau < bio_lo) | (tau > bio_hi)
        if t >= STEADY_S and tick % 2 == 0:
            worst = 0.0
            for ga, gb in pairs:
                dist = mujoco.mj_geomDistance(m, d, ga, gb, 0.05, fromto)
                worst = max(worst, -dist)
            clip.append(worst)
        if fell is None and d.qpos[2] < 0.55:
            fell = round(t, 2)
    steady = seconds - STEADY_S
    v = float(np.linalg.norm(d.qpos[:2] - x0) / steady) if x0 is not None else 0.0
    grf = np.array(grf)
    clip = np.array(clip)
    return {"cmd": cmd, "fell": fell, "speed": round(v, 2),
            "grf_peak_bw_p95": round(float(np.percentile(grf, 95)), 2), "grf_peak_bw_max": round(float(grf.max()), 2),
            "power_w": round(float(np.mean(power))), "cot": round(float(np.mean(power)) / (mass * 9.81 * v), 2) if v > 0.2 else None,
            "hill_viol_pct": round(100 * float(np.mean(hill)), 2), "cap_sat_pct": round(100 * float(np.mean(sat)), 2),
            "vel18_pct": round(100 * float(np.mean(v18)), 2),
            "qd_peak": {g: round(float(qdmax[ix].max()), 1) for g, ix in groups.items() if ix},
            "bio_cap_over_pct": {b: round(100 * float(x) / max(len(power), 1), 1) for b, x in
                                 sorted(_merge(base, over).items(), key=lambda kv: -kv[1]) if x > 0},
            "clip_cm": round(100 * float(clip.max()), 1), "clip_pct": round(100 * float(np.mean(clip > 0.01)), 1)}


def _merge(base, over):
    out = {}
    for b, x in zip(base, over):
        out[b] = max(out.get(b, 0.0), float(x))
    return out


def ath_bodies(m):
    root = m.body("pelvis").id
    return [m.body(b).name for b in range(m.nbody) if b == root or _under(m, b, root)]


def _under(m, b, root):
    while b > 0:
        b = m.body_parentid[b]
        if b == root:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--drills", default=",".join(DRILLS))
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = {k: probe(a.onnx, DRILLS[k], a.seconds) for k in a.drills.split(",")}
    for k, r in rows.items():
        print(f"{k:11s} speed {r['speed']:.2f}  GRF p95 {r['grf_peak_bw_p95']:.2f} max {r['grf_peak_bw_max']:.2f} BW  "
              f"P {r['power_w']} W  CoT {r['cot']}  hill {r['hill_viol_pct']} %  sat {r['cap_sat_pct']} %  "
              f">18 rad/s {r['vel18_pct']} %  arm clip {r['clip_cm']} cm ({r['clip_pct']} % frames)  fell {r['fell']}")
        if r['bio_cap_over_pct']:
            print(f"{'':11s} frames above the staged mattbio caps: {r['bio_cap_over_pct']}")
    if a.out:
        Path(a.out).write_text(json.dumps({"onnx": Path(a.onnx).name, "drills": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
