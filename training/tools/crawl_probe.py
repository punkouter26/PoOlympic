"""Event 8 ("30m All Fours") gate helper — crawl from a prone start, CPU MuJoCo, process body ($POOLYMPIC_BODY).

Each seed: standing joint pose, lying face down (the training reset, prone), SETUP_S of zero command to get onto all
fours, then GO: vx command + contract lane keeping until the pelvis has covered DISTANCE (or MAX_S). Reported:
finish time after GO, % of race time on all fours (pelvis in the crawl band, torso tilted > 50°), hands on the ground,
time standing up (pelvis high + torso within 40° of vertical), tumbles (all fours → pelvis below the band) and
recoveries. Heights scale with the body (λ).

Usage: [POOLYMPIC_BODY=zombie] uv run python tools/crawl_probe.py <brain.onnx> [--vx 1.2] [--seeds 5] [--out f.json]
       (--vx in the body's own units; default 1.2 × √λ)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import contract as C  # noqa: E402

LAM = C.BODY.length_scale
BAND = (0.25 * LAM, 0.8 * LAM)       # pelvis height band of a crawl (m)
DISTANCE, SETUP_S, MAX_S = 30.0, 3.0, 60.0


def crawl_yaw(q: np.ndarray) -> float:
    """Heading of mdp.crawl_heading (horizontal projection of pelvis x + z axes): valid on all fours."""
    w, x, y, z = q
    hx = (1 - 2 * (y * y + z * z)) + 2 * (x * z + w * y)
    hy = 2 * (x * y + w * z) + 2 * (y * z - w * x)
    return math.atan2(hy, hx)


def crawl_steer(q: np.ndarray, lane_offset_y: float) -> float:
    """The contract lane-keeping law (contract.steer_yaw_rate) on the crawl heading."""
    target = math.atan(-C.LANE_GAIN * lane_offset_y)
    err = (target - crawl_yaw(q) + math.pi) % (2 * math.pi) - math.pi
    return float(np.clip(C.HEADING_GAIN * err, -C.STEER_WZ_LIMIT, C.STEER_WZ_LIMIT))


def episode(onnx: str, seed: int, vx: float) -> dict:
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
    ath = C.Athlete.bind(m)
    r = ath.root_qposadr
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([rng.uniform(-0.3, 0.3), math.pi / 2 + rng.uniform(-0.2, 0.2), 0.0]), "XYZ")
    d.qpos[r:r + 3] = [0.0, 0.0, 0.22 * LAM]
    d.qpos[r + 3:r + 7] = q
    mujoco.mj_forward(m, d)
    torso = m.body("torso").id
    hands = [m.body(n).id for n in ("forearm_l", "forearm_r")]
    ground = m.geom("ground").id
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    last, phase = np.zeros(C.NUM_ACTIONS), 0.0
    dt = m.opt.timestep * C.DECIMATION
    x0, finish, n, fours, hand, stand, tumbles, recov, was_up = None, None, 0, 0, 0.0, 0, 0, 0, False
    for tick in range(int((SETUP_S + MAX_S) / dt)):
        t = tick * dt
        racing = t >= SETUP_S
        if racing and x0 is None:
            x0 = d.qpos[r]
        cmd = (np.array([vx, 0.0, crawl_steer(d.qpos[r + 3:r + 7], d.qpos[r + 1])]) if racing else np.zeros(3))
        phase = C.advance_phase(phase, cmd)
        obs = C.build_obs(ath, d.qpos, d.qvel, cmd, phase, last)
        ctrl, act = sess.run(None, {"obs": obs[None]})
        d.ctrl[ath.actuator_ids] = ctrl[0]
        last = act[0].astype(np.float64)
        for _ in range(C.DECIMATION):
            mujoco.mj_step(m, d)
        if not racing:
            continue
        z = d.qpos[r + 2]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, d.xmat[torso][8]))))
        on4 = BAND[0] < z < BAND[1] and tilt > 50.0
        n += 1
        fours += on4
        stand += int(z > BAND[1] and tilt < 40.0)
        touch = {int(c.geom1) if int(c.geom2) == ground else int(c.geom2) for c in d.contact[:d.ncon]
                 if ground in (int(c.geom1), int(c.geom2))}
        hand += sum(1 for h in hands if any(m.geom_bodyid[g] == h for g in touch)) / 2
        if was_up and z < BAND[0]:
            tumbles += 1
        if not was_up and on4 and tumbles > recov:
            recov += 1
        was_up = on4
        if d.qpos[r] - x0 >= DISTANCE:
            finish = round(t - SETUP_S, 2)
            break
    return {"seed": seed, "finish_s": finish, "progress_m": float(d.qpos[r] - (x0 or 0.0)),
            "all_fours_frac": fours / max(n, 1), "hands_frac": hand / max(n, 1), "standing_frac": stand / max(n, 1),
            "tumbles": tumbles, "recoveries": recov, "lane_offset_m": float(abs(d.qpos[r + 1]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--vx", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    vx = a.vx if a.vx is not None else 1.2 * math.sqrt(LAM)
    rows = [episode(a.onnx, 4000 + s, vx) for s in range(a.seeds)]
    fin = [x["finish_s"] for x in rows if x["finish_s"] is not None]
    mean = lambda k: np.mean([x[k] for x in rows])
    print(f"{Path(a.onnx).name} vx {vx:.2f}: finished {len(fin)}/{len(rows)}"
          + (f" in {min(fin):.1f}-{max(fin):.1f} s" if fin else f" (progress {mean('progress_m'):.1f} m)")
          + f" | all fours {100 * mean('all_fours_frac'):.0f} % | hands {100 * mean('hands_frac'):.0f} %"
          f" | standing {100 * mean('standing_frac'):.0f} % | tumbles {sum(x['tumbles'] for x in rows)}"
          f" recovered {sum(x['recoveries'] for x in rows)} | lane {mean('lane_offset_m'):.2f} m")
    if a.out:
        Path(a.out).write_text(json.dumps({"onnx": Path(a.onnx).name, "vx": vx, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
