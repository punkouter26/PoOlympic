"""R3 gate helper — get-up from supine (Event 27, The Resurrection Dash), CPU MuJoCo, zero command.

Each seed: the default standing joint pose, pelvis at 0.22 m lying on the back (pitch -90° ± 11°, roll ± 17°, random
yaw — the training reset), then the brain for 10 s. "Up" = pelvis above 0.85 m and torso within 20° of vertical,
held for 1 s; reported: time to first reach that state, still up at the end, falls after being up.

Usage: uv run python tools/getup_probe.py <brain.onnx> [--seeds 10] [--out file.json]
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

UP_Z, UP_TILT_DEG, HOLD_S, SECONDS = 0.85, 20.0, 1.0, 10.0


def episode(onnx: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
    ath = C.Athlete.bind(m)
    r = ath.root_qposadr
    roll, pitch, yaw = rng.uniform(-0.3, 0.3), -math.pi / 2 + rng.uniform(-0.2, 0.2), rng.uniform(-math.pi, math.pi)
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "XYZ")
    d.qpos[r:r + 3] = [0.0, 0.0, 0.22]
    d.qpos[r + 3:r + 7] = q
    mujoco.mj_forward(m, d)
    torso = m.body("torso").id
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    last, cmd = np.zeros(C.NUM_ACTIONS), np.zeros(3)
    dt = m.opt.timestep * C.DECIMATION
    up_since, first_up, fell_after = None, None, False
    for tick in range(int(SECONDS / dt)):
        obs = C.build_obs(ath, d.qpos, d.qvel, cmd, 0.0, last)
        ctrl, act = sess.run(None, {"obs": obs[None]})
        d.ctrl[ath.actuator_ids] = ctrl[0]
        last = act[0].astype(np.float64)
        for _ in range(C.DECIMATION):
            mujoco.mj_step(m, d)
        t = (tick + 1) * dt
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, d.xmat[torso][8]))))
        up = d.qpos[r + 2] > UP_Z and tilt < UP_TILT_DEG
        if up:
            up_since = t if up_since is None else up_since
            if first_up is None and t - up_since >= HOLD_S:
                first_up = round(up_since, 2)
        else:
            up_since = None
            if first_up is not None and d.qpos[r + 2] < 0.55:
                fell_after = True
    return {"seed": seed, "time_to_up_s": first_up, "up_at_end": bool(up_since is not None), "fell_after_up": fell_after}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [episode(a.onnx, 3000 + s) for s in range(a.seeds)]
    ups = [x["time_to_up_s"] for x in rows if x["time_to_up_s"] is not None]
    print(f"{Path(a.onnx).name}: up {len(ups)}/{len(rows)}, time to up {min(ups) if ups else '-'}..{max(ups) if ups else '-'} s"
          f" (mean {np.mean(ups):.2f})" if ups else f"{Path(a.onnx).name}: up 0/{len(rows)}",
          f"| up at end {sum(x['up_at_end'] for x in rows)}/{len(rows)} | fell after up {sum(x['fell_after_up'] for x in rows)}")
    if a.out:
        Path(a.out).write_text(json.dumps({"onnx": Path(a.onnx).name, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
