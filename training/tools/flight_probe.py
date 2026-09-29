"""Event 13 gate helper — flight phases of a MATT brain running straight (CPU MuJoCo, contract lane keeping).

For each speed: fraction of steady-state time with no foot on the ground, flights (both feet airborne for ≥ MIN_FLIGHT
so contact chatter does not count), mean / max flight, flights per second, falls. Steady state = after 3 s.

Usage: uv run python tools/flight_probe.py <brain.onnx> [--speeds 2.5,3.0,3.5,4.0] [--seconds 10] [--out file.json]
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

MIN_FLIGHT = 0.02      # s
STEADY_S = 3.0


def probe(onnx: str, vx: float, seconds: float) -> dict:
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("default").id)
    mujoco.mj_forward(m, d)
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    ath = C.Athlete.bind(m)
    feet = {m.geom(n).id for n in ("foot_l_geom0", "toe_l_geom0", "foot_r_geom0", "toe_r_geom0")}
    ground = m.geom("ground").id
    phase, last = 0.0, np.zeros(C.NUM_ACTIONS)
    air, flights, cur, fell = [], [], 0, None
    x0 = None
    for tick in range(int(seconds / (m.opt.timestep * C.DECIMATION))):
        r = ath.root_qposadr
        cmd = np.array([vx, 0.0, C.steer_yaw_rate(d.qpos[r + 3:r + 7], d.qpos[r + 1], vx)])
        phase = C.advance_phase(phase, cmd)
        obs = C.build_obs(ath, d.qpos, d.qvel, cmd, phase, last)
        ctrl, act = sess.run(None, {"obs": obs[None]})
        d.ctrl[ath.actuator_ids] = ctrl[0]
        last = act[0].astype(np.float64)
        t = tick * m.opt.timestep * C.DECIMATION
        if t >= STEADY_S and x0 is None:
            x0 = d.qpos[0]
        for _ in range(C.DECIMATION):
            mujoco.mj_step(m, d)
            on = any((int(c.geom1) in feet and int(c.geom2) == ground) or (int(c.geom2) in feet and int(c.geom1) == ground)
                     for c in d.contact[:d.ncon])
            if t >= STEADY_S:
                air.append(not on)
                if not on:
                    cur += 1
                else:
                    if cur * m.opt.timestep >= MIN_FLIGHT:
                        flights.append(cur * m.opt.timestep)
                    cur = 0
        if fell is None and d.qpos[2] < 0.55:
            fell = round(t, 2)
    steady = seconds - STEADY_S
    fl = np.array(flights) if flights else np.zeros(0)
    return {"vx": vx, "fell": fell, "speed": float((d.qpos[0] - x0) / steady) if x0 is not None else None,
            "airborne_fraction": float(np.mean(air)) if air else 0.0, "flights_per_s": len(fl) / steady,
            "flight_mean_ms": float(1000 * fl.mean()) if len(fl) else 0.0, "flight_max_ms": float(1000 * fl.max()) if len(fl) else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--speeds", default="2.5,3.0,3.5,4.0")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [probe(a.onnx, float(v), a.seconds) for v in a.speeds.split(",")]
    for r in rows:
        print(f"vx {r['vx']}: speed {r['speed']:.2f} m/s, fell {r['fell']}, airborne {100 * r['airborne_fraction']:.1f} %, "
              f"{r['flights_per_s']:.2f} flights/s, mean {r['flight_mean_ms']:.0f} ms, max {r['flight_max_ms']:.0f} ms")
    if a.out:
        Path(a.out).write_text(json.dumps({"onnx": Path(a.onnx).name, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
