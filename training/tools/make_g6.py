"""C6 — Gate G6 artifacts for one brain: lane plan, solo references, CPU meet pre-check (DESIGN.md §4).

Each of the 8 lanes gets its own command and a lane-local disturbance script (lateral shove, then a 2 kg cube dropped
where the athlete will be ~0.5 s later — found by a shove-only pre-roll, so it lands on a moving athlete too).
Writes:
  parity/g6_plan_<name>.json                      lanes: command, disturbances (lane-local), reference name
  parity/reference_trajectory_<name>_L<k>.json    solo CPU-MuJoCo reference per lane (also usable for G2-G5)
  parity/meet_run_<name>_L<k>.json                all 8 lanes run together in the CPU meet scene
  parity/g6_report_<name>_python.json             each meet lane vs its solo run (G5 criteria)
Unity then runs the same plan in Testbed_Rung1 (PoOlympic › Parity › G6) and tools/compare_g6.py scores it.

Usage: uv run python tools/make_g6.py <brain.onnx> <name>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import meet  # noqa: E402
from poolympic.closed_loop import g5_compare  # noqa: E402
from poolympic.reference import PARITY, Disturbance, rollout, write  # noqa: E402

SECONDS = 5.0
# (vx, vy, wz) per lane — float32-exact values so C# (float Vector3) and Python (float64) see identical commands
LANE_COMMANDS = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0),
                 (2.0, 0.0, 0.0), (2.5, 0.0, 0.0), (3.0, 0.0, 0.0), (1.75, 0.0, 0.25)]
SHOVE_DV = 0.3


def lane_script(onnx: Path, k: int, command) -> list[Disturbance]:
    shove = Disturbance(40 + 5 * k, "shove", "root", dqvel=[0.0, SHOVE_DV if k % 2 == 0 else -SHOVE_DV, 0.0])
    cube_tick = 100 + 5 * k
    hit_tick = cube_tick + meet.CUBE_DROP_LOOKAHEAD_TICKS
    pre = rollout(onnx, seconds=(hit_tick + 1) * 0.02, disturbances=[shove], command=command)
    x, y = pre["frames"][hit_tick]["qpos"][0:2]
    cube = Disturbance(cube_tick, "cube", "cube0_free", qpos=[round(x + 0.05, 4), round(y + 0.15, 4), 3.0, 1.0, 0.0, 0.0, 0.0],
                       qvel=[0.0] * 6)
    return [shove, cube]


def main(onnx_arg: str, name: str) -> int:
    onnx = Path(onnx_arg)
    lanes = []
    for k, cmd in enumerate(LANE_COMMANDS):
        script = lane_script(onnx, k, cmd)
        ref = rollout(onnx, seconds=SECONDS, disturbances=script, command=cmd)
        write(ref, f"{name}_L{k}")
        lanes.append({"lane": k, "command": list(cmd), "disturbances": [d.__dict__ for d in script],
                      "reference": f"{name}_L{k}"})
        fell = next((f["t"] for f in ref["frames"] if f["qpos"][2] < 0.55), None)
        dist = ref["frames"][-1]["qpos"][0]
        print(f"L{k} cmd={cmd}: solo ref x_end={dist:.2f} m, fell={fell}")
    plan = {"schema": 1, "brain": onnx.name, "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
            "fingerprint_sha256": (PARITY / "fingerprint_python.sha256").read_text().strip(),
            "seconds": SECONDS, "scene": meet.MEET_XML.name, "lanes": lanes}
    (PARITY / f"g6_plan_{name}.json").write_text(json.dumps(plan, indent=1))

    runs = meet.rollout_meet(onnx, lanes, SECONDS)
    report = {"brain": onnx.name, "source": "python_meet", "lanes": {}}
    for k, run in runs.items():
        meet.write_run(run, f"{name}_L{k}")
        ref = json.loads((PARITY / f"reference_trajectory_{name}_L{k}.json").read_text())
        r = g5_compare(ref, run, f"{name}_L{k}", f"meet_{name}_L{k}")
        report["lanes"][f"L{k}"] = r
        print(f"L{k}: meet vs solo  drift1s={r['qpos_max_abs_drift_1s']:.1e} drift5s={r['qpos_max_abs_drift_5s']:.1e} "
              f"height_rms={r['pelvis_height_rms_m']:.1e} torque={r['torque_ratio']:.4f} -> {'PASS' if r['G5_pass'] else 'FAIL'}")
    report["G6_pass"] = all(r["G5_pass"] for r in report["lanes"].values())
    (PARITY / f"g6_report_{name}_python.json").write_text(json.dumps(report, indent=1))
    print("G6 (CPU meet pre-check)", "PASS" if report["G6_pass"] else "FAIL")
    return 0 if report["G6_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
