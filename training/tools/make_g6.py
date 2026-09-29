"""C6 — Gate G6 artifacts for one brain: lane plan, solo references, CPU meet pre-check (DESIGN.md §4).

Each of the 8 lanes gets its own command and a lane-local disturbance script (lateral shove, then a 2 kg cube dropped
where the athlete will be ~0.5 s later — found by a shove-only pre-roll, so it lands on a moving athlete too).
Writes:
  parity/g6_plan_<name>.json                      lanes: command, disturbances (lane-local), reference name
  parity/reference_trajectory_<name>_L<k>.json    solo CPU-MuJoCo reference per lane (also usable for G2-G5)
  parity/meet_run_<name>_L<k>.json                all 8 lanes run together in the CPU meet scene
  parity/g6_report_<name>_python.json             each meet lane vs its solo run (G5 criteria)
Unity then runs the same plan in Testbed_Rung1 (PoOlympic › Parity › G6) and tools/compare_g6.py scores it.

Usage: uv run python tools/make_g6.py <brain.onnx> <name> [rung1|rung2]   (lane command set)
       uv run python tools/make_g6.py mixed <name> [matt_brain.onnx zombie_brain.onnx]
           Phase Z7: mixed meet scene_meet8_mzmzmzmz.xml (MATT lanes 0/2/4/6, zombie 1/3/5/7), Rung 2 commands per lane in
           MATT units (the zombie's are scaled exactly like PolicyRunner.BodyCommand), each lane vs a solo run of its own
           body; Unity runs it in Testbed_Mixed (MeetTestbed.ConfigureG6).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from poolympic import bodies, meet  # noqa: E402
from poolympic.events.iron_pedestal import body_fall_z  # noqa: E402
from poolympic.closed_loop import g5_compare  # noqa: E402
from poolympic.reference import PARITY, Disturbance, rollout, write  # noqa: E402

SECONDS = 5.0
# (vx, vy, wz) per lane — float32-exact values so C# (float Vector3) and Python (float64) see identical commands
LANE_COMMANDS = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0),
                 (2.0, 0.0, 0.0), (2.5, 0.0, 0.0), (3.0, 0.0, 0.0), (1.75, 0.0, 0.25)]
# Rung 2 (C8): every new skill in one heat — stand, backward, crab both ways, spin both ways, sprint, walking turn
LANE_COMMANDS_RUNG2 = [(0.0, 0.0, 0.0), (-1.5, 0.0, 0.0), (0.0, 0.75, 0.0), (0.0, -0.75, 0.0),
                       (0.0, 0.0, 1.5), (0.0, 0.0, -1.5), (3.5, 0.0, 0.0), (1.0, 0.0, 1.0)]
SHOVE_DV = 0.3


def lane_script(onnx: Path, k: int, command, body: str | None = None) -> list[Disturbance]:
    dv = SHOVE_DV if body in (None, "matt") else round(SHOVE_DV * bodies.BODIES[body].speed_scale, 4)
    shove = Disturbance(40 + 5 * k, "shove", "root", dqvel=[0.0, dv if k % 2 == 0 else -dv, 0.0])
    cube_tick = 100 + 5 * k
    hit_tick = cube_tick + meet.CUBE_DROP_LOOKAHEAD_TICKS
    pre = rollout(onnx, seconds=(hit_tick + 1) * 0.02, disturbances=[shove], command=command, body=body)
    x, y = pre["frames"][hit_tick]["qpos"][0:2]
    cube = Disturbance(cube_tick, "cube", "cube0_free", qpos=[round(x + 0.05, 4), round(y + 0.15, 4), 3.0, 1.0, 0.0, 0.0, 0.0],
                       qvel=[0.0] * 6)
    return [shove, cube]


def main(onnx_arg: str, name: str, commands: str = "rung1") -> int:
    onnx = Path(onnx_arg)
    lanes = []
    for k, cmd in enumerate(LANE_COMMANDS_RUNG2 if commands == "rung2" else LANE_COMMANDS):
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


MIXED_SCENE = "meet8_mzmzmzmz"


def body_command(command, body: str) -> tuple[float, float, float]:
    """PolicyRunner.BodyCommand in float32: k = (float)(0.8 / gait_hz_base); (x·k, y·k, z / k). Identity for MATT."""
    if body == "matt":
        return tuple(float(c) for c in command)
    ct = json.loads(bodies.BODIES[body].contract_json.read_text())
    k = np.float32(0.8 / ct["gait_hz_base"])
    c = np.asarray(command, np.float32)
    return float(c[0] * k), float(c[1] * k), float(c[2] / k)


def main_mixed(name: str, matt_brain: str = "../parity/brains/rung2.onnx",
               zombie_brain: str = "../parity/brains/zombie_rung2.onnx") -> int:
    xml, layout_json = ROOT / "assets" / f"scene_{MIXED_SCENE}.xml", ROOT / "assets" / f"{MIXED_SCENE}_layout.json"
    body_of = meet.lane_bodies(layout_json)
    brains = {"matt": Path(matt_brain), "zombie": Path(zombie_brain)}
    lanes = []
    for k, cmd in enumerate(LANE_COMMANDS_RUNG2):
        body = body_of[k]
        bcmd = body_command(cmd, body)
        script = lane_script(brains[body], k, bcmd, body)
        ref = rollout(brains[body], seconds=SECONDS, disturbances=script, command=bcmd, body=body)
        write(ref, f"{name}_L{k}")
        lanes.append({"lane": k, "body": body, "brain": brains[body].name, "command": list(cmd), "body_command": list(bcmd),
                      "disturbances": [d.__dict__ for d in script], "reference": f"{name}_L{k}"})
        fz = body_fall_z(body)
        fell = next((f["t"] for f in ref["frames"] if f["qpos"][2] < fz), None)
        print(f"L{k} {body} cmd={cmd} body_cmd={tuple(round(c, 4) for c in bcmd)}: solo x_end={ref['frames'][-1]['qpos'][0]:.2f} m, fell={fell}")
    plan = {"schema": 2, "scene": xml.name, "layout": layout_json.name, "seconds": SECONDS, "lanes": lanes,
            "brains": {b: p.name for b, p in brains.items()},
            "onnx_sha256": {b: hashlib.sha256(p.read_bytes()).hexdigest() for b, p in brains.items()},
            "fingerprint_sha256": {b: bodies.BODIES[b].fingerprint_json.with_suffix(".sha256").read_text().strip() for b in brains},
            "brain": brains["matt"].name}
    (PARITY / f"g6_plan_{name}.json").write_text(json.dumps(plan, indent=1))

    # CPU meet pre-check: every lane of the mixed meet vs its body's solo run (the plan's body commands)
    cpu_plan = [dict(p, command=p["body_command"]) for p in lanes]
    runs = meet.rollout_meet(brains["matt"], cpu_plan, SECONDS, scene_xml=xml, layout_json=layout_json, brains=brains)
    report = {"brains": plan["brains"], "source": "python_meet", "scene": xml.name, "lanes": {}}
    for k, run in runs.items():
        meet.write_run(run, f"{name}_L{k}")
        ref = json.loads((PARITY / f"reference_trajectory_{name}_L{k}.json").read_text())
        r = g5_compare(ref, run, f"{name}_L{k}", f"meet_{name}_L{k}")
        report["lanes"][f"L{k}"] = r
        print(f"L{k} {body_of[k]}: meet vs solo  drift1s={r['qpos_max_abs_drift_1s']:.1e} drift5s={r['qpos_max_abs_drift_5s']:.1e} "
              f"height_rms={r['pelvis_height_rms_m']:.1e} torque={r['torque_ratio']:.4f} -> {'PASS' if r['G5_pass'] else 'FAIL'}")
    report["G6_pass"] = all(r["G5_pass"] for r in report["lanes"].values())
    (PARITY / f"g6_report_{name}_python.json").write_text(json.dumps(report, indent=1))
    print("G6 mixed (CPU meet pre-check)", "PASS" if report["G6_pass"] else "FAIL")
    return 0 if report["G6_pass"] else 1


if __name__ == "__main__":
    if sys.argv[1] == "mixed":
        raise SystemExit(main_mixed(*sys.argv[2:5]))
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "rung1"))
