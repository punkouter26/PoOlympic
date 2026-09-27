"""C6 — score the 8-lane meet (DESIGN.md §4).

  g0            G0 per lane: Python fingerprint of scene_meet8 lane k vs Unity's (parity/fingerprint_unity_meet8_L<k>.json),
                plus: every meet lane == the solo training athlete except its collision bits and pelvis origin.
  g6 <name>     G6: each Unity lane run (parity/unity_run_g6_<name>_L<k>.json) vs its solo CPU reference, G5 criteria
                -> parity/g6_report_<name>_unity.json

Usage: uv run python tools/compare_g6.py g0
       uv run python tools/compare_g6.py g6 <name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import fingerprint as F  # noqa: E402
from poolympic import meet  # noqa: E402
from poolympic.closed_loop import g5_compare  # noqa: E402
from poolympic.reference import PARITY  # noqa: E402


def solo_equivalent(fp: dict) -> dict:
    """Drop what legitimately differs between a meet lane and the solo athlete: collision bits, pelvis origin,
    and which pool slot the sampled cube is."""
    out = json.loads(json.dumps(fp))
    out["geoms"]["cube_geom"]["body"] = None
    for key, g in out["geoms"].items():
        if key not in ("ground", "cube_geom"):
            g["contype"] = g["conaffinity"] = None
    out["bodies"]["pelvis"]["pos"] = None
    return out


def g0() -> int:
    mm = mujoco.MjModel.from_xml_path(str(meet.MEET_XML))
    solo = F.fingerprint(mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "scene_matt.xml")))
    ok = True
    for lane in meet.load_layout():
        py = F.fingerprint(mm, lane.prefix, f"cube{lane.cubes[0]}")
        uni = json.loads((PARITY / f"fingerprint_unity_meet8_L{lane.lane}.json").read_text())
        errs = F.compare(F.normalize_for_compare(py), F.normalize_for_compare(uni))
        same = F.compare(solo_equivalent(solo), solo_equivalent(py))
        bits = {g["contype"] for key, g in py["geoms"].items() if key not in ("ground", "cube_geom")}
        print(f"L{lane.lane}: G0 {'PASS' if not errs else 'FAIL'} ({len(errs)} mismatches)  "
              f"== solo athlete: {'yes' if not same else f'NO ({len(same)})'}  contype bits {sorted(bits)}")
        for e in (errs + same)[:10]:
            print("   ", e)
        ok &= not errs and not same
    print("G0 meet8", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def g6(name: str) -> int:
    plan = json.loads((PARITY / f"g6_plan_{name}.json").read_text())
    report = {"brain": plan["brain"], "source": "unity", "lanes": {}}
    for pl in plan["lanes"]:
        k = pl["lane"]
        ref = json.loads((PARITY / f"reference_trajectory_{pl['reference']}.json").read_text())
        run = json.loads((PARITY / f"unity_run_g6_{name}_L{k}.json").read_text())
        r = g5_compare(ref, run, pl["reference"], f"g6_{name}_L{k}")
        report["lanes"][f"L{k}"] = r
        fails = [c for c, v in r["checks"].items() if not v]
        print(f"L{k} cmd={pl['command']}: drift1s={r['qpos_max_abs_drift_1s']:.1e} drift5s={r['qpos_max_abs_drift_5s']:.1e} "
              f"cube={r['cube_qpos_max_abs_drift_after_fire']:.1e} height_rms={r['pelvis_height_rms_m']:.1e} "
              f"torque={r['torque_ratio']:.4f} fall={r['fall_time_ref']}/{r['fall_time_unity']} "
              f"-> {'PASS' if r['G5_pass'] else 'FAIL ' + ','.join(fails)}")
    report["G6_pass"] = all(r["G5_pass"] for r in report["lanes"].values())
    (PARITY / f"g6_report_{name}_unity.json").write_text(json.dumps(report, indent=1))
    print("G6", "PASS" if report["G6_pass"] else "FAIL")
    return 0 if report["G6_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(g0() if sys.argv[1] == "g0" else g6(sys.argv[2]))
