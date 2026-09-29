"""C6 — score the 8-lane meet (DESIGN.md §4).

  g0            G0 per lane: Python fingerprint of scene_meet8 lane k vs Unity's (parity/fingerprint_unity_meet8_L<k>.json),
                plus: every meet lane == the solo training athlete except its collision bits and pelvis origin.
  g6 <name>     G6: each Unity lane run (parity/unity_run_g6_<name>_L<k>.json) vs its solo CPU reference, G5 criteria
                -> parity/g6_report_<name>_unity.json

Usage: uv run python tools/compare_g6.py g0 [meet8|pedestal8|meet8_mzmzmzmz|…]
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
    out["geoms"]["ground"]["pos"] = None       # the pedestal scenes lower the ground to -0.5 m
    # lane-owned props (event 5 shaker platform: body + geom + x/y slide joints, prefixed like the athlete) are not part
    # of the solo athlete; G0 still compares them Python vs Unity
    for kind in ("bodies", "geoms", "joints"):
        for key in [k for k in out.get(kind, {}) if k.startswith("shaker")]:
            del out[kind][key]
            count = {"bodies": "athlete_bodies", "joints": "athlete_joints"}.get(kind)
            if count and kind == "bodies":
                out["counts"][count] -= 1
    if "counts" in out:
        out["counts"]["athlete_joints"] -= sum(1 for k in fp.get("joints", {}) if k.startswith("shaker"))
    return out


def g0(scene: str = "meet8") -> int:
    """scene: meet8 (scene_meet8.xml), pedestal8, … or a mixed meet (meet8_mzmzmzmz: lanes carry "body"); Unity dumps
    fingerprint_unity_<scene>_L<k>.json. Each lane is also checked against the solo scene of its own body."""
    xml = ROOT / "assets" / f"scene_{scene}.xml"
    layout = json.loads((ROOT / "assets" / f"{scene}_layout.json").read_text())
    lanes = [meet.Lane(l["lane"], l["prefix"], None, tuple(l["cubes"])) for l in layout["lanes"]]
    body_of = {l["lane"]: l.get("body", "matt") for l in layout["lanes"]}
    mm = mujoco.MjModel.from_xml_path(str(xml))
    solos = {b: F.fingerprint(mujoco.MjModel.from_xml_path(str(ROOT / "assets" / f"scene_{b}.xml"))) for b in set(body_of.values())}
    ok = True
    for lane in lanes:
        solo = solos[body_of[lane.lane]]
        py = F.fingerprint(mm, lane.prefix, f"cube{lane.cubes[0]}")
        uni = json.loads((PARITY / f"fingerprint_unity_{scene}_L{lane.lane}.json").read_text())
        errs = F.compare(F.normalize_for_compare(py), F.normalize_for_compare(uni))
        same = F.compare(solo_equivalent(solo), solo_equivalent(py))
        bits = {g["contype"] for key, g in py["geoms"].items() if key not in ("ground", "cube_geom")}
        print(f"L{lane.lane} {body_of[lane.lane]}: G0 {'PASS' if not errs else 'FAIL'} ({len(errs)} mismatches)  "
              f"== solo athlete: {'yes' if not same else f'NO ({len(same)})'}  contype bits {sorted(bits)}")
        for e in (errs + same)[:10]:
            print("   ", e)
        ok &= not errs and not same
    print(f"G0 {scene}", "PASS" if ok else "FAIL")
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
    raise SystemExit(g0(sys.argv[2] if len(sys.argv) > 2 else "meet8") if sys.argv[1] == "g0" else g6(sys.argv[2]))
