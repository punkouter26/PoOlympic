"""C2 — Gate G1: CPU-MuJoCo evaluation of an exported brain against a rung's pass bar (10 seeds).

Usage: uv run python tools/eval_cpu.py <brain.onnx> --rung 0|1|2|S [--seeds 10] [--out parity/eval_<name>.json]
       (S = Rung S stance-skill drills for events 2/3/4/6/7, contract v4 brain: poolympic/evaluate_stance.py)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


def _jsonable(o):
    """numpy scalars / arrays in the reports -> plain JSON."""
    return o.tolist() if hasattr(o, "tolist") else str(o)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import evaluate as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--rung", default="0", help="0, 1, 2 or S")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if str(a.rung).upper() == "S":
        return stance(a)
    a.rung = int(a.rung)
    sim = E.Sim(a.onnx)
    if a.rung == 2:
        results = []
        for s in range(a.first_seed, a.first_seed + a.seeds):
            r = E.rung2_episode(a.onnx, s, sim=sim)
            results.append(r)
            fails = [k for k, v in E.rung2_checks(r).items() if not v]
            seg = " ".join(f"{x['kind'][:2]}:{x['lin_rms'] or float('nan'):.2f}/{x['yaw_rms'] or float('nan'):.2f}"
                           for x in r.segments)
            print(f"seed {s}: fell={r.fell} | lin/yaw {seg} | turn {r.turntable_s} s drift {r.turntable_drift_m:.2f} m"
                  f" | brake {r.brake_m} m | back {r.backward_m:.1f} m | fails {fails}")
        verdict = E.rung2_verdict(results)
        report = {"onnx": str(a.onnx), **verdict, "episodes": [dataclasses.asdict(r) for r in results]}
        out = a.out or ROOT.parent / "parity" / f"eval_rung2_{a.onnx.stem}.json"
        out.write_text(json.dumps(report, indent=1, default=_jsonable))
        print(f"G1 rung 2: {verdict['passed_seeds']}/{verdict['seeds']} seeds pass -> "
              f"{'PASS' if verdict['PASS'] else 'FAIL'}  ({out.name})")
        return 0 if verdict["PASS"] else 1
    if a.rung == 1:
        results = []
        for s in range(a.first_seed, a.first_seed + a.seeds):
            r = E.rung1_episode(a.onnx, s, sim=sim)
            results.append(r)
            print(f"seed {s}: v_cmd={r.speed_cmd:.2f} finished={r.finished} fell={r.fell} ({r.fall_reason} @ {r.fall_time}) "
                  f"dist={r.distance_m:.1f} m vel_rms_err={r.vel_rms_err:.3f} lateral={r.lateral_drift_m:.2f} m jv_over={r.joint_vel_over_fraction:.3f}")
        verdict = E.rung1_verdict(results)
        report = {"onnx": str(a.onnx), **verdict, "episodes": [dataclasses.asdict(r) for r in results]}
        out = a.out or ROOT.parent / "parity" / f"eval_rung1_{a.onnx.stem}.json"
        out.write_text(json.dumps(report, indent=1, default=_jsonable))
        print(f"G1 rung 1: {verdict['passed_seeds']}/{verdict['seeds']} seeds pass -> {'PASS' if verdict['PASS'] else 'FAIL'}  ({out.name})")
        return 0 if verdict["PASS"] else 1
    results = []
    for s in range(a.first_seed, a.first_seed + a.seeds):
        r = E.rung0_episode(a.onnx, s, sim=sim)
        results.append(r)
        print(f"seed {s}: fell={r.fell} ({r.fall_reason} @ {r.fall_time}) hits {r.recovered_hits}/{r.hits} "
              f"worst_recovery={r.worst_recovery_s} peak_tilt={r.max_tilt_after_hit_deg:.1f}deg foot_exc={r.max_foot_excursion_m:.3f} m jv_over={r.joint_vel_over_fraction:.3f}")
    verdict = E.rung0_verdict(results)
    report = {"onnx": str(a.onnx), **verdict, "episodes": [dataclasses.asdict(r) for r in results]}
    out = a.out or ROOT.parent / "parity" / f"eval_rung{a.rung}_{a.onnx.stem}.json"
    out.write_text(json.dumps(report, indent=1, default=_jsonable))
    print(f"G1 rung {a.rung}: {verdict['passed_seeds']}/{verdict['seeds']} seeds pass -> {'PASS' if verdict['PASS'] else 'FAIL'}  ({out.name})")
    return 0 if verdict["PASS"] else 1


def stance(a) -> int:
    from poolympic import evaluate_stance as ES
    sim = ES.SkillSim(a.onnx)
    results = []
    for s in range(a.first_seed, a.first_seed + a.seeds):
        r = ES.stance_episode(a.onnx, s, sim=sim)
        results.append(r)
        print(f"seed {s}: " + " | ".join(f"{n} {'ok' if v['pass'] else 'FAIL'} "
                                         + ",".join(f"{k}={x}" for k, x in v.items() if k != "pass") for n, v in r.drills.items()))
    verdict = ES.stance_verdict(results)
    report = {"onnx": str(a.onnx), **verdict, "episodes": [ES.to_json(r) for r in results]}
    out = a.out or ROOT.parent / "parity" / f"eval_rungS_{a.onnx.stem}.json"
    out.write_text(json.dumps(report, indent=1, default=_jsonable))
    print(f"G1 rung S: {verdict['passed_seeds']}/{verdict['seeds']} seeds pass, per drill {verdict['per_drill']} -> "
          f"{'PASS' if verdict['PASS'] else 'FAIL'}  ({out.name})")
    return 0 if verdict["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
