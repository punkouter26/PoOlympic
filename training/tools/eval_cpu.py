"""C2 — Gate G1: CPU-MuJoCo evaluation of an exported brain against a rung's pass bar (10 seeds).

Usage: uv run python tools/eval_cpu.py <brain.onnx> --rung 0 [--seeds 10] [--out parity/eval_<name>.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import evaluate as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--rung", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.rung != 0:
        raise SystemExit("only rung 0 is implemented so far")
    sim = E.Sim(a.onnx)
    results = []
    for s in range(a.first_seed, a.first_seed + a.seeds):
        r = E.rung0_episode(a.onnx, s, sim=sim)
        results.append(r)
        print(f"seed {s}: fell={r.fell} ({r.fall_reason} @ {r.fall_time}) hits {r.recovered_hits}/{r.hits} "
              f"worst_recovery={r.worst_recovery_s} peak_tilt={r.max_tilt_after_hit_deg:.1f}deg foot_exc={r.max_foot_excursion_m:.3f} m jv_over={r.joint_vel_over_fraction:.3f}")
    verdict = E.rung0_verdict(results)
    report = {"onnx": str(a.onnx), **verdict, "episodes": [dataclasses.asdict(r) for r in results]}
    out = a.out or ROOT.parent / "parity" / f"eval_rung{a.rung}_{a.onnx.stem}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"G1 rung {a.rung}: {verdict['passed_seeds']}/{verdict['seeds']} seeds pass -> {'PASS' if verdict['PASS'] else 'FAIL'}  ({out.name})")
    return 0 if verdict["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
