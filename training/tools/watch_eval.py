"""Unattended checkpoint gate: wait for model_<it>.pt of a run, export it, run G1 (CPU MuJoCo) and append a summary line.

Usage: uv run python tools/watch_eval.py <run_dir> <name_prefix> --rung 2 [--every 250] [--last 1999] [--seeds 10]
Writes parity/watch_<name_prefix>.jsonl (one JSON object per checkpoint).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poolympic import evaluate as E  # noqa: E402

YAW_PROBE = ([0, 0, 1.0], [0, 0, -1.0], [0, 0, 2.2], [1.0, 0, 1.5], [2.0, 0, 0.3], [0, 1.0, 0])


def yaw_probe(onnx: Path) -> list[float]:
    sim = E.Sim(onnx)
    errs = []
    for cmd in YAW_PROBE:
        sim.reset()
        c = np.array(cmd, float)
        w = []
        for k in range(250):
            sim.control_tick(c)
            if k >= 75:
                w.append(E._vel_heading(sim)[2])
        errs.append(float(np.sqrt(np.mean((np.array(w) - c[2]) ** 2))))
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("prefix")
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--last", type=int, default=1999)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    out = ROOT.parent / "parity" / f"watch_{a.prefix}.jsonl"
    its = list(range(a.every, a.last, a.every)) + [a.last]
    for it in its:
        ckpt = a.run / f"model_{it}.pt"
        while not ckpt.exists():
            time.sleep(30)
        time.sleep(20)  # let the checkpoint finish writing
        name = f"{a.prefix}_it{it}"
        subprocess.run([sys.executable, str(ROOT / "tools" / "export_brain.py"), str(ckpt), name], cwd=ROOT,
                       capture_output=True)
        onnx = ROOT.parent / "parity" / "brains" / f"{name}.onnx"
        sim = E.Sim(onnx)
        if a.rung == 2:
            results = [E.rung2_episode(onnx, s, sim=sim) for s in range(1000, 1000 + a.seeds)]
            verdict = E.rung2_verdict(results)
            fails = {}
            for ck in verdict["per_seed_checks"]:
                for k, v in ck.items():
                    fails[k] = fails.get(k, 0) + (0 if v else 1)
            turn = [r.turntable_s for r in results]
            rec = {"it": it, "passed": verdict["passed_seeds"], "seeds": a.seeds, "fail_counts": fails,
                   "turntable_s": turn, "brake_m": [r.brake_m for r in results],
                   "yaw_probe": [round(x, 3) for x in yaw_probe(onnx)]}
        else:
            results = [E.rung1_episode(onnx, s, sim=sim) for s in range(1000, 1000 + a.seeds)]
            verdict = E.rung1_verdict(results)
            rec = {"it": it, "passed": verdict["passed_seeds"], "seeds": a.seeds}
        rec["yaw_probe_mean"] = round(float(np.mean(rec.get("yaw_probe", [0]))), 3)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
